# GKE·KFP dispatcher 운영 실행서

## 원칙

이 문서는 보존 중인 GKE Standard/KFP 2.17.0 환경에서 prebuilt Bundle을 받는 범용 CPU pipeline을
실행하는 절차다. 실제 구조는 [architecture](architecture.ko.md), 7단계 CIFAR-10 legacy production
실행의 immutable 증거는 [run record](runs/2026-09-20-dispatcher-gke.ko.md)를 따른다.

- 모든 cloud 입력은 GCS URI와 SHA-256으로 고정한다.
- Component image는 Artifact Registry manifest digest로 고정한다.
- KFP API/UI는 port-forward로만 접근한다.
- Raw/Run bucket, experiment prefix와 KSA/GSA 권한을 실행별로 분리한다.
- 현재 cluster/PVC는 보존 대상이다. 명시적 종료 요청 없이는 teardown하지 않는다.
- `terraform apply`, `kubectl apply`, IAM 변경과 run 제출 전에는 직전 read-only gate를 통과한다.

## 고정 값

```bash
PROJECT_ID=pertinence-509116
REGION=asia-northeast3
ZONE=asia-northeast3-a
CLUSTER=pertinence-dispatcher-260919-1708
NAMESPACE=kubeflow
KSA=pipeline-runner
RAW_BUCKET=pertinence-509116-pertinence-raw
RUN_BUCKET=pertinence-509116-pertinence-runs
```

실행마다 충돌하지 않는 `EXPERIMENT_PREFIX`를 만들고 동일 URI의 object를 덮어쓰지 않는다.

## 1. Read-only inventory

작업 시작 전에 project/account와 kube context를 사람이 확인하고 실제 cluster를 조회한다.

```bash
gcloud container clusters describe "$CLUSTER" \
  --project "$PROJECT_ID" --zone "$ZONE"
gcloud container node-pools list \
  --cluster "$CLUSTER" --project "$PROJECT_ID" --zone "$ZONE"

kubectl config current-context
kubectl get nodes -L cloud.google.com/gke-nodepool
kubectl get deployments,pvc -n "$NAMESPACE"
```

필수 gate:

- cluster RUNNING, context의 project/zone/name 일치;
- `system-pool` Ready 1대, `pipeline-pool` autoscaling 0–1;
- KFP deployment 14개 Available;
- MySQL과 SeaweedFS 20 GiB PVC 2개 Bound;
- public LoadBalancer/Ingress 없음.

## 2. Terraform drift와 보호 확인

KFP pipeline 반복 실행에는 baseline인 `training-current.tfvars`를 사용한다. Online benchmark를 위해
두 pool을 추가하는 `benchmark-current.tfvars`는 이 절차가 아니라
[benchmark Plan](benchmark/Plan.md)과
[KServe benchmark 실행 참고서](kserve-benchmark-execution.ko.md)의 saved-plan guard를 따른다.

```bash
terraform -chdir=cloud/iac/environments/dispatcher-current init \
  -backend-config=backend.hcl
terraform -chdir=cloud/iac/environments/dispatcher-current validate
terraform -chdir=cloud/iac/environments/dispatcher-current plan \
  -var-file=profiles/training-current.tfvars \
  -detailed-exitcode
```

정상 상태의 exit code는 0이다. 변경이 보이면 원인을 설명하고 saved plan을 guard로 검사하기 전까지
apply하지 않는다. `benchmark.tfvars.example`, `full.tfvars.example`,
`compute-only.tfvars.example`은 미적용 예제이며 현재 `dispatcher-current` state에 넣지 않는다.

현재 cluster와 bootstrap root에서 `terraform destroy`, `apply -auto-approve`, `-target`과 state 직접
편집을 사용하지 않는다. Pool 제거는 deletion policy를 해제하는 별도 검토와 실제 제거 검토의 두
단계를 요구한다.

## 3. KFP와 Workload Identity gate

Upstream KFP가 만든 `kubeflow/pipeline-runner`와 RBAC가 존재해야 한다. 저장소의
`cloud/kfp/service-account.yaml`은 이름/namespace 계약의 참조본이며 기존 SA를 대체하는 용도로
적용하지 않는다.

```bash
kubectl get serviceaccount pipeline-runner -n "$NAMESPACE"
kubectl auth can-i get pods \
  --as="system:serviceaccount:${NAMESPACE}:${KSA}" -n "$NAMESPACE"
```

Launcher의 GCS pipeline root는 저장소 manifest를 수정하지 않고 placeholder만 stream에서
치환해 적용한다.

```bash
sed "s/REPLACE_WITH_RUN_BUCKET/${RUN_BUCKET}/g" \
  cloud/kfp/kfp-launcher-configmap.yaml | kubectl apply -f -
```

KFP launcher는 외부 artifact 처리 전 bucket resource에 `storage.objects.list`를 요청한다. Experiment
GSA의 condition은 bucket resource 자체의 list와 지정 experiment object prefix read/write를 함께
허용해야 한다. Bucket 전체 object 권한으로 확대하지 않는다.

Pod credential smoke는 같은 KSA로 작은 test object의 list/read/write를 확인한 뒤 즉시 별도 prefix에
남기거나 정리한다. JSON service-account key는 생성하거나 mount하지 않는다.

## 4. Bundle input 준비

로컬 asset manifest 검증을 먼저 통과한다.

```bash
.venv/bin/python scripts/prepare_assets.py \
  --manifest experiments/cifar10/configs/assets.yaml \
  --artifacts-dir artifacts/cifar10 \
  --verify-only
```

Dataset adapter와 model adapter를 로컬 또는 별도 Job에서 실행해 다음 형태를 먼저 만든다.

```text
dataset-bundle/
├── dataset.yaml
├── train/part-*.parquet
└── evaluation/part-*.parquet

expert-bundle/
├── experts.yaml
└── models/<expert-id>/model.onnx
```

이미 Parquet/ONNX source가 준비됐다면 `package-dataset`과 `package-experts`로 manifest와 content
digest를 봉인한다. CIFAR-10 raw archive와 기존 checkpoint를 변환할 때만 보존된 legacy adapter
명령을 사용한다.

```bash
.venv/bin/pertinence package-dataset \
  --source artifacts/cifar10/portable/dataset-source \
  --output artifacts/cifar10/portable/dataset-bundle
.venv/bin/pertinence package-experts \
  --source artifacts/cifar10/portable/expert-source \
  --output artifacts/cifar10/portable/expert-bundle

DATASET_SHA256=$(.venv/bin/pertinence artifact-sha256 \
  artifacts/cifar10/portable/dataset-bundle)
EXPERT_SHA256=$(.venv/bin/pertinence artifact-sha256 \
  artifacts/cifar10/portable/expert-bundle)
RUN_CONFIG_SHA256=$(.venv/bin/pertinence artifact-sha256 \
  experiments/cifar10/configs/pipeline-run.yaml)
```

각 입력은 digest를 포함한 새 prefix에 no-overwrite 방식으로 업로드한다. DatasetBundle과
ExpertBundle의 manifest가 각각 다음 URI 바로 아래에 있어야 한다.

```text
gs://${RAW_BUCKET}/${EXPERIMENT_PREFIX}/datasets/${DATASET_SHA256}/dataset.yaml
gs://${RAW_BUCKET}/${EXPERIMENT_PREFIX}/experts/${EXPERT_SHA256}/experts.yaml
gs://${RAW_BUCKET}/${EXPERIMENT_PREFIX}/configs/${RUN_CONFIG_SHA256}/run.yaml
```

각 local digest, GCS generation/CRC32C와 최종 URI를 execution manifest에 기록한다. 같은 object나
prefix를 overwrite하지 않으며 이름만 같다는 이유로 immutable하다고 간주하지 않는다.

## 5. Component image

`pipelines/pertinence/cloudbuild.yaml`로 `linux/amd64` image를 build하고 결과 tag에서 manifest
digest를 조회한다. Production IR에는 tag가 아닌 다음 형식만 허용한다.

```text
asia-northeast3-docker.pkg.dev/pertinence-509116/pertinence/components@sha256:<digest>
```

Build 전에 전체 test, Ruff와 container runtime smoke를 통과한다. Build identity, source Git SHA,
dirty patch SHA, Cloud Build ID와 image digest를 execution manifest에 남긴다. 임시 build 권한은
완료 후 회수한다.

## 6. Compile과 제출

Compiler는 `kfp==2.17.0`과 `kfp-kubernetes==2.17.x` lock을 사용한다.

```bash
.venv/bin/python -m pip install -r pipelines/pertinence/requirements-kfp.lock
.venv/bin/python -m pipelines.pertinence.pipeline \
  --image "$COMPONENT_IMAGE" \
  --pipeline-root "gs://${RUN_BUCKET}/pertinence/runs" \
  --cpu-threads 4 \
  --output "artifacts/cifar10/portable/${EXPERIMENT_PREFIX}-generic-pipeline.yaml"
```

Compile test에서 다음을 확인한다.

- container executor 정확히 5개, importer를 포함한 전체 task 8개;
- 모든 image가 같은 digest로 고정;
- executor 5개에만 `pipeline-pool` selector+toleration 적용;
- GPU request 없음;
- search input에 final artifact/URI/volume 없음;
- `train-dispatcher` deadline 72시간, retry 0;
- registration cache off.

API server를 local `8888`로 port-forward하고 `pipelines.pertinence.submit`에 IR, pipeline root,
namespace, `pipeline-runner`, 세 Bundle/config URI와 SHA를 전달한다.

```bash
.venv/bin/python -m pipelines.pertinence.submit \
  --host http://127.0.0.1:8888 \
  --pipeline "artifacts/cifar10/portable/${EXPERIMENT_PREFIX}-generic-pipeline.yaml" \
  --pipeline-root "gs://${RUN_BUCKET}/pertinence/runs" \
  --namespace "$NAMESPACE" \
  --service-account "$KSA" \
  --dataset-bundle-uri \
    "gs://${RAW_BUCKET}/${EXPERIMENT_PREFIX}/datasets/${DATASET_SHA256}" \
  --dataset-bundle-sha256 "$DATASET_SHA256" \
  --expert-bundle-uri \
    "gs://${RAW_BUCKET}/${EXPERIMENT_PREFIX}/experts/${EXPERT_SHA256}" \
  --expert-bundle-sha256 "$EXPERT_SHA256" \
  --run-config-uri \
    "gs://${RAW_BUCKET}/${EXPERIMENT_PREFIX}/configs/${RUN_CONFIG_SHA256}/run.yaml" \
  --run-config-sha256 "$RUN_CONFIG_SHA256"
```

Production은 기본적으로 cache off다.
실패 복구 cache는 같은 immutable input과 image를 재사용할 때 compile과 submit 양쪽에서만 켠다.

제출 직후 experiment ID, KFP run ID, Argo Workflow, IR SHA와 pipeline root를 execution manifest에
추가한다.

## 7. 실행 중 관찰

```bash
kubectl get pods -n "$NAMESPACE" -o wide
kubectl get nodes -L cloud.google.com/gke-nodepool
kubectl get events -n "$NAMESPACE" --sort-by=.lastTimestamp
```

Container executor가 system pool에 배치되거나 importer가 raw byte 연산을 수행하면 중단한다.
`train-dispatcher`는 generation resume과 자동 retry를 지원하지 않으므로 최대 72시간과 비용 한도를
넘기기 전에 운영자가 run을 평가한다. Search log에서 generation/evaluation 진행률과 finite loss를
확인하되 final artifact는 열지 않는다.

## 8. 완료 gate와 보존

Run 성공만으로 완료하지 않는다.

1. KFP run과 generic container component 5개 모두 Succeeded;
2. input SHA, contract와 split/cache fingerprint 일치;
3. 50×50 NSGA-II와 retained checkpoint 전체 복원 완료;
4. final metric finite, confusion/sample 합계 일치;
5. registration report와 SQLite row/idempotency key 일치;
6. IR, run ID, Git/patch SHA, image digest, GKE/KFP version 기록;
7. GCS에서 artifact와 diagnostics를 다시 읽을 수 있음.

범용화 release gate에는 동일 IR로 synthetic 2-expert와 5-expert run 성공, 기존 CIFAR-10 Bundle
3-expert 결과 동등성, 잘못된 digest가 validate에서 실패하고 downstream task가 시작되지 않는다는
증거를 추가한다. Legacy raw-input 재현이 필요하면 `pipelines.pertinence.cifar10_pipeline`과
`pipelines.pertinence.cifar10_submit`을 사용하며 기존 7개 component gate를 적용한다.

성공/실패 run 모두 workflow, Pod, event, node, PVC/PV, deployment, network inventory와 relevant log를
run-scoped diagnostics prefix에 보존한다. KFP metadata와 PVC는 GCS artifact가 아니므로 cluster 종료
전에 별도 export가 필요하다.

Run 종료 후 `pipeline-pool`이 0대로 돌아오는지 확인한다. 현재 정책은 cluster, KFP namespace,
PVC/PV, Workload Identity binding과 두 node pool을 보존하는 것이다.

## 9. 알려진 실패 경로

- Launcher가 artifact download 전에 403: bucket-level list와 object-prefix condition을 함께 확인한다.
- Python 3.11/protobuf 6 ONNX descriptor 오류: `protobuf==6.33.5`와 modern `is_repeated` 경로가 포함된
  현재 image를 사용한다.
- Pending executor: node selector, toleration, quota와 `pipeline-pool` autoscaler event를 확인한다.
- Final/registration 재시도: 성공한 immutable upstream artifact를 확인하고 해당 stage 정책 안에서
  수행하며 search를 임의로 다시 학습하지 않는다.

## 10. Teardown

Teardown은 현재 보류 상태다. 사용자가 명시적으로 종료를 요청한 경우에만 별도 계획을 작성하고,
결과/metadata export 확인 → KFP PVC/PV와 disk 소멸 확인 → experiment IAM 정리 → node pool/cluster
보호 해제 및 삭제 → forwarding rule/disk 잔여 확인 순서로 수행한다. Cluster 삭제와 GCS bucket,
Artifact Registry digest 삭제는 별도 의사결정이다.
