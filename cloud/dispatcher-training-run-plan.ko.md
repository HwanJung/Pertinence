# Dispatcher 학습용 GKE/KFP 실행 계획

## 1. 목표와 범위

이번 실행은 CIFAR-10 입력과 세 expert checkpoint를 GCS에 올리고, 임시 GKE Standard
cluster에 OSS Kubeflow Pipelines(KFP)를 설치해 dispatcher를 학습·평가하는 것을 목표로 한다.
실행 결과와 재현성 자료는 GCS 및 Artifact Registry에 보존하고, 성공 또는 실패에 필요한 증거를
수집한 뒤 cluster와 잔여 compute resource를 제거한다.

이번 실행에 포함되는 범위는 다음과 같다.

- CIFAR-10 archive, expert checkpoint, RunConfig 검증 및 GCS 업로드
- CPU component image build와 Artifact Registry 등록
- Workload Identity Federation이 활성화된 GKE Standard cluster 생성
- OSS KFP 2.17.0 standalone 설치와 GCS pipeline root 구성
- Dispatcher 학습, Pareto 탐색, final 평가, run-scoped registration
- 결과와 실행 lineage 검증 및 보존
- Cluster, node, KFP persistent disk와 기타 잔여 compute resource 제거

다음 항목은 이번 실행에서 제외한다.

- KServe 설치
- Dispatcher 또는 expert serving 배포
- Serving artifact promotion
- Serving 및 load-generator node pool
- Load generation, latency, throughput benchmark
- Public ingress 또는 `LoadBalancer` endpoint

`register-dispatchers`는 serving 단계가 아니라 평가된 dispatcher를 run-scoped artifact로 만드는
offline 단계이므로 실행 범위에 포함한다.

## 2. 목표 구성

| 항목 | 값 |
|---|---|
| Google Cloud project | `pertinence-509116` |
| Region / zone | `asia-northeast3` / `asia-northeast3-a` |
| GKE mode | Standard, zonal |
| Release channel | Regular |
| KFP | OSS KFP `2.17.0` standalone |
| Pipeline SDK | `kfp==2.17.0` |
| 실행 장치 | CPU only |
| `system-pool` | `e2-standard-4`, 1 node |
| `pipeline-pool` | `n2-standard-8`, autoscaling 0–1 |
| Node boot disk | `pd-standard`, 50 GiB (두 pool 모두) |
| Pipeline namespace / KSA | `kubeflow` / `pipeline-runner` |
| 영속 저장소 | Raw GCS bucket, run GCS bucket, Artifact Registry |
| 외부 노출 | 없음; KFP API와 UI는 `kubectl port-forward`만 사용 |

```text
로컬 입력 ──► GCS raw bucket ──┐
코드 ──► Cloud Build ──► AR digest ─┼─► GKE/KFP ─► dispatcher 학습·평가
                                   │                         │
                                   └───────────────► GCS run bucket
                                                              │
                                                   결과·진단 검증 및 보존
                                                              │
                                                   GKE와 잔여 disk 삭제
```

Raw bucket과 run bucket 이름은 전역 유일성을 확인한 뒤 정한다. 기본 후보는
`pertinence-509116-pertinence-raw`와 `pertinence-509116-pertinence-runs`다. Serving 및 benchmark
bucket은 이번 실행에서 만들지 않는다.

## 3. 사전 준비

### 3.1 저장소와 입력 고정

1. 실행할 worktree를 확정하고 Git SHA를 기록한다.
2. Worktree가 clean하지 않다면 commit SHA만 기록하지 말고 diff 또는 patch의 SHA-256도 함께
   보존한다.
3. 다음 검사를 모두 통과시킨다.

   ```bash
   .venv/bin/python -m pytest
   .venv/bin/ruff check .
   ```

4. `scripts/prepare_assets.py --manifest experiments/cifar10/configs/assets.yaml
   --artifacts-dir artifacts --verify-only`로 CIFAR-10 archive와 checkpoint 원본 hash를 확인한다.
5. 다음 네 입력과 RunConfig의 SHA-256을 별도 실행 manifest에 기록한다.

   - `cifar-10-python.tar.gz`
   - ShuffleNetV2 x0.5 checkpoint
   - MobileNetV2 x0.5 checkpoint
   - ResNet56 checkpoint
   - `experiments/cifar10/configs/pipeline-run.yaml`

현재 pipeline이 기대하는 고정 asset hash는
[`pipelines/pertinence/docs/gke-cpu.ko.md`](../pipelines/pertinence/docs/gke-cpu.ko.md)에 정의돼 있다.

### 3.2 Pipeline scheduling 보완

현재 pipeline은 각 task의 CPU와 memory request/limit은 지정하지만 `pipeline-pool`을 선택하는
node selector와 taint toleration은 넣지 않는다. Cluster 생성 전에 다음을 구현하고 compile test로
검증한다.

1. Compiler dependency에 KFP SDK와 호환되는 `kfp-kubernetes==2.17.x`를 고정한다.
2. 일곱 component task 모두에 다음 scheduling 계약을 적용한다.

   ```text
   node selector: cloud.google.com/gke-nodepool=pipeline-pool
   toleration:    workload=pertinence-pipeline:NoSchedule
   ```

3. 생성된 IR의 일곱 component executor에 두 설정이 들어 있는지 자동 검사한다.
4. KFP controller에는 해당 toleration을 주지 않아 `system-pool`에 남도록 한다.

KFP 2.17.0은 importer executor의 Kubernetes platform 설정을 실제 importer Pod spec에 적용하지
않는다. 현재 importer는 `reimport=False`, `downloadToWorkspace=False`로 외부 artifact URI만 MLMD에
등록하고 GCS byte를 읽지 않으므로, 다섯 importer와 KFP driver/root Pod는 `system-pool`에서
실행하도록 허용한다. 실제 dataset/checkpoint I/O와 연산을 수행하는 일곱 container component만
`pipeline-pool`에 강제한다.

KFP 2.17 generic 설치의 `cacheNodeRestrictions=false` 기본값은 유지한다. 따라서 cache hit 때 실제
component 대신 실행되는 경량 dummy Pod는 system pool에서 실행될 수 있다. Scale-up을 유발하는
것보다 효율적이며, 배치 검증은 uncached 실제 component executor Pod 7개에 적용한다.

### 3.3 Cloud readiness 확인

Resource를 만들기 전에 다음을 읽기 전용으로 확인한다.

- 활성 gcloud 계정과 project가 올바른지
- Billing이 활성화돼 있는지
- GKE, Artifact Registry, Cloud Build, Cloud Storage, IAM Credentials API 상태
- `asia-northeast3`의 CPU, external IP, persistent disk quota
- 같은 이름의 cluster, bucket, Artifact Registry repository가 이미 존재하는지
- Cluster와 IAM을 만들고 bucket IAM을 변경할 운영자 권한이 있는지

Node에는 Compute Engine 기본 service account 대신 전용 최소권한 service account를 사용한다.
최소한 GKE node 기본 역할과 Artifact Registry read 권한이 필요하다.

## 4. GCS와 component image 준비

### 4.1 Bucket 생성과 IAM

Raw bucket과 run bucket은 `asia-northeast3`에 생성하고 다음 정책을 적용한다.

- Uniform bucket-level access
- Public access prevention
- Object versioning
- 실험 ID별 immutable prefix
- 삭제 복구 절차가 확인되기 전에는 retention lock을 설정하지 않음

KFP 설치 후 다음 Workload Identity principal을 사용한다.

```text
principal://iam.googleapis.com/projects/<PROJECT_NUMBER>/locations/global/
workloadIdentityPools/pertinence-509116.svc.id.goog/subject/ns/kubeflow/sa/pipeline-runner
```

권한은 다음처럼 제한한다. KFP 2.17 launcher는 외부 artifact URI를 내려받을 때 객체를 직접
`GET`하기 전에 `storage.objects.list`를 bucket resource에 요청한다. 따라서 IAM condition은
`resource.name == <bucket-resource>` 또는 `resource.name.startsWith(<experiment-object-prefix>)`인
복합식으로 구성한다. 이 방식은 bucket 목록 조회는 허용하지만 객체 read/write는 해당 experiment
prefix로 제한한다. Binding은 teardown에서 정확히 회수한다.

| 대상 | 권한 |
|---|---|
| Raw bucket | `roles/storage.objectViewer` |
| Run bucket | `roles/storage.objectUser` |

JSON service-account key를 생성하거나 Pod에 mount하지 않는다.

### 4.2 입력 업로드

모든 입력은 덮어쓰지 않는 experiment prefix에 업로드한다.

```text
gs://<RAW_BUCKET>/pertinence/inputs/<EXPERIMENT_ID>/
├── cifar-10-python.tar.gz
├── cifar10_shufflenetv2_x0_5.pt
├── cifar10_mobilenetv2_x0_5.pt
├── cifar10_resnet56.pt
└── pipeline-run.yaml
```

업로드 뒤 GCS에서 다시 내려받지 않고 object metadata와 pipeline 입력 SHA-256을 기록한다. Pipeline의
첫 단계와 validation 단계가 실제 byte hash를 다시 검증하도록 한다.

### 4.3 Image build

`pipelines/pertinence/cloudbuild.yaml`로 component image를 Cloud Build에서 `linux/amd64`로 만들고
`asia-northeast3`의 Artifact Registry repository `pertinence`에 push한다.

신규 프로젝트의 기본 Cloud Build identity는 권한이 없는 Compute 기본 SA이므로 이를 승격하지
않는다. 전용 `pertinence-cloud-build` IAM SA를 사용하고 `pertinence` repository writer만
부여한다. Custom build SA는 source/log bucket 자체 권한을 요구하므로 raw/run artifact와 격리된
임시 build bucket을 만든다. 이 SA에는 해당 bucket에만 Storage Admin을 부여하며, build source와
GCS-only log는 `cloud-build/<EXPERIMENT_ID>/` 아래에 둔다. Image push와 build evidence 확인 뒤 두
권한을 모두 회수한다.

실행에 사용하는 값은 mutable tag가 아니라 다음 형태의 manifest digest여야 한다.

```text
asia-northeast3-docker.pkg.dev/pertinence-509116/pertinence/components@sha256:<digest>
```

Git SHA, Cloud Build ID, image digest를 실행 manifest에 기록한다.

## 5. GKE와 KFP 생성

### 5.1 Cluster와 node pool

1. Workload Identity Federation과 VPC-native networking을 활성화한 zonal Standard cluster를 만든다.
2. gcloud stable은 초기 node pool 이름을 지정하지 못하므로 동일한 최소 사양의 임시
   `default-pool` 한 대로 cluster를 만든다. 즉시 `system-pool` (`e2-standard-4`, 1 node)을
   생성하고 system Pod 정상 배치 확인 뒤 `default-pool`을 삭제한다.
3. `pipeline-pool`은 다음 조건으로 만든다.

   - Machine type: `n2-standard-8`
   - Initial/min/max: 0/0/1
   - On-demand VM
   - Workload metadata: `GKE_METADATA`
   - Taint: `workload=pertinence-pipeline:NoSchedule`
   - Boot disk: `pd-standard`, 50 GiB

4. Serving 및 loadgen pool은 만들지 않는다.
5. Node와 control-plane version, machine type, node service account를 실행 manifest에 기록한다.

프로젝트의 `CPUS_ALL_REGIONS` quota가 12 vCPU이므로 이번 실행의 `pipeline-pool` 상한은 1이다.
`system-pool`의 4 vCPU와 합쳐 quota를 정확히 12 vCPU 사용한다. 시작 시 병렬인 두 task의 요청
합계가 6 CPU/12 GiB이고 나머지 task는 최대 4 CPU/12 GiB이므로 `n2-standard-8` 한 대의
allocatable resource 안에 들어간다. 상한 2가 필요해지면 cluster 생성 전에 global CPU quota를
20 이상으로 올리고 별도 검증한다.

`SSD_TOTAL_GB` quota는 250 GiB뿐이므로 GKE 기본 100 GiB `pd-balanced` boot disk를 사용하지 않는다.
두 node pool의 boot disk를 50 GiB `pd-standard`로 명시해 KFP PVC용 SSD quota를 보존한다.

### 5.2 KFP 설치와 GCS 연동

공식 release ref `2.17.0`의 standalone manifest를 사용한다.

```bash
export PIPELINE_VERSION=2.17.0
kubectl apply -k \
  "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=${PIPELINE_VERSION}"
kubectl wait --for condition=established --timeout=60s crd/applications.app.k8s.io
kubectl apply -k \
  "github.com/kubeflow/pipelines/manifests/kustomize/env/platform-agnostic?ref=${PIPELINE_VERSION}"
```

설치 후 다음을 적용한다.

- `kfp-launcher.defaultPipelineRoot`를
  `gs://<RUN_BUCKET>/pertinence/runs/<EXPERIMENT_ID>`로 설정
- GCS provider credential은 `fromEnv: true`로 설정
- KFP가 제공하는 `pipeline-runner` KSA와 실행 RBAC 존재 여부 확인
- KSA에 Raw/Run bucket IAM binding 적용
- Raw/Run 권한 condition이 bucket resource 자체(list)와 experiment object prefix(read/write)를
  함께 허용하는 복합식인지 확인한다. 객체 prefix만 허용하면 smoke의 직접 객체 I/O는 통과해도
  KFP launcher의 `storage.objects.list`가 실제 artifact download 전에 실패한다.
- KFP API server는 기본 internal object store를 유지

KFP API와 artifact pipeline root는 서로 다른 저장소다. Cluster를 삭제하면 KFP 내부 metadata는
사라질 수 있으므로 필요한 run metadata는 종료 전에 별도로 보존한다.

## 6. 실행 전 smoke gate

별도의 축소 dispatcher run은 기본 절차에 포함하지 않는다. 현재 DAG는 NSGA-II population과
generation을 줄여도 세 expert의 CIFAR-10 60,000장 inference를 그대로 수행하므로 비용과 시간이
충분히 작아지지 않기 때문이다. 대신 다음 control-plane smoke를 수행한다.

1. KFP controller Pod가 모두 Ready이고 `system-pool`에서 실행되는지 확인한다.
2. KFP PVC가 Bound인지 확인한다.
3. KFP API health check를 통과시킨다.
4. `pipeline-runner` KSA로 Raw bucket object를 읽는다.
5. Run bucket에 sentinel object를 쓰고 읽고 삭제한다.
6. 공용 component image를 pull하는 smoke Pod를 `pipeline-pool`에 실행한다.
7. `pipeline-pool`이 0에서 1 이상으로 확장되는지 확인한다.
8. 작은 artifact를 GCS에 출력하는 최소 KFP pipeline을 실행한다.
9. 일곱 container component가 `pipeline-pool`에만 배치될 수 있고 importer는 system pool에서
   실행되는 IR/backend 계약인지 확인한다.
10. Public `LoadBalancer`와 ingress가 생성되지 않았는지 확인한다.

하나라도 실패하면 production run을 제출하지 않는다.

## 7. Production dispatcher 학습

Production RunConfig는 다음 주요 값을 유지한다.

```text
dispatcher epochs: 20
batch size:         256
NSGA-II population: 50
NSGA-II generations: 50
평가 횟수:          2,500
search/final split: 8,000 / 2,000
```

입력 URI, 입력 hash와 image digest를 immutable하게 고정한 상태에서 compile과 submit 양쪽에 KFP
cache를 활성화한다. 이는 실패 후 완료된 upstream stage를 안전하게 재사용하기 위한 것이다. 같은
GCS URI를 덮어쓰는 운영은 금지한다.

실행 순서는 다음과 같다.

```text
prepare-dataset-bundle ─┐
export-expert-bundle ───┼─► validate-run ─► build-routing-dataset
RunConfig ──────────────┘                         │
                                                   ▼
                                  train-dispatcher ─► evaluate-final
                                                               │
                                                               ▼
                                                    register-dispatchers
```

제출 후 즉시 다음을 기록한다.

- Experiment ID와 KFP run ID
- 시작 UTC 시각
- Git SHA와 worktree patch hash
- Pipeline IR SHA-256
- Component image digest
- 각 GCS 입력 URI, generation과 SHA-256
- GKE/KFP/Kubernetes version

실행 중에는 Pod phase, node 배치, restart/OOM, component log와 GCS artifact 생성을 감시한다.
`train-dispatcher`는 중간 resume과 자동 retry를 지원하지 않으므로 초반 evaluation의 실제 시간을
기준으로 총 wall time과 비용을 다시 추정한다. 무한 대기를 막기 위한 최대 wall time과 비용 한도는
production 제출 전에 확정한다.

이번 실행의 한도는 다음과 같이 고정한다. 로컬 4-thread, 50,000×1,024 feature microbenchmark에서
20 epoch 평가 1회가 40.58초였고, 2,500회 학습만 단순 투영해도 약 28.2시간이다. Cloud CPU 차이,
search 평가와 expert inference를 고려해 45–60시간을 예상한다.

- `train-dispatcher` hard deadline: 72시간 (`activeDeadlineSeconds=259200`)
- Cluster 전체 hard lifetime: 80시간
- 비용 warning / 재시도 금지 / hard limit: USD 40 / 45 / 55
- Pymoo 세대별 progress log와 Kubernetes log timestamp로 첫 50 evaluation 뒤 ETA 재산정
- ETA가 72시간 deadline 또는 USD 55 hard limit을 넘으면 run을 중단하고 diagnostics 보존 후 teardown

## 8. 완료 조건

다음 조건을 모두 만족해야 성공으로 판정한다.

- KFP run 상태가 `Succeeded`
- 일곱 component가 모두 성공
- Validation report가 유효하고 input hash가 일치
- 50×50 NSGA-II search 완료
- Final report metric이 유한값이고 contract/cache fingerprint가 일치
- 모든 Pareto dispatcher state 복원 성공
- Evaluated dispatcher bundle 생성
- Registration report와 run-scoped registry artifact 생성
- 필요한 artifact가 Run bucket에 존재하며 읽을 수 있음
- 실행 manifest에 run ID, IR, Git SHA, image digest와 GKE/KFP version 기록 완료

이번 실행에서는 결과를 serving bucket으로 승격하거나 endpoint에 배포하지 않는다.

## 9. 실패 처리

실패 시 cluster를 무기한 유지하지 않는다. 먼저 다음 증거를 Run bucket의
`diagnostics/<RUN_ID>/` prefix에 저장한다.

- KFP run 상태와 실패 component
- Pod description과 container 종료 사유
- 관련 namespace events
- 실패 component log
- Node 상태와 autoscaler event
- PVC/PV 상태
- 성공한 upstream artifact URI와 hash

Cache key에 사용한 입력 URI/hash와 image digest가 동일한 경우에만 재제출한다. 원인이 코드,
RunConfig 또는 image 변경을 요구하면 새 experiment ID와 새 image digest를 사용한다. 재시도하지
않기로 결정한 경우에도 진단 보존 뒤 아래 teardown을 실행한다.

## 10. 결과 보존과 teardown

원래 계획에서는 cluster 삭제를 성공 또는 실패 처리의 마지막 필수 단계로 두었다.

> 실행 중 사용자 보존 지시(2026-09-20): 이번 실행이 끝난 뒤에도 GKE cluster를 현재 상태로
> 유지한다. 이 지시는 아래 teardown 절차보다 우선한다. 따라서 PVC/PV, KFP namespace,
> `pipeline-runner`의 Raw/Run bucket IAM binding, node pool과 cluster를 제거하지 않는다. 아래 절차는
> 향후 사용자가 명시적으로 종료를 요청할 때 적용할 보류 작업으로 남기며, 최종 보고서에는 보존된
> 리소스와 계속 발생할 수 있는 비용을 기록한다.

### 10.1 삭제 전 gate

1. Final report, evaluated bundle, registration report와 diagnostics를 GCS에서 읽을 수 있는지
   확인한다.
2. Pipeline IR, run ID와 실행 manifest를 Run bucket에 보존한다.
3. KFP 내부 metadata 중 필요한 내용을 export한다.
4. `kubectl get pvc,pv`와 cluster가 만든 persistent disk 목록을 기록한다.
5. Public Service 또는 forwarding rule이 없음을 확인한다.

### 10.2 삭제 순서

1. KFP namespace의 PVC를 제거하고 동적 persistent disk 삭제가 끝났는지 확인한다.
2. Raw/Run bucket에 추가한 `pipeline-runner` Workload Identity IAM binding을 제거한다.
3. Cluster를 삭제한다.

   ```bash
   gcloud container clusters delete "${CLUSTER_NAME}" \
     --project pertinence-509116 \
     --location asia-northeast3-a \
     --quiet
   ```

4. 다음 잔여 resource가 없는지 확인한다.

   - GKE cluster
   - Cluster node VM
   - KFP용 persistent disk
   - GKE가 만든 forwarding rule과 load balancer
   - 임시 external IP

5. 로컬 kubeconfig에서 삭제된 cluster context를 정리한다.

GKE cluster 삭제만으로 persistent disk 제거가 보장되지 않으므로 disk 확인을 생략하지 않는다.
Raw/Run bucket과 Artifact Registry image는 재현성과 결과 보존을 위해 삭제하지 않는다. 장기 보존
기간과 lifecycle policy는 결과 검토 후 별도로 결정한다.

## 11. 실행 체크리스트

### 준비

- [x] Worktree와 Git/patch SHA 고정
- [x] 전체 test와 Ruff 통과
- [x] Raw asset verify 통과
- [x] Pipeline scheduling modifier와 IR test 완료
- [x] Billing/API/IAM/quota 확인
- [x] 최대 wall time과 비용 한도 확정

### Cloud 생성

- [x] Raw/Run bucket 생성 및 보호 설정
- [x] Artifact Registry와 component image digest 준비
- [x] GKE cluster와 두 node pool 생성
- [x] KFP 2.17.0 설치
- [x] Workload Identity와 bucket IAM 적용
- [x] GCS pipeline root 설정

### 실행

- [x] Control-plane smoke gate 통과
- [x] Production IR compile 및 hash 기록
- [x] Production run 제출 및 run ID 기록
- [x] Run 상태와 resource 사용 감시
- [x] Final/registration artifact 검증

### 종료

- [x] 실행 manifest와 필요한 KFP metadata를 GCS에 보존
- [x] 성공 결과와 실패 diagnostics 보존
- [ ] PVC와 persistent disk 정리 — 사용자 보존 지시로 `DEFERRED`
- [ ] `pipeline-runner` Workload Identity IAM binding 제거 — 사용자 보존 지시로 `DEFERRED`
- [ ] GKE cluster 삭제 — 사용자 보존 지시로 `DEFERRED`
- [x] Public LoadBalancer, Ingress, forwarding rule과 예약 address 없음 확인
- [x] Raw/Run bucket과 Artifact Registry 보존 확인

## 12. 참고 문서

- [저장소 GKE 실행 가이드](../pipelines/pertinence/docs/gke-cpu.ko.md)
- [Cloud 환경 설계](README.md)
- [KFP standalone 설치](https://www.kubeflow.org/docs/components/pipelines/operator-guides/installation/)
- [KFP pipeline root](https://www.kubeflow.org/docs/components/pipelines/concepts/pipeline-root/)
- [KFP object store 구성](https://www.kubeflow.org/docs/components/pipelines/operator-guides/configure-object-store/)
- [GKE Workload Identity Federation](https://cloud.google.com/kubernetes-engine/docs/how-to/workload-identity)
- [GKE zonal cluster 생성](https://cloud.google.com/kubernetes-engine/docs/how-to/creating-a-zonal-cluster)
- [GKE cluster 삭제](https://cloud.google.com/kubernetes-engine/docs/how-to/deleting-a-cluster)
- [Cloud Storage uniform bucket-level access](https://cloud.google.com/storage/docs/uniform-bucket-level-access)
- [Artifact Registry와 Cloud Build](https://cloud.google.com/artifact-registry/docs/configure-cloud-build)
