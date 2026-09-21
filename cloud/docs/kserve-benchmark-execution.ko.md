# GKE KServe benchmark 실제 실행 절차

이 문서는 [로컬 세션 게이트](kserve-session-gate.ko.md)가 성공하여 승인 변수와 새 evidence
directory가 이미 봉인됐다는 전제에서 시작한다. 준비물의 선정·승인은 반복하지 않고, 그 다음에
실행할 20단계만 다룬다.

기존 cluster, KFP, PVC/PV와 모든 성공·실패 증거를 보존한다. 기존 GCS object, image tag,
ConfigMap, Job, chart release, IAM binding 또는 Terraform resource를 충돌 해결 목적으로 삭제하거나
덮어쓰지 않는다. 충돌은 중단 사유이며 재실행에는 새 build tag와 benchmark ID를 사용한다.

## 공통 실행 규칙

승인된 `session.env`는 검증 입력이지 shell script가 아니므로 `source`하지 않는다. 운영자 shell에는
session gate에서 승인된 값을 별도로 설정한다.

```bash
set -euo pipefail
set -o noclobber
umask 077

REPO=/home/hwan/projects/pertinence
PROJECT=pertinence-509116
REGION=asia-northeast3
ZONE=asia-northeast3-a
CLUSTER=pertinence-dispatcher-260919-1708
EXPECTED_CONTEXT=gke_pertinence-509116_asia-northeast3-a_pertinence-dispatcher-260919-1708
AR_REPOSITORY=pertinence
IMAGE_REPOSITORY=asia-northeast3-docker.pkg.dev/pertinence-509116/pertinence
RUN_BUCKET=pertinence-509116-pertinence-runs
NODE_SA=pertinence-gke-nodes@pertinence-509116.iam.gserviceaccount.com

: "$EXPECTED_PRINCIPAL" "$BENCHMARK_GSA" "$EVIDENCE_ROOT"
: "$RUN_SET_ID" "$BENCHMARK_ID_FINAL" "$BENCHMARK_ID_C101"
: "$BUILD_TAG" "$APPROVED_PROMOTION_ID" "$GOLDEN_LOGITS_ATOL"
: "$APPROVED_FINAL_LABEL_SHA256" "$APPROVED_BY" "$APPROVED_AT"
: "$IMMUTABLE_BUILD_BUNDLE_PREFIX" "$EVALUATION_NPZ_URI"

cd "$REPO"
jq -e '.valid == true and .cloud_calls_performed == false' \
  "$EVIDENCE_ROOT/00-session/session-gate-receipt.json" >/dev/null
```

단계 표시는 다음 뜻이다.

- **LOCAL/READ-ONLY**: 로컬 파일 생성 또는 cloud/cluster 조회만 한다.
- **GCS CREATE-ONLY**: `ifGenerationMatch=0`으로 새 object만 만든다.
- **MUTATION**: IAM, image push, Terraform apply 또는 Kubernetes object 변경이다.
- **STOP GATE**: 하나라도 실패하면 다음 단계로 진행하지 않는다.

기존 파일에 대한 redirection 실패를 우회하지 않는다. Mutation 후 오류가 나면 20단계 종료를
시도하되 이미 만들어진 resource나 증거를 삭제하지 않는다.

## 1. KFP artifact 획득과 통합 검증

**LOCAL/READ-ONLY → GCS CREATE-ONLY · STOP GATE**

성공 KFP run의 expert/evaluated/dataset bundle과 split manifest를 object generation까지 확정한다.
각 object의 describe JSON을 `01-artifacts/`에 저장하고 `gs://OBJECT#GENERATION`으로 새 로컬 경로에
내려받는다. prefix의 최신 version이나 wildcard 결과를 build 입력으로 사용하지 않는다.

```bash
gcloud storage ls --recursive --all-versions --long \
  'gs://pertinence-509116-pertinence-runs/pertinence/runs/dispatcher-gke-20260920t015118z/**' \
  > "$EVIDENCE_ROOT/01-artifacts/gcs-inventory.txt"

# 확정한 각 object에 반복한다.
gcloud storage objects describe "gs://BUCKET/OBJECT#GENERATION" --format=json \
  > "$EVIDENCE_ROOT/01-artifacts/descriptions/OBJECT.json"
gcloud storage cp "gs://BUCKET/OBJECT#GENERATION" "$LOCAL_NEW_PATH"
```

다운로드한 실물로 통합 driver를 실행한다.

```bash
.venv/bin/python cloud/kserve/scripts/integrate-artifacts.py \
  --dataset-bundle "$DATASET_ROOT" \
  --split-manifest "$SPLIT_MANIFEST" \
  --expert-bundle "$EXPERT_ROOT" \
  --evaluated-bundle "$EVALUATED_ROOT" \
  --dispatcher-id dispatcher-000733 \
  --approved-dispatcher-sha256 \
    d82d3243bfec4f0c5889aa7e241735c4f212c849a6bcc2f461f60ca6c1524720 \
  --approved-label-sha256 "$APPROVED_FINAL_LABEL_SHA256" \
  --output "$EVIDENCE_ROOT/01-artifacts/integration"
```

`receipt.json.valid`가 true여야 한다. ONNX signature, dispatcher shape/finite 값, lineage, sample-order,
label hash, 2,000개 inference의 정확도 `1809/1858/1892/1893`과 route count `[0,640,1360]`을 함께
확인한다. `evaluation-final.npz`, `local-predictions.npz`, `request.json`, `build-bundle/`이 모두
있어야 한다. ONNX SHA는 순서대로
`04c7c69ea710375a3f8e7314c4b58f9ca225dd5bf8f4d13069e1c4b3f3833ea7`,
`69225030b26f6de25f85f2b6873b6afb9095cac79b68fa153c59e5a07534d62b`,
`9494958830e238f0cb35ad4f490256a4f07794936403559ac3651c4af4f96858`이어야 한다.

`build-bundle/`의 정확한 11개 파일과 `evaluation-final.npz`를 각각 create-only로 publish한다.
동일하고 완전한 기존 prefix만 generation을 재사용할 수 있다. 일부 object만 있거나 bytes가 다르면
삭제하지 않고 중단한다. 실제 generation으로 build manifest를 봉인한다.

```bash
.venv/bin/python cloud/kserve/scripts/build_inputs.py seal \
  --bundle "$EVIDENCE_ROOT/01-artifacts/integration/build-bundle" \
  --object experts/shufflenetv2_x0_5/model.onnx=gs://BUCKET/PREFIX/experts/shufflenetv2_x0_5/model.onnx#GENERATION \
  --object experts/shufflenetv2_x0_5/model.sha256=gs://BUCKET/PREFIX/experts/shufflenetv2_x0_5/model.sha256#GENERATION \
  --object experts/mobilenetv2_x0_5/model.onnx=gs://BUCKET/PREFIX/experts/mobilenetv2_x0_5/model.onnx#GENERATION \
  --object experts/mobilenetv2_x0_5/model.sha256=gs://BUCKET/PREFIX/experts/mobilenetv2_x0_5/model.sha256#GENERATION \
  --object experts/resnet56/model.onnx=gs://BUCKET/PREFIX/experts/resnet56/model.onnx#GENERATION \
  --object experts/resnet56/model.sha256=gs://BUCKET/PREFIX/experts/resnet56/model.sha256#GENERATION \
  --object evaluated/contract.json=gs://BUCKET/PREFIX/evaluated/contract.json#GENERATION \
  --object evaluated/search-result.json=gs://BUCKET/PREFIX/evaluated/search-result.json#GENERATION \
  --object evaluated/final-report.json=gs://BUCKET/PREFIX/evaluated/final-report.json#GENERATION \
  --object evaluated/manifest.json=gs://BUCKET/PREFIX/evaluated/manifest.json#GENERATION \
  --object evaluated/dispatcher-000733.npz=gs://BUCKET/PREFIX/evaluated/dispatcher-000733.npz#GENERATION \
  --output "$EVIDENCE_ROOT/01-artifacts/build-input-manifest.json"
```

Manifest도 create-only로 publish하고 실제 `#GENERATION` URI를 `BUILD_INPUT_MANIFEST_URI`에 기록한다.

## 2. chart와 image lock 확인

**LOCAL/READ-ONLY · STOP GATE**

승인된 chart archive를 evidence로 복사해 SHA-256을 다시 확인한다.

```bash
install -m 0600 /tmp/pertinence-cert-chart.SILbTy/cert-manager-v1.20.4.tgz \
  "$EVIDENCE_ROOT/02-charts/cert-manager-v1.20.4.tgz"
install -m 0600 /tmp/pertinence-kserve-release.DwQy3J/kserve-crd.tgz \
  "$EVIDENCE_ROOT/02-charts/kserve-crd-v0.20.0.tgz"
install -m 0600 /tmp/pertinence-kserve-release.DwQy3J/kserve-resources.tgz \
  "$EVIDENCE_ROOT/02-charts/kserve-resources-v0.20.0.tgz"
sha256sum "$EVIDENCE_ROOT/02-charts/"*.tgz \
  > "$EVIDENCE_ROOT/02-charts/SHA256SUMS"
```

기대 SHA는 cert-manager `aa916e8fb726ee573f8c2fbed6899de7dbb77a017ac1deef63fee91cc5bfcd03`,
KServe CRD `0b5437558fa3a77684a08b1480252404db2b7729f46f0605f5744f41a8efd44c`, resources
`1d43f36f47037f5e604c69fb78dace2c0e30c80c2a44324c701efd8a8b7631ce`다. 승인된 inventory의
controller 11개와 Python/BusyBox는 모두 실제 `linux/amd64` child digest여야 한다. sentinel,
index digest 또는 비-amd64 값이 있으면 중단한다.

## 3. Cloud Build 실행

**MUTATION: image push · STOP GATE**

`BUILD_TAG`가 Artifact Registry에 없음을 먼저 확인한다. 이미 있으면 삭제·재사용하지 않는다.
`PYTHON_BASE_IMAGE`와 `MODELCAR_BASE_IMAGE`는 2단계 inventory에 승인된 source와 linux/amd64 digest를
결합한 `name:tag@sha256:...` reference여야 한다.

```bash
gcloud artifacts docker images list "$IMAGE_REPOSITORY" \
  --project "$PROJECT" --include-tags --filter="tags:$BUILD_TAG" --format=json \
  > "$EVIDENCE_ROOT/03-cloud-build/preexisting-tag.json"
jq -e 'length == 0' "$EVIDENCE_ROOT/03-cloud-build/preexisting-tag.json" >/dev/null

CLOUD_BUILD_ID=$(gcloud builds submit "$REPO" \
  --project "$PROJECT" --region "$REGION" \
  --config cloud/kserve/cloudbuild.yaml --async \
  --substitutions="_REPOSITORY=$IMAGE_REPOSITORY,_BUILD_ID=$BUILD_TAG,_PYTHON_BASE_IMAGE=$PYTHON_BASE_IMAGE,_MODELCAR_BASE_IMAGE=$MODELCAR_BASE_IMAGE,_BUILD_INPUT_MANIFEST_URI=$BUILD_INPUT_MANIFEST_URI" \
  --format='value(id)')
printf '%s\n' "$CLOUD_BUILD_ID" > "$EVIDENCE_ROOT/03-cloud-build/build-id.txt"
```

`gcloud builds describe`를 polling해 terminal status, 전체 JSON과 log URI를 보존한다. `SUCCESS`가
아니거나 generation-pinned fetch, 다섯 image push, `seal-deployment` 중 하나라도 실패하면 중단한다.
재시도는 새 tag만 사용한다.

Python과 BusyBox build base는 child digest로 고정하지만, 현재 `cloudbuild.yaml`의 Cloud SDK과
Docker builder image는 tag reference다. 따라서 build request/status/log과 최종 image digest를 모두
보존하고, builder까지 완전한 byte-for-byte 재현성을 달성했다고 표현하지 않는다.

## 4. deployment image manifest 검증

**READ-ONLY · STOP GATE**

Build ID 경로에서 `deployment-images.json`, `build-input-manifest.json`,
`build-input-receipt.json`의 generation을 먼저 describe하고 그 generation으로 받는다. Input receipt가
1단계 manifest fingerprint와 generation을 가리키는지 확인한다.

```bash
gcloud storage cp \
  "gs://$RUN_BUCKET/pertinence/builds/$CLOUD_BUILD_ID/deployment-images.json#DEPLOYMENT_MANIFEST_GENERATION" \
  "$EVIDENCE_ROOT/04-deployment-images/deployment-images.json"

jq -e '
  .schema_version == 1 and (.images | length) == 5 and
  ([.images[].name] | sort) ==
    (["serving-runtime","shufflenetv2_x0_5-modelcar",
      "mobilenetv2_x0_5-modelcar","resnet56-modelcar",
      "dispatcher-000733-modelcar"] | sort) and
  ([.images[].architecture] | unique) == ["linux/amd64"] and
  ([.images[].user] | unique) == ["65532:65532"] and
  .runtime_versions.numpy == "1.26.4" and
  .runtime_versions.onnxruntime == "1.20.1" and .cli_smoke == true
' "$EVIDENCE_ROOT/04-deployment-images/deployment-images.json" >/dev/null
```

Canonical fingerprint와 각 Artifact Registry digest를 독립 재확인한다. Repository 불일치, tag-only
reference, 중복 이름 또는 digest 부재는 중단 사유다.

## 5. promotion 봉인

**LOCAL/READ-ONLY · STOP GATE**

실제 runtime/modelcar digest와 generation-pinned evaluated manifest를 promotion draft에 넣는다.
Expert 순서는 `shufflenetv2_x0_5`, `mobilenetv2_x0_5`, `resnet56`이며 dispatcher는
`dispatcher-000733`, `feature_dim=1024`, `expert_count=3`이어야 한다. Draft의 promotion ID는
`APPROVED_PROMOTION_ID`이고 approver/time은 `APPROVED_BY`/`APPROVED_AT`과 정확히 일치해야 한다.

```bash
.venv/bin/pertinence promotion create \
  --manifest "$EVIDENCE_ROOT/05-promotion/promotion-draft.json" \
  --output "$EVIDENCE_ROOT/05-promotion/promotion.json" \
  --evaluated-bundle "$EVALUATED_ROOT"
.venv/bin/pertinence promotion verify \
  --manifest "$EVIDENCE_ROOT/05-promotion/promotion.json" \
  --evaluated-bundle "$EVALUATED_ROOT" \
  > "$EVIDENCE_ROOT/05-promotion/verification.txt"
```

같은 promotion ID의 기존 bytes가 다르면 교체하지 않고 중단한다.

## 6. 두 trace 생성·검증·publish

**LOCAL → GCS CREATE-ONLY · STOP GATE**

```bash
.venv/bin/pertinence trace create-final \
  --evaluation-npz "$EVIDENCE_ROOT/01-artifacts/integration/evaluation-final.npz" \
  --split-manifest "$SPLIT_MANIFEST" \
  --output "$EVIDENCE_ROOT/06-traces/cifar10-final-20260824" \
  --trace-id cifar10-final-20260824 \
  --expected-label-sha256 "$APPROVED_FINAL_LABEL_SHA256" \
  --evaluation-uri "$EVALUATION_NPZ_URI" \
  --evaluation-generation "$EVALUATION_NPZ_GENERATION" \
  --split-uri "$SPLIT_MANIFEST_URI" \
  --split-generation "$SPLIT_MANIFEST_GENERATION"

.venv/bin/pertinence trace create-cifar10.1-v6 \
  --data-npy "$C101_DATA_NPY" --labels-npy "$C101_LABELS_NPY" \
  --output "$EVIDENCE_ROOT/06-traces/cifar10.1-v6" --trace-id cifar10.1-v6

for trace_id in cifar10-final-20260824 cifar10.1-v6; do
  .venv/bin/pertinence trace verify --trace "$EVIDENCE_ROOT/06-traces/$trace_id"
  .venv/bin/pertinence trace publish \
    --trace "$EVIDENCE_ROOT/06-traces/$trace_id" \
    --uri "gs://$RUN_BUCKET/pertinence/benchmarks/traces/$trace_id"
done
```

CIFAR-10.1 NPY는 다음 commit-pinned URL로 새 파일에 받고 승인된 hash/size/shape를 먼저 확인한다.

```bash
curl --fail --location --output "$C101_DATA_NPY" \
  https://raw.githubusercontent.com/modestyachts/CIFAR-10.1/d9982abb0bfc4846b8d13a11e66b887d946205d0/datasets/cifar10.1_v6_data.npy
curl --fail --location --output "$C101_LABELS_NPY" \
  https://raw.githubusercontent.com/modestyachts/CIFAR-10.1/d9982abb0bfc4846b8d13a11e66b887d946205d0/datasets/cifar10.1_v6_labels.npy
sha256sum "$C101_DATA_NPY" "$C101_LABELS_NPY" \
  > "$EVIDENCE_ROOT/06-traces/cifar10.1-v6-source-SHA256SUMS"
```

Publish는 동일하고 완전한 trace만 재사용한다. 부분 prefix나 다른 fingerprint는 삭제하지 않고
실패한다. `trace.json`과 `trace.npz`의 generation/size/hash를 기록한다.

## 7. Workload Identity와 IAM 적용

**MUTATION → READ-ONLY VERIFICATION · STOP GATE**

이 단계는 12단계의 전체 workload render보다 먼저 실행된다. `BENCHMARK_GSA`만 담은
strict env로 Workload Identity template 하나만 별도 render하고 검증한다.

```bash
printf 'BENCHMARK_GSA=%s\n' "$BENCHMARK_GSA" \
  > "$EVIDENCE_ROOT/07-iam/workload-identity.env"
.venv/bin/python cloud/kserve/scripts/render_config.py \
  --env "$EVIDENCE_ROOT/07-iam/workload-identity.env" \
  --output-dir "$EVIDENCE_ROOT/07-iam/k8s-render" \
  cloud/kserve/manifests/05-benchmark-workload-identity.yaml.tmpl=05-benchmark-workload-identity.yaml
.venv/bin/python cloud/kserve/scripts/validate_rendered.py \
  "$EVIDENCE_ROOT/07-iam/k8s-render/05-benchmark-workload-identity.yaml"
kubectl --context "$EXPECTED_CONTEXT" apply --server-side --dry-run=server \
  -f "$EVIDENCE_ROOT/07-iam/k8s-render/05-benchmark-workload-identity.yaml"
kubectl --context "$EXPECTED_CONTEXT" apply --server-side \
  -f "$EVIDENCE_ROOT/07-iam/k8s-render/05-benchmark-workload-identity.yaml"
```

그 다음 GSA의 `workloadIdentityUser`, 두 trace prefix의 objectViewer, 두 run prefix의
objectCreator만 추가한다. 전후 GSA/bucket policy 전체 JSON과 etag, GSA key 목록을 보존한다.
Binding title/expression과 적용 명령은 [IAM 경계 문서](../kserve/iam/README.md)의 exact 형식을
사용한다. Bucket-wide writer, `objectAdmin`, Editor/Owner 또는 JSON key를 추가하지 않는다.

```bash
cloud/kserve/scripts/verify-benchmark-iam.py \
  --project "$PROJECT" --bucket "$RUN_BUCKET" --gsa "$BENCHMARK_GSA" \
  --run-set-id "$RUN_SET_ID" \
  --benchmark-id-final "$BENCHMARK_ID_FINAL" \
  --benchmark-id-c101 "$BENCHMARK_ID_C101" \
  --gsa-policy "$EVIDENCE_ROOT/07-iam/gsa-policy-after.json" \
  --bucket-policy "$EVIDENCE_ROOT/07-iam/bucket-policy-after.json" \
  --service-account-keys "$EVIDENCE_ROOT/07-iam/gsa-keys.json" \
  --output "$EVIDENCE_ROOT/07-iam/verification.json"
jq -e '.valid == true' "$EVIDENCE_ROOT/07-iam/verification.json" >/dev/null
```

Verifier는 정확히 네 conditional bucket binding, KSA 한 개의 impersonation, broad role 부재와
user-managed key 부재를 요구한다. 부분 실패에서는 기존 binding을 지우지 않고 누락된 exact
binding만 보완한다.

## 8. cluster preflight 봉인

**READ-ONLY · STOP GATE**

```bash
.venv/bin/python cloud/kserve/scripts/preflight-gke.py \
  --context "$EXPECTED_CONTEXT" --project "$PROJECT" --region "$REGION" \
  --zone "$ZONE" --cluster "$CLUSTER" --repository "$AR_REPOSITORY" \
  --principal "$EXPECTED_PRINCIPAL" --node-service-account "$NODE_SA" \
  --output "$EVIDENCE_ROOT/08-preflight/sealed"
jq -e '.valid == true' "$EVIDENCE_ROOT/08-preflight/sealed/receipt.json" >/dev/null
```

Context/project/zone/cluster/principal, GKE version, regional CPU와 N2 quota 각 12 vCPU, node SA,
benchmark RBAC, KFP namespace/PVC/workflow, pipeline node 0과 외부 exposure 0 중 하나라도 어긋나면
중단한다.

## 9. Terraform saved plan과 receipt

**READ-ONLY PLAN · STOP GATE**

```bash
cd "$REPO/cloud/iac/environments/dispatcher-current"
terraform init -backend-config=backend.hcl -lockfile=readonly
terraform plan -var-file=profiles/benchmark-current.tfvars \
  -out="$EVIDENCE_ROOT/09-terraform-plan/benchmark-pools.tfplan"
terraform show -json "$EVIDENCE_ROOT/09-terraform-plan/benchmark-pools.tfplan" \
  > "$EVIDENCE_ROOT/09-terraform-plan/benchmark-pools.tfplan.json"
python ../../../../scripts/terraform_plan_guard.py --mode benchmark-pools-create \
  "$EVIDENCE_ROOT/09-terraform-plan/benchmark-pools.tfplan.json" \
  > "$EVIDENCE_ROOT/09-terraform-plan/guard.txt"
./seal-plan.sh \
  "$EVIDENCE_ROOT/09-terraform-plan/benchmark-pools.tfplan" \
  "$EVIDENCE_ROOT/09-terraform-plan/benchmark-pools.tfplan.json" \
  profiles/benchmark-current.tfvars "$(git rev-parse HEAD)" \
  "$EVIDENCE_ROOT/09-terraform-plan/benchmark-pools.receipt.json"
```

허용 change는 `serving-pool`과 `loadgen-pool`의 정확한 pure create 두 개뿐이다. Cluster나 기존 pool의
update/delete/replace, 다른 create 또는 resource drift가 있으면 중단한다.

## 10. 봉인한 동일 tfplan apply

**MUTATION · STOP GATE**

```bash
./apply-saved-plan.sh \
  "$EVIDENCE_ROOT/09-terraform-plan/benchmark-pools.tfplan" \
  profiles/benchmark-current.tfvars \
  "$EVIDENCE_ROOT/09-terraform-plan/benchmark-pools.receipt.json" \
  > "$EVIDENCE_ROOT/10-terraform-apply/apply.txt"
```

새 plan을 만들거나 tfvars를 `terraform apply`에 직접 주지 않는다. 적용 후 state와 live GKE에서
machine type, autoscaling 0–1, initial 0, taint, GKE_METADATA, node SA, disk, non-spot,
`deletion_policy=PREVENT`를 확인하고 후속 no-op plan을 보존한다.

Guard 자체는 `benchmark-pool-recovery`를 지원하지만 현재 `seal-plan.sh`와
`apply-saved-plan.sh`는 `benchmark-pools-create`를 고정한다. 따라서 정확히 한 pool만 생긴 부분
실패라도 일반 wrapper를 우회하지 않는다. 별도로 code-reviewed되고 create-only receipt를 봉인하는
recovery workflow가 준비되지 않았다면 여기서 중단하고 live/state 증거만 보존한다.

## 11. cert-manager와 KServe addon 적용

**LOCAL RENDER → MUTATION · STOP GATE**

승인된 chart와 controller digest로 `02-charts/chart-lock.env`를 만들고 render한다.

```bash
cd "$REPO"
cloud/kserve/scripts/render-install.sh \
  "$EVIDENCE_ROOT/02-charts/chart-lock.env" \
  "$EVIDENCE_ROOT/11-addons/rendered"
cloud/kserve/scripts/audit-exposure.sh "$EXPECTED_CONTEXT" "$PROJECT"

for manifest in 00-namespaces.yaml cert-manager.yaml kserve-crd.yaml kserve-resources.yaml; do
  kubectl --context "$EXPECTED_CONTEXT" apply --server-side --dry-run=server \
    -f "$EVIDENCE_ROOT/11-addons/rendered/$manifest"
done

kubectl --context "$EXPECTED_CONTEXT" apply --server-side \
  -f "$EVIDENCE_ROOT/11-addons/rendered/00-namespaces.yaml"
kubectl --context "$EXPECTED_CONTEXT" apply --server-side \
  -f "$EVIDENCE_ROOT/11-addons/rendered/cert-manager.yaml"
# cert-manager deployment와 webhook Ready를 확인한 뒤에만 다음 두 apply를 실행한다.
kubectl --context "$EXPECTED_CONTEXT" apply --server-side \
  -f "$EVIDENCE_ROOT/11-addons/rendered/kserve-crd.yaml"
kubectl --context "$EXPECTED_CONTEXT" apply --server-side \
  -f "$EVIDENCE_ROOT/11-addons/rendered/kserve-resources.yaml"
```

네 manifest를 각각 server-side dry-run한 뒤 namespaces → cert-manager → webhook Ready → KServe CRD
→ KServe resources → controller Ready 순서로 apply한다. 기존 addon이 다른 version/config이면 upgrade나
uninstall을 하지 않고 중단한다. 모든 image digest pin, controller `system-pool`, Standard mode,
`disableIngressCreation=true`, Gateway API disabled, modelcar enabled를 확인한다.

## 12. 두 workload render와 apply

**LOCAL RENDER → MUTATION · STOP GATE**

같은 promotion/image를 쓰되 서로 다른 benchmark ID, trace URI와 output URI를 가진 final/C101 env를
사용한다. 두 env를 서로 다른 새 directory에 한 번씩 render하고 덮어쓰지 않는다.

```bash
cloud/kserve/scripts/render-manifests.sh \
  "$EVIDENCE_ROOT/12-workloads/benchmark-images-final.env" \
  "$EVIDENCE_ROOT/12-workloads/render-final"
cloud/kserve/scripts/render-manifests.sh \
  "$EVIDENCE_ROOT/12-workloads/benchmark-images-c101.env" \
  "$EVIDENCE_ROOT/12-workloads/render-c101"

.venv/bin/python cloud/kserve/scripts/derive-cifar101-job.py \
  --source "$EVIDENCE_ROOT/12-workloads/render-c101/10-workloads.yaml" \
  --output "$EVIDENCE_ROOT/12-workloads/pertinence-loadgen-cifar101-v6.yaml" \
  --receipt "$EVIDENCE_ROOT/12-workloads/pertinence-loadgen-cifar101-v6.receipt.json"
.venv/bin/python cloud/kserve/scripts/validate_rendered.py \
  "$EVIDENCE_ROOT/12-workloads/pertinence-loadgen-cifar101-v6.yaml"

cmp \
  "$EVIDENCE_ROOT/07-iam/k8s-render/05-benchmark-workload-identity.yaml" \
  "$EVIDENCE_ROOT/12-workloads/render-final/05-benchmark-workload-identity.yaml"

for manifest in \
  "$EVIDENCE_ROOT/12-workloads/render-final/00-platform.yaml" \
  "$EVIDENCE_ROOT/12-workloads/render-final/05-benchmark-workload-identity.yaml" \
  "$EVIDENCE_ROOT/12-workloads/render-final/10-workloads.yaml" \
  "$EVIDENCE_ROOT/12-workloads/pertinence-loadgen-cifar101-v6.yaml"; do
  kubectl --context "$EXPECTED_CONTEXT" apply --server-side --dry-run=server -f "$manifest"
done

kubectl --context "$EXPECTED_CONTEXT" apply --server-side \
  -f "$EVIDENCE_ROOT/12-workloads/render-final/00-platform.yaml"
kubectl --context "$EXPECTED_CONTEXT" apply --server-side \
  -f "$EVIDENCE_ROOT/12-workloads/render-final/05-benchmark-workload-identity.yaml"
kubectl --context "$EXPECTED_CONTEXT" apply --server-side \
  -f "$EVIDENCE_ROOT/12-workloads/render-final/10-workloads.yaml"
kubectl --context "$EXPECTED_CONTEXT" apply --server-side \
  -f "$EVIDENCE_ROOT/12-workloads/pertinence-loadgen-cifar101-v6.yaml"
```

파생기는 Job 이름 이외의 차이를 거부하고 동일 validator를 다시 실행한다. Final identity
manifest는 7단계에 적용한 bytes와 정확히 같아야 한다. Server-side dry-run 후 final render의
platform/identity/workloads와 파생 C101 Job만 apply한다. 두 Job은 suspended여야 한다.
ConfigMap bytes나 Job 이름 충돌은 삭제로 해결하지 않는다.

## 13. exposure, readiness와 alias smoke

**READ-ONLY + ALIAS MUTATION · STOP GATE**

```bash
cloud/kserve/scripts/audit-exposure.sh "$EXPECTED_CONTEXT" "$PROJECT"
kubectl --context "$EXPECTED_CONTEXT" -n pertinence-serving wait \
  --for=condition=Ready --timeout=15m inferenceservice --all

REQUEST_JSON="$EVIDENCE_ROOT/01-artifacts/integration/request.json"
cloud/kserve/scripts/switch-active.sh "$EXPECTED_CONTEXT" static "$REQUEST_JSON"
cloud/kserve/scripts/switch-active.sh "$EXPECTED_CONTEXT" dispatcher "$REQUEST_JSON"
cloud/kserve/scripts/switch-active.sh "$EXPECTED_CONTEXT" static "$REQUEST_JSON"
```

전환마다 기존 connection을 drain하고 새 connection 20회의 strict identity를 검사한다. 마지막 alias는
static이어야 한다. 혼합 identity, tag image, restart, public endpoint 또는 Ready 실패가 있으면 Job을
시작하지 않는다.

## 14. 두 dataset offline/online parity

**CONTROLLED TRAFFIC + ALIAS MUTATION · STOP GATE**

각 trace의 2,000 samples를 다음 5개 arm 순서로 한 번씩 보내 create-only arm receipt를 만든다.

```text
shufflenetv2_x0_5
mobilenetv2_x0_5
resnet56
static-resnet-router
dispatcher-000733
```

Direct expert는 각 predictor ClusterIP를 사용한다. Router는 `switch-active.sh`로 alias를 바꾸고
`/v2/models/static-resnet-router/infer` 또는 `/v2/models/dispatcher-000733/infer`를 사용한다.

```bash
.venv/bin/pertinence benchmark parity-collect \
  --trace "$TRACE_DIRECTORY" --arm "$ARM" --endpoint "$ENDPOINT" \
  --expected-identity model_name="$ARM" \
  --expected-identity model_version=1 \
  --output "$ARM_OUTPUT_DIRECTORY"
```

Static/dispatcher arm에는 `promotion_id`, `dispatcher_id`와 static의 `expert_id=resnet56`까지 필요한
identity key 전체를 전달한다. 다섯 output을 승인 arm 순서로 `benchmark parity-combine`한 뒤 다음
verifier를 실행한다.

```bash
.venv/bin/pertinence benchmark parity-combine \
  --trace "$TRACE_DIRECTORY" \
  --input shufflenetv2_x0_5="$SHUFFLENET_OUTPUT_DIRECTORY" \
  --input mobilenetv2_x0_5="$MOBILENET_OUTPUT_DIRECTORY" \
  --input resnet56="$RESNET56_OUTPUT_DIRECTORY" \
  --input static-resnet-router="$STATIC_OUTPUT_DIRECTORY" \
  --input dispatcher-000733="$DISPATCHER_OUTPUT_DIRECTORY" \
  --output "$COMBINED_OUTPUT"

.venv/bin/pertinence benchmark parity \
  --trace "$TRACE_DIRECTORY" \
  --local-predictions "$LOCAL_PREDICTIONS" \
  --online-records "$COMBINED_OUTPUT/records.jsonl" \
  --promotion-fingerprint "$PROMOTION_FINGERPRINT" \
  --image-digest runtime="$RUNTIME_DIGEST" \
  --golden-logits-atol "$GOLDEN_LOGITS_ATOL" \
  --output "$PARITY_RECEIPT"
```

Final은 1단계의 `local-predictions.npz`를 사용하며 승인 accuracy/route count까지 검사한다.
`parity-collect`는 online record만 만들며 CIFAR-10.1 offline local predictions를 생성하지 않는다.
현재 저장소에는 C101 local-predictions 생성기가 없으므로 session gate에서 봉인된
`REVIEWED_PARITY_DRIVER`가 C101 trace의 5-arm logits/class/route를 생성·봉인해야 한다. 해당 driver와
receipt가 없거나 review SHA가 달라지면 C101 parity 및 이후 benchmark를 실행하지 않는다.

두 receipt 모두 `matched: true`이고 두 dataset 모두 승인된 `GOLDEN_LOGITS_ATOL`로 golden logits를
검사해야 한다. 불일치는 전체 실행 중단 사유다. 완료 후 alias는 static으로 돌린다.

## 15. cold-start 측정

**MUTATION + MEASUREMENT · STOP GATE**

```bash
cloud/kserve/scripts/stop.sh "$EXPECTED_CONTEXT"
cloud/kserve/scripts/measure-cold-start.sh \
  "$EXPECTED_CONTEXT" \
  "$EVIDENCE_ROOT/12-workloads/render-final/10-workloads.yaml" \
  "$REQUEST_JSON" \
  "$EVIDENCE_ROOT/15-cold-start/cold-start-$RUN_SET_ID"
```

두 Job suspended와 serving Pod 0을 먼저 확인한다. Script는 apply부터 5개 ISVC Ready까지와 각 arm의
최초 20요청/identity를 기록한다. 실패 trap은 다시 stop하고 실패 증거를 보존한다. 실패한 output
directory를 재사용하지 않는다.

## 16. CIFAR-10 final benchmark

**MUTATION + MEASUREMENT · STOP GATE**

직전 snapshot에서 active KFP workflow 0, serving/loadgen node 각 1개, restart 0과 두 Job 상태를
확인한다. 외부 collector를 시작한 뒤 final Job만 unsuspend한다.

```bash
cloud/kserve/scripts/collect-evidence.sh "$EXPECTED_CONTEXT" \
  "$EVIDENCE_ROOT/datasets/$BENCHMARK_ID_FINAL/external/window" 240 5 &
FINAL_COLLECTOR_PID=$!
kubectl --context "$EXPECTED_CONTEXT" -n pertinence-serving patch \
  job pertinence-loadgen --type=merge -p '{"spec":{"suspend":false}}'
```

Job Complete/Failed를 polling한다. 종료 즉시 end snapshot, logs, describe와 events를 보존하고 collector도
wait한다. Job은 native sidecar의 전체-run evidence와 raw 결과를 지정 GCS prefix에 create-only로
publish한다. Exit 2/invalid run도 삭제하거나 같은 ID로 다시 실행하지 않는다.

## 17. CIFAR-10.1 v6 benchmark

**MUTATION + MEASUREMENT · STOP GATE**

16단계의 Job, collector와 GCS publish가 모두 끝난 뒤 별도 evidence window를 시작하고 C101 Job만
unsuspend한다.

```bash
cloud/kserve/scripts/collect-evidence.sh "$EXPECTED_CONTEXT" \
  "$EVIDENCE_ROOT/datasets/$BENCHMARK_ID_C101/external/window" 240 5 &
C101_COLLECTOR_PID=$!
kubectl --context "$EXPECTED_CONTEXT" -n pertinence-serving patch \
  job pertinence-loadgen-cifar101-v6 --type=merge -p '{"spec":{"suspend":false}}'
```

Final과 C101의 Job, prefix, raw requests, validity, parity receipt와 summary를 합치지 않는다.

## 18. dataset별 validity와 summary

**LOCAL/READ-ONLY · INVALID도 보존**

각 output prefix를 generation inventory한 뒤 새 local directory로 받고 dataset마다 별도로 실행한다.

```bash
.venv/bin/pertinence benchmark summarize-evidence \
  --evidence "$RUN_LOCAL/evidence.jsonl" \
  --requests "$RUN_LOCAL/requests.jsonl" \
  --markers "$RUN_LOCAL/markers.jsonl" \
  --output "$ANALYSIS/metric-summary.json"

.venv/bin/pertinence benchmark validate-evidence \
  --request-validity "$RUN_LOCAL/validity.json" \
  --parity-receipt "$DATASET_PARITY_RECEIPT" \
  --snapshot "$ANALYSIS/snapshot-validation.json" \
  --metric-summary "$ANALYSIS/metric-summary.json" \
  --output "$ANALYSIS/environment.json"
```

`validate-evidence` exit 2는 삭제 사유가 아니라 invalid 봉인 신호다. Environment가 valid하고 complete
request matrix가 있을 때만 다음을 실행한다.

```bash
.venv/bin/pertinence benchmark analyze \
  --records "$RUN_LOCAL/requests.jsonl" \
  --run-manifest "$RUN_LOCAL/run.json" \
  --environment-manifest "$ANALYSIS/environment.json" \
  --output "$ANALYSIS/summary.json" \
  --baseline-arm static-resnet-router --comparison-arm dispatcher-000733 \
  --repeats 5 --samples 2000 --seed 20260920
```

Complete matrix가 없으면 현재 `benchmark analyze`가 실패하며 전용 invalid-summary 생성 CLI는 없다.
이 경우 numeric effect를 만들지 말고 raw/validity/inventory를 그대로 보존한다. 별도로 review된
create-only invalid-summary 절차가 없으면 이 단계에서 중단한다. 없는 summary를 성공 결과처럼
대체하지 않는다.

## 19. 추가 증거 create-only publish

**GCS CREATE-ONLY · STOP GATE**

Job object와 겹치지 않게 추가 파일은 `analysis/`, `external-evidence/`, `receipts/` 아래에만 staging한다.

```bash
find "$POST_ANALYSIS_STAGING" -type f -print0 | sort -z |
while IFS= read -r -d '' file; do
  relative=${file#"$POST_ANALYSIS_STAGING/"}
  gcloud storage cp --if-generation-match=0 \
    "$file" "$BENCHMARK_OUTPUT_URI/$relative"
done
```

각 generation/hash를 receipt에 기록한다. 부분 실패 시 기존 object를 삭제하지 않고 가능한 경우 동일
prefix에 create-only `INVALID.json`을 추가한 뒤 중단한다. 이 upload는 final과 C101 각각의
`BENCHMARK_OUTPUT_URI`를 설정해 별도로 실행하며 두 staging tree를 합치지 않는다.

## 20. 최종 stop과 scale-to-zero 확인

**MUTATION → READ-ONLY FINALIZATION**

성공, invalid, 오류 어느 경우에도 static 복귀와 stop을 시도한다.

```bash
cloud/kserve/scripts/switch-active.sh "$EXPECTED_CONTEXT" static "$REQUEST_JSON" || true
cloud/kserve/scripts/stop.sh "$EXPECTED_CONTEXT"
```

`stop.sh`는 다섯 ISVC와 final/C101 Job을 모두 처리하며, 일부 실패에도 나머지를 계속 시도하고
read-back한다. 다음을 `20-stop/`에 봉인한다.

- 다섯 ISVC의 `serving.kserve.io/stop=true`
- 두 Job이 suspended 또는 terminal
- serving/loadgen active Pod 0
- `serving-pool=0`, `loadgen-pool=0`, `pipeline-pool=0`
- KServe/cert-manager controller와 ISVC/ConfigMap/Job/artifact 보존
- KFP namespace/PVC/PV 보존
- exposure audit 재통과

제한 시간 안에 node가 0이 되지 않으면 resource를 삭제하지 않는다. Scheduling, PDB, autoscaler event와
node inventory를 남기고 incomplete/invalid로 종료한다.

## 실패 처리 불변식

- GCS object/prefix는 overwrite/delete하지 않는다. 부분 결과도 `INVALID.json`과 함께 보존한다.
- Cloud Build tag는 재사용·삭제하지 않는다. 재시도는 새 승인 tag다.
- Promotion과 ConfigMap은 동일 fingerprint만 재사용한다. 다른 bytes면 중단한다.
- IAM은 정확한 binding만 추가한다. Broad binding은 자동 제거하지 않고 중단한다.
- Terraform은 봉인한 saved plan만 적용한다. 일반 re-plan/apply로 부분 실패를 수습하지 않는다.
- Addon partial apply는 동일 verified render의 idempotent 재적용만 허용한다. Uninstall/CRD 삭제는 금지다.
- 실패 Job을 삭제하거나 같은 ID로 다시 실행하지 않는다.
- Invalid run도 raw JSONL/Parquet, logs, events, metrics, snapshot과 generation receipt를 보존하며 수치
  효과를 해석하지 않는다.
- 최종 stop이 일부 실패하면 성공으로 보고하지 않는다. 삭제 대신 실패 원인과 live inventory를 남긴다.

두 dataset이 각자 sealed valid summary를 갖고, 모든 hash/digest/generation이 추적 가능하며, exposure와
최종 stop/scale-to-zero가 확인된 경우에만 20단계 완료로 판정한다.
