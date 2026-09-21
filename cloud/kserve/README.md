# KServe dispatcher benchmark assets

이 디렉터리는 기존 GKE cluster에 CPU 전용 serving/load-generator workload를 내부 전용으로
배포하기 위한 정본이다. 대상 버전은 Kubernetes 1.35, KServe `v0.20.0` Standard mode,
cert-manager `v1.20.4`이다. 어떤 스크립트도 cluster를 만들거나 node pool을 변경하지 않는다.

## 안전 경계

- 모든 workload image와 OCI modelcar는 `@sha256:<64 hex>` reference만 허용한다.
- `InferenceService`는 replica 1, HPA 비활성(`minReplicas == maxReplicas == 1`)이고
  `serving-pool`에만 배치된다. load generator는 `loadgen-pool`에만 배치된다.
- KServe가 Ingress/Gateway를 만들지 않도록 전역 `disableIngressCreation: true`와
  `enableGatewayApi: false`를 사용한다. 유일한 benchmark entry point는 ClusterIP Service
  `pertinence-active`이다.
- `serving-runner`에는 GCP IAM annotation이나 JSON key를 부여하지 않는다. Artifact Registry
  pull은 기존 node service account만 사용한다.
- 현재 cluster는 NetworkPolicy enforcement가 꺼져 있다. 따라서 이 디렉터리는
  `NetworkPolicy`를 포함하지 않으며 격리를 주장하지 않는다. enforcement 활성화에는 node
  재생성이 필요하므로 이번 benchmark 범위 밖이다.
- `audit-exposure.sh`가 배포 전과 후 모두 `LoadBalancer`, Ingress, Gateway, 외부 forwarding
  rule 0개를 확인해야 한다. project에 다른 합법적인 외부 forwarding rule이 있다면 이
  benchmark를 같은 project에서 실행하지 말고 별도 검토한다.

## 디렉터리

```text
cloud/kserve/
├── install/                 # chart lock/values와 offline render 절차
├── runtime/                 # 공통 Python 3.11 non-root image
├── modelcar/                # /models OCI image build template
├── manifests/               # namespace/KSA, 5 ISVC, active Service, loadgen Job
└── scripts/                 # render validation, audit, switch/rollback, stop
```

## 입력 고정과 render

실제 실행을 시작하기 전에 [로컬 세션 게이트](../docs/kserve-session-gate.ko.md)로 사람 승인값,
검토된 integration/parity driver, linux/amd64 image inventory와 새 evidence root를 봉인한다. 이
gate는 cloud API를 호출하지 않으며 하나라도 미확정이면 evidence directory도 만들지 않는다.
Gate가 성공한 뒤의 artifact 통합부터 최종 stop까지는
[실제 실행 절차](../docs/kserve-benchmark-execution.ko.md)를 순서대로 따른다.

Kubernetes workload나 node pool을 변경하기 전에는 현재 context/project/cluster, KFP PVC/PV와
workflow, regional CPU/N2 quota, node SA image-pull IAM, benchmark KSA RBAC, external exposure를
create-only receipt로 봉인한다. 출력 디렉터리가 이미 있으면 실패한다.

```bash
.venv/bin/python cloud/kserve/scripts/preflight-gke.py \
  --context EXPECTED_KUBE_CONTEXT \
  --project pertinence-509116 --region asia-northeast3 \
  --zone asia-northeast3-a --cluster EXPECTED_CLUSTER \
  --repository pertinence --principal EXPECTED_PRINCIPAL \
  --node-service-account EXPECTED_NODE_GSA \
  --output /secure/evidence/08-preflight/sealed
```

1. Integration의 `build-bundle/` 파일을 create-only GCS object로 올린 뒤 각 실제 generation을
   `PATH=gs://OBJECT#GENERATION` 형식으로 주어 build input manifest를 봉인한다. 입력 경로는 세 expert의
   `model.onnx`/`model.sha256`과 evaluated JSON 4개, `dispatcher-000733.npz`로 정확히 제한된다.

   ```bash
   .venv/bin/python cloud/kserve/scripts/build_inputs.py seal \
     --bundle /secure/evidence/integration/build-bundle \
     --object experts/shufflenetv2_x0_5/model.onnx=gs://BUCKET/PREFIX/experts/shufflenetv2_x0_5/model.onnx#GENERATION \
     --object ... \
     --output /secure/evidence/build-input-manifest.json
   ```

   Manifest 자체도 create-only로 publish하고 그 `gs://...json#GENERATION`을 Cloud Build의
   `_BUILD_INPUT_MANIFEST_URI`로 넘긴다. Build는 manifest에 적힌 object generation만 내려받아
   size/SHA-256을 재검증하며 `build-input-manifest.json`과 `build-input-receipt.json`을 결과에 남긴다.
2. `cloudbuild.yaml`에 승인된 linux/amd64 Python/BusyBox base digest와 위 manifest URI를 넘겨 공통
   runtime, 세 ONNX modelcar와 `dispatcher-000733` modelcar를 build/push한다.
3. build가 만든 `deployment-images.json`에서 repository/digest, linux/amd64, non-root user,
   Python/NumPy/ONNX Runtime version과 CLI smoke receipt를 검토한다.
4. `manifests/images.env.example`을 복사하되 복사본은 commit하지 않고 모든 reference와
   promotion fingerprint, sealed promotion/config의 base64, trace/run GCS URI를 실제 값으로 채운다.
5. 다음 명령으로 render한다. renderer는 placeholder/all-zero digest, tag-only image, 미해결
   변수, 불안전한 path를 거부하고 결과를 create-only directory에 쓴다.

```bash
cloud/kserve/scripts/render-manifests.sh \
  /secure/path/benchmark-images.env \
  /tmp/pertinence-kserve-rendered
```

render된 `00-platform.yaml`과 `10-workloads.yaml`을 검토한 다음 pre-audit, server-side dry-run,
apply, Ready wait, post-audit 순서로 진행한다. example 파일의 digest는 의도적인 sentinel이라
render에 사용할 수 없다.

```bash
cloud/kserve/scripts/audit-exposure.sh EXPECTED_KUBE_CONTEXT pertinence-509116
kubectl apply --server-side --dry-run=server -f /tmp/pertinence-kserve-rendered/00-platform.yaml
kubectl apply --server-side --dry-run=server \
  -f /tmp/pertinence-kserve-rendered/05-benchmark-workload-identity.yaml
kubectl apply --server-side --dry-run=server -f /tmp/pertinence-kserve-rendered/10-workloads.yaml
kubectl apply --server-side -f /tmp/pertinence-kserve-rendered/00-platform.yaml
kubectl apply --server-side \
  -f /tmp/pertinence-kserve-rendered/05-benchmark-workload-identity.yaml
kubectl apply --server-side -f /tmp/pertinence-kserve-rendered/10-workloads.yaml
kubectl -n pertinence-serving wait --for=condition=Ready --timeout=15m \
  inferenceservice --all
cloud/kserve/scripts/audit-exposure.sh EXPECTED_KUBE_CONTEXT pertinence-509116
```

CIFAR-10.1은 별도로 렌더한 `10-workloads.yaml`에서 loadgen Job 하나만 추출한다. 아래 명령은
`metadata.name`만 `pertinence-loadgen-cifar101-v6`로 바꾸며, 그 외 필드의 canonical bytes가
동일한지 재검증한다. 출력 YAML이나 receipt가 이미 있으면 덮어쓰지 않고 실패한다.

```bash
.venv/bin/python cloud/kserve/scripts/derive-cifar101-job.py \
  --source /secure/evidence/render-c101/10-workloads.yaml \
  --output /secure/evidence/render-c101/pertinence-loadgen-cifar101-v6.yaml \
  --receipt /secure/evidence/render-c101/pertinence-loadgen-cifar101-v6.receipt.json
```

`benchmark-runner`의 Workload Identity IAM binding은 Kubernetes manifest가 소유하지 않는다.
별도 create-only IAM 절차에서 trace prefix read와 해당 benchmark run prefix write만 허용하고,
annotation의 GSA가 그 binding과 일치하는지 apply 전에 확인한다.
loadgen은 Workload Identity로 trace GCS prefix를 직접 읽고 결과를 benchmark-ID GCS prefix에
`ifGenerationMatch=0`으로 직접 쓴다. 작업 공간은 `emptyDir`이며 PVC staging/uploader는 없다.
promotion은 modelcar가 아니라 immutable ConfigMap에서 read-only mount한다. 충돌이나 부분 upload가
발생하면 객체를 삭제하지 않고 `INVALID.json` 생성을 시도한 뒤 실패로 종료한다.

## active alias 전환

초기 selector는 static ResNet router이다. 실제 label selector를 바꾸기 전에 새 ISVC Ready를
확인하고, 바꾼 뒤에는 port-forward를 통해 새 TCP connection으로 20회 응답 identity를 검사한다.
하나라도 섞이면 이전 selector로 자동 rollback한다. payload에는 label을 넣지 않는다.

```bash
cloud/kserve/scripts/switch-active.sh EXPECTED_KUBE_CONTEXT dispatcher request.json
cloud/kserve/scripts/switch-active.sh EXPECTED_KUBE_CONTEXT static request.json
```

순서는 `static -> dispatcher-000733 -> static`이다. benchmark client의 기존 connection pool은
각 전환 전에 닫고, drain 완료 후 새 pool을 만들어야 한다.

## 종료

`stop.sh`는 다섯 ISVC에 `serving.kserve.io/stop=true`를 붙이고 두 loadgen Job을 모두 suspend한다.
KServe/cert-manager, ISVC object, modelcar와 결과 artifact는 삭제하지 않는다.

```bash
cloud/kserve/scripts/stop.sh EXPECTED_KUBE_CONTEXT
```

그 뒤 active Pod가 없고 `serving-pool`/`loadgen-pool`이 0으로 축소되는지 확인한다. KServe
Standard mode의 HTTP scale-to-zero 미지원과 NetworkPolicy 미적용은 roadmap 미완료 항목으로
남는다.

## 실행 증거

Cold-start는 steady-state와 별도로 workload apply부터 다섯 ISVC Ready까지의 wall time 및 active
router 첫 20요청을 기록한다. `measure-cold-start.sh`의 출력 경로는 존재하면 거부된다.

```bash
cloud/kserve/scripts/measure-cold-start.sh EXPECTED_KUBE_CONTEXT \
  /tmp/pertinence-kserve-rendered/10-workloads.yaml request.json \
  /secure/evidence/cold-start-001
```

steady-state 중에는 Job의 native sidecar collector가 5초 간격으로 필터링된 cAdvisor counter와
traffic/arm/repeat marker를 공유 `emptyDir`에 기록한다. 아래 외부 collector는 시작/종료 node·Pod·ISVC
inventory, restart/event와 KFP workflow snapshot, Metrics API CPU/memory, 관련 두 노드의 raw cAdvisor
metric(usage, memory, CFS throttling 포함)을 보존한다. Metrics Server 또는 node proxy RBAC가 없으면
script가 실패하며 해당 run은 성능 판정에 쓰지 않는다. 예시는 5초 간격 240개 sample이다.

```bash
cloud/kserve/scripts/collect-evidence.sh EXPECTED_KUBE_CONTEXT \
  /secure/evidence/steady-run-001 240 5
```

run manifest에는 이 directory의 재귀 SHA-256 inventory와 cluster/GKE/KServe/cert-manager version,
promotion fingerprint, image/model digest를 함께 기록한다. 시작/종료 inventory에서 Pod restart, node
identity/topology 변경, active KFP workflow가 발견되거나 raw metric에 빈 구간이 있으면 run을
무효화한다.
