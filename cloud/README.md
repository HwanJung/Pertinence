# PERTINENCE GKE cloud environment

## 1. 문서 목적

이 디렉터리는 PERTINENCE의 cloud 전용 구성과 운영 계약을 모으는 정본이다. 첫 GKE
환경은 다음 두 실행을 지원한다.

1. Kubeflow Pipelines(KFP)에서 dataset과 expert bundle을 준비하고 dispatcher를 학습·평가한다.
2. 평가를 통과한 expert 후보와 dispatcher를 KServe에 배포하고 실제 latency, throughput,
   resource 사용량과 전환 동작을 측정한다.

두 실행은 하나의 GKE cluster와 Artifact Registry를 공유하지만 namespace, Kubernetes
ServiceAccount, GCS 권한과 workload node pool은 분리한다. Offline accuracy/MFLOPs 결과와 online
latency 결과를 섞지 않으며, KFP의 final 평가를 통과하지 않은 dispatcher는 serving 입력으로
사용하지 않는다.

이 문서는 cluster와 내부 resource의 목표 상태를 정의한다. 아직 구현되지 않은 manifest나
application은 구현된 것처럼 가정하지 않고 마지막의 구현 상태 표에 명시한다.

## 2. 고정 환경

| 항목 | 값 | 이유 |
|---|---|---|
| Google Cloud project | `pertinence-509116` (`Pertinence`) | 실험 resource의 단일 소유 경계 |
| Project number | `423661327395` | Workload Identity principal 구성 |
| GKE mode | Standard, zonal | node pool별 scale-to-zero와 고정 benchmark topology 제어 |
| Region / zone | `asia-northeast3` / `asia-northeast3-a` | bucket, registry, compute 위치 통일 |
| Release channel | Regular | 지원되는 Kubernetes patch 자동 갱신 |
| Workload identity pool | `pertinence-509116.svc.id.goog` | service-account key 없는 GCS 접근 |
| KFP SDK | `2.17.0` | 저장소 lock과 compile test 기준 |
| KFP backend | v2 IR 호환 standalone | 현재 `submit.py`와 single-user API 계약 |
| KServe | `0.20.0`, Standard mode | Knative 없이 고정 replica 성능 측정 |
| Container architecture | `linux/amd64` | 현재 component build 계약 |

KServe 0.20은 Kubernetes 1.32 이상을 요구한다. Cluster 생성 시 특정 patch를 문서에 고정하기보다
Regular channel에서 제공되는 버전을 사용하되, 실제 control-plane/node/KFP/KServe 버전을 각
실험 report에 기록한다. KFP와 KServe upgrade는 실험 도중 수행하지 않고 별도의 maintenance
window에서 smoke test 후 적용한다.

이 환경은 single-user 연구 환경이다. KFP API와 UI를 public LoadBalancer로 노출하지 않고
`kubectl port-forward`로만 접근한다. 다중 사용자 인증, 외부 공개 endpoint와 production SLA는
현재 범위가 아니다.

## 3. 전체 구조

```text
Google Cloud project: pertinence-509116

GCS raw bucket ───────────────┐
Artifact Registry ────────────┼──► KFP tasks on pipeline-pool
                              │        │
                              │        ▼
                              │   evaluated dispatcher bundle
                              │        │
                              │        ▼ explicit promotion
                              └──► GCS serving bucket
                                       │
                                       ▼
Load generator ──► PERTINENCE router ──┬──► ShuffleNet expert
                    on serving-pool    ├──► MobileNet expert
                                       └──► ResNet56 expert
                                               │
                                               ▼
                                      benchmark report bucket

Always on: system-pool
Scale to zero while idle: pipeline-pool, serving-pool, loadgen-pool
```

KFP와 KServe controller는 상시 system node에서 실행한다. 실제 학습, inference server와 부하
발생기는 서로 다른 tainted node pool에서 실행한다. Pipeline run과 serving benchmark는 기본적으로
동시에 실행하지 않는다. 노드가 분리돼도 control plane, Artifact Registry, GCS와 network egress를
공유하므로 동시 실행은 latency 실험에 불필요한 변동을 만들 수 있다.

## 4. Node pool 설계

### 4.1 목표 구성

| Pool | Machine | 크기 | Taint | 실행 대상 |
|---|---|---:|---|---|
| `system-pool` | `e2-standard-4` | 고정 1 | 없음 | KFP, importer/driver, Argo, MySQL, SeaweedFS, controller |
| `pipeline-pool` | `n2-standard-8` | 0–2 | `workload=pertinence-pipeline:NoSchedule` | 일곱 pipeline container component |
| `serving-pool` | `n2-standard-8` | 0–3 | `workload=pertinence-serving:NoSchedule` | Router와 expert `InferenceService` Pod |
| `loadgen-pool` | `n2-standard-4` | 0–1 | `workload=pertinence-loadgen:NoSchedule` | 고정 trace를 재생하는 benchmark Job |

모든 pool은 on-demand VM으로 시작한다. 현재 dispatcher search는 자동 retry가 없고, Spot VM 회수는
학습 손실과 latency 분산을 만들기 때문에 첫 재현과 성능 결과에는 사용하지 않는다. Spot 비교가
필요하면 별도 pool과 별도 실험 ID로 수행한다.

`pipeline-pool`의 한 node는 현재 최대 task request인 4 CPU/12 GiB를 수용한다. Pipeline 시작의
`prepare-dataset-bundle` 2 CPU/4 GiB와 `export-expert-bundle` 4 CPU/8 GiB가 병렬 실행될 수 있으므로
8 vCPU/32 GiB를 기준으로 한다. 두 번째 node는 scheduler와 DaemonSet overhead, 재시도 및 향후
병렬화를 위한 상한이며 평상시에는 생성되지 않는다.

GKE Standard cluster 전체는 system Pod 때문에 자동으로 0 node가 되지 않는다. 대신 세 workload
pool만 0으로 축소한다. 따라서 idle 비용에는 control plane, `system-pool` 한 대, KFP PVC와 GCS
storage가 계속 포함된다.

### 4.2 Scheduling 계약

Taint만으로는 원하는 pool을 선택할 수 없고 toleration만으로는 다른 pool 배치를 막을 수 없다.
모든 workload는 node selector와 toleration을 함께 사용한다.

| Workload | Node selector | Toleration |
|---|---|---|
| KFP container component | `cloud.google.com/gke-nodepool=pipeline-pool` | `workload=pertinence-pipeline:NoSchedule` |
| KServe predictor/router | `cloud.google.com/gke-nodepool=serving-pool` | `workload=pertinence-serving:NoSchedule` |
| Benchmark Job | `cloud.google.com/gke-nodepool=loadgen-pool` | `workload=pertinence-loadgen:NoSchedule` |

KFP에서는 `kfp-kubernetes`의 node selector와 toleration modifier를 실제 연산을 수행하는 일곱
container component에 적용한다. KFP 2.17 backend는 importer executor의 Kubernetes platform
설정을 Pod에 적용하지 않으며, 현재 importer는 URI만 등록하므로 importer와 driver/root Pod는
`system-pool`에서 실행하도록 허용한다. Compile test는 일곱 component executor를 검사한다.
KFP의 `cacheNodeRestrictions=false` 기본값 때문에 cache-hit dummy Pod도 system pool에서 실행될 수
있으며, 실제 component 연산이 아니므로 이 동작을 허용한다.

KServe `InferenceService`도 predictor spec에 같은 selector와 toleration을 설정한다. KServe
controller와 cert-manager에는 workload toleration을 주지 않아 `system-pool`에 남긴다.

### 4.3 Scale-to-zero와 측정 구간

- KFP run 제출로 Pending task가 생기면 `pipeline-pool`이 0에서 확장된다.
- Pipeline의 모든 Pod가 종료되고 autoscaler 유예 시간이 지나면 `pipeline-pool`이 0으로 돌아간다.
- KServe Standard mode의 Pod replica는 자동으로 0이 되지 않는다. 실험이 끝나면
  `InferenceService`를 제거하거나 명시적으로 종료해야 `serving-pool`이 0이 될 수 있다.
- 성능 측정 중에는 serving node 수, Pod replica와 CPU/memory를 고정한다. Cluster autoscaler나
  HPA가 측정 중 topology를 바꾸지 못하게 하고 KServe annotation
  `serving.kserve.io/autoscalerClass: none`을 사용한다.
- Cold-start 실험은 node 0→1, image pull, model load, readiness 시간을 각각 기록한다. Steady-state
  latency 측정에는 이 시간을 포함하지 않고 별도 지표로 보고한다.
- Load generator는 serving node에 배치하지 않는다. Client CPU 경쟁이 server latency로 집계되는
  것을 방지하기 위해 `loadgen-pool`에서 실행한다.

## 5. Namespace와 Kubernetes resource

| Namespace | 주요 resource | Cloud 권한 |
|---|---|---|
| `kubeflow` | KFP API/UI, Argo Workflow, MySQL, ML Metadata, SeaweedFS, task Pod | `pipeline-runner`만 raw 읽기/run 쓰기 |
| `kserve` | KServe CRD/controller/webhook | application bucket 권한 없음 |
| `cert-manager` | KServe webhook certificate | application bucket 권한 없음 |
| `pertinence-serving` | expert/router `InferenceService`, catalog ConfigMap | serving artifact 읽기 전용 |
| `pertinence-benchmark` | load generator Job, trace ConfigMap | benchmark report 쓰기 |

### 5.1 ServiceAccount

| Identity | 종류 | 용도 |
|---|---|---|
| `pertinence-gke-nodes@pertinence-509116.iam.gserviceaccount.com` | IAM SA | node logging/monitoring과 private image pull |
| `kubeflow/pipeline-runner` | Kubernetes SA | KFP task 실행과 GCS artifact I/O |
| `pertinence-serving/serving-runner` | Kubernetes SA | 승인된 model/dispatcher bundle 읽기 |
| `pertinence-benchmark/benchmark-runner` | Kubernetes SA | benchmark report 쓰기 |

Node IAM SA에는 최소 GKE node 역할과 Artifact Registry reader만 부여한다. Application Pod는 node
identity를 사용하지 않고 Workload Identity Federation principal로 bucket 권한을 받는다. JSON
service-account key는 생성하거나 Pod에 mount하지 않는다.

`pipeline-runner`는 KFP 배포판이 제공하는 Role과 RoleBinding을 그대로 사용한다. 같은 이름의 빈
ServiceAccount만 별도로 만들지 않는다. KFP task 실행 전 다음 권한을 검증한다.

KFP 2.17 launcher는 외부 GCS artifact를 내려받을 때 bucket-level `storage.objects.list`를 먼저
호출한다. 따라서 아래 pipeline 전용 raw/run bucket binding의 IAM condition은 bucket resource
자체 또는 허용된 experiment object prefix인 경우를 허용하는 복합식으로 만든다. 목록 조회 요구를
충족하면서 객체 read/write는 immutable prefix로 제한하며, binding은 cluster teardown 때 회수한다.

```text
raw bucket:       pipeline-runner → roles/storage.objectViewer
run bucket:       pipeline-runner → roles/storage.objectUser
serving bucket:   serving-runner  → roles/storage.objectViewer
benchmark bucket: benchmark-runner → roles/storage.objectUser
```

KFP 결과를 serving bucket으로 승격하는 주체는 runtime KSA가 아니라 운영자 또는 별도 promotion
job이다. 학습 pipeline이 곧바로 serving 정본을 덮어쓰지 못하게 한다.

### 5.2 KFP 내부 resource

KFP는 `kubeflow` namespace에 single-user standalone backend로 설치한다.

- API/UI와 persistence agent
- Argo Workflow controller
- MySQL과 ML Metadata
- SeaweedFS
- `pipeline-runner` RBAC
- `kfp-launcher` ConfigMap
- MySQL 20 GiB PVC
- SeaweedFS 20 GiB PVC

KFP API server의 pipeline definition/metadata 저장소와 task artifact의 GCS pipeline root는 서로
다르다. `kfp-launcher.defaultPipelineRoot`는 run bucket으로 설정하지만 MySQL과 SeaweedFS PVC를
자동으로 GCS로 대체하지 않는다. Cluster를 제거하기 전에는 필요한 KFP metadata를 별도로
백업한다.

현재 pipeline은 다음 순서로 실행된다.

```text
prepare-dataset-bundle ─┐
export-expert-bundle ───┼─► validate-run ─► build-routing-dataset
run-config ─────────────┘                         │
                                                  ▼
                                  train-dispatcher ─► evaluate-final
                                                               │
                                                               ▼
                                                    register-dispatchers
```

`register-dispatchers`의 SQLite 결과는 run artifact이지 여러 run이 공유하는 중앙 Registry가 아니다.
Serving promotion은 evaluated bundle의 lineage, expert order, preprocessing, checkpoint hash와 image
digest를 다시 검증해야 한다.

### 5.3 KServe 내부 resource

첫 성능 실험은 KServe Standard mode를 사용한다. Knative, Istio, scale-to-zero와 request queueing을
동시에 도입하면 PERTINENCE routing 비용과 platform overhead를 분리하기 어렵기 때문이다.
Standard mode에는 cert-manager가 필요하며, 외부 ingress는 만들지 않는다. Load generator는
cluster 내부 Service/ClusterIP로 요청한다.

초기 serving topology는 다음과 같다.

```text
pertinence-router-<dispatcher-id>
  ├── frozen feature extractor
  ├── selected linear dispatcher head
  ├── immutable expert index → service catalog
  └── internal calls
      ├── expert-shufflenet-v2-05
      ├── expert-mobilenet-v2-05
      └── expert-resnet56
```

Router가 feature extractor와 동일한 expert를 선택하면 이미 계산한 logits를 반환하고 같은 model을
두 번 실행하지 않는다. 다른 expert를 선택했을 때만 해당 internal service를 호출한다. 각 요청에는
최소한 다음 값을 기록한다.

```text
experiment_id
request_id
dispatcher_id
dispatcher_version
selected_expert_id
route_latency_ms
expert_latency_ms
total_latency_ms
response_status
```

Dispatcher head 파일 하나만 배포하지 않는다. 다음 값이 같은 immutable serving bundle에 있어야
한다.

- Dispatcher checkpoint와 SHA-256
- Feature extractor ID/version/hash
- FC output index 순서와 expert service mapping
- Preprocessing contract와 class index 의미
- Dataset/run/split fingerprint
- Offline search/final metric
- Container image digest와 runtime version

## 6. Google Cloud resource

### 6.1 GCS bucket

Bucket 이름은 전역 유일해야 하므로 실제 생성 직전에 사용 가능 여부를 확인한다. 기본 후보는
다음과 같다.

| 변수 | 기본 후보 | 내용 |
|---|---|---|
| `RAW_BUCKET` | `pertinence-509116-pertinence-raw` | CIFAR archive, checkpoint, RunConfig |
| `RUN_BUCKET` | `pertinence-509116-pertinence-runs` | KFP output artifact와 run-scoped registry |
| `SERVING_BUCKET` | `pertinence-509116-pertinence-models` | 승인·승격된 expert/dispatcher bundle |
| `BENCHMARK_BUCKET` | `pertinence-509116-pertinence-benchmarks` | 요청 trace, raw measurement, summary |

모든 bucket은 `asia-northeast3`에 만들고 uniform bucket-level access와 object versioning을 사용한다.
URI를 immutable identifier로 간주하지 않고 모든 pipeline, promotion과 serving manifest에 object
generation 또는 SHA-256을 기록한다. Retention lock은 복구 불가능하므로 smoke run과 삭제 정책을
검증한 뒤 별도 승인으로 설정한다.

### 6.2 Artifact Registry

`asia-northeast3`의 Docker repository `pertinence`를 공용 registry로 사용한다.

```text
components/kfp:<build-id>
serving/router:<build-id>
serving/expert:<build-id>
benchmark/loadgen:<build-id>
```

실행 manifest와 KFP IR에는 tag가 아니라 manifest digest를 기록한다. Build provenance와 source Git
SHA를 image label 및 실험 report에 남긴다.

### 6.3 Network와 노출 정책

- GKE는 VPC-native cluster로 구성한다.
- 첫 단계의 KFP UI/API는 port-forward로만 접근한다.
- KServe benchmark endpoint는 cluster 내부에서만 호출한다.
- `LoadBalancer` Service와 public ingress는 기본적으로 금지한다.
- NetworkPolicy로 benchmark namespace에서 router로, router에서 승인된 expert service로만 통신을
  허용한다.
- Private node를 선택하면 Artifact Registry, GHCR와 GCS 접근을 위한 Cloud NAT 또는 Private
  Google Access 구성을 함께 검증한다. Network 형태는 cluster 생성 전에 비용과 접속 방식을 보고
  최종 확정한다.

## 7. 두 실행의 lifecycle

### 7.1 Dispatcher 학습 pipeline

1. Raw asset의 SHA-256을 로컬 manifest와 대조한다.
2. Component image를 build하고 digest를 기록한다.
3. KFP IR을 `pipeline-pool` scheduling 정보와 함께 compile한다.
4. 작은 NSGA-II smoke RunConfig로 end-to-end 실행한다.
5. `pipeline-pool`이 0→1 이상으로 확장되고 일곱 container component가 해당 pool에서 실행되는지
   확인한다. Importer와 driver/root Pod는 `system-pool` 배치를 확인한다.
6. Smoke 결과를 production 결과와 다른 GCS prefix에 저장한다.
7. Production RunConfig의 50×50 search를 실행한다.
8. Final 평가와 lineage 검증을 통과한 bundle만 promotion 대상으로 표시한다.
9. Task 종료 후 `pipeline-pool`이 0으로 돌아가는지 확인한다.

Production run은 expert 3개의 60,000장 추론과 dispatcher 2,500회 학습을 수행한다. 입력 URI와 hash,
component image digest가 immutable할 때만 실패 복구용 KFP cache를 활성화한다.

### 7.2 KServe 성능 실험

1. 평가된 bundle 중 실험할 dispatcher ID를 명시적으로 선택한다.
2. Bundle과 expert artifact를 serving bucket의 새 immutable version으로 승격한다.
3. Expert service와 router image digest, resources, replica 수를 고정한다.
4. `serving-pool`과 `loadgen-pool`을 준비하고 실제 node 이름/machine/version을 기록한다.
5. Expert endpoint Ready와 router catalog/hash 일치를 검사한다.
6. Cold-start 시간을 별도 측정한다.
7. Warm-up 뒤 동일한 sample 순서, concurrency와 request rate로 각 비교군을 실행한다.
8. Raw request log와 server metric을 benchmark bucket에 원자적으로 저장한다.
9. Offline accuracy/MFLOPs와 online latency/throughput의 차이를 보고한다.
10. `InferenceService`와 loadgen Job을 제거하고 두 node pool이 0으로 돌아가는지 확인한다.

최소 비교군은 각 expert 직접 호출, 저비용/고정확도 dispatcher, operator-selected dispatcher다.
동일 실험 중에는 model replica, node 수, CPU frequency 계열, container digest와 request trace를
바꾸지 않는다.

## 8. 성능 측정 계약

각 실험은 다음 설정과 결과를 보존한다.

### 고정 입력

- `experiment_id`와 UTC 시작/종료 시각
- Git SHA와 모든 container digest
- Dispatcher/expert bundle SHA-256
- GKE control-plane/node version과 machine type
- Node 수, Pod replica, CPU/memory request와 limit
- Dataset/sample manifest hash
- Warm-up 횟수 또는 시간
- Concurrency, target RPS, duration, timeout
- 호출 경로와 protocol

### 필수 결과

- 성공/실패 요청 수와 오류 유형
- Throughput과 achieved RPS
- End-to-end p50/p90/p95/p99 latency
- Router, feature extractor와 selected expert latency
- Expert별 route 비율
- System Top-1 accuracy와 offline result 차이
- Pod CPU/memory, restart/OOM/throttling
- Node scale-up, image pull, model load와 readiness 시간
- Dispatcher 전환 시간, 실패 요청 수와 rollback 결과

Cold start가 포함된 결과와 steady-state 결과는 별도 표로 보고한다. Client와 server clock이 필요한
분산 측정은 clock skew를 기록하고, 기본 end-to-end latency는 load generator의 monotonic clock으로
측정한다.

## 9. 생성 및 검증 순서

1. Project billing, API, IAM과 regional quota를 read-only로 점검한다.
2. Network, node IAM SA, Artifact Registry와 네 bucket을 만든다.
3. Workload Identity가 활성화된 zonal Standard cluster와 `system-pool`을 만든다.
4. 세 workload pool을 min 0으로 만들고 label/taint를 확인한다.
5. KFP backend를 설치하고 system Pod와 PVC가 정상인지 확인한다.
6. cert-manager와 KServe Standard controller를 설치한다.
7. Namespace, KSA, Workload Identity와 bucket IAM을 적용한다.
8. KFP scheduler modifier와 KServe scheduling manifest를 적용한다.
9. 각 workload별 작은 smoke Pod로 0→1→0 node lifecycle을 검증한다.
10. KFP smoke run, production run, serving smoke, 고정 topology benchmark 순서로 진행한다.

Cluster 준비 완료 gate는 다음과 같다.

- 모든 controller Pod가 `system-pool`에서 Ready다.
- KFP PVC가 Bound이고 KFP API health check가 성공한다.
- `pipeline-runner`의 Workflow/RBAC와 GCS read/write 검사가 성공한다.
- Serving KSA는 serving bucket을 읽을 수 있지만 run/raw bucket을 쓸 수 없다.
- Benchmark KSA는 benchmark bucket에 쓸 수 있지만 model artifact를 변경할 수 없다.
- KFP container component, KServe Pod와 loadgen Job이 각각 지정 pool에서만 실행된다.
- Pipeline IR과 serving manifest에 mutable image tag가 없다.
- Workload 종료 후 workload pool이 모두 0으로 축소된다.

## 10. 운영과 삭제 정책

- Cluster autoscaler의 scale-down은 즉시 실행되지 않으므로 실험 종료 시 Pod/Job이 남지 않았는지
  먼저 확인한다.
- KFP metadata와 SQLite registration artifact는 중앙 Registry가 아니다. Cluster 삭제 전 필요한
  pipeline definition, run ID, report와 promoted bundle을 GCS에서 검증한다.
- Bucket retention policy를 lock하기 전에는 삭제/복구 절차를 smoke bucket에서 시험한다.
- Cluster 삭제와 bucket 삭제는 별도 작업으로 취급한다. Cluster를 지워도 연구 정본인 GCS object와
  Artifact Registry digest는 자동 삭제하지 않는다.
- 성능 실험 중 GKE/KFP/KServe 자동 upgrade를 피하도록 maintenance window를 설정한다.
- 비용 분석은 idle, KFP run, serving warm-up, steady-state benchmark 구간을 분리한다.

## 11. 저장소 구조와 구현 상태

Cloud 전용 file은 다음 구조로 확장한다.

```text
cloud/
├── README.md                    # 이 문서: architecture와 resource 정본
├── iac/                         # Terraform state bootstrap, GKE module과 환경별 profile
├── kfp/                         # KFP overlay와 launcher 설정
├── kserve/                      # controller, InferenceService, router manifest
└── benchmark/                   # load generator와 실험 config
```

`iac/`는 Terraform 1.16.3과 Google provider 7.42.0으로 기존 VPC를 참조하고 GKE, node service
account/IAM과 네 종류의 조건부 node pool을 관리한다. 현재 수동 생성 cluster와 system/pipeline
pool은 변경 없이 원격 state에 import되어 있다. KFP/KServe와 실험별 GCS IAM은 수명주기가 달라
Terraform GKE state에 넣지 않는다. 기존 cloud 관련 문서는 구현 세부의 참고 자료로 유지하지만,
상충할 경우 이 문서와 [`iac/README.md`](iac/README.md)의 계약을 우선한다.

| 항목 | 현재 상태 | 다음 작업 |
|---|---|---|
| GKE Terraform module과 원격 state | 구현·import 완료 | 신규 disposable 환경 생성 smoke |
| CPU/GCS KFP component와 7단계 DAG | 구현·production 검증 | 반복 실행 시 lineage 보존 |
| GCS Workload Identity | 구현·실환경 검증 | 실험별 prefix IAM 수명주기 유지 |
| KFP task node selector/toleration | 구현·compile/실환경 검증 | pool 이름 계약 유지 |
| system/pipeline pool | 구현·Terraform 관리 | 삭제 보호 유지 |
| serving/loadgen pool | Terraform에서 선택 지원 | workload 구현 후 신규 profile로 생성 |
| 작은 smoke RunConfig | 미구현 | production config와 별도 작성 |
| Serving promotion 계약 | 설계만 존재 | bundle manifest와 promotion command 구현 |
| KServe expert runtime | 미구현 | ONNX/PyTorch runtime 선택과 image 구현 |
| PERTINENCE router | 미구현 | feature 공유와 internal expert call 구현 |
| KServe manifest | 미구현 | Standard mode scheduling/resources 작성 |
| Load generator와 metric 수집 | 미구현 | 고정 trace 기반 benchmark 구현 |
| 중앙 Model Registry | 범위 밖 | run artifact와 promoted manifest 우선 사용 |

## 12. 관련 문서

- [GKE CPU + GCS KFP 실행 가이드](../pipelines/pertinence/docs/gke-cpu.ko.md)
- [구성 독립형 KFP 최종 계획](../pipelines/pertinence/docs/final-plan.ko.md)
- [Operator-Selected Cloud-PERTINENCE](../pipelines/pertinence/docs/operator-selected-cloud.ko.md)
- [현재 production RunConfig](../experiments/cifar10/configs/pipeline-run.yaml)
- [KFP Kubernetes scheduling API](https://kfp-kubernetes.readthedocs.io/en/kfp-kubernetes-2.17/source/kubernetes.html)
- [GKE cluster autoscaler](https://cloud.google.com/kubernetes-engine/docs/concepts/cluster-autoscaler)
- [GKE dedicated node pool](https://cloud.google.com/kubernetes-engine/docs/how-to/isolate-workloads-dedicated-nodes)
- [KServe 0.20 설치](https://kserve.github.io/website/docs/install/kserve-install)
- [KServe Standard mode autoscaling](https://kserve.github.io/website/docs/model-serving/predictive-inference/autoscaling/hpa-autoscaler)
