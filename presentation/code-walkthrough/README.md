# 발표용 핵심 코드 모음

발표 7~15페이지에서 구현을 설명할 때 열어 볼 파일을 모았습니다. 번호 순서로 읽으면
GKE 환경 → KFP 자동화 → Dispatcher 학습 → 온라인 추론 → 측정 흐름으로 이어집니다.

이 폴더는 **설명용 스냅샷**입니다. 원본 코드를 그대로 복사했으며 독립 실행에 필요한 모든
의존 모듈·모델·데이터는 포함하지 않습니다. 실행은 원본 저장소의 환경과 절차를 사용합니다.
Dockerfile의 COPY 경로와 Python import도 원본 저장소 기준으로 유지했습니다.
복사 시점의 작업 트리에는 미커밋 변경이 포함되어 있습니다.

- [발표 PDF](../../pertinence-integrated-presentation-v4.pdf)
- [7~15페이지 발표 대본](../presentation-script-pages-07-15.ko.md)
- [발표 개요](../../presentation_outline.md)
- [원본 실행 안내](../../README.md)

## 발표 페이지별 읽기 순서

| 페이지 | 설명할 내용 | 열어 볼 코드와 지점 |
|---|---|---|
| 7 | 역할별 노드 풀 분리와 CPU 실행 환경 | [benchmark-current.tfvars](01-infrastructure/cloud/iac/environments/dispatcher-current/profiles/benchmark-current.tfvars)의 `node_pools` → [main.tf](01-infrastructure/cloud/iac/modules/gke/main.tf)의 `google_container_node_pool` → [pipeline.py](02-kfp/pipelines/pertinence/pipeline.py)의 `_schedule_on_pipeline_pool` |
| 8 | KFP 전체 DAG와 컨테이너 입출력 | [pipeline.py](02-kfp/pipelines/pertinence/pipeline.py)의 `make_pipeline` → [components.py](02-kfp/pipelines/pertinence/components.py)의 `make_training_components` → [cli.py](03-training/src/pertinence/pipeline/cli.py)의 `run_command` |
| 8 | 라우팅 정답·Dispatcher 학습·후보 탐색 | [cache.py](03-training/src/pertinence/pipeline/cache.py)의 `build_routing_dataset` → [routing.py](03-training/src/pertinence/routing.py)의 `LinearDispatcher`, `paper_penalty_loss` → [training.py](03-training/src/pertinence/training.py)의 `train_linear_dispatcher` → [optimization.py](03-training/src/pertinence/optimization.py)의 `DispatcherNSGA2Problem` |
| 8 | 최종 평가와 결과 등록 | [search.py](03-training/src/pertinence/pipeline/search.py)의 `evaluate_final` → [registry.py](03-training/src/pertinence/pipeline/registry.py)의 `register_dispatcher_bundle`, `SQLiteRegistry` |
| 9 | Direct와 Distributed의 요청 경로 | [runtime.py](04-serving/src/pertinence/serving/runtime.py)의 `OnnxExpertRunner`, `DynamicRouter` → [10-workloads.yaml.tmpl](04-serving/cloud/kserve/manifests/10-workloads.yaml.tmpl)의 `InferenceService` |
| 10 | JSON 변환과 내부 RPC 계측 | [runtime.py](04-serving/src/pertinence/serving/runtime.py)의 `HttpExpertClient._predict` → [diagnostics.py](05-benchmark/src/pertinence/serving/diagnostics.py)의 `Diagnostics.span` |
| 11 | Binary·Keep-alive와 Fused | [binary.py](04-serving/src/pertinence/serving/binary.py)의 `encode`, `decode` → [runtime.py](04-serving/src/pertinence/serving/runtime.py)의 `HttpExpertClient`, `FusedDynamicRouter` → [http.py](04-serving/src/pertinence/serving/http.py)의 `make_server` |
| 12 | Hybrid의 로컬 특징 추출과 원격 Expert | [runtime.py](04-serving/src/pertinence/serving/runtime.py)의 `HybridDynamicRouter` → [transport_deployment.py](04-serving/src/pertinence/serving/transport_deployment.py)의 `render_services` → [runtime.py](04-serving/src/pertinence/serving/cli/runtime.py)의 `run_command` |
| 13~14 | 측정 조건·지연시간·선택 분포 해석 | [benchmark.py](05-benchmark/src/pertinence/serving/benchmark.py)의 `warm_up_arm`, `run_scheduled_arm` → [transport_suite.py](05-benchmark/src/pertinence/serving/transport_suite.py)의 `compare_runs`, `join_diagnostics` |
| 15 | 실측 비용 반영이라는 후속 과제 | [pipeline-run.yaml](02-kfp/experiments/cifar10/configs/pipeline-run.yaml)의 `routing.cost_key: mflops` → [optimization.py](03-training/src/pertinence/optimization.py)의 `DispatcherNSGA2Problem` |

## KFP에서 실제 학습 코드까지

```text
pipeline.py: make_pipeline
  → components.py: make_training_components (container command/args)
  → pertinence.pipeline.cli: run_command
      validate-run          → validation.py: validate_run_contract
      build-routing-dataset → cache.py: build_routing_dataset
      train-dispatcher      → search.py: train_dispatcher
                              → optimization.py: run_nsga2
                              → training.py: train_linear_dispatcher
      evaluate-final        → search.py: evaluate_final
      register-dispatchers  → registry.py: register_dispatcher_bundle
```

DatasetBundle·ExpertBundle·RunConfig를 Importer로 등록한 뒤 계산 컴포넌트 5개를 실행합니다.
입력 검증과 분할 → 특징·Expert 예측·라우팅 정답 캐시 → Dispatcher 학습과 후보 탐색 →
checkpoint 복원 및 final 평가 → 실행별 결과 등록 순서입니다. Expert는 고정하고 Dispatcher를
학습합니다. 탐색에는 train/search 캐시를 전달하고 final 평가는 별도 단계에서 수행합니다.

현행 기본 DAG는 **5단계**입니다. 2026-09-20 검증 실행의 **7단계 legacy DAG**는 앞에
CIFAR-10 데이터 변환과 Expert ONNX 내보내기를 포함했습니다. 발표 중 두 구조를 혼동하지 않도록
구분하며, 이 모음에서는 현행 경로를 중심으로 설명합니다. `components.py`와 CLI에 함께 정의된
legacy 진입점은 원본 보존을 위해 남아 있지만, 별도의 legacy 구현 파일은 복사하지 않았습니다.
과거 실행 근거는 [GKE 실행 기록](../../cloud/docs/runs/2026-09-20-dispatcher-gke.ko.md)을 봅니다.

## 서빙 요청 흐름

| 구조 | 코드에서 볼 흐름 | 내부 RPC |
|---|---|---|
| Direct | `http.py`의 `ExpertModel` → `OnnxExpertRunner`로 ResNet56 실행 | 부하 생성기에서 Expert로 직접 요청, 내부 RPC 없음 |
| Distributed | `DynamicRouter` → 원격 특징 추출 → Dispatcher → 선택 Expert | 선택 Expert가 특징 추출 모델이면 결과 재사용, 그 외에는 총 2회 |
| Fused | `FusedDynamicRouter` → 로컬 특징 추출·Dispatcher·Expert | 없음 |
| Hybrid | `HybridDynamicRouter` → 로컬 특징 추출·Dispatcher → 선택 Expert | 로컬 모델 선택 시 없음, 원격 Expert 선택 시 1회 |

`StaticRouter`는 고정 Expert를 원격 호출하는 별도 대조군 구현이며, 발표의 Direct 경로와는
구분합니다. Direct 조건은 `benchmark-single-arm-v2.yaml`의 `resnet56-direct`에서 확인합니다.

발표의 선택 분포에서는 ShuffleNet 종료가 없었으므로 Distributed는 요청당 2회,
Hybrid는 1회가 됩니다. 일반 코드 흐름의 호출 수와 이 실험의 호출 수를 구분합니다.
`HttpExpertClient`는 JSON/Binary 데이터 표현과 연결 재사용을 담당하고,
`http.py`는 서버의 응답과 연결 처리를 담당합니다.

KServe 기본 템플릿만으로 Fused·Hybrid 설정이 모두 표현되지는 않습니다.
`transport_deployment.py`의 `render_services`가 비교 조건에 따라 서비스 구성과 실행 인자를
변환하고, 서빙 CLI가 해당 Router를 생성하는 연결까지 함께 보여줍니다.
매니페스트는 `${...}` 치환이 필요한 **배포 템플릿**이며 배포 완료 증거가 아닙니다.

## 측정 결과를 설명할 때

설정 파일은 실험 의도를, benchmark·diagnostics 코드는 측정 방법을 보여줍니다.
이 파일들만으로 발표의 정확도·지연시간 수치나 현재 클러스터 상태를 입증하지는 않습니다.
구간별 CPU 시간과 요청 wall time도 서로 다른 지표입니다.

결과 설명에는 원본의 [통신 최적화 결과](../../experiments/cifar10/docs/distributed-transport-results-20261011.ko.md),
[실제 실행시간 비교](../../experiments/cifar10/docs/expert-once-results-20261017.ko.md),
[세 구조 비교 기록](../../cloud/docs/benchmark/fused-three-structure-20261009.ko.md)을 함께 참고합니다.
15페이지의 실측 비용 학습 반영은 후속 과제이며 이 복사본에 새로 구현한 기능이 아닙니다.

## 복사 파일과 원본 경로

각 주제 폴더 안에는 원본 상대 경로를 유지했습니다. 파일명 링크는 복사본을,
원본 경로 링크는 저장소의 현행 파일을 엽니다. 코드 내용은 수정하지 않았습니다.

| 주제 | 복사본 | 원본 경로 | 역할 |
|---|---|---|---|
| 01-infrastructure | [main.tf](01-infrastructure/cloud/iac/modules/gke/main.tf) | [cloud/iac/modules/gke/main.tf](../../cloud/iac/modules/gke/main.tf) | GKE 클러스터와 역할별 노드 풀 리소스 |
| 01-infrastructure | [variables.tf](01-infrastructure/cloud/iac/modules/gke/variables.tf) | [cloud/iac/modules/gke/variables.tf](../../cloud/iac/modules/gke/variables.tf) | 노드 풀과 클러스터 입력 변수 |
| 01-infrastructure | [benchmark-current.tfvars](01-infrastructure/cloud/iac/environments/dispatcher-current/profiles/benchmark-current.tfvars) | [cloud/iac/environments/dispatcher-current/profiles/benchmark-current.tfvars](../../cloud/iac/environments/dispatcher-current/profiles/benchmark-current.tfvars) | 시스템·학습·추론·부하 생성 노드 풀 설정 |
| 01-infrastructure | [service-account.yaml](01-infrastructure/cloud/kfp/service-account.yaml) | [cloud/kfp/service-account.yaml](../../cloud/kfp/service-account.yaml) | KFP 실행 Pod의 ServiceAccount |
| 01-infrastructure | [kfp-launcher-configmap.yaml](01-infrastructure/cloud/kfp/kfp-launcher-configmap.yaml) | [cloud/kfp/kfp-launcher-configmap.yaml](../../cloud/kfp/kfp-launcher-configmap.yaml) | KFP launcher의 artifact 저장 설정 |
| 02-kfp | [pipeline.py](02-kfp/pipelines/pertinence/pipeline.py) | [pipelines/pertinence/pipeline.py](../../pipelines/pertinence/pipeline.py) | Importer 3개와 계산 컴포넌트 5개의 DAG·리소스·재시도 정책 |
| 02-kfp | [components.py](02-kfp/pipelines/pertinence/components.py) | [pipelines/pertinence/components.py](../../pipelines/pertinence/components.py) | 컨테이너 컴포넌트의 입출력과 CLI 호출 정의 |
| 02-kfp | [submit.py](02-kfp/pipelines/pertinence/submit.py) | [pipelines/pertinence/submit.py](../../pipelines/pertinence/submit.py) | 파이프라인 제출과 입력 URI·해시 전달 |
| 02-kfp | [Dockerfile](02-kfp/pipelines/pertinence/Dockerfile) | [pipelines/pertinence/Dockerfile](../../pipelines/pertinence/Dockerfile) | CPU 학습 컴포넌트 이미지 구성 |
| 02-kfp | [pipeline-run.yaml](02-kfp/experiments/cifar10/configs/pipeline-run.yaml) | [experiments/cifar10/configs/pipeline-run.yaml](../../experiments/cifar10/configs/pipeline-run.yaml) | Expert 선택·데이터 분할·학습·NSGA-II 설정 |
| 03-training | [cli.py](03-training/src/pertinence/pipeline/cli.py) | [src/pertinence/pipeline/cli.py](../../src/pertinence/pipeline/cli.py) | 컨테이너 명령을 실제 구현 함수에 연결 |
| 03-training | [validation.py](03-training/src/pertinence/pipeline/validation.py) | [src/pertinence/pipeline/validation.py](../../src/pertinence/pipeline/validation.py) | 입력 해시·계약 검증과 데이터 분할 |
| 03-training | [cache.py](03-training/src/pertinence/pipeline/cache.py) | [src/pertinence/pipeline/cache.py](../../src/pertinence/pipeline/cache.py) | 특징·Expert 예측·라우팅 정답 캐시 생성 |
| 03-training | [search.py](03-training/src/pertinence/pipeline/search.py) | [src/pertinence/pipeline/search.py](../../src/pertinence/pipeline/search.py) | Dispatcher 후보 탐색과 checkpoint 복원·최종 평가 |
| 03-training | [registry.py](03-training/src/pertinence/pipeline/registry.py) | [src/pertinence/pipeline/registry.py](../../src/pertinence/pipeline/registry.py) | 평가된 모델과 run별 SQLite registry 등록 |
| 03-training | [training.py](03-training/src/pertinence/training.py) | [src/pertinence/training.py](../../src/pertinence/training.py) | Dispatcher 학습 루프와 캐시 기반 평가 |
| 03-training | [routing.py](03-training/src/pertinence/routing.py) | [src/pertinence/routing.py](../../src/pertinence/routing.py) | FC Dispatcher·라우팅 정답·가중치·penalty loss |
| 03-training | [optimization.py](03-training/src/pertinence/optimization.py) | [src/pertinence/optimization.py](../../src/pertinence/optimization.py) | NSGA-II chromosome과 정확도·연산량 목적함수 |
| 03-training | [metrics.py](03-training/src/pertinence/metrics.py) | [src/pertinence/metrics.py](../../src/pertinence/metrics.py) | 시스템 정확도·연산량·Pareto 판정 |
| 04-serving | [00-platform.yaml.tmpl](04-serving/cloud/kserve/manifests/00-platform.yaml.tmpl) | [cloud/kserve/manifests/00-platform.yaml.tmpl](../../cloud/kserve/manifests/00-platform.yaml.tmpl) | Namespace·ServiceAccount·불변 ConfigMap |
| 04-serving | [05-benchmark-workload-identity.yaml.tmpl](04-serving/cloud/kserve/manifests/05-benchmark-workload-identity.yaml.tmpl) | [cloud/kserve/manifests/05-benchmark-workload-identity.yaml.tmpl](../../cloud/kserve/manifests/05-benchmark-workload-identity.yaml.tmpl) | Benchmark identity와 Kubernetes RBAC |
| 04-serving | [10-workloads.yaml.tmpl](04-serving/cloud/kserve/manifests/10-workloads.yaml.tmpl) | [cloud/kserve/manifests/10-workloads.yaml.tmpl](../../cloud/kserve/manifests/10-workloads.yaml.tmpl) | KServe InferenceService와 benchmark Job의 기본 템플릿 |
| 04-serving | [20-single-arm-job-v2.yaml.tmpl](04-serving/cloud/kserve/manifests/20-single-arm-job-v2.yaml.tmpl) | [cloud/kserve/manifests/20-single-arm-job-v2.yaml.tmpl](../../cloud/kserve/manifests/20-single-arm-job-v2.yaml.tmpl) | 단일 비교 조건의 부하 생성 Job |
| 04-serving | [runtime.py](04-serving/src/pertinence/serving/runtime.py) | [src/pertinence/serving/runtime.py](../../src/pertinence/serving/runtime.py) | ONNX 실행·Dispatcher·원격 Expert·네 가지 추론 구조 |
| 04-serving | [http.py](04-serving/src/pertinence/serving/http.py) | [src/pertinence/serving/http.py](../../src/pertinence/serving/http.py) | Expert HTTP 엔드포인트와 JSON·Binary 응답 |
| 04-serving | [binary.py](04-serving/src/pertinence/serving/binary.py) | [src/pertinence/serving/binary.py](../../src/pertinence/serving/binary.py) | FP32 tensor의 Binary 직렬화·역직렬화 |
| 04-serving | [transport_deployment.py](04-serving/src/pertinence/serving/transport_deployment.py) | [src/pertinence/serving/transport_deployment.py](../../src/pertinence/serving/transport_deployment.py) | 기본 매니페스트를 통신·Fused·Hybrid 조건별로 변환 |
| 04-serving | [runtime.py](04-serving/src/pertinence/serving/cli/runtime.py) | [src/pertinence/serving/cli/runtime.py](../../src/pertinence/serving/cli/runtime.py) | CLI 설정에 따라 Expert·Router 프로세스 구성 |
| 04-serving | [Dockerfile](04-serving/cloud/kserve/runtime/Dockerfile) | [cloud/kserve/runtime/Dockerfile](../../cloud/kserve/runtime/Dockerfile) | CPU ONNX 서빙 이미지 구성 |
| 05-benchmark | [benchmark.yaml](05-benchmark/experiments/cifar10/configs/benchmark.yaml) | [experiments/cifar10/configs/benchmark.yaml](../../experiments/cifar10/configs/benchmark.yaml) | 기본 서빙 비교 실험 설정 |
| 05-benchmark | [benchmark-transport-suite.yaml](05-benchmark/experiments/cifar10/configs/benchmark-transport-suite.yaml) | [experiments/cifar10/configs/benchmark-transport-suite.yaml](../../experiments/cifar10/configs/benchmark-transport-suite.yaml) | 통신 최적화 비교 실험 설정 |
| 05-benchmark | [benchmark-single-arm-v2.yaml](05-benchmark/experiments/cifar10/configs/benchmark-single-arm-v2.yaml) | [experiments/cifar10/configs/benchmark-single-arm-v2.yaml](../../experiments/cifar10/configs/benchmark-single-arm-v2.yaml) | 단일 조건의 워밍업·요청·측정 설정 |
| 05-benchmark | [benchmark.py](05-benchmark/src/pertinence/serving/benchmark.py) | [src/pertinence/serving/benchmark.py](../../src/pertinence/serving/benchmark.py) | 워밍업과 요청 스케줄·결과 레코드 수집 |
| 05-benchmark | [diagnostics.py](05-benchmark/src/pertinence/serving/diagnostics.py) | [src/pertinence/serving/diagnostics.py](../../src/pertinence/serving/diagnostics.py) | 구간별 wall time·CPU time 계측 |
| 05-benchmark | [transport_suite.py](05-benchmark/src/pertinence/serving/transport_suite.py) | [src/pertinence/serving/transport_suite.py](../../src/pertinence/serving/transport_suite.py) | 비교 조건 정의·지연 통계·진단 로그 결합 |

## 스냅샷과 제외 항목

[manifest.json](manifest.json)에 복사 시각(UTC), 복사본·원본 상대 경로, 역할과 SHA-256을
기록했습니다. SHA-256은 복사 시점의 내용 기준입니다. 원본이 바뀌어도 복사본은 자동으로
갱신되지 않으므로 갱신할 때는 복사본과 manifest를 함께 갱신합니다.

데이터셋·체크포인트·캐시·실행 산출물·Terraform state·인증정보·가상환경은 포함하지 않았습니다.
전체 의존 코드, 운영 스크립트와 legacy 호환 모듈도 포함하지 않았습니다.
발표 PDF와 대본은 기존 파일로 연결하고 중복 복사하지 않았습니다.
