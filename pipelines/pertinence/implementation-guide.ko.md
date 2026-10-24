# PERTINENCE KFP 학습 파이프라인 구현 이해 가이드

이 문서는 PERTINENCE 학습 파이프라인이 저장소에서 어떻게 구성되고 실행되는지를 코드와
artifact의 흐름을 따라 설명한다. 파이프라인을 처음 읽는 사람이 다음 질문에 답할 수 있게 하는 것이
목적이다.

- KFP DAG는 어디에서 정의하는가?
- 각 KFP component는 실제로 어떤 Python 함수를 실행하는가?
- DatasetBundle, cache, checkpoint와 report는 단계 사이에서 어떻게 전달되는가?
- Search가 final 데이터를 볼 수 없도록 어떻게 막았는가?
- 실행 결과가 이전 단계의 결과와 같은 계보에 속하는지 어떻게 검증하는가?
- 기능을 수정하려면 어느 파일부터 읽어야 하는가?

가장 중요한 구조는 단순하다. `pipelines/pertinence/`는 작업의 연결과 컨테이너 실행 조건을 정의하고,
실제 검증·추론·학습·평가 로직은 `src/pertinence/`가 담당한다. 따라서 KFP 코드는 얇은 orchestration
계층이며 핵심 알고리즘을 중복 구현하지 않는다.

## 1. 전체 구조

파이프라인은 compile 시점과 run 시점으로 나누어 이해하는 것이 좋다.

```text
compile 시점

pipeline.py + components.py
        │
        │ KFP Compiler
        ▼
pipeline IR YAML
        │
        │ submit.py
        ▼
KFP API에 Run 생성


run 시점

GCS의 immutable 입력
        │
        │ KFP importer와 launcher가 Pod의 local path로 staging
        ▼
validate → build cache → train/search → final evaluation → register
        │          │             │               │              │
        └──────────┴─────────────┴───────────────┴──────────────┘
                       KFP output artifact로 GCS에 보존
```

코드는 다음 여섯 층으로 나뉜다.

| 층 | 역할 | 주요 코드 |
|---|---|---|
| DAG | component 사이의 의존성과 자원 정책 선언 | [`pipeline.py`](pipeline.py), [`cifar10_pipeline.py`](cifar10_pipeline.py) |
| Container interface | KFP 입출력을 CLI 인자로 변환 | [`components.py`](components.py) |
| CLI adapter | 문자열 명령을 실제 Python 함수로 dispatch | [`pipeline/cli.py`](../../src/pertinence/pipeline/cli.py) |
| Domain logic | 검증, cache, 탐색, 평가, 등록 | [`src/pertinence/pipeline/`](../../src/pertinence/pipeline) |
| Contracts | 입력 schema와 platform 상한 검증 | [`src/pertinence/contracts/`](../../src/pertinence/contracts) |
| Platform | Image, KFP launcher, ServiceAccount와 GKE | [`Dockerfile`](Dockerfile), [`cloud/kfp/`](../../cloud/kfp) |

이 분리는 로컬 테스트와 KFP 실행이 같은 함수를 사용하게 한다. 예를 들어 KFP의
`build-routing-dataset` component와 로컬 테스트는 모두
[`build_routing_dataset()`](../../src/pertinence/pipeline/cache.py#L40)을 호출한다.

## 2. 범용 파이프라인과 CIFAR-10 legacy 파이프라인

저장소에는 의도적으로 두 DAG가 있다.

| 구분 | 범용 bundle 기반 경로 | CIFAR-10 legacy 경로 |
|---|---|---|
| Pipeline 이름 | `pertinence-routing-training-cpu` | `pertinence-cifar10-cpu` |
| DAG 정의 | [`pipeline.py`](pipeline.py#L52) | [`cifar10_pipeline.py`](cifar10_pipeline.py#L12) |
| 입력 | DatasetBundle, ExpertBundle, RunConfig | CIFAR-10 archive, checkpoint 3개, RunConfig |
| Importer 수 | 3 | 5 |
| Container task 수 | 5 | 7 |
| Raw asset 변환 | 파이프라인 외부 adapter가 담당 | 첫 두 component가 담당 |
| 용도 | 새 실행의 기본 구조 | 2026-09-20 production 실행 재현과 과거 IR 호환 |

범용 경로는 이미 표준 bundle로 준비된 입력만 받는다. 데이터셋 종류나 expert 개수는 Bundle v1
계약 안에서 바꿀 수 있다. CIFAR-10 legacy 경로는 raw archive와 PyTorch checkpoint에서 바로
시작해야 했던 기존 실행을 보존하기 위해 앞에 두 단계를 추가한다.

```text
범용 5단계

DatasetBundle ─┐
ExpertBundle ──┼─ validate-run ─ build-routing-dataset
RunConfig ─────┘        │                 │
                       │                 ├─ train cache
                       │                 ├─ search cache ─ train-dispatcher
                       │                 └─ final cache ───────────────┐
                       │                              │                │
                       └─ normalized contract ────────┴─ evaluate-final
                                                                  │
                                                        register-dispatchers
```

```text
CIFAR-10 legacy 7단계

CIFAR-10 archive ─ prepare-dataset-bundle ─ DatasetBundle ─┐
checkpoint 3개 ─── export-expert-bundle ─── ExpertBundle ──┼─ 범용 5단계
RunConfig ──────────────────────────────────────────────────┘
```

실제 학습·평가 로직은 두 경로가 공유한다. Legacy 경로도 bundle을 만든 다음부터는 범용 5단계와
같은 component를 사용한다.

## 3. 입력 artifact 계약

범용 파이프라인의 외부 입력은 세 개이며, 각 URI와 SHA-256을 함께 받는다.

### 3.1 DatasetBundle

DatasetBundle은 directory artifact다.

```text
dataset-bundle/
├── dataset.yaml
├── train/
│   └── part-*.parquet
└── evaluation/
    └── part-*.parquet
```

Parquet row는 다음 세 column을 가져야 한다.

| Column | 형식 | 의미 |
|---|---|---|
| `sample_id` | string | split과 계보를 추적하는 안정적인 ID |
| `image` | binary | PNG 또는 JPEG byte |
| `target` | int64 | class index |

Schema는 [`DatasetBundle`](../../src/pertinence/contracts/dataset.py)을 통해 검증하고, streaming
reader는 [`DatasetBundleReader`](../../src/pertinence/pipeline/bundles.py#L28)에 구현되어 있다.
이미 준비된 shard를 bundle로 봉인하는 코드는
[`package_dataset()`](../../src/pertinence/pipeline/bundles.py#L97)이다.

### 3.2 ExpertBundle

ExpertBundle 역시 directory artifact다.

```text
expert-bundle/
├── experts.yaml
└── models/
    ├── expert-a/model.onnx
    ├── expert-b/model.onnx
    └── expert-c/model.onnx
```

`experts.yaml`에는 expert ID, ONNX 경로와 hash, class 수, 전처리, output name, 비용이 들어간다.
Schema는 [`ExpertBundle`](../../src/pertinence/contracts/experts.py)이 검증한다. ONNX는 임의의 Python
코드를 실행하지 않고 [`ExpertRuntime`](../../src/pertinence/runtimes/onnx.py#L28)이 읽는다.

Runtime v1은 다음 범위만 허용한다.

- 단일 파일 ONNX
- NCHW float32 RGB 입력
- 동적 batch dimension
- 선언된 `logits` output
- Feature extractor의 경우 `[batch, D]` 형태의 `features` output
- 기본 ONNX domain의 operator

External tensor data, custom operator domain과 ONNX local function은 validation에서 거부한다.

### 3.3 RunConfig

CIFAR-10 실행 설정 예시는
[`experiments/cifar10/configs/pipeline-run.yaml`](../../experiments/cifar10/configs/pipeline-run.yaml)에
있다.

RunConfig에는 다음 정보가 포함된다.

- 선택할 expert와 비용 오름차순의 순서
- Feature extractor와 fallback expert
- Search/final split 크기와 seed
- Dispatcher optimizer, epoch와 batch size
- NSGA-II population, generation, crossover, mutation과 weighting 후보

엄격한 schema 검증은 [`RunConfig`](../../src/pertinence/contracts/run.py#L39)가 수행한다. 알 수 없는
field, 중복 YAML key, NaN/Infinity, 잘못된 expert ID와 지원하지 않는 optimizer는 허용하지 않는다.

### 3.4 Bundle digest와 내부 file hash의 차이

두 종류의 hash를 구분해야 한다.

1. 외부 입력 SHA-256은 KFP parameter로 들어오며, 전달된 artifact 전체가 기대한 byte인지 확인한다.
2. Manifest 내부의 `content_sha256` 또는 `model_sha256`은 bundle 내부 파일이 manifest와 일치하는지
   확인한다.

Directory의 digest는 파일을 단순 연결해서 계산하지 않는다. 상대 경로와 각 파일 SHA-256의 정렬된
mapping을 canonical JSON으로 만든 뒤 다시 hash한다. 구현은
[`artifact_sha256()`](../../src/pertinence/pipeline/artifacts.py#L70)과
[`file_inventory()`](../../src/pertinence/pipeline/artifacts.py#L58)에 있다.

## 4. KFP DAG를 만드는 코드

범용 DAG의 시작점은 [`make_pipeline()`](pipeline.py#L52)이다. 함수 안의 `@dsl.pipeline` 함수가
실제 그래프를 선언한다.

핵심 순서는 다음과 같다.

```python
dataset = dsl.importer(...)
experts = dsl.importer(...)
config = dsl.importer(...)

validate = components["validate"](...)
build = components["build"](...)
train = components["train"](...)
evaluate = components["evaluate"](...)
register = components["register"](...)
```

KFP에서 실행 순서는 별도의 `after()` 호출보다 artifact 연결로 결정된다. 예를 들어 `train`의
`search_cache` 입력에 `build.outputs["search_cache"]`를 전달했기 때문에 build가 끝난 뒤 train이
시작된다.

### 4.1 Importer의 역할

`dsl.importer`는 외부 GCS URI를 KFP artifact로 그래프에 등록한다. Component 내부 코드는 GCS URI를
직접 다운로드하지 않는다. KFP launcher가 입력을 container의 local artifact path로 staging하고,
component에는 `dataset_bundle.path`와 같은 local path가 전달된다.

이 설계 덕분에 domain 함수는 GCS SDK나 KFP SDK에 의존하지 않고 일반 local path만 처리한다.

### 4.2 Resource와 node 배치

[`_resources()`](pipeline.py#L25)는 CPU/memory request와 limit을 같은 값으로 설정하고
`OMP_NUM_THREADS`, `MKL_NUM_THREADS`도 CPU 수에 맞춘다.

| Component | CPU | Memory |
|---|---:|---:|
| validate | 4 | 6 GiB |
| build cache | 4 | 12 GiB |
| train/search | 4 | 8 GiB |
| final evaluation | 2 | 4 GiB |
| registration | 1 | 2 GiB |

[`_schedule_on_pipeline_pool()`](pipeline.py#L36)은 모든 실제 executor에 다음 두 조건을 함께 넣는다.

```text
nodeSelector: cloud.google.com/gke-nodepool=pipeline-pool
toleration: workload=pertinence-pipeline:NoSchedule
```

Toleration은 해당 taint가 있는 node에서 실행할 수 있게 하고, node selector는 실제로
`pipeline-pool`을 선택한다. Importer와 driver는 이 함수를 적용받지 않으므로 system pool에서 실행될
수 있다.

### 4.3 Timeout, retry와 cache

- Search timeout은 기본 259,200초, 즉 72시간이다.
- Search는 중간 generation resume을 지원하지 않으므로 retry가 0이다.
- Validate, build, evaluate, register는 retry가 1이다.
- 모든 task cache는 기본적으로 꺼져 있다.
- Registration cache는 항상 꺼져 있다.
- 같은 immutable input URI, digest와 image를 사용하는 복구 실행에서만 나머지 cache를 켤 수 있다.

Task cache를 compile 시점과 submit 시점 양쪽에서 명시하는 이유는 복구 실행의 정책이 IR과 실제
Run 모두에서 일치하도록 하기 위해서다.

### 4.4 Compile

[`compile_pipeline()`](pipeline.py#L135)은 Python DAG를 KFP IR YAML로 변환한다. 이 단계는 cluster에
접속하지 않는다.

```bash
.venv/bin/python -m pipelines.pertinence.pipeline \
  --image "REGISTRY/pertinence@sha256:<digest>" \
  --pipeline-root gs://RUN_BUCKET/pertinence/runs \
  --output artifacts/cifar10/runs/pipeline/pertinence-routing-training.yaml
```

Mutable image tag는 [`_validate_component_settings()`](components.py#L9)에서 거부한다. 따라서 모든
container task는 동일한 `linux/amd64` image digest를 실행한다.

## 5. Component interface와 CLI 연결

[`make_training_components()`](components.py#L86)은 다섯 개의 KFP container component를 만든다.
각 component는 알고리즘을 직접 구현하지 않고 다음 형태의 command를 생성한다.

```text
python -m pertinence.pipeline_cli <subcommand> <input/output paths>
```

예를 들어 `train-dispatcher` component는 KFP artifact path를 다음 CLI 인자로 변환한다.

```text
train-dispatcher
--train-cache <local input path>
--search-cache <local input path>
--normalized-contract <local input path>
--search-result <local output path>
--pareto-bundle <local output path>
```

호환 entry point인 [`src/pertinence/pipeline_cli.py`](../../src/pertinence/pipeline_cli.py)는 실제
구현인 [`pipeline/cli.py`](../../src/pertinence/pipeline/cli.py)로 위임한다.
[`run_command()`](../../src/pertinence/pipeline/cli.py#L94)은 subcommand를 다음 함수에 연결한다.

| CLI subcommand | 실제 함수 |
|---|---|
| `prepare-dataset-bundle` | `prepare_cifar10_dataset_bundle()` |
| `export-expert-bundle` | `export_cifar10_expert_bundle()` |
| `validate-run` | `validate_run_contract()` |
| `build-routing-dataset` | `build_routing_dataset()` |
| `train-dispatcher` | `train_dispatcher()` |
| `evaluate-final` | `evaluate_final()` |
| `register-dispatcher-bundle` | `register_dispatcher_bundle()` |

KFP 실행과 로컬 실행의 차이는 입출력 path를 누가 준비하는지뿐이다. KFP에서는 launcher가 path를
준비하고, 로컬에서는 사용자가 명시적인 path를 전달한다.

## 6. 단계별 내부 구현

### 6.1 Legacy 단계 1 prepare dataset bundle

코드: [`prepare_cifar10_dataset_bundle()`](../../src/pertinence/pipeline/bootstrap.py#L126)

이 단계는 CIFAR-10 Python tar archive를 portable DatasetBundle로 바꾼다.

1. Raw archive SHA-256을 검증한다.
2. 허용된 NumPy 관련 global만 읽을 수 있는 restricted unpickler를 사용한다.
3. CIFAR-10 class 이름과 순서가 고정 계약과 같은지 확인한다.
4. 각 이미지를 PNG byte로 변환한다.
5. Train 50,000장과 evaluation 10,000장을 Parquet shard로 저장한다.
6. 파일 inventory hash를 포함한 `dataset.yaml`을 작성한다.
7. 완성된 directory의 bundle digest를 출력한다.

Tar를 임의 경로에 extract하지 않고 정확한 archive member만 읽기 때문에 path traversal과 부분
추출 상태를 피한다.

### 6.2 Legacy 단계 2 export expert bundle

코드: [`export_cifar10_expert_bundle()`](../../src/pertinence/pipeline/bootstrap.py#L310)

이 단계는 세 PyTorch checkpoint를 portable ONNX bundle로 변환한다.

1. 세 checkpoint SHA-256을 검증한다.
2. 고정된 ShuffleNetV2 x0.5, MobileNetV2 x0.5, ResNet56 구조를 생성한다.
3. Checkpoint state dict를 strict mode로 적재한다.
4. 모델을 frozen evaluation mode로 전환한다.
5. Logits와 classifier 직전 feature를 함께 출력하도록 감싼다.
6. Dynamic batch ONNX opset 17로 export한다.
7. ONNX checker를 실행하고 external tensor data가 없는지 확인한다.
8. 같은 입력에 대한 PyTorch와 ONNX Runtime의 logits와 feature를 비교한다.
9. 모델 hash, 전처리, 출력 이름과 MFLOPs를 `experts.yaml`에 기록한다.

이 단계 이후 공용 component는 PyTorch checkpoint의 출처나 Python 모델 구현을 알 필요가 없다.

### 6.3 공용 단계 1 validate run

코드: [`validate_run_contract()`](../../src/pertinence/pipeline/validation.py#L48)

Validation은 잘못된 장시간 실행을 초기에 중단하는 fail-fast gate다. 단순 schema 검사보다 훨씬 많은
작업을 수행한다.

#### 입력 무결성

- DatasetBundle, ExpertBundle, RunConfig의 외부 digest 검증
- Bundle 내부 content/model hash 검증
- Symlink와 상위 directory escape 경로 거부
- Dataset 및 model byte 상한 검사

#### Dataset 검증

- Parquet schema와 실제 row 수 검사
- `sample_id`의 유일성 및 train/evaluation 간 비중복 검사
- Target 범위와 image codec 검사
- Manifest의 class 이름과 class 수 검사

#### Expert 검증

- 모든 expert의 class 수와 class index 순서 일치 검사
- 선택 expert가 존재하는지 확인
- Expert 비용이 오름차순인지 확인
- ONNX input/output와 dynamic batch 검사
- Feature extractor의 feature 차원이 batch size에 따라 변하지 않는지 smoke inference

#### 실행량 상한

- 최대 sample, class, expert와 feature dimension
- 예상 cache byte
- NSGA-II 평가 횟수
- 전체 training sample visit 수
- 예상 dispatcher checkpoint state 크기

#### Split 생성

Evaluation sample ID를 먼저 정렬한 뒤 split seed로 permutation한다. 따라서 Parquet shard나 row
저장 순서가 바뀌어도 같은 sample 집합이면 split은 동일하다.

```text
evaluation IDs
  → stable sort
  → seeded permutation
  ├─ 앞 search_size개: search
  └─ 나머지: final
```

#### 출력

| 출력 | 내용 |
|---|---|
| `normalized_contract` | 검증된 입력, 파생 차원 C/D/K, 비용, split, runtime lineage |
| `split_manifest` | train/search/final sample ID와 fingerprint |
| `validation_report` | valid 상태, 차원, 예상 cache byte와 contract fingerprint |

Normalized contract에는 code fingerprint, container image digest와 주요 package version도 기록된다.
Downstream 단계의 [`load_contract()`](../../src/pertinence/pipeline/validation.py#L33)는 현재 code와
package가 validation 당시와 달라졌으면 실행을 거부한다.

### 6.4 공용 단계 2 build routing dataset

코드: [`build_routing_dataset()`](../../src/pertinence/pipeline/cache.py#L40)

이 단계의 목적은 비싼 image decoding과 expert inference를 한 번만 수행하고, NSGA-II가 작은 배열만
반복해서 사용하게 만드는 것이다.

각 split에는 다음 배열을 만든다.

| 파일 | Shape | 의미 |
|---|---|---|
| `features.npy` | `[N, D]` | Feature extractor 출력 |
| `expert_predictions.npy` | `[N, K]` | 각 expert의 예측 class |
| `targets.npy` | `[N]` | 실제 class |
| `route_labels.npy` | `[N]` | Cheapest-correct expert index |
| `sample_ids.npy` | `[N]` | 정렬과 계보 확인용 sample ID |
| `manifest.json` | - | Shape, hash, split와 lineage |

배열은 `numpy.open_memmap`으로 생성하여 전체 cache를 메모리에 올리지 않는다. 데이터셋을 expert별로
streaming하면서 자신의 split과 위치에 prediction을 기록한다.

Cheapest-correct label은 [`route_labels()`](../../src/pertinence/pipeline/cache.py#L35)에서 만든다.
Expert 순서가 비용 오름차순이므로, 정답을 맞힌 첫 expert index가 가장 저렴한 정답 expert다. 어느
expert도 정답을 맞히지 못하면 RunConfig에 선언한 fallback index를 사용한다.

```text
correct = expert_predictions == target
route = first correct expert, or fallback
```

Train, search, final output path가 서로 같은지, sample ID가 겹치는지, validation 이후 데이터셋이
바뀌었는지도 검사한다. Cache는 임시 directory에 완성한 뒤 한 번에 publish한다.

### 6.5 공용 단계 3 train dispatcher

파이프라인 코드: [`train_dispatcher()`](../../src/pertinence/pipeline/search.py#L73)

알고리즘 코드:

- [`DispatcherNSGA2Problem`](../../src/pertinence/optimization.py#L144)
- [`run_nsga2()`](../../src/pertinence/optimization.py#L241)
- [`train_linear_dispatcher()`](../../src/pertinence/training.py#L141)
- [`class_weights()`](../../src/pertinence/routing.py#L31)
- [`paper_penalty_loss()`](../../src/pertinence/routing.py#L63)
- [`system_metrics()`](../../src/pertinence/metrics.py#L18)

이 함수의 signature에는 의도적으로 final cache가 없다.

```text
inputs: train cache + search cache + normalized contract
outputs: search result + Pareto checkpoint bundle
```

처리 순서는 다음과 같다.

1. Train과 search cache의 manifest, file hash, shape와 의미를 다시 검증한다.
2. 두 cache의 sample ID가 겹치지 않는지 확인한다.
3. Contract에서 dispatcher와 NSGA-II 설정을 구성한다.
4. 각 chromosome마다 새 linear dispatcher를 동일 seed로 초기화한다.
5. Train feature와 route label로 dispatcher를 학습한다.
6. Search cache에서 system accuracy와 평균 비용을 계산한다.
7. 목적함수 `[-system_accuracy, average_cost]`를 NSGA-II에 반환한다.
8. 전체 evaluation과 Search Pareto front를 계산한다.
9. Pareto 해의 FC weight와 bias만 `.npz` checkpoint로 저장한다.

Expert가 `K`개라면 chromosome은 `K²` penalty gene과 weighting gene 하나로 구성된다. CIFAR-10의
현행 `K=3`에서는 10개 gene이다. Weighting gene은 INS, ISNS, ENS 중 하나로 decode된다.

System cost에는 feature extractor, dispatcher FC와 선택 expert 비용이 포함된다. Feature extractor가
선택 expert와 같으면 이미 계산한 결과를 재사용하므로 backbone 비용을 두 번 더하지 않는다.

#### Search 출력

`search_result`에는 전체 evaluation의 다음 정보가 들어간다.

- Evaluation ID
- 원 chromosome과 decode된 penalty matrix
- Weighting 방식
- System/route accuracy와 평균 비용
- Routing confusion matrix와 expert별 선택 횟수
- Epoch loss
- Pareto ID와 front fingerprint

`pareto_bundle`에는 Pareto checkpoint와 manifest가 들어간다. Manifest는 checkpoint SHA-256,
feature/expert 차원, expert 순서, cache fingerprint를 checkpoint마다 기록한다.

### 6.6 공용 단계 4 evaluate final

코드: [`evaluate_final()`](../../src/pertinence/pipeline/search.py#L245)

Final 단계는 모델 선택 단계가 아니라 이미 선택된 Search Pareto checkpoint의 일반화 상태를
측정하는 단계다.

1. Final cache의 lineage와 내용을 검증한다.
2. Search evaluation ID가 `0..evaluation_count-1`인지 검사한다.
3. Search metric에서 Pareto mask를 다시 계산하고 보존된 Pareto ID와 비교한다.
4. 모든 checkpoint의 hash, tensor key, shape, dtype와 finite 값을 검사한다.
5. FC weight와 bias를 복원한다.
6. 재학습 없이 final cache에서 system metric을 계산한다.
7. Checkpoint, contract, search result와 final report를 하나의 evaluated bundle로 묶는다.

Final 결과를 보고 일부 checkpoint를 제거하거나 Pareto front를 다시 고르지 않는다. Search에서
보존한 checkpoint를 모두 평가한다. 이 규칙은 final holdout이 사실상 validation set으로 바뀌는 것을
막는다.

### 6.7 공용 단계 5 register dispatchers

코드:

- [`register_dispatcher_bundle()`](../../src/pertinence/pipeline/registry.py#L149)
- [`register_pareto_dispatchers()`](../../src/pertinence/pipeline/registry.py#L79)
- [`SQLiteRegistry`](../../src/pertinence/pipeline/registry.py#L22)

현재 registry는 중앙 서비스가 아니라 실행별 SQLite artifact다. 등록 전에 evaluated bundle 전체를
다시 검증한다.

각 dispatcher의 idempotency key는 다음 값에서 만든다.

```text
contract fingerprint
+ Pareto front ID
+ evaluation ID
+ checkpoint SHA-256
```

SQLite transaction은 `BEGIN IMMEDIATE`로 시작한다. 동일 idempotency key와 같은 내용의 재시도는
같은 version을 반환하고, 같은 key에 다른 내용이 들어오면 실패한다. Batch 중 하나라도 실패하면 전체
transaction을 rollback한다.

최종 출력은 다음과 같다.

```text
registry-bundle/
├── registry.sqlite3
└── registration-report.json
```

`RegistryAdapter` protocol이 분리되어 있으므로 향후 원격 registry를 구현하더라도 동일한 atomic
idempotent batch 계약을 지켜 교체할 수 있다.

## 7. Final 격리를 코드로 보장하는 방법

Final 누수 방지는 개발자 주의사항만으로 유지하지 않고 interface와 IR에서 강제한다.

1. [`train_dispatcher()`](../../src/pertinence/pipeline/search.py#L73)의 인자에 final path가 없다.
2. KFP `train-dispatcher` component도 train/search cache만 입력받는다.
3. DAG에서 final cache는 build에서 evaluate로만 연결된다.
4. Pipeline은 shared volume을 사용하지 않는다.
5. IR test는 train task 직렬화 결과에 `final` 문자열과 volume이 없는지 검사한다.
6. Final component test는 학습 함수를 실패하도록 monkeypatch한 상태에서도 평가가 끝나는지 확인한다.

관련 검증은
[`test_compile_generic_dag_and_no_final_input_to_search()`](../../tests/test_kfp_pipeline.py#L22)와
[`test_all_components_dynamic_and_registry_idempotent()`](../../tests/test_pipeline_components.py#L240)에
있다.

## 8. Artifact 원자성과 계보 검증

파이프라인의 각 단계는 단순히 이전 단계의 파일을 신뢰하지 않는다.

### 8.1 Sealed JSON

[`seal()`](../../src/pertinence/pipeline/artifacts.py#L123)은 JSON body의 canonical hash를
`fingerprint` field로 추가한다. [`read_sealed()`](../../src/pertinence/pipeline/artifacts.py#L127)은
읽을 때 다시 계산하여 내용이 변했는지 확인한다.

이 fingerprint들은 다음과 같이 연결된다.

```text
input digests
   ↓
contract fingerprint + split fingerprint
   ↓
train/search/final cache fingerprints
   ↓
search fingerprint + Pareto front ID
   ↓
checkpoint hashes
   ↓
final report fingerprint
   ↓
registry idempotency keys
```

중간 artifact 하나를 다른 실행의 artifact로 바꾸면 downstream 비교 중 하나가 일치하지 않아
실행이 중단된다.

### 8.2 Atomic directory publication

[`atomic_directory()`](../../src/pertinence/pipeline/artifacts.py#L92)는 output 옆의 임시 directory에
모든 내용을 먼저 작성한 뒤 `os.replace`로 publish한다. 기존 non-empty output은 덮어쓰지 않는다.
따라서 실패 중간 상태를 완성된 artifact로 오인하지 않는다.

JSON 파일도 temporary file 작성, flush, `fsync`, replace 순서로 저장한다.

### 8.3 안전한 경로

`safe_path()`는 절대 경로, `..`, 역슬래시, drive 표기와 symlink를 거부한다. Bundle manifest가
container filesystem의 다른 파일을 가리키지 못하게 하는 방어다.

## 9. Container image와 KFP platform 연결

### 9.1 Component image

[`Dockerfile`](Dockerfile)은 Python 3.11 slim image에 CPU 전용 PyTorch, ONNX Runtime, PyArrow,
Pymoo와 프로젝트 package를 설치한다. KFP SDK는 runtime image에 넣지 않는다. KFP SDK는 DAG
compile/submit 환경에만 필요하기 때문이다.

Image의 기본 entry point는 다음과 같다.

```text
python -m pertinence.pipeline_cli
```

ComponentSpec도 같은 command를 명시하므로 각 Pod는 동일한 application image에서 다른 subcommand만
실행한다. Dependency pin은 [`requirements-components.lock`](requirements-components.lock)에 있다.

Image build는 [`cloudbuild.yaml`](cloudbuild.yaml)이 담당하며 `linux/amd64` platform으로 Artifact
Registry에 게시한다. Compile에는 tag가 아니라 build 후 확인한 manifest digest를 전달한다.

### 9.2 KFP launcher와 ServiceAccount

[`cloud/kfp/kfp-launcher-configmap.yaml`](../../cloud/kfp/kfp-launcher-configmap.yaml)은 기본 pipeline
root와 `gs:` provider가 environment credential을 사용하도록 선언한다.

[`cloud/kfp/service-account.yaml`](../../cloud/kfp/service-account.yaml)은 기존
`kubeflow/pipeline-runner` 이름과 namespace 계약을 확인하는 참조 manifest다. 실제 Pod는 Workload
Identity를 통해 GCS 권한을 얻으며 JSON service-account key를 mount하지 않는다.

### 9.3 Submit

범용 submit client는 [`submit.py`](submit.py#L25)에 있다.

1. GCS URI와 lowercase SHA-256 형식을 검사한다.
2. Local port-forward의 KFP API에 연결한다.
3. Compiled IR과 여섯 입력 parameter를 보낸다.
4. Namespace, pipeline root와 `pipeline-runner` ServiceAccount를 명시한다.
5. 생성된 KFP run ID를 출력한다.

Submit client는 object 존재 여부나 bundle 내부 구조까지 검사하지 않는다. 이는 container 안의
`validate-run`이 실제 staged byte를 대상으로 검사해야 race와 잘못된 신뢰 경계를 피할 수 있기
때문이다.

## 10. 테스트가 보장하는 내용

### 10.1 DAG와 IR 테스트

파일: [`tests/test_kfp_pipeline.py`](../../tests/test_kfp_pipeline.py)

주요 검증은 다음과 같다.

- 범용 DAG가 importer 3개와 container task 5개인지 확인
- Legacy DAG가 importer 5개와 container task 7개인지 확인
- Train 입력에 final artifact가 없는지 확인
- Shared volume과 GPU request가 없는지 확인
- 모든 executor가 고정 image digest를 사용하는지 확인
- Node selector, toleration, timeout, retry와 cache 정책 확인
- Mutable image tag와 GCS가 아닌 pipeline root 거부
- Submit parameter와 ServiceAccount 전달 확인

### 10.2 Component end-to-end 테스트

파일: [`tests/test_pipeline_components.py`](../../tests/test_pipeline_components.py)

이 테스트는 네트워크 다운로드 없이 작은 PNG/Parquet와 실제 ONNX model을 생성한다. Class 수
3/10/100, expert 수 2/3/5와 여러 feature dimension을 사용해 validation부터 SQLite registration까지
실행한다.

실패 경로도 다음과 같이 검사한다.

- 잘못된 schema, row 수, class 순서와 split 크기
- Expert 비용 순서, model path와 ONNX output 오류
- Custom operator, external tensor, static batch와 non-finite output
- Cache file 변조와 의미상 잘못된 route label
- 오래되거나 다른 실행의 checkpoint
- Symlink와 path escape
- Registry transaction rollback과 동시 재시도의 idempotency
- Final 단계에서 dispatcher 재학습이 발생하지 않는지 확인

Legacy 변환은 [`tests/test_pipeline_bootstrap.py`](../../tests/test_pipeline_bootstrap.py)가 CIFAR archive
변환과 export model architecture를 별도로 검사한다.

## 11. 코드를 처음 읽는 권장 순서

한 번에 모든 파일을 읽기보다 다음 순서를 권장한다.

1. [`pipeline.py`](pipeline.py#L52)에서 다섯 task와 artifact 연결을 본다.
2. [`components.py`](components.py#L86)에서 각 task가 만드는 CLI command를 본다.
3. [`pipeline/cli.py`](../../src/pertinence/pipeline/cli.py#L94)에서 command가 어느 함수로 가는지 본다.
4. [`validation.py`](../../src/pertinence/pipeline/validation.py#L48)에서 normalized contract가 어떻게
   생성되는지 본다.
5. [`cache.py`](../../src/pertinence/pipeline/cache.py#L40)에서 image가 학습용 배열로 바뀌는 과정을
   본다.
6. [`search.py`](../../src/pertinence/pipeline/search.py#L73)에서 NSGA-II와 checkpoint 보존 흐름을
   본다.
7. [`optimization.py`](../../src/pertinence/optimization.py#L144)와
   [`training.py`](../../src/pertinence/training.py#L141)에서 실제 학습을 본다.
8. [`search.py`](../../src/pertinence/pipeline/search.py#L245)의 final 평가를 본다.
9. [`registry.py`](../../src/pertinence/pipeline/registry.py#L79)에서 등록과 idempotency를 본다.
10. 마지막으로 [`test_pipeline_components.py`](../../tests/test_pipeline_components.py#L240)의 작은
    end-to-end 예제를 따라간다.

## 12. 변경 목적별 코드 위치

| 바꾸려는 것 | 먼저 볼 파일 |
|---|---|
| Task 연결 또는 새 KFP 단계 | [`pipeline.py`](pipeline.py), [`components.py`](components.py) |
| Task CPU, memory, timeout, retry | [`pipeline.py`](pipeline.py#L25) |
| 새 dataset adapter | [`bundles.py`](../../src/pertinence/pipeline/bundles.py), dataset별 adapter 추가 |
| 새 expert export adapter | [`bootstrap.py`](../../src/pertinence/pipeline/bootstrap.py), [`experts.py`](../../src/pertinence/contracts/experts.py) |
| RunConfig field | [`contracts/run.py`](../../src/pertinence/contracts/run.py), [`validation.py`](../../src/pertinence/pipeline/validation.py) |
| Split 정책 | [`validation.py`](../../src/pertinence/pipeline/validation.py#L48) |
| Cheapest-correct label | [`cache.py`](../../src/pertinence/pipeline/cache.py#L35) |
| Dispatcher loss와 class weight | [`routing.py`](../../src/pertinence/routing.py) |
| NSGA-II chromosome 또는 operator | [`optimization.py`](../../src/pertinence/optimization.py) |
| Accuracy와 system cost | [`metrics.py`](../../src/pertinence/metrics.py) |
| Checkpoint 또는 final report 형식 | [`pipeline/search.py`](../../src/pertinence/pipeline/search.py) |
| Registry backend | [`pipeline/registry.py`](../../src/pertinence/pipeline/registry.py) |
| Artifact hash와 atomic write | [`pipeline/artifacts.py`](../../src/pertinence/pipeline/artifacts.py) |
| Container dependency | [`requirements-components.lock`](requirements-components.lock), [`Dockerfile`](Dockerfile) |
| Submit parameter | [`submit.py`](submit.py) |

변경할 때는 해당 domain 코드뿐 아니라 contract, artifact fingerprint와 실패 경로 테스트도 함께
수정해야 한다. 예를 들어 cache 배열을 하나 추가하면 cache 생성 코드, manifest, `load_cache()`의
shape·dtype 검증, search 입력 변환과 corruption test까지 같은 변경 단위로 다루는 것이 안전하다.

## 13. 로컬에서 구조를 확인하는 방법

선택 dependency가 설치된 환경에서 다음 테스트를 실행한다.

```bash
.venv/bin/python -m pytest \
  tests/test_pipeline_components.py \
  tests/test_kfp_pipeline.py \
  tests/test_pipeline_bootstrap.py
```

범용 IR을 compile한 뒤 YAML에서 task와 executor를 직접 확인할 수도 있다.

```bash
.venv/bin/python -m pipelines.pertinence.pipeline \
  --image "example/pertinence@sha256:$(printf 'a%.0s' {1..64})" \
  --pipeline-root gs://example-bucket/pertinence/runs \
  --output /tmp/pertinence-routing-training.yaml
```

이 명령은 cluster에 접속하지 않고 IR만 생성한다. 실제 image가 아니어도 digest 형식이면 compile
구조를 관찰할 수 있다.

Component의 핵심 함수는 KFP 없이 local path로 직접 호출할 수 있다. 이 방식은
[`test_pipeline_components.py`](../../tests/test_pipeline_components.py)가 사용하는 방법이며, 새 기능을
개발할 때 전체 cluster 실행보다 먼저 검증하기 좋다.

## 14. 자주 혼동하기 쉬운 점

### KFP가 모델을 학습하는가?

KFP 자체는 실행 순서와 artifact 전달을 관리한다. 실제 dispatcher 학습은 container 안에서
`pertinence.training`과 `pertinence.optimization` 코드가 수행한다.

### Importer가 데이터를 변환하는가?

아니다. Importer는 외부 GCS artifact를 KFP 그래프에 등록한다. Dataset 변환은 legacy의
`prepare-dataset-bundle` 또는 파이프라인 외부 dataset adapter가 수행한다.

### KFP cache와 routing cache는 같은가?

아니다. KFP cache는 이전 task output을 재사용하는 orchestration 기능이다. Routing cache는
`features.npy`, `expert_predictions.npy`처럼 NSGA-II 학습을 빠르게 하기 위해 직접 생성한 데이터
artifact다.

### Registry가 중앙 Model Registry인가?

아니다. 현재 구현은 run-scoped SQLite artifact다. Atomic 등록과 idempotency 계약을 검증하기 위한
구현이며 여러 run을 모으는 중앙 서비스는 아니다.

### 범용 파이프라인이 임의 모델을 실행할 수 있는가?

아니다. Bundle v1 계약을 만족하는 단일 파일 ONNX image classifier만 지원한다. Arbitrary Python
source, remote endpoint, custom ONNX operator와 GPU 자동 선택은 지원하지 않는다.

### Final 결과로 가장 좋은 모델을 다시 선택해도 되는가?

현재 실험 계약에서는 안 된다. Final은 Search에서 보존된 해의 일반화 상태를 기술하기 위한
holdout이다. Final 결과로 다시 선택하려면 별도 validation set이나 새로운 test set이 필요하다.

