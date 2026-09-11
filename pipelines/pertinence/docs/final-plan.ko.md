# PERTINENCE 구성 독립형 Kubeflow Pipeline 최종 계획

## 1. 목표

이 pipeline의 최우선 목표는 **지원되는 입력 계약 안에서 dataset과 expert 모델 구성만
바꾸어도 pipeline 코드, component 코드, container image를 수정하지 않고 dispatcher를 학습**할
수 있게 하는 것이다.

모델 개발자가 해야 할 일은 다음 세 입력을 준비하고 기존 pipeline을 제출하는 것뿐이다.

1. `DatasetBundle`: 학습 및 평가 대상 dataset
2. `ExpertBundle`: 사전학습된 expert 모델과 비용/입출력 metadata
3. `RunConfig`: 사용할 expert 조합, feature extractor, split, dispatcher, NSGA-II 설정

Pipeline은 입력을 먼저 검증하고, expert 추론 cache 생성, dispatcher 탐색, final 평가, Model
Registry 등록까지 수행한다. Expert 자체를 다시 학습하거나 final 결과로 단일 best dispatcher를
자동 선정하는 것은 범위가 아니다.

### 1.1 No-code onboarding의 범위

초기 지원 범위는 고정된 class 집합을 갖는 **image classification**이다. 새 dataset이나 모델이
아래 versioned contract를 만족하면 내부 구현 변경 없이 사용할 수 있다.

- Dataset은 표준 `DatasetBundle` 형식으로 패키징돼야 한다.
- Expert는 pipeline runtime이 지원하는 portable model 형식으로 export돼야 한다.
- 모든 expert는 같은 class index 의미를 사용해야 한다.
- 선택된 feature extractor는 logits와 classifier 직전의 2차원 feature를 함께 출력해야 한다.
- 비용 목적함수에 사용할 동일 단위의 cost가 모든 expert에 제공돼야 한다.

새 task 유형(object detection, generation 등), 새 model runtime/custom operator, 원격 API 전용
모델은 contract 자체를 확장해야 하므로 no-code 범위 밖이다. 임의 Python source를 입력으로 받아
실행하지 않는다. 이는 dependency 충돌, 원격 코드 실행, 재현성 저하를 막기 위한 경계다.

## 2. 개발자 사용 경험

Pipeline의 공개 입력은 다음 세 immutable URI와 SHA-256이다.

| 입력 | 내용 | 변경 시 pipeline 코드 수정 |
|---|---|---:|
| `dataset_bundle_uri`, `dataset_bundle_sha256` | dataset manifest와 shard | 없음 |
| `expert_bundle_uri`, `expert_bundle_sha256` | model artifact와 expert manifest | 없음 |
| `run_config_uri`, `run_config_sha256` | 모델 조합과 학습/탐색 설정 | 없음 |

Cluster 접속 정보, service account, object-store credential, Registry credential, GPU node 정책은
플랫폼 운영 설정으로 관리하며 모델 개발자가 `RunConfig`에 넣지 않는다.

권장 사용자 흐름은 다음과 같다.

```text
pertinence package-dataset ...  → DatasetBundle
pertinence package-experts ...  → ExpertBundle
pertinence validate-run ...     → 로컬 계약 검증
pertinence submit ...           → 동일한 compile된 KFP pipeline 실행
```

패키징 명령은 manifest와 SHA-256을 자동 생성해야 한다. 개발자가 KFP artifact 내부 경로,
dispatcher feature dimension, expert 개수에 따른 penalty matrix 크기 등을 직접 계산해서는 안 된다.

## 3. 입력 계약

### 3.1 `DatasetBundle`

Dataset은 dataset 이름에 종속되지 않는 하나의 immutable artifact로 전달한다.

```text
dataset-bundle/
  dataset.yaml
  train/part-*.parquet
  evaluation/part-*.parquet
```

각 row는 최소한 다음 필드를 가진다.

- `sample_id: string`: bundle 전체에서 유일하고 안정적인 ID
- `image: binary`: manifest에 선언된 codec으로 인코딩된 이미지
- `target: int64`: `[0, num_classes)` 범위의 class index

`dataset.yaml`에는 다음을 기록한다.

```yaml
schema_version: 1
task: image_classification
dataset_id: example-dataset
dataset_version: v3
num_classes: 3
class_names: [class-0, class-1, class-2]
image_codec: jpeg
train_glob: train/part-*.parquet
evaluation_glob: evaluation/part-*.parquet
train_samples: 50000
evaluation_samples: 10000
content_sha256: <sha256>
```

`class_names`의 실제 길이는 반드시 `num_classes`와 같아야 한다. Train과 evaluation의
`sample_id`는 서로 겹치지 않아야 한다. Search/final 분리는 pipeline이 evaluation pool에서
`RunConfig`의 seed와 크기 또는 비율로 생성한다. 생성된 split manifest와 fingerprint는 별도
artifact로 보존한다.

### 3.2 `ExpertBundle`

초기 portable runtime은 ONNX를 기준으로 한다. 모델 구조가 달라도 아래 입출력 계약으로 export할
수 있으면 pipeline 내부 코드를 수정하지 않는다.

```text
expert-bundle/
  experts.yaml
  models/<expert-id>/model.onnx
```

`experts.yaml`의 각 expert는 다음 metadata를 가진다.

```yaml
schema_version: 1
bundle_id: example-experts
experts:
  - id: small-model
    version: v5
    format: onnx
    model_path: models/small-model/model.onnx
    model_sha256: <sha256>
    input:
      layout: NCHW
      dtype: float32
      shape: [3, 224, 224]
      preprocessing:
        resize: [224, 224]
        mean: [0.485, 0.456, 0.406]
        std: [0.229, 0.224, 0.225]
    outputs:
      logits: logits
      features: features
    num_classes: 3
    costs:
      mflops: 85.4
```

계약은 다음과 같다.

- 모든 선택 expert는 batch 입력과 `[batch, num_classes]` logits를 지원한다.
- `RunConfig`에서 feature extractor로 선택된 expert만 `features` 출력이 필수이며, shape는
  `[batch, feature_dim]`이어야 한다.
- `feature_dim`은 smoke inference에서 읽는다. Config에 `1024` 같은 값을 중복 기입하지 않는다.
- 전처리는 versioned allow-list의 선언형 연산만 허용하며 model fingerprint에 포함한다.
- Model artifact에는 임의 import나 network download가 없어야 한다.
- Cost는 선택한 목적함수 key와 단위가 모든 expert에서 같아야 한다.

ONNX로 export할 수 없는 사내 모델은 별도 pipeline 코드를 추가하는 대신, 향후 정의할 표준
inference service protocol을 구현하고 immutable endpoint/model revision을 제공하는 방식으로
확장한다. 이 backend가 구현되기 전까지 원격 endpoint는 입력으로 허용하지 않는다.

### 3.3 `RunConfig`

`RunConfig`는 bundle에 들어 있는 모델 중 실제 routing pool과 실행 설정을 선택한다.

```yaml
schema_version: 1
run_name: example-routing-run
seed: 20260824

routing:
  expert_ids: [small-model, medium-model, large-model]
  feature_extractor_id: small-model
  fallback_expert_id: small-model
  cost_key: mflops
  require_cost_sorted: true

split:
  search_size: 8000
  final_size: 2000
  seed: 20260824

dispatcher:
  epochs: 20
  batch_size: 256
  optimizer: adam
  learning_rate: 0.001
  weight_decay: 0.0
  ens_beta: 0.9999
  normalize_class_weights: mean_one

nsga2:
  population_size: 50
  generations: 50
  crossover: SBX
  crossover_eta: 20
  crossover_probability: 0.9
  mutation: polynomial
  mutation_eta: 25
  mutation_probability: null
  penalty_min: 0.0
  penalty_max: 100.0
  weighting_schemes: [INS, ISNS, ENS]
```

`expert_ids` 순서는 cheapest-correct label과 under/overestimation 의미를 결정하므로 cost 오름차순이어야
한다. `require_cost_sorted: true`일 때 pipeline은 순서가 잘못된 입력을 자동 재정렬하지 않고 실패시킨다.
조용한 재정렬로 route class 의미가 달라지는 것을 방지하기 위해서다.

## 4. 최종 Pipeline

```text
DatasetBundle ─┐
ExpertBundle ──┼─→ validate-run-contract
RunConfig ─────┘             │
                              ↓
                    build-routing-dataset
                      ├─ train-cache ───┐
                      ├─ search-cache ──┴─→ train-dispatcher ─→ search-result/Pareto bundle ─┐
                      └─ final-cache ────────────────────────────────────────────────────────┴─→ evaluate-final
                                                                                                      │
                                                                                                      ↓
                                                                                 register-pareto-dispatchers
```

| Component | 입력 | 기능 | 출력 |
|---|---|---|---|
| `validate-run-contract` | 세 input bundle/설정 | schema, hash, sample, model I/O, cost, 자원 범위를 fail-fast 검증 | `normalized-run-contract`, `split-manifest`, validation report |
| `build-routing-dataset` | 검증된 입력과 split manifest | expert 추론, feature 추출, cheapest-correct label 생성 | 독립된 `train-cache`, `search-cache`, `final-cache` |
| `train-dispatcher` | train/search cache, normalized config | 동적 크기 dispatcher 학습과 NSGA-II Pareto 탐색 | `search-result`, `pareto-dispatcher-bundle` |
| `evaluate-final` | final cache, search result, Pareto bundle | 재학습 없이 모든 Search Pareto dispatcher 평가 | `final-report`, evaluated dispatcher bundle |
| `register-pareto-dispatchers` | evaluated bundle과 전체 lineage | 멱등적으로 dispatcher version 등록 | Registry model/version ID 목록 |

Pipeline DAG와 component image digest는 입력 dataset이나 expert 조합이 바뀌어도 동일하다.

## 5. Component 상세 설계

### 5.1 `validate-run-contract`

모든 비싼 GPU 작업 전에 다음을 검사한다.

1. 세 입력의 schema version, URI, SHA-256과 내부 path 안전성
2. Dataset의 실제 row 수, unique sample ID, target 범위, train/evaluation 비중복
3. `search_size + final_size == evaluation_samples`
4. 선택 expert ID의 존재·중복 여부와 extractor/fallback membership
5. 모든 expert의 class 수가 dataset `num_classes`와 일치하는지
6. Cost 누락, 단위 불일치, 비유한 값, model order 위반
7. ONNX load 및 작은 batch smoke inference
8. Logits shape/유한성, extractor feature rank와 `feature_dim`
9. 모델 수와 예상 cache 크기/탐색량이 플랫폼 운영 한도 안인지

성공하면 원본 manifest를 그대로 downstream에 넘기지 않고, 검증 결과가 고정된
`normalized-run-contract.json`을 만든다. 다음 값은 입력에서 동적으로 유도한다.

```text
C = dataset.num_classes
K = len(routing.expert_ids)
D = feature extractor smoke output width
dispatcher = Linear(D, K)
penalty matrix = K × K
chromosome genes = K² + 1
route confusion matrix = K × K
evaluation target count = population_size × generations
```

### 5.2 `build-routing-dataset`

1. Dataset shard를 streaming 방식으로 읽는다.
2. 선택된 expert를 manifest 순서대로 frozen/eval 상태에서 실행한다.
3. Expert별 선언형 preprocessing을 적용한다.
4. Feature extractor의 feature output과 각 expert prediction을 저장한다.
5. 정답을 맞힌 expert 중 가장 앞선(cost가 가장 낮은) expert를 route label로 지정한다.
6. 모두 틀리면 `fallback_expert_id`를 사용한다.
7. Split별 cache를 서로 다른 KFP output artifact로 생성한다.

Cache schema는 고정 크기 대신 다음 shape metadata를 포함한다.

```text
features:           [samples, D]
expert_predictions: [samples, K]
targets:            [samples]
route_labels:       [samples]
sample_ids:          [samples]
```

각 cache에는 dataset, expert bundle, selected expert order, preprocessing, cost vector, split,
config, code/container fingerprint를 저장한다.

### 5.3 `train-dispatcher`

`K`와 `D`를 cache manifest에서 읽어 `Linear(D, K)`를 생성한다. Config나 component 코드에
expert 수와 feature dimension을 고정하지 않는다. Penalty matrix, chromosome, class weight,
metric 배열도 모두 `K`에서 유도한다.

Search component의 입력 signature에는 `final-cache`, final URI, pipeline-wide shared volume을
포함하지 않는다. 전체 후보의 search metric과 chromosome은 `search-result`에 저장하고, Search
Pareto 후보의 checkpoint는 별도 bundle로 저장한다.

### 5.4 `evaluate-final`

1. Search Pareto ID와 checkpoint manifest를 대조한다.
2. Checkpoint의 `D`, `K`, expert order, cache fingerprint를 확인한다.
3. Dispatcher를 재학습하지 않고 final cache에서 평가한다.
4. System accuracy, average cost, route accuracy, expert별 선택 수, under/overestimation,
   `K × K` route confusion matrix를 기록한다.

Final metric은 보고에만 사용하며 후보를 제거하거나 다시 선택하지 않는다.

### 5.5 `register-pareto-dispatchers`

Registry에는 checkpoint만 아니라 실행에 필요한 다음 계약을 함께 등록한다.

- Dataset ID/version/schema와 dataset fingerprint
- Expert bundle ID, 선택 expert 순서, 각 model version/SHA-256
- Feature extractor/fallback ID, preprocessing, `C`, `D`, `K`
- Cost key/unit/vector와 router cost 계산 규칙
- Search evaluation ID, Pareto front ID, penalty와 weighting
- Search/final metric, split fingerprint, code와 image digest

Idempotency key는 최소한
`(run_contract_fingerprint, front_id, evaluation_id, checkpoint_sha256)`로 만든다. 등록 상태는
`candidate` 또는 `evaluated`이며 production 승격은 pipeline 밖의 승인 정책에서 수행한다.

## 6. Artifact와 실행 정책

- Tensor cache, model, bundle, 전체 report는 KFP artifact로 전달한다.
- URI, digest, seed와 작은 scalar만 parameter로 전달한다.
- Pipeline root object storage를 정본으로 사용한다.
- Container image와 입력 bundle은 immutable digest로 고정한다.
- KFP cache와 별개로 artifact 내부 fingerprint를 검증한다.
- 모든 component는 허용된 output path에 원자적으로 기록한다.
- 모델 수와 입력 크기의 지원 상한은 platform profile로 고정하고 validation 단계에서 초과 입력을
  명확히 거부한다. 모델 개발자가 pod resource 코드를 수정하게 만들지 않는다.

| Component | GPU | KFP cache | Retry |
|---|---:|---:|---:|
| `validate-run-contract` | smoke용 1 | 활성 | 일시적 저장소 오류 1회 |
| `build-routing-dataset` | 1 | 활성 | 일시적 저장소 오류 1회 |
| `train-dispatcher` | 1 | 활성 | 0회 |
| `evaluate-final` | 1 | 활성 | 1회 |
| `register-pareto-dispatchers` | 0 | 비활성 | 멱등 등록 전제로 1회 |

Search는 generation checkpoint/resume이 구현되기 전까지 자동 retry하지 않는다.

## 7. 필요한 구현 변경

### Phase 1: 입력 계약과 generic core

1. `DatasetBundle`, `ExpertBundle`, `RunConfig`, `NormalizedRunContract` schema와 parser를 추가한다.
2. CIFAR-10 이름, 50,000/10,000/8,000/2,000, 10 classes, 고정 expert inventory 검증을 제거하고
   계약 기반 검증으로 교체한다.
3. `torchvision.CIFAR10` loader를 versioned `DatasetBundleReader`로 교체한다.
4. chenyaofo registry/source loader를 portable `ExpertRuntime`으로 교체한다.
5. Cost를 Python registry가 아니라 normalized expert metadata에서 읽는다.
6. 기존 CIFAR-10 재현 구성을 세 contract의 fixture로 변환한다.

### Phase 2: Artifact 중심 CLI

1. 기존 `--cache-dir`을 명시적인 split별 input/output 파일 인자로 바꾼다.
2. `search.json`과 Pareto checkpoint bundle을 분리한다.
3. Final report의 pod-local 경로를 bundle 안의 logical model ID로 바꾼다.
4. 패키징, 로컬 validation, submission CLI를 추가한다.

### Phase 3: KFP와 Registry

1. 다섯 component를 container component로 구현한다.
2. 동일 component/DAG를 연결한 KFP v2 pipeline을 작성하고 IR YAML로 compile한다.
3. Search input에 final artifact가 없는지 IR 구조 테스트를 추가한다.
4. Registry adapter와 idempotency transaction을 구현한다.
5. Container image와 KFP SDK/runtime 버전을 lock한다.

제안 파일 구조:

```text
src/pertinence/contracts/
  dataset.py
  experts.py
  run.py
src/pertinence/runtimes/
  onnx.py
pipelines/pertinence/
  components.py
  pipeline.py
  Dockerfile
  requirements-components.lock
  requirements-kfp.lock
  docs/
  tests/
schemas/
  dataset-bundle-v1.schema.json
  expert-bundle-v1.schema.json
  run-config-v1.schema.json
```

## 8. 검증 계획

### 8.1 Contract 및 core test matrix

동일한 pipeline/core 코드에 대해 다음 조합을 parameterize한다.

| Dataset | Classes | Experts | Feature dim | Split |
|---|---:|---:|---:|---:|
| synthetic-small | 3 | 2 | 16 | 60/20/20 |
| CIFAR-10 fixture | 10 | 4 | 실제 추론값 | 50k/8k/2k |
| CIFAR-100 fixture | 100 | 3 또는 5 | 실제 추론값 | config 기반 |

필수 failure test는 hash 불일치, 중복 ID, class 수 불일치, 잘못된 logits/features shape, cost 누락,
cost 순서 위반, split overlap, unsupported schema/runtime, stale cache/checkpoint를 포함한다.

### 8.2 KFP test

1. Component interface unit test
2. Pipeline compile test와 IR snapshot test
3. Search task에 final artifact/URI/공용 volume이 없는지 검사
4. Synthetic bundle을 사용한 container end-to-end smoke run
5. 두 개의 서로 다른 dataset과 두 개 이상의 expert 조합으로 같은 IR을 실행
6. 동일 입력 재실행 시 artifact cache와 Registry idempotency 검증

## 9. 완료 기준

- Dataset/model 조합을 바꿀 때 `pipelines/`, `src/pertinence/`, container image에 변경이 없다.
- 모델 개발자는 세 input URI와 digest만으로 기존 compile된 pipeline을 제출할 수 있다.
- 최소 두 dataset과 expert 수 2/3/5 구성이 동일 pipeline IR에서 끝까지 실행된다.
- `C`, `D`, `K`, split 크기, penalty/chromosome/confusion shape가 입력에서 동적으로 유도된다.
- 잘못된 입력은 첫 validation component에서 actionable message와 함께 실패한다.
- Train/Search/Final cache의 sample ID, shape, expert order와 fingerprint가 일치한다.
- Search는 final cache에 접근할 수 없다.
- 평가 수는 config의 population과 generation에서 유도되고 Pareto checkpoint가 모두 보존된다.
- Final은 checkpoint를 재학습하지 않는다.
- Registry에는 Search Pareto dispatcher만 중복 없이 등록되며 전체 실행 계약을 조회할 수 있다.
- 기존 CIFAR-10 재현 결과는 contract migration 전후 허용 오차 안에서 동일하다.

이 완료 기준을 만족하기 전에는 “설정만 바꾸면 새 dataset/model을 사용할 수 있다”고 간주하지
않는다.
