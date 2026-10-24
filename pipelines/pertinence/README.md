# PERTINENCE portable pipeline

이 package는 이미 생성된 `DatasetBundle`, `ExpertBundle`, `RunConfig`를 입력받는 범용 CPU 학습
파이프라인과, 기존 CIFAR-10 raw asset을 같은 bundle로 변환하는 legacy 파이프라인을 분리한다.
재사용 가능한 contract, ONNX runtime, cache, 학습과 평가는 `src/pertinence/`에 있고 이 디렉터리는
container interface, orchestration, image build와 compile/submit client만 소유한다.

```text
pipelines/pertinence/
├── components.py                 # 공용 5개 component와 CIFAR-10 adapter 2개
├── pipeline.py                   # 범용 5단계 DAG와 IR compiler
├── submit.py                     # 범용 OSS KFP submit client
├── cifar10_pipeline.py           # 보존된 CIFAR-10 7단계 legacy DAG
├── cifar10_submit.py             # legacy submit client
├── Dockerfile                    # 공용 CPU component image
├── cloudbuild.yaml               # Artifact Registry build
├── requirements-components.lock # image runtime pins
└── requirements-kfp.lock         # compiler/client pins
```

Component E2E와 두 IR 구조 검사는 root `tests/test_pipeline_components.py`와
`tests/test_kfp_pipeline.py`가 소유한다.

코드를 처음 읽는 사람을 위한 DAG, component, artifact와 domain 함수의 상세 연결은
[`implementation-guide.ko.md`](implementation-guide.ko.md)에 정리한다.

GKE, Workload Identity, launcher ConfigMap과 production 운영 절차는
[cloud runbook](../../cloud/docs/runbook.ko.md)이 정본이다. Repository-owned Kubernetes manifest는
`cloud/kfp/`에 둔다.

## 설치와 검증

Python 3.11을 기준으로 pipeline/KFP 선택 의존성을 설치한다.

```bash
.venv/bin/python -m pip install -r pipelines/pertinence/requirements-kfp.lock
.venv/bin/python -m pip install -e . --no-deps
.venv/bin/python -m pytest tests/test_pipeline_components.py tests/test_kfp_pipeline.py
.venv/bin/ruff check .
```

Component image는 CPU PyTorch와 CPU ONNX Runtime을 사용하고 KFP SDK를 포함하지 않는다. Unit
test는 다운로드 없이 synthetic PNG/Parquet와 실제 ONNX model을 만들어 class 3/10/100,
선택 expert 2/3/5, 여러 feature dimension에서 validation부터 registration까지 검사한다. 이 test는
실제 CIFAR 결과를 대신하지 않는다.

## 입력 계약과 지원 범위

범용 workflow는 세 immutable 입력과 각각의 SHA-256을 받는다.

| 입력 | 내용 |
|---|---|
| `DatasetBundle` | `dataset.yaml`과 train/evaluation Parquet shard를 가진 GCS prefix |
| `ExpertBundle` | `experts.yaml`과 ONNX expert를 가진 GCS prefix |
| `RunConfig` | expert 선택, split, dispatcher와 NSGA-II 설정을 가진 GCS object |

Bundle digest는 정렬된 `{relative POSIX path: file SHA-256}` mapping의 canonical JSON digest이고,
RunConfig digest는 파일 byte SHA-256이다. Bundle URI는 manifest가 바로 아래에 있는 prefix를
가리켜야 한다. URI는 digest를 포함한 content-addressed 경로로 만들고 기존 prefix를 overwrite하지
않는다.

범용이라는 표현은 Bundle v1 계약 안에서 dataset, class 수, expert 선택 수와 feature dimension을
바꿀 수 있다는 의미다. 현재 지원 범위는 JPEG/PNG RGB image classification, Parquet
`sample_id/image/target`, 단일 파일 ONNX, NCHW float32 입력과 동적 batch다. 모든 expert는 같은
class index를 사용해야 하고 `routing.expert_ids`는 비용 오름차순이어야 한다. Feature extractor에는
`[batch, D]` feature 출력이 필요하다. ONNX external data, custom operator, arbitrary Python source,
remote model endpoint, GPU와 resource auto-scaling은 지원하지 않는다.

`K`는 ExpertBundle 전체 모델 수가 아니라 `RunConfig.routing.expert_ids`의 선택 수다. Dispatcher
출력과 penalty matrix, `K²+1` chromosome은 이 값에서 파생된다. `C`는 DatasetBundle에서, `D`는
feature extractor smoke inference에서 파생된다.

## Bundle 준비

범용 KFP DAG는 raw dataset이나 framework checkpoint를 변환하지 않는다. Dataset별 adapter가 raw
data를 Parquet와 `dataset.yaml`로 만들고, model별 adapter가 ONNX와 `experts.yaml`을 만든다. 이미
계약 형태로 준비된 source는 다음 명령으로 봉인한다.

```bash
.venv/bin/pertinence package-dataset \
  --source artifacts/cifar10/runs/pipeline/dataset-source \
  --output artifacts/cifar10/runs/pipeline/dataset-bundle

.venv/bin/pertinence package-experts \
  --source artifacts/cifar10/runs/pipeline/expert-source \
  --output artifacts/cifar10/runs/pipeline/expert-bundle

.venv/bin/pertinence artifact-sha256 artifacts/cifar10/runs/pipeline/run.yaml
```

`prepare-dataset-bundle`과 `export-expert-bundle`은 기존 CIFAR-10 archive와 세 checkpoint를 위한
legacy adapter 명령이다. 기존 compiled IR과의 호환성을 위해 이름을 유지한다.

## 범용 KFP DAG

범용 production DAG에는 external GCS artifact를 등록하는 importer 3개와 container task 5개가
있다.

```text
DatasetBundle ─┐
ExpertBundle ──┼─ validate-run ─ build-routing-dataset
RunConfig ─────┘    ─ train-dispatcher ─ evaluate-final ─ register-dispatchers
```

Search에는 train/search cache와 normalized contract만 전달된다. Final cache나 그 URI, 공유 volume은
없다. Final은 search가 보존한 모든 checkpoint를 재학습 없이 복원하며 final metric으로 제거하거나
재선정하지 않는다. Registration은 중앙 registry가 아니라 run-scoped SQLite artifact를 만든다.

다섯 executor에는 `pipeline-pool` selector+toleration을 적용한다. Search는 retry 0과 72시간
deadline을 사용하고, validate/build/evaluate/register는 retry 1이다. Registration cache는 항상
꺼져 있다. 다른 task의 cache도 기본값은 off이며 immutable input URI, digest와 image를 그대로 쓰는
복구 IR에서 compile과 submit 양쪽에만 켠다.

Compile은 cluster에 접속하지 않는다.

```bash
.venv/bin/python -m pipelines.pertinence.pipeline \
  --image "registry/name@sha256:<64-hex-digest>" \
  --pipeline-root gs://bucket/pertinence/runs \
  --output artifacts/cifar10/runs/pipeline/pertinence-routing-training.yaml
```

Submit 예시는 다음과 같다.

```bash
.venv/bin/python -m pipelines.pertinence.submit \
  --host http://127.0.0.1:8888 \
  --pipeline artifacts/cifar10/runs/pipeline/pertinence-routing-training.yaml \
  --pipeline-root gs://RUN_BUCKET/pertinence/runs \
  --namespace kubeflow \
  --service-account pipeline-runner \
  --dataset-bundle-uri "gs://INPUT_BUCKET/datasets/${DATASET_SHA256}" \
  --dataset-bundle-sha256 "$DATASET_SHA256" \
  --expert-bundle-uri "gs://INPUT_BUCKET/experts/${EXPERT_SHA256}" \
  --expert-bundle-sha256 "$EXPERT_SHA256" \
  --run-config-uri "gs://INPUT_BUCKET/configs/${RUN_CONFIG_SHA256}/run.yaml" \
  --run-config-sha256 "$RUN_CONFIG_SHA256"
```

Submit client는 URI와 digest 형식만 검사한다. 실제 object 존재 여부, bundle byte hash, manifest와
ONNX 계약은 `validate-run`이 fail-fast로 검증하며 실패하면 downstream task는 시작하지 않는다.

## CIFAR-10 legacy 경로

기존 raw asset부터 재현해야 할 때는 보존된 7단계 compiler와 submit client를 사용한다.

```bash
.venv/bin/python -m pipelines.pertinence.cifar10_pipeline \
  --image "registry/name@sha256:<64-hex-digest>" \
  --pipeline-root gs://bucket/pertinence/runs \
  --output artifacts/cifar10/runs/pipeline/legacy-pipeline.yaml

.venv/bin/python -m pipelines.pertinence.cifar10_submit --help
```

Legacy pipeline 이름은 `pertinence-cifar10-cpu`이고 기존 10개 parameter, importer 5개와 container
task 7개를 유지한다. 범용 pipeline 이름은 `pertinence-routing-training-cpu`다. 새 범용 경로가
검증되어도 과거 IR, image와 GCS run artifact는 overwrite하지 않는다.

2026-09-20 검증 실행은 raw archive와 세 checkpoint, RunConfig를 입력받아 dataset/expert 변환,
계약 검증, routing cache 생성, dispatcher 탐색, final 평가, run-scoped SQLite 등록을 수행했다.
이 일곱 단계의 실행 식별자와 보존 artifact는
[해당 GKE 실행 기록](../../cloud/docs/runs/2026-09-20-dispatcher-gke.ko.md)에 남긴다.
