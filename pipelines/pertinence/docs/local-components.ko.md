# Portable Kubeflow pipeline 컴포넌트 로컬 실행

관련 설계와 운영 범위는 [설계 문서](design.ko.md), [최종 계획](final-plan.ko.md),
[Cloud 운영 아이디어](operator-selected-cloud.ko.md)에 함께 정리되어 있다.

최종 계획의 다섯 컴포넌트는 `src/pertinence/`에 구현되어 있으며 Python API와
`pertinence` CLI에서 개별 실행할 수 있다. 클라우드 접속이나 배포 없이 사용할 수 있다.
기존 CIFAR 재현용 `baseline`, `fixed`, `search`, `final` 명령도 유지된다.

## 설치와 검증

기존 환경에 선택 의존성을 추가한다. Python 3.11이 기준이며 현재 저장소의 Python 3.12
환경에서도 검증했다.

```bash
.venv/bin/python -m pip install -r pipelines/pertinence/requirements-kfp.lock
.venv/bin/python -m pip install -e . --no-deps
.venv/bin/python -m pytest pipelines/pertinence/tests
.venv/bin/ruff check .
```

`pipelines/pertinence/requirements-kfp.lock`은 portable runtime과 KFP SDK의 고정 버전 목록이다.
전체 전이 의존성의 lock은 아니다. KFP를 쓰지 않는 실행 환경은 `.[pipeline]`만 설치해도 된다.
컨테이너 runtime은 `pipelines/pertinence/requirements-components.lock`과
`pipelines/pertinence/Dockerfile`을 사용하며
KFP SDK를 이미지에 포함하지 않는다. Dockerfile은 프로젝트 안의 재현 가능한 build 정의만
제공한다. 현재 단계에서는 이미지를 build하거나 registry로 push하지 않는다.
기본 runtime은 CPU ONNX Runtime이다. CUDA 환경은 호환되는 `onnxruntime-gpu`와 PyTorch/CUDA
환경을 별도로 준비하고 validation의 `--device cuda`로 선택한다. CUDA provider가 없으면
명확하게 실패한다.

테스트는 다운로드 없이 작은 PNG/Parquet 데이터와 실제 ONNX 모델을 생성한다.
클래스 3/10/100, expert 2/3/5, feature 차원 16/7/9 조합으로 검증부터 등록까지 실행한다.
이 입력은 합성 데이터이며 실제 CIFAR 학습 결과의 재현성 검증을 대신하지 않는다.

## 입력 패키징

`DatasetBundle`과 `ExpertBundle`의 manifest 필드는
[최종 계획](final-plan.ko.md)의 v1 형식을 따른다.
다음 예시는 이미 준비된 CIFAR-10 Parquet shard와 ONNX 모델을 패키징한다.
생성물은 모두 `artifacts/cifar10/portable/` 아래에 둔다.

```bash
.venv/bin/pertinence package-dataset \
  --source artifacts/cifar10/portable/dataset-source \
  --output artifacts/cifar10/portable/dataset-bundle

.venv/bin/pertinence package-experts \
  --source artifacts/cifar10/portable/expert-source \
  --output artifacts/cifar10/portable/expert-bundle

.venv/bin/pertinence artifact-sha256 artifacts/cifar10/portable/run.yaml
```

- `dataset-source/dataset.yaml`에는 ID/version/classes/codec/glob을 선언한다.
  패키징 명령은 실제 row 수와 `content_sha256`을 계산하고 전체 row를 검증한다.
- `expert-source/experts.yaml`에는 모델 경로와 I/O, 전처리, 비용을 선언한다.
  `model_sha256`은 명령이 채운다. ONNX smoke inference는 validation에서 수행한다.
- 패키징 명령은 완성된 번들의 SHA-256을 stdout에 출력한다.
- 파일 digest는 파일 바이트의 SHA-256이다. 디렉터리 digest는 정렬된
  `{상대 POSIX 경로: 파일 SHA-256}` 매핑의 canonical JSON SHA-256이다.
  Canonical JSON은 Python `json.dumps(..., sort_keys=True, separators=(",", ":"))` 규칙이다.
- `content_sha256`은 `dataset.yaml`을 제외한 동일한 inventory의 digest다.
  Dataset manifest 자체를 포함한 번들 digest와 구별한다.
- 입력은 로컬 경로나 `file://` URI다. 원격 object-store artifact의 다운로드는 플랫폼이
  수행해야 한다. 심볼릭 링크, 경로 탈출, 임의 Python 코드, 원격 모델 endpoint는 지원하지 않는다.

전처리 v1은 RGB 변환, PIL bilinear resize, `[0, 1]` float32 변환, 채널별 mean/std 정규화와
NCHW 배치만 지원한다. Resize는 `[height, width]`다. ONNX는 동적 batch를 지원해야 하며
external tensor data와 custom operator domain을 포함할 수 없다. Feature extractor에는
`[batch, D]` features 출력이 필요하다. Class index 의미는 모든 모델에서 일치해야 하며,
expert의 선택 필드 `class_names`를 제공하면 dataset의 순서와 대조한다.

비용 key 자체를 단위로 사용한다. `mflops`에서는 router 비용을 `2 * D * K / 1e6`으로 계산한다.
다른 key는 `routing.router_cost`에 같은 단위의 값을 명시해야 한다. Expert 순서는 항상
cost 오름차순이어야 한다. `require_cost_sorted: false`도 순서 변경이나 자동 정렬을 허용하지 않는다.
비율 기반 split은 아직 지원하지 않으며 `search_size`와 `final_size`를 명시해야 한다.

## 다섯 컴포넌트 실행

아래 digest 변수에는 패키징 명령이 출력한 실제 SHA-256을 넣는다.
`run.yaml`은 최종 계획의 RunConfig 예시를 선택한 모델 ID에 맞게 작성한다.

```bash
.venv/bin/pertinence validate-run \
  --dataset-bundle-uri artifacts/cifar10/portable/dataset-bundle \
  --dataset-bundle-sha256 "$DATASET_SHA256" \
  --expert-bundle-uri artifacts/cifar10/portable/expert-bundle \
  --expert-bundle-sha256 "$EXPERT_SHA256" \
  --run-config-uri artifacts/cifar10/portable/run.yaml \
  --run-config-sha256 "$RUN_CONFIG_SHA256" \
  --normalized-contract artifacts/cifar10/portable/contract.json \
  --split-manifest artifacts/cifar10/portable/split.json \
  --validation-report artifacts/cifar10/portable/validation.json

.venv/bin/pertinence build-routing-dataset \
  --dataset-bundle artifacts/cifar10/portable/dataset-bundle \
  --expert-bundle artifacts/cifar10/portable/expert-bundle \
  --normalized-contract artifacts/cifar10/portable/contract.json \
  --split-manifest artifacts/cifar10/portable/split.json \
  --train-cache artifacts/cifar10/portable/train-cache \
  --search-cache artifacts/cifar10/portable/search-cache \
  --final-cache artifacts/cifar10/portable/final-cache

.venv/bin/pertinence train-dispatcher \
  --train-cache artifacts/cifar10/portable/train-cache \
  --search-cache artifacts/cifar10/portable/search-cache \
  --normalized-contract artifacts/cifar10/portable/contract.json \
  --search-result artifacts/cifar10/portable/search-result.json \
  --pareto-bundle artifacts/cifar10/portable/pareto

.venv/bin/pertinence evaluate-final \
  --final-cache artifacts/cifar10/portable/final-cache \
  --search-result artifacts/cifar10/portable/search-result.json \
  --pareto-bundle artifacts/cifar10/portable/pareto \
  --normalized-contract artifacts/cifar10/portable/contract.json \
  --final-report artifacts/cifar10/portable/final-report.json \
  --evaluated-bundle artifacts/cifar10/portable/evaluated

.venv/bin/pertinence register-pareto-dispatchers \
  --evaluated-bundle artifacts/cifar10/portable/evaluated \
  --registry-db artifacts/cifar10/portable/registry.sqlite3 \
  --registration-report artifacts/cifar10/portable/registration.json
```

다섯 컴포넌트의 입출력은 각각 명시적인 파일/디렉터리 경로다. 캐시는 NPY 배열과 JSON manifest,
checkpoint는 pickle 없이 읽는 NPZ다. 파일 내용의 SHA-256과 manifest fingerprint를 확인한다.
Split은 정렬한 안정적 sample ID에 seed permutation을 적용하므로 shard의 순서에 의존하지 않는다.
Source 코드, 주요 runtime 패키지 버전, image digest도 실행 계약에 포함한다. 코드나 의존성 변경
후에는 새 실행 계약과 artifact를 생성해야 한다.

기존 비어 있지 않은 번들 출력 디렉터리는 덮어쓰지 않는다. 재실행에는 새 output 경로를 쓴다.
캐시/모델 번들은 임시 디렉터리를 완성한 후 rename하고, JSON도 임시 파일을 거쳐 기록한다.
여러 output artifact 전체를 묶는 파일시스템 transaction은 아니므로 일부 output이 이미
존재하는 실패 실행의 재시도는 새 작업 디렉터리에서 수행해야 한다.

Search는 train/search artifact와 normalized contract만 받는다. Final cache의 경로나 데이터는
입력에 없다. 학습은 기존 `training.py`와 `optimization.py` 코어를 재사용한다. 모든 후보의
metric과 chromosome을 저장하고 Search Pareto checkpoint를 별도 보존한다. Final은 해당
checkpoint 전체를 재학습 없이 평가하며 final metric에 따른 제거·순위 재선정을 하지 않는다.

Registry는 로컬 SQLite 구현이다. Checkpoint 바이트와 전체 실행 계약, search/final metric을
하나의 transaction으로 저장한다. 동일 idempotency key의 재등록은 동일 model/version ID를
반환하며, 같은 key에 다른 내용이 들어오면 transaction 전체를 rollback한다. 상태는 `evaluated`다.
원격 Model Registry용 구현은 `RegistryAdapter.register_many` 계약에 맞춰 별도로 연결할 수 있다.

## KFP 인터페이스와 로컬 compile

`pipelines/pertinence/components.py`는 다섯 container component를 정의한다.
`pipeline.py`는 URI/digest 여섯 parameter와 importer를 고정 DAG에 연결한다. 실제 사용할
container image의 digest를 얻은 뒤 다음과 같이 IR을 로컬에서 생성할 수 있다.

```bash
.venv/bin/python -m pipelines.pertinence.pipeline \
  --image "$PERTINENCE_COMPONENT_IMAGE" \
  --output artifacts/cifar10/portable/pipeline.yaml
```

Image는 `registry/name@sha256:<64 hex>` 형식이어야 한다. Compile에는 image pull, cluster 접속,
submission이 없다. 테스트에서는 가상 digest로 인터페이스와 DAG 구조만 확인한다.
GPU 설정은 compile 시 `--device cuda`로 지정하며 모든 연산 component에 GPU 1개를 요청한다.
Search retry는 0, 나머지는 1이고 Registry만 KFP caching을 끈다. KFP의 task retry는 오류 종류를
구분하지 않으므로 validation의 계약 오류에도 한 번 재시도할 수 있다.

현재 작업에는 container build/push, object store, credential, PVC, 원격 Registry, cluster 제출을
포함하지 않는다. 특히 기본 Registry 경로 `/registry/registry.db`는 추후 영속 저장소 연결이
필요하다. 현 상태의 IR을 배포 준비 완료 상태로 간주하면 안 된다. 실제 CIFAR ONNX export와
migration 전후 수치 비교도 별도 검증 대상이다.

인터페이스 구현은 공식 [KFP container component 문서](https://www.kubeflow.org/docs/components/pipelines/user-guides/components/container-components/),
[ONNX Runtime Python API](https://onnxruntime.ai/docs/api/python/api_summary),
[Arrow Parquet streaming API](https://arrow.apache.org/docs/python/generated/pyarrow.parquet.ParquetFile.html)를 참고한다.
