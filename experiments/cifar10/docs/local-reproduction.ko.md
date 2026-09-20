# CIFAR-10 로컬 재현 계약과 실행서

## 범위

이 문서는 현재 `experiment.yaml`을 사용해 native PyTorch/CUDA 경로를 실행하는 단일 정본이다.
PERTINENCE 방법론과 논문 공통 가정은 [공통 방법론 문서](../../method.ko.md)를 따른다.

정적 비교 경계와 routing expert를 구분한다.

- 정적 경계: `data/model_catalog.csv`의 Pareto 모델 10종. 표와 그림 비교에만 사용한다.
- routing expert: ShuffleNetV2 x0.5, MobileNetV2 x0.5, ResNet56의 3종.
- feature extractor와 fallback: ShuffleNetV2 x0.5.

현재 expert가 3개이므로 penalty matrix는 3×3이고 NSGA-II chromosome은 penalty gene 9개와
weighting gene 1개, 총 10개다. 과거 4-expert 로컬 결과는 현재 config의 결과가 아니며
[별도 결과 문서](local-4expert-results.ko.md)에 보존한다.

## 재현 계약

### 자산과 환경

- Python 3.11, repository lock과 `experiment.yaml`의 CUDA device를 사용한다.
- upstream source는 commit `786c16252c0fc58ee9adac063f8337cc4a7a497a`에 고정한다.
- CIFAR-10 archive, source archive와 checkpoint는 `assets.yaml`의 SHA-256으로 검증한다.
- 모델은 로컬 source와 checkpoint만 읽으며 import 또는 model load 중 네트워크를 사용하지 않는다.
- 전처리는 `ToTensor()`와 CIFAR-10 mean/std normalization을 사용하고 dispatcher feature에는
  augmentation을 적용하지 않는다.

### Split과 holdout

```text
official train 50,000 -> dispatcher_train 50,000
official test  10,000 -> ga_search 8,000 + final_evaluation 2,000
```

Split은 seed `20260824`의 결정론적 permutation으로 만들고 manifest hash를 cache/run fingerprint에
포함한다. Final 2,000장은 checkpoint 무결성 검증을 제외하면 `final` 단계 전까지 분석하지 않는다.
`baseline`, `fixed`, `search`는 final cache를 입력으로 받지 않는다.

### 학습과 탐색

- fixed dispatcher: INS, underestimation 10, overestimation 0.001.
- dispatcher: Adam, learning rate 1e-3, batch 256, 20 epochs.
- search: population 50 × generations 50, SBX eta 20/probability 0.9,
  polynomial mutation eta 25, INS/ISNS/ENS.
- objectives: `[-system_top1, average_system_mflops]`.
- final: search Pareto checkpoint를 재학습 없이 복원해 전부 평가한다.

## 준비

저장소 루트에서 환경을 설치한다.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock
.venv/bin/python -m pip install -e . --no-deps
```

자산을 준비하고 검증한다. 다운로드가 일어나는 단계는 첫 명령뿐이다.

```bash
.venv/bin/python scripts/prepare_assets.py \
  --manifest experiments/cifar10/configs/assets.yaml \
  --artifacts-dir artifacts/cifar10

.venv/bin/python scripts/prepare_assets.py \
  --manifest experiments/cifar10/configs/assets.yaml \
  --artifacts-dir artifacts/cifar10 \
  --verify-only

tar -xzf artifacts/cifar10/datasets/cifar-10-python.tar.gz \
  -C artifacts/cifar10/datasets
tar -xzf artifacts/cifar10/sources/pytorch-cifar-models-786c16252c0fc58ee9adac063f8337cc4a7a497a.tar.gz \
  -C artifacts/cifar10/sources
```

일부만 풀린 디렉터리는 재사용하지 않는다. Production preflight는 config의 `device: cuda`를
유지해야 하며 CPU override 결과는 자산 smoke 증거일 뿐 production 합격으로 보지 않는다.

## 연속 실행

장시간 search가 터미널 종료와 함께 중단되지 않도록 batch scheduler나 지속 세션에서 실행한다.

```bash
set -euo pipefail

PYTHON_BIN=.venv/bin/python
CONFIG=experiments/cifar10/configs/experiment.yaml
MANIFEST=experiments/cifar10/configs/assets.yaml
ARTIFACTS=artifacts/cifar10
DATASET_DIR=artifacts/cifar10/datasets
SOURCE_DIR=artifacts/cifar10/sources/pytorch-cifar-models-786c16252c0fc58ee9adac063f8337cc4a7a497a
CACHE_DIR=artifacts/cifar10/cache/pareto
RUN_DIR=artifacts/cifar10/runs/pareto

mkdir -p "$RUN_DIR"

"$PYTHON_BIN" -m pytest
"$PYTHON_BIN" -m ruff check .

"$PYTHON_BIN" scripts/preflight.py \
  --config "$CONFIG" \
  --manifest "$MANIFEST" \
  --artifacts-dir "$ARTIFACTS" \
  --dataset-dir "$DATASET_DIR" \
  --source-dir "$SOURCE_DIR" \
  --cache-dir "$CACHE_DIR" \
  | tee "$RUN_DIR/preflight.json"

"$PYTHON_BIN" scripts/build_cache.py \
  --config "$CONFIG" \
  --manifest "$MANIFEST" \
  --artifacts-dir "$ARTIFACTS" \
  --dataset-dir "$DATASET_DIR" \
  --source-dir "$SOURCE_DIR" \
  --cache-dir "$CACHE_DIR" \
  | tee "$RUN_DIR/cache-build.json"

"$PYTHON_BIN" -m pertinence.cli baseline \
  --config "$CONFIG" --cache-dir "$CACHE_DIR" \
  --output "$RUN_DIR/baselines.json"

"$PYTHON_BIN" -m pertinence.cli fixed \
  --config "$CONFIG" --cache-dir "$CACHE_DIR" \
  --output "$RUN_DIR/fixed.json"

"$PYTHON_BIN" -u -m pertinence.cli search \
  --config "$CONFIG" --cache-dir "$CACHE_DIR" \
  --output "$RUN_DIR/search.json" \
  | tee "$RUN_DIR/search-progress.log"

"$PYTHON_BIN" -m pertinence.cli final \
  --config "$CONFIG" --cache-dir "$CACHE_DIR" \
  --search-result "$RUN_DIR/search.json" \
  --output "$RUN_DIR/final.json"
```

## 합격 조건

| 단계 | 필수 조건 | 주요 산출물 |
|---|---|---|
| tests/lint | 두 명령 모두 exit 0 | 없음 |
| preflight | `status=passed`, CUDA, 세 expert load/forward 성공 | `preflight.json` |
| cache | expert accuracy gate 통과, 세 split written/reused | cache 3개 |
| baseline/fixed | fingerprint 일치, finite metric/loss | JSON, fixed checkpoint |
| search | 50세대와 2,500 evaluations 완료 | `search.json` |
| final | 모든 retained state 복원, finite metric | `final.json`, checkpoints |

Expert accuracy gate는 고정 checkpoint의 공식 test 10,000장 Top-1과 catalog 값의 차이가
`-0.25pp` 이상 `+0.75pp` 이하여야 통과한다. Catalog가 best-validation log 값이므로 소폭 개선은
허용하지만 regression이나 설명되지 않는 큰 개선은 새 cache를 commit하기 전에 거부한다.

## 재실행과 결과 해석

- Cache fingerprint와 tensor 구조가 모두 같으면 재사용하고, 다르면 기존 cache를 덮어쓰지 않는다.
- 설정을 바꾼 실행은 새 cache/run 경로를 사용한다.
- NSGA-II는 generation resume을 지원하지 않는다. `search.json` 생성 전에는 final을 실행하지 않는다.
- Final만 실패하면 같은 완성된 `search.json`으로 final을 재실행한다.
- Final 결과는 retained search front의 일반화 상태를 기술할 뿐, 새 모델 선택에 사용하지 않는다.
- 정적 catalog와 dynamic result는 cost 정의와 평가 표본이 다를 수 있으므로 동일 split expert 비교를
  정량 결론의 우선 근거로 사용한다.
