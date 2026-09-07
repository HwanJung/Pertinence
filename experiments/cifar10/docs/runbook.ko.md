# PERTINENCE CIFAR-10 실험 실행서

## 실행 원칙

이 문서는 준비가 끝난 GPU 작업에서 사람의 추가 판단 없이 순서대로 실행하기
위한 운영 절차다. 명령 하나라도 실패하면 다음 단계로 넘어가지 않는다. 실제
실험에서는 `experiments/cifar10/configs/experiment.yaml`의 `device: cuda`를 유지하며 CPU override를
사용하지 않는다.

final 2,000장은 expert checkpoint 무결성 검증을 제외하면 마지막 단계까지 열지
않는다. Cache build가 공식 test 10,000장의 고정 expert Top-1을 확인하는 이유는
checkpoint, source, transform 조합이 upstream catalog와 같은지 검증하기 위해서다.
이 값은 dispatcher 해 선택이나 NSGA-II fitness에 사용하지 않는다.

## 연속 실행 명령

저장소 루트에서 아래 블록을 한 GPU allocation 안에서 실행한다. `set -euo
pipefail` 때문에 실패를 숨긴 채 다음 단계로 진행하지 않는다. 장시간 search가
터미널 종료와 함께 죽지 않도록 batch scheduler나 지속 세션 안에서 실행한다.

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

"$PYTHON_BIN" scripts/prepare_assets.py \
  --manifest "$MANIFEST" \
  --artifacts-dir "$ARTIFACTS" \
  --verify-only

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
  --config "$CONFIG" \
  --cache-dir "$CACHE_DIR" \
  --output "$RUN_DIR/baselines.json"

"$PYTHON_BIN" -m pertinence.cli fixed \
  --config "$CONFIG" \
  --cache-dir "$CACHE_DIR" \
  --output "$RUN_DIR/fixed.json"

"$PYTHON_BIN" -u -m pertinence.cli search \
  --config "$CONFIG" \
  --cache-dir "$CACHE_DIR" \
  --output "$RUN_DIR/search.json" \
  | tee "$RUN_DIR/search-progress.log"

"$PYTHON_BIN" -m pertinence.cli final \
  --config "$CONFIG" \
  --cache-dir "$CACHE_DIR" \
  --search-result "$RUN_DIR/search.json" \
  --output "$RUN_DIR/final.json"
```

압축 해제 디렉터리가 없는 새 작업 공간이라면 자산 해시 검증 뒤, preflight 전에
README의 두 `tar -xzf` 명령을 한 번 실행한다. 압축이 일부만 풀린 디렉터리를
재사용하지 않는다.

## 단계별 합격 조건

| 단계 | 자동 합격 조건 | 생성물 |
|---|---|---|
| tests/lint | 두 명령 모두 exit 0 | 없음 |
| asset verify | 12개 asset 모두 `verified` | 없음 |
| preflight | `status: passed`, CUDA device, 네 모델 output `[2, 10]` | `preflight.json` |
| cache build | `expert_accuracy_gate: passed`, 세 split이 `written` 또는 `reused` | cache 3개, `cache-build.json` |
| baseline | cache fingerprint 검증 후 exit 0 | `baselines.json` |
| fixed | loss/parameter가 전 구간 finite이고 exit 0 | `fixed.json`, `fixed.pt` |
| search | 50세대 완료, 모든 학습 loss/parameter finite, exit 0 | `search.json` |
| final | search/cache fingerprint 일치, 모든 Pareto state 복원 성공 | `final.json`, dispatcher checkpoint들 |

`preflight.json`의 CPU 결과는 자산 smoke 증거일 뿐 production 합격으로 간주하지
않는다. Production preflight는 config 그대로 실행해 `device: cuda`여야 한다.
Expert accuracy gate는 `measured - catalog`가 각 모델에서 -0.25pp 이상,
+0.75pp 이하여야 통과한다. Catalog가 학습 로그의 best-validation 값이라는 점과
upstream maintainer가 runtime별 재평가 차이를 확인한 점을 반영한 비대칭 범위다.
근거는 upstream
[`default.log`](https://cdn.jsdelivr.net/gh/chenyaofo/pytorch-cifar-models@logs/logs/cifar10/shufflenetv2_x0_5/default.log)와
[`pytorch-cifar-models` issue #13](https://github.com/chenyaofo/pytorch-cifar-models/issues/13)에
남긴다.
Baseline의 `route_distribution`에 0건인 class가 있어도 실행 오류로 간주하지
않는다. Class weighting은 관측 class에서만 계산·정규화하고 미관측 class는 0으로
처리한다. 단, 이 class에 직접 supervised signal이 없다는 사실은 최종 결과의
제한으로 반드시 기록한다.

## 재실행과 중단 대응

- Asset 준비는 기존 파일의 SHA-256을 다시 확인하고 재사용한다.
- Cache 세 개가 모두 같은 fingerprint이면 inference 없이 `reused`된다. 하나라도
  fingerprint나 tensor 구조가 다르면 덮어쓰지 않고 즉시 실패한다. 설정을 바꾼
  실험은 새 cache/run 디렉터리를 사용한다.
- JSON 결과는 임시 파일을 거쳐 원자적으로 교체된다. 기존 결과를 보존해야 하면
  새 `RUN_DIR`을 사용한다.
- NSGA-II는 세대 진행률을 stdout에 출력하지만 현재 세대 중간 checkpoint resume은
  지원하지 않는다. Search 시작 전에 scheduler walltime을 충분히 확보한다. Fixed
  1회 wall time을 측정한 뒤 대략 `2500 × fixed 학습 시간`을 search 상한 추정치로
  사용하되, I/O와 평가 시간을 추가한다.
- `search.json`이 완전히 생성되기 전에는 final을 실행하지 않는다. Final 실패 시
  search를 다시 돌리지 않고 같은 `search.json`으로 final만 재실행한다.

## 결과를 열어 보는 시점

Search가 끝날 때까지 `final_evaluation.pt`, `final.json` 또는 final dispatcher
결과를 수동 분석하지 않는다. Cache 파일의 존재 자체는 허용하지만 `baseline`,
`fixed`, `search` 명령은 코드상 final cache를 로드하지 않는다. 최종 보고서는
`final.json`의 dynamic solution과 `experiments/cifar10/data/model_catalog.csv`의 정적
10-model Pareto 경계를 함께 사용한다.
