# Distributed serving 전송 비교 suite

기존 V1/V2 benchmark 계약은 유지한다. 이 suite는 `transport-suite-run` manifest의
구조·tensor transport·연결 모드·계측 모드·runtime digest·node UID를 fingerprint에 포함한다.
Router가 반환하는 `run_fingerprint`도 loadgen이 확인한다. 기존 promotion은 수정하지 않고
모델 provenance로 사용하며, 실험 runtime은 별도 manifest에 고정한다. Production promotion은 없다.

## 구현과 실행

- `serve router --expert-transport=json|binary`: 기본 JSON. Binary는 V2 JSON header와
  little-endian contiguous FP32 본문이며 `Inference-Header-Content-Length`로 경계를 표시한다.
  이미지 12,288 bytes, feature 4,096 bytes. 요청별 이미지 바이트를 RPC 간 공유한다.
- `serve expert --keep-alive --idle-timeout 10`: 선택 가능한 HTTP/1.1. 기본 HTTP/1.0.
  연결별 thread와 실행 semaphore를 분리해 health probe가 idle 연결에 막히지 않는다.
  Content-Length를 항상 전송하고 socket idle timeout 뒤 연결을 종료한다. TCP_NODELAY를 사용한다.
  끊긴 지속 연결의 현재 요청은 실패하며 자동 재시도하지 않는다. 다음 요청은 새 연결을 연다.
- `serve router --router-mode=hybrid-dynamic --expert-transport=binary`: 로컬 ShuffleNet과
  Dispatcher, 원격 MobileNet/ResNet. `--bundle-path`에는 `evaluated/`와
  `experts/shufflenetv2_x0_5/model.onnx`만 필요하다. 모델 SHA와 promotion lineage를 검증한다.
- `--diagnostics-path /diagnostics/spans.jsonl`: 지정한 경우에만 상세 계측. 요청 ID·expert·RPC 순번으로
  연결한다. JSON 응답에 상세 span을 추가하지 않는다. `request_total`은 보조 지표이며 child span과
  합산하지 않는다. 기록 자체의 파일 I/O는 개별 span 밖에 있으며 ON/OFF 비교로 비용을 확인한다.

설정: `experiments/cifar10/configs/benchmark-transport-suite.yaml`.
공통 runtime: Python 3.11, ORT 1.20.1, intra/inter-op 각 1 thread.
Predictor request/limit 합계는 각 arm 4 CPU/4 GiB다. Modelcar는 별도 계산한다.
Hybrid modelcar 준비는 전체 고정 입력을 검증한 뒤 Dispatcher와 ShuffleNet만 복사한다.

```bash
.venv/bin/python -m pertinence.serving.transport_cli schedule --output artifacts/cifar10/suite-order.json
.venv/bin/python experiments/cifar10/scripts/transport-parity.py --help
.venv/bin/python experiments/cifar10/scripts/render-transport-suite.py --help
.venv/bin/python experiments/cifar10/scripts/run-transport-suite.py --help
```

`run-transport-suite.py`는 기존 distributed service/job YAML을 입력으로 받아 새 arm별 manifest를
생성한다. 기존 serving workload를 stop한 뒤 다음 arm을 시작한다. 원격 Job의 결과는 로컬에서
회수·보존한 뒤 handoff를 확인하므로 새 GCS IAM 권한을 요구하지 않는다. 상세 diagnostic 파일도
Pod 종료 전에 회수한다. 입력 trace 다운로드에는 기존 benchmark Workload Identity를 사용한다.
실험 중 serving/loadgen pool은 각각 1대로 유지해야 한다. 종료 시 serving workload는 stop한다.
Node pool min/size 복원은 실행자가 수행한다.
Warm-up 전에 serving/loadgen 첫 cAdvisor cycle을 확인해 측정 시작 경계의 sample을 확보한다.
측정 종료 후 artifact 회수만 최대 3회 시도하며 실패 내용과 ZIP CRC를 확인한다. 추론은 재시도하지 않는다.

진단 ON 6 arm → OFF 6 block(각 block 6 arm) → D0 및 선택 후보 ON/OFF 순서다.
첫 순서는 seed 20261011로 결정하며 block마다 회전한다. Warm-up 200, 측정 2,000, 5 RPS,
concurrency 1, retry 0이다. Node UID 변화·실패·계측 누락은 fail-closed로 중단한다.
해당 block의 전체 6 arm을 새로운 run prefix로 다시 실행해야 하며 일부 arm을 섞으면 안 된다.

## 분석 계약

외부 latency는 전체 및 route별 mean/p50/p95/p99로 보고한다. cAdvisor는 측정 시작 전과 종료 후
sample을 확보하고 경계 counter를 선형 보간해 warm-up을 제외한다. 5초 sample 간격에서 7.5초를
넘는 gap, container series 누락, counter reset, Pod UID 변화, restart는 결과를 무효화한다.
CPU는 predictor와 modelcar 합계이며 modelcar 부분도 따로 저장한다.

Router RPC wall − expert의 decode/model/output/encode/write wall 합계는
**전송·스케줄링·대기 잔차**다. 서로 다른 Pod의 timestamp를 빼지 않는다. 파일 기록·미계측 framework
처리 및 계측 오버헤드도 잔차에 영향을 주므로 순수 네트워크 시간으로 해석하지 않는다.

성능 비교는 36개의 valid run만 수용한다. 요청을 독립 반복으로 취급하지 않고 6개의 대응 block을
bootstrap 10,000회 재표집한다. Mean 절감량 95% 구간의 하한이 양수이고 p95 및 전체 Pod CPU/request
평균 악화가 5% 이내이면 통과한다. D1~D3 중 최저 mean을 우선하며 D3의 추가 변경 효과가 불확실하면
통과한 단일 변경 후보를 우선한다. H1은 D3 및 F0와 별도로 비교한다.

이 suite의 결론 범위는 같은 node의 저부하 지연이다. 최대 처리량, autoscaling, cross-node에 대한
결론을 내리지 않는다. 기존 2026-10-09의 54% CPU/46% 통신 분해는 근거로 사용하지 않는다.

## 이번 실행의 범위 변경 (2026-10-11)

사용자 요청에 따라 이번 실행은 `--phase diagnostics`로 구조별 계측 ON 1회만 수행한다.
36회 OFF 반복과 별도 ON/OFF 비교는 생략한다. 이미 완료된 유효 진단은 재사용한다.
보고서는 `report-transport-suite.py --single-run`으로 생성하며 관측 차이를 제시하되,
반복 실험 기반 신뢰구간·개선 통과 판정·계측 OFF 최종 성능을 주장하지 않는다.
위 반복 실험 계약과 기능은 향후 명시적으로 실행할 경우를 위해 유지한다.
