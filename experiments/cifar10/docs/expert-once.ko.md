# Expert 직접 호출 단회 비교 — 10월 계측 기준

기준은 [10월 11일 Dispatcher 진단 실험](distributed-transport-results-20261011.ko.md)의
5 RPS·준비 200건·본 측정 2,000건·diagnostics ON 조건이다.
[이전 실패 결과](expert-once-results-20261017.ko.md)는 보존하며 참고값으로만 구분한다.
이번에는 Router를 정지하고 expert를 직접 호출한다. 새 node UID에서 실행한다.

ShuffleNetV2 x0.5 → MobileNetV2 x0.5 → ResNet56 순서로 각각 한 번만 실행한다.
기존 `cifar10-final-20260824` trace 전체 2,000개를 같은 순서로 사용하고 모델마다
앞 200개로 준비 요청을 보낸다. 준비 요청은 모든 통계에서 제외한다.
FP32 JSON, batch 1, concurrency 1, 5 RPS, 요청당 timeout 10초, 요청·Job 재시도 0회다.
반복 실험·자동 재실행·학습·모델 변환은 수행하지 않는다.
준비 포함 모델당 약 7분 20초, 총 약 22분이며 배포·회수·정리 시간은 별도다.

```bash
.venv/bin/python -m pytest tests/test_expert_once.py tests/test_kserve_single_arm_v2.py tests/test_documentation_layout.py
.venv/bin/ruff check .
.venv/bin/python experiments/cifar10/scripts/run-expert-once.py \
  --run-id expert-once-YYYYMMDD-HHMM \
  --base-job <기존에 검증된 loadgen Job YAML>
```

현재 보존된 dispatcher GKE context와 `pertinence-serving` namespace를 사용한다.
사전 조건은 serving/loadgen/pipeline pool에 node가 없고 serving Pod와 active workflow가
없으며 서비스가 정지된 상태다. serving node는 `n2-standard-8`, loadgen node는
`n2-standard-4`다. 세 expert를 같은 serving node에 replica 1로 배치한다.
predictor CPU request/limit 1, memory request/limit 1Gi, ORT intra/inter thread 1,
worker 1, logits-only, JSON·HTTP keep-alive·idle timeout 10초를 적용한다.
Python 3.11, ONNX Runtime 1.20.1을 실제 Pod에서 확인한다.

실제 Pod의 환경변수에는 KServe가 주입한 메타데이터가 포함될 수 있으므로 전체 목록의
동일성을 요구하지 않는다. 실험에 필요한 ORT intra/inter thread·worker 값, 자원·image·args와
diagnostics mount를 검증한다. 검증 전에 `pods-start.json`·`nodes-start.json`을 저장하고,
실패 메시지에는 기대값과 실제값을 남긴다. 추론 없이 모델 파일의 실제 SHA-256,
runtime의 request ID 지원, diagnostics 쓰기 권한을 검사한다.
loadgen Job은 node 준비 전에 렌더링해 검증하며 controller·worker 코드도 실행 경로에 보존한다.
로컬 실제 HTTP 테스트로 worker → predictor → server span → 연결·결과 문서 경로를 확인한다.

10월 11일 runtime digest
`sha256:ce9729d2d6170636ebd1bcad51aecb23765c7d775302aff0d2f9bb629c4008ca`와
기존 expert modelcar digest·모델 SHA-256을 재사용한다. 모델 hash는 runtime이 파일을
검증하고 응답의 `run_fingerprint`로 연결한다. 실제 배포 digest, model hash, trace fingerprint,
Pod/node UID와 실행 조건은 실행 manifest에 기록한다. 새 worker만 immutable ConfigMap으로
제공하며 runtime은 재빌드하지 않는다. diagnostics는 각 expert Pod의 `emptyDir`에 저장한다.

| 지표 | 측정 범위 |
| --- | --- |
| 클라이언트 응답시간 | 기존 benchmark와 같은 단조 시계로 predictor 호출부터 응답 파싱·결과 및 identity 검증·stage 정보 추출까지 |
| 모델 실행 지연시간 | 서버 `expert_model.wall_seconds`: `runner.predict()`의 ONNX 실행과 내부 입력·출력 검증. HTTP·JSON·네트워크 전송 제외 |
| 모델 thread CPU 시간 | 동일 span의 `thread_cpu_seconds`. 모델 wall 시간과 구분하는 보조 지표 |

모델 wall 시간은 순수 ONNX 연산자 시간이나 CPU 사용시간이 아니다.
Dispatcher 원격 호출을 포함한 `expert_ns`를 대체값으로 사용하지 않는다.
cAdvisor 기반 CPU·메모리 상세 분석은 이번 범위에서 제외한다.
이미지 hash 계산, 전송 간격 대기, JSONL 기록은 응답시간 측정 밖에 둔다.

준비·본 측정 모두 직전 요청 시작부터 200ms 간격을 적용한다. 이미 200ms가 지났으면
이전 요청 완료 후 다음 요청을 보낸다. 준비 마지막 요청과 본 측정 첫 요청 사이도 같다.
지연 시 동시 요청이나 밀린 요청의 몰아 보내기 없이 계속 진행한다.
전송 지연이나 전송률 저하는 중단 사유가 아니다. 공용 스케줄러는 변경하지 않는다.

모든 요청은 실행 ID·모델명·연속 순번으로 고유 request ID를 가진다. phase와 sample ID도
저장한다. 회수한 span을 `(request_id, model_name)`으로 직접 연결한다. 서버의 모델명은
`expert` 필드에 있으며 직접 호출의 `call_index`는 사용하지 않는다. 본 측정마다
`expert_model` span이 정확히 하나여야 하며 누락·중복·모델 불일치·비정상 시간은 무효다.
모델별 요청 수·오류 수·연결 span 수·실제 전송률을 표시한다. 실제 전송률은
`요청 수 / (마지막 완료 시각 − 첫 시작 시각)`이다. 응답시간과 모델 wall 시간은 각각
mean·p50·p95·p99·max(ms), 모델 thread CPU는 mean(ms)을 보고한다.
백분위수는 NumPy 기본 선형 보간으로 계산한다.

컨트롤러는 Pod UID·배치 node·readiness·container image ID·재시작 수와 serving/loadgen
node UID를 반복 확인하고 환경 이력을 보존한다. 변경, 요청 실패·timeout·잘못된 응답·
모델/hash 불일치, 감시 heartbeat 20초 만료 시 전체 실행을 중단한다. 부분 기록을 보존하고
이후 모델이나 실패 모델을 자동 실행하지 않는다. 동일 run ID 출력 경로가 있으면 거부한다.
컨트롤러 측정 제한은 1,800초, Job `activeDeadlineSeconds`는 2,400초다.

`artifacts/cifar10/runs/kserve/<run-id>/`에 다음을 보존한다.

- `raw/results/requests.jsonl`: 준비·본 측정·실패 원본. `requests-live.jsonl`: 감시 중 로컬 복사본.
- `<model>-spans.jsonl`: Pod별 서버 span. `joined-requests.jsonl`: 본 측정 연결 기록.
- `manifest.json`, `raw/trace.json`: 실행 조건·digest·hash·trace fingerprint·Pod/node UID.
- `environment-start.json`, `environment-latest.json`, `environment-history.jsonl`: 환경 감시 증거.
- `summary.csv`, `result.json`, `results.ko.md`: 검증 후 최종 통계·결과.
- `INVALID.json`: 무효 사유. `cleanup.json`, `verification.json`, pool/node 전후 기록: 정리 증거.

클라이언트와 서버 span을 서비스 정지 전에 회수한 뒤 서비스를 stop하고 Job을 삭제한다.
변경한 pool 크기와 autoscaling을 실행 전 0..1·size 0으로 복원하고 확인한다.
실험용 서비스 spec과 immutable ConfigMap은 감사용으로 남기며 서비스는 정지한다.
완료 조건은 세 모델 각각 본 측정 2,000건과 대응 span 2,000건, 오류·연결 누락·중복 0건,
측정 중 환경 변경 없음과 정리 완료다. 단회 저부하 관측으로 한정하며 신뢰구간이나
통계적 우월성을 주장하지 않는다.
