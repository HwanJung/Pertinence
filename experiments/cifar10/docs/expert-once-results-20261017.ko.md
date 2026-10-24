# Expert 단회 비교 실행 기록 — 2026-10-17

실행 ID는 `expert-once-20261017-1435`다. ShuffleNet의 준비 요청 11건이 성공한 뒤,
다음 준비 요청에서 schedule miss가 감지되어 전체 실행을 중단했다. 본 측정은 시작하지 않았다.
MobileNet과 ResNet56은 호출하지 않았으며, 요청·Job·모델을 재실행하지 않았다.
따라서 이번 실행으로 모델별 응답시간을 비교할 수 없다.

| 모델 | 준비 요청 | 본 측정 요청 | 요청 오류 | schedule miss | mean / p50 / p95 / max(ms) | 상태 |
|---|---:|---:|---:|---:|---|---|
| ShuffleNetV2 x0.5 | 11 | 0 | 0 | 1 | — | 준비 중 중단 |
| MobileNetV2 x0.5 | 0 | 0 | 0 | 0 | — | 미실행 |
| ResNet56 | 0 | 0 | 0 | 0 | — | 미실행 |

오류 원문은 `request_error:sample=11:RuntimeError:missed_schedule:warmup:sample=11`이다.
인덱스는 0부터 시작하므로 12번째 준비 요청을 보내기 전에 중단했다. 준비 요청은
본 측정 통계에서 제외했으며 지연시간 통계, 신뢰구간, 우월성 판단을 산출하지 않는다.
Schedule miss의 원인은 이번 기록만으로 확정하지 않는다.

기존 runtime digest는 `sha256:ce9729d2d6170636ebd1bcad51aecb23765c7d775302aff0d2f9bb629c4008ca`이며,
세 expert의 기존 modelcar digest를 재사용했다. CPU 1, memory 1Gi, ORT intra/inter thread 1,
replica 1, logits-only, HTTP keep-alive를 동일하게 적용했다. 세 expert는 같은
`n2-standard-8` serving node에 배치했고 loadgen은 별도 `n2-standard-4` node에서 실행했다.
Batch 1, concurrency 1, FP32 JSON, 5 RPS, timeout 10초, retry 0 조건이었다.
Router는 정지 상태였다.

고정 trace는 `cifar10-final-20260824`이며 fingerprint는
`3185ad8c7d3663eda8e7e5710929d83afdf255f35a6837caba5f644767ce6b7e`다.
요청별 모델명과 모델 hash identity 검사가 모두 통과했다. 감시 시작과 마지막 환경 기록이 같고,
관측된 Pod 재시작과 Pod/node 변경은 없다.

부분 기록과 실행 조건은 repository root 기준
`artifacts/cifar10/runs/kserve/expert-once-20261017-1435/`에 보존했다.
주요 파일은 `raw/results/requests.jsonl`, `raw/results/summary.csv`,
`raw/results/result.json`, `raw/trace.json`, `manifest.json`, `INVALID.json`,
`environment-start.json`, `environment-latest.json`, `archive.zip`이다.

단회 실행 모듈과 GKE 실행 스크립트를 추가했다. 오프라인 테스트 17개와 Ruff 검사가 통과했다.
전체 pytest에서는 390개 통과, 7개 실패가 발생했다. 이 중 6개는 sandbox의 localhost socket
제한 때문에 실패했고, 권한을 확장한 관련 HTTP 테스트 19개는 모두 통과했다.
나머지는 기존 `pipelines/pertinence/implementation-guide.ko.md`의 `...` 링크 실패다.

모든 expert/router를 stop하고 단회 Job을 삭제했다. Serving/loadgen pool 모두 size 0,
autoscaling 0..1로 원복되었으며 기존 system node만 남아 있다.
`cleanup.json`의 오류는 0건이고 `pools-after.json`, `nodes-after.json`,
`services-after.json`으로 종료 상태를 확인했다. 세 모델 각각 본 측정 300건이라는 완료 기준은 미충족이다.
