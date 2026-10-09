# Dispatcher와 direct ResNet56 benchmark 실행 계획

## 목적과 범위

`cifar10-final-20260824`의 고정 2,000 samples로 direct `resnet56`과
`dispatcher-000733`을 한 번씩 비교한다. 결과는 동일 환경의 단일-pass 기술 비교이며 최대 처리량,
cold start, 반복 실행 또는 통계적 우월성을 주장하지 않는다.

## 실행 계약

- 순서: `resnet56` 다음 `dispatcher-000733`
- arm별 warm-up: 서로 다른 첫 200 samples, 측정 record에서 제외
- arm별 측정: 원래 순서의 2,000 samples를 한 번
- batch 1, concurrency/max-in-flight 1, 고정 2 RPS, retry 0
- direct 응답은 `model_name=resnet56`, dispatcher 응답은 promotion ID와
  `dispatcher_id=dispatcher-000733`을 검증
- 요청 오류, identity mismatch, schedule miss, 누락·중복 또는 offline 기준 불일치는 run invalid
- invalid run도 이미 생성된 request/evidence 파일을 보존

## 분석 계약

- 정확도: arm별 correct/accuracy와 dispatcher-minus-ResNet56 percentage-point 차이
- offline 기준: ResNet56 `1892/2000`, dispatcher `1893/2000`, dispatcher route
  `[0, 640, 1360]`
- 지연시간: arm별 mean/p50/p95/p99/achieved RPS와 절대 차이·비율
- 자원: direct arm은 `expert-resnet56` Pod, dispatcher arm은 router와 세 expert Pod의 합계
- 자원 지표: CPU seconds, CPU ms/request, 평균·peak working-set memory, throttling
- dispatcher Pod별 breakdown과 Kubernetes request/limit 총량을 관측값과 분리
- memory peak는 같은 수집 시점의 관련 Pod memory 합계 중 최댓값
- load generator와 collector 자원은 serving 비교에서 제외

## 배포와 실행

1. final trace의 검증된 GCS generation과 최신 build-input manifest의 선택 자산 11개를 재사용한다.
2. 변경된 benchmark 코드를 포함한 runtime과 세 expert/dispatcher modelcar를 새 고유 tag로 build하고
   새 promotion을 만든다.
3. quota·내부 전용 노출 preflight 후 serving/loadgen pool과 KServe addon을 준비한다.
4. 세 expert와 dispatcher router InferenceService, suspended final loadgen Job 하나를 배포한다.
5. endpoint Ready와 active KFP workflow 0을 확인하고 Job을 시작한다.
6. sidecar가 5초 주기 cAdvisor evidence 수집을 마치면 같은 Job에서 summary를 생성한다.
7. summary를 포함한 결과 directory 전체를 새 GCS run prefix에 create-only로 한 번 게시한다.
8. Job과 네 InferenceService를 stop하고 serving/loadgen/pipeline pool을 0으로 돌린다. KFP deployment와
   기존 artifact는 보존한다.

## 완료 조건

- pytest, Ruff, manifest render 검증 통과
- arm별 2,000개, 총 4,000개 measured record가 누락·중복 없이 존재
- 요청 오류와 Pod restart 0, online accuracy/route가 offline 기준과 일치
- accuracy, latency, 전체 serving 자원 비교가 하나의 valid summary에 존재
- 결과가 새 immutable GCS prefix에 보존되고 benchmark 자원이 안전 정지됨

실행 상태와 다음 행동은 [Progress.md](Progress.md)에만 기록한다. 이전 준비 이력은
[run-history-20260922.ko.md](run-history-20260922.ko.md)에 보존한다.
