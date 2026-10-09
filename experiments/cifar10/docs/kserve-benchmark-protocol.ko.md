# CIFAR-10 KServe 비교 protocol

## 대상

검증된 `cifar10-final-20260824` trace의 2,000 samples만 사용한다. 비교 arm은 direct
`resnet56`과 `dispatcher-000733`이며 이 순서를 바꾸지 않는다.

각 arm은 첫 200개 서로 다른 samples로 warm-up한 다음, trace의 2,000 samples를 원래 순서대로
한 번 처리한다. warm-up 응답은 request record와 분석에서 제외한다. 공통 요청 조건은 batch 1,
concurrency/max-in-flight 1, 고정 2 RPS, retry 0이다.

## 유효성

direct 응답은 `model_name=resnet56`이어야 한다. dispatcher 응답은 실행 promotion ID와
`dispatcher_id=dispatcher-000733`이 일치해야 한다. 요청 오류, identity mismatch, schedule miss,
누락·중복, Pod restart가 있으면 run은 invalid다.

online correct count는 ResNet56 `1892`, dispatcher `1893`이어야 하며 dispatcher route count는
`[0, 640, 1360]`이어야 한다. 다르면 numeric 비교를 유효 결과로 게시하지 않는다.

## 보고값

- arm별 correct count, accuracy, dispatcher-minus-ResNet56 percentage-point 차이
- arm별 latency mean/p50/p95/p99/achieved RPS와 절대 차이·비율
- 전체 serving stack의 CPU seconds, CPU ms/request, 평균·peak working-set memory, throttling
- dispatcher router/세 expert Pod별 breakdown
- 관측 사용량과 분리한 Kubernetes CPU/memory request와 limit

memory peak는 관련 Pod 각각의 peak 합이 아니라 같은 시점의 working-set 합계 중 최댓값이다.
load generator와 collector는 serving 자원 비교에서 제외한다.

이 결과는 한 환경의 단일-pass 기술 비교이며 최대 처리량 또는 통계적 유의성을 평가하지 않는다.
