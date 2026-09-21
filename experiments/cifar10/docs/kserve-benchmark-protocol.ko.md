# CIFAR-10 KServe 저부하 benchmark protocol

이 문서는 `dispatcher-000733`과 세 expert의 online 효과를 재현하기 위한 사전 등록된 실행 계약이다.
승인 checkpoint SHA-256은
`d82d3243bfec4f0c5889aa7e241735c4f212c849a6bcc2f461f60ca6c1524720`이며 다른 digest는
promotion contract가 거부한다.
실행값의 정본은 [`../configs/benchmark.yaml`](../configs/benchmark.yaml)이다. 이 계약은 최대 처리량
시험이나 사전 pass/fail 효과 기준을 정의하지 않는다.

## Dataset provenance

기존 final trace는 seed `20260824`로 평가 pool을 나눈 당시의 final 2,000개 stable sample ID를
그 순서대로 사용한다. 전체 ID 배열의 canonical JSON SHA-256은
`323c5f237cf1a7dd81d5e726ae684c067dd28213c6a1460b023801f3c05f07f5`다. Trace 생성 시 promoted
dataset bundle의 URI, GCS generation, object SHA-256, label hash를 함께 봉인한다. 서버 요청에는
normalized FP32 image만 넣고 label은 보내지 않는다.

CIFAR-10.1 v6 trace는 공식 `modestyachts/CIFAR-10.1` upstream revision
`d9982abb0bfc4846b8d13a11e66b887d946205d0`의 두 NPY 객체를 순서 변경 없이 사용한다.
2026-09-20에 commit-pinned URL과 default-branch 객체의 byte hash 및 배열을 독립 확인했다.

| Object | SHA-256 | Shape/dtype |
|---|---|---|
| `cifar10.1_v6_data.npy` | `2997188e5816f5bd545dc77771b6227828c28146049fcecf3fa10775474cacc6` | `[2000,32,32,3]`, `uint8` |
| `cifar10.1_v6_labels.npy` | `ae40beda001693674edc94d925ee8268cfe68905f8f9aff800c8dcdfcd6c9448` | `[2000]`, `int32` |

각 class는 정확히 200개다. 이 데이터는 CIFAR-10과 마찬가지로 Tiny Images에서 유래한 별도
test set이다. Repository에 image byte를 재배포하지 않으며, 사용자는 CIFAR-10.1 upstream의
provenance와 Tiny Images/CIFAR 관련 이용 조건을 직접 검토해야 한다. Artifact에는 upstream URL,
두 object hash, sample-order hash와 label-value hash를 남긴다.

## Arms와 일정

모든 arm은 같은 `[1,3,32,32]` FP32 요청과 batch 1을 받는다. 비교 대상은 세 expert direct endpoint,
static ResNet router, dynamic `dispatcher-000733` router다. Static/dynamic router 비교는 공통
router→expert hop을 통제하고 direct ResNet은 전체 serving overhead를 보여준다.

모든 endpoint는 같은 digest-pinned Python 3.11 runtime, `onnxruntime==1.20.1`,
`numpy==1.26.4`, 단일 worker와 ONNX intra/inter-op thread 1을 사용한다. Router의 expert connection
pool과 timeout은 arm 사이에서 고정하고 retry는 0이다.

각 arm을 200회 warm-up한 다음 200회 serial 요청으로 보정한다. 공통 요청률은 각 arm p95로 구한
용량 중 가장 느린 값의 10%와 2 RPS 중 작은 값이다. Dataset별로 2,000요청을 5회 반복하고 다섯
arm은 5×5 cyclic Latin square로 순서를 바꾼다. 동시성·최대 in-flight는 1이고, 예정 시각에 직전
요청이 끝나지 않았으면 queue나 retry를 만들지 않고 해당 run을 무효화한다. Cold-start는 manifest
적용부터 Ready까지와 최초 20요청을 steady-state에서 분리해 기록한다.

## 결과와 무효화

Raw request Parquet에는 correctness, route, endpoint identity와 end-to-end/router/extractor/expert
latency를 저장한다. Pod/node 시계열에서는 router와 선택 expert의 CPU time/request, memory와 CPU
throttling을 계산한다. 환경 manifest에는 GKE/KServe/cert-manager version, node/Pod inventory,
image/model digest와 promotion fingerprint를 포함한다. Offline MFLOPs는 online CPU/latency와 같은
양으로 합치지 않고 별도 표로 둔다.

Accuracy는 고유 2,000개 sample에 대한 McNemar 통계와 paired interval을, 성능은 sample과 반복
구조를 보존한 seeded paired bootstrap 95% CI를 보고한다. 사전 합격선을 두지 않으며 interval이
명확하지 않으면 결론을 `불확실`로 기록한다.

Active KFP workflow, Pod restart, node topology 변경, 응답 identity 또는 offline parity 불일치,
request error, missed schedule 중 하나라도 있으면 효과를 해석하기 전에 해당 run 전체를 무효화한다.
