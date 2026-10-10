# Fused PERTINENCE 3구조 비교 실행 기록 (2026-10-09 UTC)

## 결론

동일한 GKE serving/loadgen node, CIFAR-10 trace, `dispatcher-000733` 모델 byte와 ORT 설정에서
`direct → fused → distributed`를 순차 실행했다. 세 최종 arm은 모두 200회 warm-up 뒤 2,000개
요청을 5 RPS, concurrency 1, retry 0으로 한 번 처리했고 valid 판정을 받았다.

비교 결과는 `distributed_overhead_removed`다. Fused는 distributed보다 mean latency, p95 latency,
request CPU와 cAdvisor CPU/request가 모두 낮았다. Fused는 direct보다 이 지표들이 높았으므로
`serving_efficiency_improvement` 판정은 받지 않았다. 자동 promotion은 수행하지 않았다.

이 결과는 단일 pass의 기술 통계다. 통계적 우월성, cold start 또는 최대 throughput을 주장하지
않는다.

## 실행 식별자와 보존 위치

| 항목 | 값 |
|---|---|
| Session ID | `fused-20261009t152737z` |
| 최종 promotion | `fused-promotion-20261009t152737z-r3` |
| Promotion fingerprint | `37ca11705ff4120de9efa85f643e3e608b851e419d8224be52b6f3c52d953b2e` |
| Cloud Build | `43bfb945-59e7-4d75-ba0f-1ae40dcf2db2` (`SUCCESS`) |
| Runtime image | `serving-runtime@sha256:a87e27dabb8b71b4a2c3a760625acd9b693bf527bdd51adadabed7bc2d97c21c` |
| Direct run | `fused-20261009t152737z-direct-r4` |
| Fused run | `fused-20261009t152737z-fused-r3` |
| Distributed run | `fused-20261009t152737z-distributed-r3` |
| Comparison prefix | `gs://pertinence-509116-pertinence-runs/pertinence/benchmarks/fused-v2/comparisons/fused-20261009t152737z/` |
| Local ignored evidence | `artifacts/fused-session-20261009T152737Z/` |

최종 comparison summary SHA-256은
`ad82509c5cd9d373fd51ae55699a5140dcf317f8ceba25f17e98f076560d9ecd`다. Comparison prefix의
12개 본문 파일은 `comparison-files.sha256`을 함께 게시하고 재다운로드해 전부 byte 일치를
검증했다. 각 run prefix에는 raw JSONL/Parquet, markers, summary와 별도로 `postrun/` 아래에
environment manifest, node/Pod snapshot, modelcar 포함 resource evidence와 validated summary를
create-only로 게시했다.

## 환경 동일성과 유효성

- Trace fingerprint: `3185ad8c7d3663eda8e7e5710929d83afdf255f35a6837caba5f644767ce6b7e`
- Serving node UID/type: `4f3e9d7a-f931-4125-b91d-62cd3b118307` / `n2-standard-8`
- Loadgen node UID/type: `1964e09c-77e9-45a4-9f8e-216bce55fbc5` / `n2-standard-4`
- ORT: Python 3.11, ONNX Runtime 1.20.1, workers 1, intra/inter-op threads 1
- 세 arm 모두 request 오류·누락·중복·schedule miss·Pod restart·metric gap·topology change 0
- Direct: correct `1892/2000`, route 없음
- Fused/Distributed: 각각 correct `1893/2000`, route `[0, 640, 1360]`
- Fused↔distributed parity: prediction mismatch 0, route mismatch 0, max absolute logit error `0.0`
  (`atol=1e-6`)
- Active KFP workflow 0, public LoadBalancer/Ingress/Gateway/external forwarding rule 없음

## 성능 결과

단위는 latency와 CPU/request가 ms, memory가 bytes다.

| 구조 | mean | p50 | p95 | p99 | achieved RPS | request CPU mean | cAdvisor CPU/request | memory avg | memory peak |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Direct | 10.379 | 10.042 | 12.646 | 14.648 | 5.00238 | 2.935 | 4.869 | 43,876,086 | 44,306,432 |
| Fused | 11.466 | 11.508 | 13.519 | 16.154 | 5.00237 | 3.779 | 6.576 | 57,716,310 | 58,183,680 |
| Distributed | 25.845 | 25.341 | 30.379 | 38.966 | 5.00220 | 11.610 | 20.384 | 166,562,422 | 167,223,296 |

주요 차이는 다음과 같다. 비율은 오른쪽 구조를 왼쪽 구조로 나눈 값이다.

| 비교 | mean 차이 / 비율 | p95 차이 / 비율 | request CPU 차이 / 비율 | cAdvisor CPU/request 차이 / 비율 |
|---|---:|---:|---:|---:|
| Direct → Fused | +1.087 / 1.105× | +0.873 / 1.069× | +0.844 / 1.288× | +1.707 / 1.351× |
| Fused → Distributed | +14.379 / 2.254× | +16.860 / 2.247× | +7.831 / 3.072× | +13.809 / 3.100× |
| Direct → Distributed | +15.466 / 2.490× | +17.733 / 2.402× | +8.675 / 3.956× | +15.516 / 4.187× |

따라서 같은 routing 결과를 내면서 process 내부 fused 실행이 HTTP·직렬화·다중 Pod 구조보다
상당히 낮은 latency와 CPU 비용을 보였다. 반면 extractor·routing을 포함한 fused 경로는 direct
ResNet56보다 낮지 않았다.

## 실제 Pod 자원

| 구조 | Predictor 합계 | Modelcar 포함 실제 request/limit |
|---|---|---|
| Direct | 1 CPU / 1 GiB | 1.010 CPU / 1,089,470,464 bytes |
| Fused | 4 CPU / 4 GiB | 4.010 CPU / 4,310,695,936 bytes |
| Distributed | 4 CPU / 4 GiB | 4.040 CPU / 4,357,881,856 bytes |

각 Pod의 regular container image, request/limit, Pod UID, node UID와 machine type은 run별
`postrun/pod-resources.json`에 보존했다.

## 실행 중 발견하고 수정한 사항

기존 결과를 삭제하거나 덮어쓰지 않고 실패 Job과 prefix를 모두 보존했다.

1. 첫 direct Job은 GSA가 trace object를 읽지 못해 요청 전 실패했다. Bucket list와 고정 trace
   prefix에 한정한 viewer condition을 추가했다.
2. `direct-r2`는 측정 완료 뒤 기존 2-arm GCS root validator가 fused-v2 root를 거부했다.
   `single-arm-run` create-only kind와 전용 root를 추가해 기존 `run` 계약을 유지했다.
3. `direct-r3`는 2,000개 valid record를 만들었지만, GSA가 업로드 객체를 byte-read 검증할 권한이
   없었고 native collector sidecar가 완료 직후 재시작했다. Viewer condition을 이 session run
   prefix까지 좁게 확장하고, collector가 completion receipt 뒤 Kubernetes 종료까지 idle 상태를
   유지하도록 수정했다.

최종 runtime/modelcar는 수정 뒤 새 tag로 다시 build하고 새 digest와 promotion으로 세 arm 전체를
실행했다. 최종 로컬 검증은 pytest `346 passed`, Ruff 통과, `git diff --check` 통과다.

## 종료 상태

결과 게시 뒤 다섯 InferenceService를 모두 stop 처리했고 active Job은 0이다. Job과 결과는 삭제하지
않았다. Serving/loadgen/pipeline pool은 autoscaling min 0, max 1 및 실제 size 0으로 축소했다.
최종 Kubernetes node는 system pool 한 대뿐이며 `pertinence-serving` namespace의 active Pod는 0이다.
