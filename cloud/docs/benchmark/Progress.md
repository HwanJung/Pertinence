# ResNet56 대 dispatcher benchmark 진행 상태

현재 상태는 `COMPLETED`이다. `final-20260922t170252z` 실행은 valid summary를 생성했고,
결과 9개 객체를 아래의 새 immutable GCS prefix에 create-only로 게시했다.

```text
gs://pertinence-509116-pertinence-runs/pertinence/benchmarks/runs/final-20260922t170252z
```

- evidence root: `artifacts/kserve-session-20260922T170252Z`
- run set: `kserve-20260922t170252z`
- promotion: `promotion-20260922t170252z`
- Cloud Build: `d6604c9e-dc41-45a4-9fbb-acdaf3ea614c` (`SUCCESS`)
- promotion fingerprint: `f0e69ca328f3506e3c53c87a5760e7de923a0220cd9b11b642042dd377f643e2`
- run fingerprint: `6ad16351fec14ec7eae18ad7768d04816b1c7bbc35fdb7a4c6872da507814aef`
- summary fingerprint: `38fc07b163151fe4ee99dfdafa30640896d0aa451daf8dcd714751bab0376eca`

## 측정 결과

| 항목 | ResNet56 direct | dispatcher-000733 | dispatcher/direct |
| --- | ---: | ---: | ---: |
| 정확도 | 94.60% (1892/2000) | 94.65% (1893/2000) | +0.05%p |
| 평균 지연시간 | 10.600 ms | 26.593 ms | 2.509x |
| p50 지연시간 | 9.967 ms | 25.012 ms | 2.509x |
| p95 지연시간 | 14.167 ms | 35.201 ms | 2.485x |
| p99 지연시간 | 24.306 ms | 46.634 ms | 1.919x |
| achieved RPS | 2.00098 | 2.00095 | 1.000x |
| CPU | 5.410 ms/request | 20.357 ms/request | 3.763x |
| 평균 working-set memory | 43,655,168 B | 166,671,829 B | 3.818x |
| peak working-set memory | 44,060,672 B | 167,399,424 B | 3.799x |

이 결과는 계획대로 고정 trace의 두 arm 단일-pass 기술 비교이며 최대 처리량, cold start 또는 통계적
우월성을 의미하지 않는다.

## 완료 조건 검증

- 로컬 검증: pytest `306 passed`, Ruff, manifest render와 `git diff --check` 통과
- measured record: arm별 2,000개, 총 4,000개; JSONL/Parquet 모두 일치
- 누락, 중복, invalid record, 요청 오류, schedule miss: 모두 0
- online 기준: correct `[1892, 1893]`, dispatcher route `[0, 640, 1360]`으로 offline 기준과 일치
- 자원 증적: metric gap 0, Pod restart delta 0, throttling 0
- 외부 재검증: `valid=true`, evidence 770 rows, GCS 객체 9개와 고정 generation 확인
- 내부 전용 노출 감사: LoadBalancer, Ingress, Gateway, external forwarding rule 없음

## 재사용 자산

- 검증된 final trace: `cifar10-final-20260824`, 2,000 samples
- 기존 build-input manifest의 선택 자산 11개와 고정 generation/hash
- 기존 GCS 객체, image, GSA, IAM binding은 삭제하지 않음
- 이전 준비 실행의 상세 기록은
  [run-history-20260922.ko.md](run-history-20260922.ko.md)에 보존

## 안전 종료

- loadgen Job은 `spec.suspend=true`이다.
- 네 InferenceService는 모두 `serving.kserve.io/stop=true`이다.
- serving, loadgen, pipeline node pool은 모두 0으로 축소했다.
- 활성 benchmark Pod와 KFP workflow는 0이고, system-pool node 하나만 남았다.
- KFP deployment 14개와 Bound PVC 2개, 기존 GCS artifact는 보존했다.

따라서 [Plan.md](Plan.md)의 완료 조건을 모두 충족했으며 추가 실행 행동은 없다.
