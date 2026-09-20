# Cloud serving·benchmark 로드맵

> 상태: **미구현·미배포**
> 이 문서는 현행 GKE/KFP capability나 완료된 실험 결과를 설명하지 않는다.

현재 검증된 범위는 GKE Standard와 OSS KFP에서 dispatcher를 학습·평가하고 evaluated bundle을
GCS artifact로 보존하는 것까지다. 실제 상태는 [architecture](architecture.ko.md), 반복 실행은
[runbook](runbook.ko.md)을 따른다.

## 목표

검증된 expert/dispatcher bundle 중 운영자가 accuracy–cost 요구에 맞는 지점을 명시적으로 승인해
활성화하고, 고정된 GKE topology에서 latency, throughput, resource와 전환 동작을 측정한다.

```text
verified KFP evaluated bundle
  └─ explicit promotion manifest
       └─ PERTINENCE router on serving-pool
            ├─ ShuffleNet expert
            ├─ MobileNet expert
            └─ ResNet56 expert

loadgen-pool ── fixed trace ──► router ──► benchmark artifact
```

Cloud가 load나 model availability를 근거로 dispatcher를 자동 선택하지 않는다. Offline final
accuracy/MFLOPs와 online latency/throughput은 서로 다른 실험 결과로 보존한다.

## 필요한 구현

1. **Promotion contract**
   - evaluated bundle hash, selected dispatcher ID와 승인자/시점을 가진 immutable manifest;
   - search/final lineage와 expert model digest 재검증;
   - 같은 promotion ID의 idempotency와 충돌 거부.
2. **Serving runtime**
   - feature extractor와 dispatcher를 실행하는 router image;
   - expert별 KServe `InferenceService` 또는 동등한 internal runtime;
   - selected expert가 extractor와 같을 때 logits 재사용;
   - request/route/model identity telemetry.
3. **격리된 GKE workload**
   - `serving-pool`과 `loadgen-pool` 생성 및 selector+toleration;
   - `serving-runner`, `benchmark-runner` Workload Identity와 최소 bucket 권한;
   - cluster-internal endpoint, NetworkPolicy와 public exposure 부재 검사.
4. **Benchmark**
   - immutable request trace와 warm-up/steady-state/cold-start 구간 분리;
   - p50/p95/p99 latency, throughput, error, route distribution, CPU/memory 기록;
   - 고정 node/replica/topology와 반복 run 통계;
   - raw measurement, environment manifest와 summary를 함께 보존.
5. **전환과 rollback**
   - 운영자가 승인한 bundle만 활성화;
   - in-flight request 정책과 readiness gate;
   - 이전 bundle로 deterministic rollback;
   - 전환 전후 correctness와 route telemetry 검증.

## Terraform 예제의 의미

`cloud/iac/environments/dispatcher-current/profiles/`의 다음 파일은 설계 검토용 미적용 예제다.

- `benchmark.tfvars.example`: system+serving+loadgen;
- `full.tfvars.example`: system+pipeline+serving+loadgen;
- `compute-only.tfvars.example`: addon을 지원하지 않는 pipeline-only 예제.

현재 `dispatcher-current` remote state에는 적용하지 않는다. 실제 구현 시 새 environment root와
고유 backend prefix를 만들고 quota, KServe/Kubernetes compatibility와 disposable smoke를 먼저
검증한다. 현행 적용 profile은 `training-current.tfvars`뿐이다.

## 단계별 완료 조건

### 1. Promotion

- 같은 evaluated bundle과 selected ID가 로컬/cluster에서 동일 hash로 검증된다.
- Final metric으로 자동 재선정하지 않으며 승인 기록이 manifest에 남는다.
- 잘못된 lineage, expert order 또는 runtime version은 serving 배포 전에 실패한다.

### 2. Runtime smoke

- Synthetic request와 고정 CIFAR sample에서 route와 prediction이 offline result와 일치한다.
- Router/expert Pod는 지정 pool에만 배치되고 public Service/Ingress가 없다.
- 같은 bundle의 재배포와 이전 bundle rollback이 idempotent하다.

### 3. Benchmark

- Cold-start와 steady-state 수치를 분리하고 topology 변경 없이 반복 측정한다.
- 단일 expert, static choice와 PERTINENCE routing을 같은 trace에서 비교한다.
- Offline accuracy/MFLOPs와 online latency를 별도 표로 보고한다.

이 조건을 모두 충족하기 전에는 repository README나 architecture에서 KServe serving 또는 GKE
online 성능을 구현 완료로 표시하지 않는다.
