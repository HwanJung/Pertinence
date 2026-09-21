# GKE KServe dispatcher benchmark 실행서

## 상태와 범위

이 문서는 보존 중인 `pertinence-dispatcher-260919-1708` cluster에서 준비된 KServe benchmark를
실행하기 위한 절차다. 저장소의 manifest와 Terraform profile은 실행 준비물이며 실제 배포·측정
완료의 증거가 아니다. 성공한 run의 immutable ID와 결과가 새 run record에 기록되기 전에는
[현재 architecture](architecture.ko.md)의 미배포 상태가 우선한다.

사람 승인과 session gate가 끝난 뒤 실제 명령을 수행할 때는 이 설계 요약과 함께
[20단계 실제 실행 절차](kserve-benchmark-execution.ko.md)를 사용한다.

이번 측정은 low-load descriptive comparison이다. 최대 처리량 시험을 하지 않고 사전 효과 합격선을
두지 않는다. Cluster NetworkPolicy enforcement는 꺼져 있으므로 NetworkPolicy manifest를 적용하거나
격리가 완료됐다고 주장하지 않는다. 활성화에는 node 재생성이 필요하므로 별도 작업으로 남긴다.

## 1. 변경 전 gate

다음을 읽기 전용으로 확인한다.

- cluster/context가 project `pertinence-509116`, zone `asia-northeast3-a`, 대상 cluster와 일치;
- control plane/node가 KServe Standard mode와 cert-manager의 pinned version을 지원;
- KFP deployment/PVC가 정상이고 active KFP workflow가 없음;
- `pipeline-pool` autoscaling은 0–1이지만 현재 node 수는 0;
- application `LoadBalancer`, Ingress, Gateway, 외부 forwarding rule이 각각 0;
- project quota가 선언상 네 pool 최대 24 vCPU를 허용.

KFP run은 benchmark가 끝날 때까지 제출하지 않는다. 이 gate와 이후 inventory를 run-scoped
environment manifest에 저장한다.

## 2. 두 node pool만 생성

`cloud/iac/environments/dispatcher-current`에서 remote-state init/validate 후 다음 saved plan을 만든다.

```bash
terraform plan \
  -var-file=profiles/benchmark-current.tfvars \
  -out=benchmark-pools.tfplan
terraform show -json benchmark-pools.tfplan > benchmark-pools.tfplan.json
python ../../../../scripts/terraform_plan_guard.py \
  --mode benchmark-pools-create benchmark-pools.tfplan.json
./seal-plan.sh benchmark-pools.tfplan benchmark-pools.tfplan.json \
  profiles/benchmark-current.tfvars "$(git rev-parse HEAD)" benchmark-pools.receipt.json
./apply-saved-plan.sh benchmark-pools.tfplan profiles/benchmark-current.tfvars \
  benchmark-pools.receipt.json
```

허용되는 mutation은 다음 두 개의 create뿐이다.

```text
module.gke.google_container_node_pool.this["serving-pool"]
module.gke.google_container_node_pool.this["loadgen-pool"]
```

Cluster, `system-pool`, `pipeline-pool`의 update/delete/replace 또는 다른 create가 있으면 중단한다.
Guard 이후 plan을 재생성하지 않고 검사한 동일 saved plan만 적용한다. 적용 후 모든 pool의
deletion policy가 `PREVENT`인지 확인한다.
부분 적용 복구는 live pool 사양을 별도 JSON으로 캡처하고 `benchmark-pool-recovery` mode가 정본과
완전히 일치시킨 경우에만 누락된 pool 하나의 create plan을 허용한다.

선언상 최대 topology는 system 4 + pipeline 8 + serving 8 + loadgen 4 = 24 vCPU다. Benchmark의
실제 topology는 pipeline 0, 나머지 pool 1개씩인 16 vCPU다.

## 3. Addon·workload 배포 gate

KServe `v0.20.0` Standard mode와 cert-manager `v1.20.4`를 저장소가 고정한 chart/image digest로만
설치한다. Standard mode가 요구하는 Kubernetes 1.32+ 조건을 실제 cluster version으로 다시 확인한다.
Controller는
`system-pool`, predictor/router는 selector+toleration 쌍으로 `serving-pool`, loadgen Job은 같은
방식으로 `loadgen-pool`에만 배치한다. 다섯 serving workload는 replica 1, CPU 1, memory 1 GiB의
동일 request/limit와 HPA 비활성을 유지한다.

KServe ingress 생성은 전역 비활성화하고 endpoint는 ClusterIP만 허용한다. 설치 직후 다음을
검사한다.

- 모든 image와 modelcar가 tag 없이 digest로 고정됨;
- container가 non-root이며 예상 selector+toleration/resource/replica를 가짐;
- `serving-runner` KSA에 Cloud IAM annotation/binding이 없고 node SA의 Artifact Registry reader만
  modelcar pull에 사용됨;
- `benchmark-runner`만 trace prefix read와 새 benchmark-ID prefix create-only write 권한을 가짐;
- JSON service-account key와 public Service/Ingress/Gateway/forwarding rule이 없음.

## 4. Promotion, smoke와 전환

자동 dispatcher 선택은 금지한다. `dispatcher-000733` promotion을 검증하면서 evaluated object
URI/generation/SHA, contract/search/final lineage, `D=1024`, `K=3`, expert 순서와 model/runtime
digest를 다시 확인한다. 같은 promotion ID는 같은 fingerprint일 때만 재사용한다.

세 expert의 고정 sample logits/class를 offline 값과 비교하고 dynamic router의 route/class 및
extractor logits reuse를 확인한다. Active ClusterIP alias는 다음 순서로만 바꾼다.

1. static ResNet router를 Ready 상태로 만들고 alias selector 전환;
2. `dispatcher-000733`을 Ready 상태로 만들고 전환;
3. rollback 시 Ready인 static router로 복귀.

각 전환 후 기존 connection을 drain하고 모든 응답 identity가 단일 target을 가리키는지 확인한다.
혼합 identity는 run 무효화 사유다.

## 5. 측정·보존

Dataset, arm, warm-up, calibration, Latin-square 실행 순서와 통계 계약은
[CIFAR-10 benchmark protocol](../../experiments/cifar10/docs/kserve-benchmark-protocol.ko.md)을
따른다. Cold-start와 steady-state를 분리한다. Raw request Parquet, Pod/node 시계열, environment
manifest와 summary를 새 GCS benchmark ID prefix에 create-only로 기록한다.

Active KFP workflow, Pod restart, node topology 변경, identity/parity 불일치, request error 또는
missed schedule이 있으면 결과를 해석하지 않고 run을 invalid로 봉인한다. 성공 여부와 무관하게
진단 artifact를 보존한다.

## 6. 종료와 기록

완료 후 다섯 `InferenceService`에 저장소가 정의한 stop annotation을 적용하고 loadgen Job이 종료됐는지
확인한다. `serving-pool`과 `loadgen-pool` node 수가 0이 될 때까지 관찰한다. KServe/cert-manager,
stopped manifest, OCI modelcar와 benchmark artifact는 삭제하지 않는다. `pipeline-pool`도 0을 유지한다.

성공 run 뒤에만 architecture와 새 run record에 실제 version, inventory, digest, promotion fingerprint,
결과와 confidence interval을 기록한다. NetworkPolicy enforcement 미적용과 descriptive-only 평가
때문에 성공하더라도 전체 roadmap은 완료로 표시하지 않는다.
