# Preserved dispatcher cluster adoption

이 root는 현재 실행 중인 `pertinence-dispatcher-260919-1708`과 기존 두 node pool을 새로 만들지
않고 Terraform state에 가져온다. `imports.tf`는 현재 project와 resource ID에만 사용하며 다른
환경을 만들 때 복사하지 않는다.

현행 remote state에는 보존 상태를 확인하는 `profiles/training-current.tfvars`와 두 benchmark pool만
추가하는 `profiles/benchmark-current.tfvars`를 사용할 수 있다. Benchmark profile은 아래의 전용
saved-plan guard를 통과할 때만 적용한다. 이 디렉터리의 `.example` profile은 다른 topology의 설계
예제이며 이 backend에서 plan 또는 apply하지 않는다.

## 1. Import-only 단계

먼저 [`../../README.md`](../../README.md)의 bootstrap을 완료한다. 이후 repository root에서 다음을
실행한다.

```bash
cd cloud/iac/environments/dispatcher-current
terraform init -backend-config=backend.hcl
terraform plan \
  -var-file=profiles/import-baseline.tfvars \
  -out=import.tfplan
terraform show -json import.tfplan > import.tfplan.json
python ../../../../scripts/terraform_plan_guard.py \
  --mode no-op import.tfplan.json
terraform apply import.tfplan
```

Guard가 거부하면 apply하지 않는다. Import plan에는 resource별 `importing` metadata가 있을 수 있지만
change action은 `no-op` 또는 data-source `read`만 허용한다. Apply는 검토한 cloud resource를 state에
연결할 뿐 실제 cluster 설정을 변경하지 않아야 한다.

Import 직후 같은 baseline으로 다시 plan하여 exit code 0을 확인한다.

```bash
terraform plan \
  -var-file=profiles/import-baseline.tfvars \
  -detailed-exitcode
```

## 2. 삭제 보호 단계

Zero-drift가 확인된 뒤에만 보호 프로필을 별도 saved plan으로 만든다.

```bash
terraform plan \
  -var-file=profiles/training-current.tfvars \
  -out=protection.tfplan
terraform show -json protection.tfplan > protection.tfplan.json
python ../../../../scripts/terraform_plan_guard.py \
  --mode protection protection.tfplan.json
terraform apply protection.tfplan
```

현재 cluster에는 이미 `deletion_protection`이 켜져 있다. 이 단계는 그 값을 유지하면서 node pool의
Terraform `deletion_policy`를 `PREVENT`로 바꾸는 변경만 포함해야 한다. 그 외 변경이 섞이면 guard가
plan을 거부한다.

현재 일회성 maintenance exclusion은 `2026-09-24T01:10:46Z`까지 실험 런타임 상태로 그대로 둔다.
Google provider는 exclusion을 선언할 때 실제 환경에 없는 정기 maintenance window도 요구하므로,
zero-drift adoption을 위해 base GKE state가 이 일회성 설정을 소유하지 않는다. 만료 여부는 운영
inventory에서 확인하고, 장기 반복 window가 필요하면 별도 비파괴 변경으로 설계한다.

## 3. Benchmark pool 생성

`benchmark-current.tfvars`는 기존 cluster, `system-pool`, `pipeline-pool`과 KFP를 그대로 유지하면서
다음 두 resource만 추가한다.

- `module.gke.google_container_node_pool.this["serving-pool"]`: `n2-standard-8`, 0–1,
  `workload=pertinence-serving:NoSchedule`;
- `module.gke.google_container_node_pool.this["loadgen-pool"]`: `n2-standard-4`, 0–1,
  `workload=pertinence-loadgen:NoSchedule`.

선언된 네 pool의 최대치는 24 vCPU다. 측정 시에는 `pipeline-pool=0`, `system-pool=1`,
`serving-pool=1`, `loadgen-pool=1`인 16-vCPU topology만 허용하며 active KFP workflow가 없어야 한다.
먼저 project quota와 remote-state drift를 읽기 전용으로 확인한 뒤 saved plan을 만든다.

```bash
terraform plan \
  -var-file=profiles/benchmark-current.tfvars \
  -out=benchmark-pools.tfplan
terraform show -json benchmark-pools.tfplan > benchmark-pools.tfplan.json
python ../../../../scripts/terraform_plan_guard.py \
  --mode benchmark-pools-create benchmark-pools.tfplan.json
terraform apply benchmark-pools.tfplan
```

Guard는 위 두 indexed node-pool 주소의 순수 create가 모두 있을 때만 허용한다. Cluster나 기존 pool의
update/delete/replace, 예상 밖 create, 두 pool 중 하나만 있는 partial plan은 거부한다. Guard를 통과한
JSON과 실제 apply 입력은 같은 saved plan에서 나온 것이어야 하며 guard 뒤에 plan을 다시 만들지 않는다.
모든 pool의 `deletion_policy`는 `PREVENT`를 유지한다.

## 4. Profile 사용 규칙

- `training-current.tfvars`: 현재 system+pipeline 환경의 최종 보호 상태
- `benchmark-current.tfvars`: 기존 두 pool을 유지하며 serving+loadgen을 각각 최대 1개 추가;
  `benchmark-pools-create` guard 필수
- `benchmark.tfvars.example`: pipeline을 끄는 별도 system+serving+loadgen topology 예제
- `full.tfvars.example`: 네 pool 전체, 실제 48-vCPU quota 확인 전 apply 금지
- `compute-only.tfvars.example`: system/loadgen 없이 pipeline만 사용하는 addon 비지원 예제

Example profile은 [cloud roadmap](../../../docs/roadmap.ko.md)의 미구현 구성을 설명할 뿐이다.
현재 state에 `.example` profile을 넣으면 기존 pool 삭제나 sizing 변경이 plan될 수 있으므로 사용하지
않는다.

KFP/KServe 설치 전에는 `terraform output -json`의 `kfp_addons_supported`가 `true`인지 검사한다.
Node selector와 toleration은 `node_scheduling` output을 사용하며, Kubernetes resource와 PVC는 이
Terraform state에 넣지 않는다.
