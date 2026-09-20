# Preserved dispatcher cluster adoption

이 root는 현재 실행 중인 `pertinence-dispatcher-260919-1708`과 기존 두 node pool을 새로 만들지
않고 Terraform state에 가져온다. `imports.tf`는 현재 project와 resource ID에만 사용하며 다른
환경을 만들 때 복사하지 않는다.

현행 remote state에 사용할 수 있는 최종 profile은 `profiles/training-current.tfvars`뿐이다.
이 디렉터리의 `.example` profile은 미구현 serving/benchmark 설계 예제이며 이 backend에서 plan
또는 apply하지 않는다.

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

## 3. Profile 사용 규칙

- `training-current.tfvars`: 현재 system+pipeline 환경의 최종 보호 상태
- `benchmark.tfvars.example`: system+serving+loadgen, 실제 16-vCPU quota 확인 전 apply 금지
- `full.tfvars.example`: 네 pool 전체, 실제 48-vCPU quota 확인 전 apply 금지
- `compute-only.tfvars.example`: system/loadgen 없이 pipeline만 사용하는 addon 비지원 예제

Example profile은 [cloud roadmap](../../../docs/roadmap.ko.md)의 미구현 구성을 설명할 뿐이다.
구현을 시작하더라도 신규 cluster 이름과 별도 backend prefix를 가진 새 environment root에서만 사용한다.
현재 state에 다른 profile을 넣으면 기존 pool 삭제가 plan될 수 있으므로 사용하지 않는다.

KFP/KServe 설치 전에는 `terraform output -json`의 `kfp_addons_supported`가 `true`인지 검사한다.
Node selector와 toleration은 `node_scheduling` output을 사용하며, Kubernetes resource와 PVC는 이
Terraform state에 넣지 않는다.
