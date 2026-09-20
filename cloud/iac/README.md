# PERTINENCE GKE Infrastructure as Code

이 디렉터리는 PERTINENCE GKE 기반 환경을 재현하고 기존 dispatcher cluster를 안전하게
Terraform 관리 대상으로 가져오기 위한 정본이다. Terraform `1.16.3`과 Google provider
`7.42.0`을 고정하며, 인증에는 Application Default Credentials(ADC) 또는 service account
impersonation만 사용한다. JSON service-account key는 만들거나 저장소에 넣지 않는다.

## 구성

```text
cloud/iac/
├── bootstrap/                       # 원격 state GCS bucket
├── modules/gke/                     # GKE, node SA/IAM, 독립 node pool
└── environments/dispatcher-current/ # 현재 cluster import와 환경별 설정
```

Terraform이 소유하는 범위는 GKE cluster, node pool, node service account와 그 기반 IAM이다.
KFP/KServe 설치, Kubernetes PVC, raw/run/build bucket, Artifact Registry repository 자체,
`pipeline-runner`의 실험 prefix별 GCS IAM은 Terraform state에 넣지 않는다. 이 항목들은 GKE
output을 입력으로 받는 별도 addon runbook으로 관리하며 `null_resource`나 `local-exec`으로
Terraform 안에 숨기지 않는다.

현재 보존 중인 cluster는 재생성하거나 삭제하지 않는다. 최초 도입은 import 이후 create,
delete, replace가 모두 0인 plan을 확인한 다음에만 보호 설정을 적용한다. 현재 cluster에서 node
pool 활성화 토글을 끄는 것도 삭제이므로 일반 apply 절차로 수행하지 않는다.

2026-09-20 기준 bootstrap state bucket 생성과 state 이전을 완료했고, 현재 cluster 관련 12개
resource를 `gke/dispatcher-current` state에 `0 added, 0 changed, 0 destroyed`로 import했다. Cluster의
기존 deletion protection을 유지하며 두 node pool의 Terraform deletion policy도 `PREVENT`로
전환했다. 최종 `training-current` plan은 `No changes`다.

## 사전 준비

1. Terraform `1.16.3`, Google Cloud CLI와 접근 대상 project 권한을 준비한다.
2. 사용자 ADC를 사용할 때는 다음처럼 로그인하고 quota project를 지정한다.

   ```bash
   gcloud auth application-default login
   gcloud auth application-default set-quota-project pertinence-509116
   ```

3. 자동화에서는 사용자 로그인을 대신해 단기 자격 증명과 impersonation을 사용한다.

   ```bash
   export GOOGLE_IMPERSONATE_SERVICE_ACCOUNT="<terraform-sa>@pertinence-509116.iam.gserviceaccount.com"
   ```

4. `terraform version`이 정확히 `1.16.3`인지 확인한다. 모든 root에서 생성한
   `.terraform.lock.hcl`은 검토 후 커밋해 provider checksum을 재현한다.

## 원격 state bootstrap

State bucket 자체를 만들기 전에는 GCS backend를 초기화할 수 없다. 따라서 bootstrap root만
최초 한 번 local state로 적용한 뒤, 같은 state를 새 bucket으로 이전한다. Bucket은
`asia-northeast3`의 `pertinence-509116-pertinence-iac-state`이며 다음 보호를 적용한다.

- Uniform bucket-level access
- Public access prevention `enforced`
- Object versioning
- `force_destroy = false`와 Terraform `prevent_destroy`
- retention lock 없음: backend lock object와 state 복구 작업을 방해하지 않음

다음 명령은 repository root에서 실행한다.

```bash
cd cloud/iac/bootstrap
cp terraform.tfvars.example terraform.tfvars
terraform init
terraform fmt -check
terraform validate
terraform plan -out=bootstrap.tfplan
terraform apply bootstrap.tfplan
```

Apply 전에 plan이 state bucket 하나만 생성하는지 확인한다. 생성이 끝나면 출력의 bucket 이름이
예상값인지 확인하고 remote backend를 활성화한다.

```bash
cp backend.tf.example backend.tf
terraform init -migrate-state
terraform state pull > /tmp/pertinence-bootstrap-state.json
terraform plan -detailed-exitcode
```

`terraform init -migrate-state` 질문에는 local state를 GCS로 복사하도록 승인한다. 마지막 plan의
exit code `0`은 drift가 없음을 의미한다. `/tmp/pertinence-bootstrap-state.json`은 migration 확인용
임시 파일이며 자격 증명은 포함하지 않지만 state가 민감할 수 있으므로 공유하거나 커밋하지 않는다.
확인 후 운영체제의 안전한 임시 파일 정리 절차로 제거한다. `backend.tf`, `terraform.tfvars`,
`*.tfstate`, `*.tfplan`과 `.terraform/`은 Git에서 제외된다.

이미 같은 이름의 bucket이 존재한다면 apply하지 않는다. 소유 project, region, UBLA, PAP,
versioning을 읽기 전용으로 확인하고, 이 root의 관리 대상으로 확정된 경우에만 local state에 import한
뒤 zero-drift plan을 만든다.

```bash
terraform import google_storage_bucket.terraform_state \
  pertinence-509116-pertinence-iac-state
terraform plan -detailed-exitcode
```

State bucket 삭제는 정상 운영 절차가 아니다. `prevent_destroy`를 우회하거나 object version을
수동 삭제하지 않는다. 복구가 필요하면 GCS object version에서 직전 state를 식별하고 별도 파일로
검증한 다음 `terraform state push` 권한과 영향을 동료 검토한다.

## 안전한 환경 workflow

환경 root마다 서로 다른 backend prefix를 사용한다. 현재 cluster는
`gke/dispatcher-current`를 사용하고, 신규 또는 disposable 환경은 고유한 환경 이름을 prefix에
포함한다. 서로 다른 프로필을 같은 state에서 교체하지 않는다.

모든 변경은 다음 순서로 수행한다.

```bash
terraform fmt -check -recursive
terraform init
terraform validate
terraform plan -out=change.tfplan
terraform show change.tfplan
terraform apply change.tfplan
```

현재 cluster의 import/protection 단계는 plan JSON을
[`scripts/terraform_plan_guard.py`](../../scripts/terraform_plan_guard.py)에 각각 `no-op`과
`protection` mode로 통과시킨 saved plan만 적용한다. 신규 환경도 `fresh-create` mode로 delete와
replace를 거부한다.

- 현재 cluster import에서는 `create/update/delete/replace = 0`이 될 때까지 apply하지 않는다.
- 최초 보호 단계에서는 cluster deletion protection과 node pool 삭제 보호만 변경한다.
- node pool 제거는 삭제 보호를 해제하는 별도 plan/apply와 실제 제거 plan/apply의 두 단계로
  나누고, 정확한 대상과 state를 사람이 승인한다.
- `terraform destroy`는 현재 cluster나 bootstrap root에서 실행하지 않는다. 삭제 검증은 별도
  state의 disposable 환경에서만 한다.
- `terraform apply -auto-approve`, `-target`과 state 직접 편집은 정상 workflow에서 사용하지 않는다.

Node pool은 `system`, `pipeline`, `serving`, `loadgen`을 독립적으로 활성화할 수 있다. 따라서
system과 loadgen을 모두 포함하거나 모두 제외할 수 있다. 단, system 없는 구성은 명시적인
`allow_without_system_pool`, 전체 pool이 없는 control-plane-only 구성은 추가적인
`allow_control_plane_only` 승인이 필요하다. KFP/KServe addon 검사는 system pool이 없는 환경을
지원 대상으로 간주하지 않는다.

## 검증과 addon 인계

정적 검사는 실제 cloud 변경 없이 다음을 통과해야 한다.

```bash
terraform fmt -check -recursive
terraform init -backend=false
terraform validate
terraform -chdir=cloud/iac/modules/gke test
```

신규 환경 생성 후에는 실제 node pool/taint, workload pool 0→1→0 autoscaling, Artifact Registry
pull과 GCS 접근을 smoke test한다. KFP/KServe addon은 Terraform output의 cluster 이름, 위치,
Workload Identity pool, 활성 pool 목록과 selector/toleration을 받아 설치 전 조건을 검사한다.
Addon 설치가 Terraform state를 변경하거나 Terraform이 Kubernetes resource를 소유하게 만들면
안 된다.
