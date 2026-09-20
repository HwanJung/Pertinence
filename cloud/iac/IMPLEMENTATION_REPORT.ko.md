# GKE IaC 구현 보고서

작성일: 2026-09-20

## 결과

기존 `pertinence-dispatcher-260919-1708` GKE 클러스터를 재생성하거나 중단하지 않고 Terraform
관리 대상으로 가져왔다. 원격 state bootstrap, 재사용 가능한 GKE 모듈, 현재 환경 import root,
노드 풀 구성 프로필과 비파괴 plan guard를 구현했다.

현재 클러스터는 `RUNNING`이며 system 노드 한 대는 `Ready`다. Pipeline 풀은 0–1 autoscaling으로
0대 상태를 유지한다. Kubeflow namespace의 KFP deployment 14개는 모두 Available이고 MySQL과
SeaweedFS의 20 Gi PVC 두 개는 `Bound`다. 실험 종료 후에도 클러스터와 PVC는 삭제하지 않았다.

## 구현 범위

- `bootstrap/`: versioning, UBLA, PAP, 이중 삭제 방어가 적용된 GCS state bucket
- `modules/gke/`: 기존 VPC/subnet 기반 zonal Standard GKE, node service account/IAM, 네 개의 독립
  node pool 역할
- `environments/dispatcher-current/`: 현재 리소스 12개 import와 보호 상태의 정본
- `scripts/terraform_plan_guard.py`: no-op, 보호 강화, 신규 생성 plan만 허용하는 fail-closed guard
- system/loadgen 포함 예제(`benchmark`, `full`)와 둘 다 제외하는 예제(`compute-only`)

KFP/KServe, Kubernetes resource/PVC, 실험 데이터 bucket과 pipeline IAM은 이 Terraform state가
소유하지 않는다.

## 실제 적용 내역

1. `pertinence-509116-pertinence-iac-state` bucket을 `asia-northeast3`에 생성하고 bootstrap state를
   `bootstrap/default.tfstate`로 이전했다.
2. 현재 cluster, system/pipeline pool, API 6개, node service account, IAM binding 2개 등 12개
   리소스를 `gke/dispatcher-current/default.tfstate`에 import했다.
3. Import plan은 `12 imported, 0 added, 0 changed, 0 destroyed`였다.
4. Cluster의 기존 deletion protection을 유지하고 두 node pool의 Terraform
   `deletion_policy`를 `PREVENT`로 설정했다. 이 적용은 `0 added, 2 changed, 0 destroyed`였으며
   실제 node pool 재생성은 없었다.
5. 추가 output을 state에 저장한 적용은 `0 added, 0 changed, 0 destroyed`였다.
6. 마지막 `training-current` plan과 sizing-only 부분 override plan은 모두 `No changes`였다.

원격 state 객체 두 개는 GCS에 존재하며 migration 뒤 남은 로컬 bootstrap state/backup은
삭제했다. State bucket의 object versioning은 유지된다.

## 안전성 보강

- 보호 mode는 `deletion_protection=true`와 `deletion_policy=PREVENT` 방향만 허용한다.
- 형식이 불완전한 plan JSON은 안전하다고 가정하지 않고 거부한다.
- 환경의 node pool 부분 override는 생략한 `enabled` 값을 모듈 기본값에서 상속하므로 sizing만
  바꿨을 때 기존 pool이 삭제 대상으로 바뀌지 않는다.
- Imported pool의 taint 표현 차이는 plan drift에서 제외하지만, provider가 읽은
  `effective_taints`에 필수 workload taint가 없으면 resource postcondition이 실패한다.
- 현재 일회성 maintenance exclusion은 2026-09-24T01:10:46Z까지 그대로 유지하며 base state가
  소유하지 않는다.
- 현재 cluster와 bootstrap root에서는 `terraform destroy`를 사용하지 않는다.

## 검증

- Python/파이프라인 테스트: 135 passed
- Plan guard 테스트: 17 passed
- Terraform GKE module 테스트: 11 passed(taint 실패 경로 포함)
- Bootstrap, module, dispatcher environment: `terraform validate` 통과
- `terraform fmt -check -recursive cloud/iac` 통과
- `ruff check .` 통과
- `git diff --check` 통과
- 독립 에이전트 재감사: 기존 HIGH/MEDIUM 지적 해소, 새 중대 문제 없음

Pytest에는 `pymoo`의 기존 deprecation warning 144건이 있으나 실패는 없다.

## 운영 시 주의사항

Example profile은 현재 dispatcher state에 적용하지 않는다. 신규 cluster에는 별도 environment root와
backend prefix를 만들고 `fresh-create` guard를 통과한 saved plan만 적용한다. Pool 제거는 먼저
삭제 정책을 별도 검토된 plan에서 해제한 뒤, 두 번째 plan에서 제거해야 한다. 실제 quota와 addon
호환성은 신규 환경 생성 전 별도로 확인한다.
