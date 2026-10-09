# 2026-09-20 Dispatcher GKE/KFP 실행·IaC 도입 기록

## 1. 결론

CIFAR-10 dispatcher production pipeline을 GKE Standard와 OSS KFP 2.17.0에서 완료했다.
최종 KFP run과 Argo Workflow는 모두 `SUCCEEDED`이며, 50세대 × population 50의 NSGA-II
2,500회 평가, final 평가, run-scoped registry 등록까지 성공했다.

사용자의 2026-09-20 보존 지시에 따라 GKE cluster는 삭제하지 않았다. KFP가 계속 작동하도록
PVC/PV, namespace, Workload Identity binding과 두 node pool도 유지했다. 따라서 원래 계획의
teardown 항목은 완료가 아니라 명시적으로 `DEFERRED` 상태다.

## 2. 최종 실행 식별자

| 항목 | 값 |
|---|---|
| Cloud project | `pertinence-509116` (`423661327395`) |
| Experiment prefix | `dispatcher-gke-20260920t015118z` |
| KFP experiment ID | `49a3a60b-d869-446d-bbaa-515d7a13f132` |
| KFP run ID | `196dc4e3-e604-440f-b9ca-7c12b66bcdb0` |
| Argo Workflow | `pertinence-cifar10-cpu-bmhbj` |
| 최종 상태 | `SUCCEEDED` |
| 실행 시간 | `2026-09-20T02:10:09Z`–`09:57:32Z` (7시간 47분 23초) |
| Git SHA | `9649df668402f3492b72d752f2158c542c04acb8` |
| Image source worktree patch SHA-256 | `6937ceb56a292d2da647e71d7f8421a4a40a3aef6a5a6514b1086534ccc5913a` |
| Pipeline IR SHA-256 | `ff09e528efa349879f8383480f42261cffe6750c9d813d8ec9ba54065ed416e5` |
| Component image | `asia-northeast3-docker.pkg.dev/pertinence-509116/pertinence/components@sha256:615dfd0be92b1314aabaa1d9b2e2450793243eb62a99b1f965df257c87f55360` |
| Cloud Build | `c6c13405-7c7c-4df6-a208-3d1863f2753d` (`SUCCESS`) |
| Run root | `gs://pertinence-509116-pertinence-runs/pertinence/runs/dispatcher-gke-20260920t015118z` |

## 3. 실행 결과

| 지표 | 결과 |
|---|---:|
| NSGA-II generations / evaluations | 50 / 2,500 |
| 전체 평가 기준 Pareto dispatcher | 233 |
| 복원·final 평가한 checkpoint | 233 |
| Registry 등록 version | 233 |
| Final system accuracy 범위 | 90.40%–94.65% |
| Final cost 범위 | 21.806144–232.825224 MFLOPs |
| 최고 정확도 모델 | `dispatcher-000733`: 94.65%, 210.726944 MFLOPs |
| 최저 비용 모델 | `dispatcher-000267`: 90.45%, 21.806144 MFLOPs |

핵심 lineage fingerprint는 다음과 같다.

- Contract: `5da2c50b3b718923326b55ce21fbdd5b74beed9220df6baecfbfe6091df4bcc0`
- Search: `8c8a2491ed065d6fe100b76273be723c0d8bcb5ef60687f9ad631ef568ef2b07`
- Pareto front: `4c31958c59a2c3f4f30b2c4ec86b24b504dd85a5bfbc4616f61f0405d4f52c10`
- Final report: `1c1d253694392e54b71e4e44b4e4248e0084da294cc38d160d93a9de35b8423c`
- Registry report: `455188a978c6cea1df72d2b919cd162318fb5794bb090174b00a31c930657874`

학습은 `02:14:47Z`에 시작했고, 첫 50회 평가는 9분 27.8초가 걸렸다. 10·25·40세대에서
평균은 각각 9분 14.7초, 9분 14.7초, 9분 13.6초였으며 실제 50세대 완료는
`09:55:34.279Z`였다. Search 본체는 7시간 40분 47초가 걸렸다. 최대 관측 자원은 약
3.9 vCPU와 529 MiB였고 restart, OOM, workflow Warning은 없었다.

## 4. 검증 결과

### 로컬·이미지 검증

- 수정 전 전체 테스트: 116 passed
- protobuf 호환 수정 후 전체 테스트: 118 passed
- Ruff: passed
- `pip check`: no broken requirements
- Raw asset SHA-256 검증: passed
- 새 image runtime smoke: `COMPONENT_PROTOBUF_SMOKE_OK x86_64 6.33.5`
- 새 experiment KSA/GCS smoke: `NEW_EXPERIMENT_GCS_SMOKE_OK`
- IR: 12 tasks, 실제 container executor 7개에만 `pipeline-pool` selector/toleration 적용
- Train Pod `activeDeadlineSeconds`: 259,200초

### Cloud 실행 검증

- Workflow Pod 20개 전부 `Succeeded`
- 실제 component Pod 7개는 `pipeline-pool`, importer/driver 13개는 `system-pool`
- 전체 container restart 합계 0
- Validation report: `valid: true`, `D=1024`, `K=3`, `C=10`, evaluation 2,500
- Train/search/final cache: 50,000 / 8,000 / 2,000 samples
- 독립 artifact 검증: `FINAL_ARTIFACT_VERIFY_OK`
- Search evaluation ID 0–2,499와 Pareto ID 재계산 일치
- 233개 checkpoint의 SHA-256, tensor key/shape/dtype/finiteness와 state 복원 검증
- 233개 final confusion matrix가 모두 3×3이고 표본 합계가 2,000인지 검증
- 233개 registry idempotency key가 모두 고유하고 SQLite row/status와 report가 일치
- Standalone final/registration report와 bundle 내부 사본의 size·MD5·CRC32C가 각각 일치
- Public `LoadBalancer`, Ingress, forwarding rule, 예약 address 없음

로컬 verifier의 PyTorch version 문자열은 `2.5.1`, cloud contract는 `2.5.1+cpu`여서 환경
동일성 검사는 의도대로 거부됐다. 이후 contract의 sealed fingerprint를 먼저 검증하고 해당 contract를
verifier에 주입해 artifact lineage, 모델 복원과 registry replay를 독립 확인했다. Production의
`evaluate-final`과 `register-dispatchers`는 원래 cloud image(`2.5.1+cpu`) 안에서 환경 동일성 검사까지
포함해 성공했다.

## 5. 실행 중 발견·수정한 사항

### KFP GCS 권한

첫 run `d32fe6d4-526e-402f-a192-2677697cad55`는 KFP launcher가 입력 artifact를 받기 전에
bucket resource에 `storage.objects.list`를 요청해 실패했다. 객체 prefix만 허용한 condition은
직접 객체 I/O smoke는 통과하지만 launcher list는 통과하지 못했다.

권한을 bucket 전체 객체 접근으로 넓히지 않고 다음 복합 condition으로 수정했다.

```text
resource.name == <bucket-resource>
|| resource.name.startsWith(<one-experiment-object-prefix>)
```

따라서 bucket 목록 조회는 허용하되 객체 read/write는 한 experiment prefix에만 한정한다.

### Python 3.11 / protobuf 6 ONNX 호환성

두 번째 run `e8ed8396-c028-46df-aefb-0dc37a9bc237`는 `validate-run`에서
`FieldDescriptor.label`이 없는 Python 3.11 upb runtime 때문에 실패했다.

- message traversal을 `field.is_repeated`로 변경
- legacy protobuf용 `label == LABEL_REPEATED` fallback 유지
- modern/legacy descriptor 회귀 테스트 추가
- component lock과 `pipeline` extra에 `protobuf==6.33.5` 명시
- 새 experiment, 새 input prefix, 새 image digest로 실행

두 실패 run의 Workflow, Pod, log, event, node, PVC/PV와 upstream artifact inventory는 각각
다음 prefix에 보존했다.

```text
gs://pertinence-509116-pertinence-runs/pertinence/runs/dispatcher-gke-20260919t170821z/diagnostics/d32fe6d4-526e-402f-a192-2677697cad55/
gs://pertinence-509116-pertinence-runs/pertinence/runs/dispatcher-gke-20260919t170821z/diagnostics/e8ed8396-c028-46df-aefb-0dc37a9bc237/
```

## 6. 보존 artifact

```text
<run-root>/metadata/
  pipeline.yaml
  execution-manifest.submitted.json
  execution-manifest.run.json
  execution-manifest.gen1.json
  execution-manifest.gen10.json
  execution-manifest.gen25.json
  execution-manifest.gen40.json
  dispatcher-training-run-plan.ko.md

<run-root>/pertinence-cifar10-cpu/196dc4e3-e604-440f-b9ca-7c12b66bcdb0/
  train-dispatcher/3e707e03-a374-4f3c-ae82-3c10514ccc83/
  evaluate-final/7abe9377-7f61-4bfe-8753-278d43cf4542/
  register-dispatchers/b6991c2e-bbff-470a-a855-c64316beed5d/

<run-root>/diagnostics/196dc4e3-e604-440f-b9ca-7c12b66bcdb0/
  kfp-run.json, workflow.yaml, pods.json, events, nodes, PVC/PV,
  deployments, services/ingress, GKE cluster/node-pool, network inventory, train log
```

`dispatcher-training-run-plan.ko.md`는 제출 당시 GCS에 고정한 historical artifact 이름이다.
현재 저장소의 운영 정본은 [runbook](../runbook.ko.md)이며 이 object를 최신 절차로 해석하지 않는다.

Raw/run/build bucket은 Uniform bucket-level access, public access prevention과 versioning을 유지한다.
최종 image와 두 build의 source/log evidence도 Artifact Registry와 build bucket에 보존했다. 임시
Cloud Build SA의 build bucket `Storage Admin`과 Artifact Registry writer 권한은 모두 회수했다.

## 7. 보존한 GKE 상태와 운영 주의

| 리소스 | 보존 상태 |
|---|---|
| Cluster | `pertinence-dispatcher-260919-1708`, GKE 1.35.8-gke.1036000 |
| `system-pool` | e2-standard-4, 1 node, pd-standard 50 GiB |
| `pipeline-pool` | n2-standard-8, autoscaling 0–1, pd-standard 50 GiB |
| KFP | 14 deployments 모두 Available |
| MySQL PVC | Bound, 20 GiB, reclaim `Delete` |
| SeaweedFS PVC | Bound, 20 GiB, reclaim `Delete` |
| Raw KSA IAM | bucket list + 새 experiment input prefix read |
| Run KSA IAM | bucket list + 새 experiment run prefix write |

보고 시점에는 `system-pool` 1대가 `Ready`이고 `pipeline-pool`은 설정된 autoscaling 0–1에 따라
유휴 후 자동으로 0대가 됐다. 다만 `system-pool`, GKE control plane, 40 GiB KFP persistent disk와
저장소 비용은 계속 발생한다. 계획 단가 `$0.65537/h`를 두 node 동시 실행 상한으로 단순 적용하면
cluster 생성부터 성공 종료까지 약 `$5.80`이며 실제 청구액은 Billing에서 확인해야 한다. 보존
이후에도 비용이 계속 누적된다.

애플리케이션 `LoadBalancer`나 Ingress는 없지만 현재 Standard cluster의 system node에는 ephemeral
external IP가 있다. 외부 서비스 endpoint로 사용되지는 않으며 방화벽/IAM을 임의로 넓히지 않아야
한다.

향후 종료 요청 시에는 결과/metadata 재확인 후 KFP PVC 삭제와 PV/disk 소멸 확인 → 두 KSA IAM
binding 제거 → cluster 삭제 → VM/disk/LB/IP 잔여 확인 → kubeconfig context 정리 순서로 수행한다.

## 8. Terraform 관리 도입

Production 실행 뒤 기존 cluster를 재생성하거나 중단하지 않고 Terraform remote state에 가져왔다.

- `pertinence-509116-pertinence-iac-state` bucket을 `asia-northeast3`에 만들고 bootstrap state를
  `bootstrap/default.tfstate`로 이전했다.
- Cluster, system/pipeline pool, API 6개, node service account와 IAM binding 2개 등 12개 resource를
  `gke/dispatcher-current/default.tfstate`에 import했다.
- Import plan은 `12 imported, 0 added, 0 changed, 0 destroyed`였다.
- Cluster의 deletion protection을 유지하고 두 node pool의 Terraform deletion policy를
  `PREVENT`로 바꿨다. 이 적용은 `0 added, 2 changed, 0 destroyed`였고 node 재생성은 없었다.
- 마지막 `training-current.tfvars` plan과 sizing-only 확인 plan은 모두 `No changes`였다.

Base state는 KFP/KServe, Kubernetes resource/PVC, experiment bucket과 pipeline IAM을 소유하지
않는다. Imported pool의 taint 표현 차이는 plan drift에서 제외하지만 provider가 읽은 effective
taint에 필수 workload taint가 없으면 postcondition이 실패한다.

도입 시 다음 검증을 통과했다.

| 검증 | 결과 |
|---|---|
| Python/pipeline tests | 135 passed |
| Terraform plan-guard tests | 17 passed |
| GKE module tests | 11 passed |
| Terraform validate/fmt | passed |
| Ruff / diff check | passed |

예제 profile은 이 실행에 적용하지 않았다. 현행 state에는 `training-current.tfvars`만 사용하며,
신규 cluster는 별도 environment root와 backend prefix에서 `fresh-create` guard를 통과해야 한다.

## 관련 정본

- [현재 cloud 상태와 문서 색인](../../README.md)
- [보존된 7단계 DAG와 현행 범용 pipeline](../../../pipelines/pertinence/README.md)
- [실제 아키텍처](../architecture.ko.md)
- [반복 운영 runbook](../runbook.ko.md)
- [Terraform guide](../../iac/README.md)
