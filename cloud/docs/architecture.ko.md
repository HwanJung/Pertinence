# 실제 GKE·KFP 아키텍처

> 기준일: 2026-09-20
> 근거: production KFP run, read-only GKE/Kubernetes inventory, Terraform remote-state adoption

## 범위와 상태

현행 cloud 구성은 zonal GKE Standard cluster에서 CIFAR-10 dispatcher를 CPU로 학습·평가하는
single-user OSS KFP 환경이다. 일곱 container component와 GCS artifact, immutable image digest,
node-pool scheduling, final 평가와 run-scoped SQLite registration까지 실제 실행으로 검증했다.

다음 항목은 현행 아키텍처에 없다.

- KServe, `InferenceService`와 online router;
- serving/loadgen node pool의 실제 생성;
- evaluated bundle promotion과 online latency benchmark;
- public KFP endpoint와 multi-user authentication;
- 중앙 Model Registry.

미배포 목표와 남은 완료 조건은 [roadmap](roadmap.ko.md)에 기록한다.

저장소의 serving runtime, KServe manifest와 현재 cluster용 Terraform benchmark profile은 실행
준비 자산이다. 실제 cluster 배포와 성공 benchmark evidence가 없으면 이 문서의 현행 미배포 상태를
바꾸지 않는다. 실행 전후 gate와 기록 기준은
[KServe benchmark runbook](kserve-benchmark-runbook.ko.md)을 따른다.

## 시스템 컨텍스트

```text
operator workstation
  ├─ Terraform/gcloud/kubectl ───────────────► GKE control plane
  ├─ Cloud Build ─► Artifact Registry ───────► component image pull
  ├─ immutable raw input + SHA-256 ──────────► Raw GCS bucket
  └─ KFP SDK over local port-forward ────────► KFP API

GKE: pertinence-dispatcher-260919-1708
  system-pool
    └─ KFP API/UI, Argo, MLMD, MySQL, SeaweedFS, importer/driver Pods
  pipeline-pool (0–1)
    └─ seven PERTINENCE container executors

KFP launcher ────────────────────────────────► Run GCS bucket
```

KFP API metadata와 task artifact 저장 위치는 다르다. `defaultPipelineRoot`는 GCS Run bucket을
가리키지만 KFP API metadata, MySQL, ML Metadata와 internal object-store 상태는 cluster PVC에
남는다. Cluster를 제거하기 전에는 필요한 metadata를 별도로 export해야 한다.

## GKE topology

| Pool | Machine/disk | Scaling | Taint | 실제 역할 |
|---|---|---|---|---|
| `system-pool` | `e2-standard-4`, `pd-standard` 50 GiB | 고정 1 | 없음 | KFP system, importer/driver |
| `pipeline-pool` | `n2-standard-8`, `pd-standard` 50 GiB | 0–1 | `workload=pertinence-pipeline:NoSchedule` | 일곱 executor |

Cluster는 `asia-northeast3-a`의 VPC-native Standard cluster이고 Regular release channel과
Workload Identity Federation을 사용한다. 2026-09-20 확인 버전은 control plane/node 모두
`1.35.8-gke.1036000`이다.

Container executor에는 다음 selector와 toleration을 함께 적용한다.

```yaml
nodeSelector:
  cloud.google.com/gke-nodepool: pipeline-pool
tolerations:
  - key: workload
    operator: Equal
    value: pertinence-pipeline
    effect: NoSchedule
```

Taint만으로는 pool 선택이 보장되지 않고 toleration만으로는 다른 pool 배치를 막지 못하므로 둘을
하나의 계약으로 검사한다. KFP 2.17 importer는 URI를 MLMD에 등록할 뿐 raw byte를 처리하지 않으며,
driver/root와 importer Pod는 `system-pool`에서 실행될 수 있다. Cache-hit 보조 Pod도 실제 component
연산이 아니므로 system pool 실행을 허용한다.

## KFP pipeline

Production IR은 외부 GCS artifact를 등록하는 importer 5개와 container executor 7개로 구성된다.

```text
prepare-dataset-bundle ─┐
export-expert-bundle ───┴─ validate-run ─ build-routing-dataset
  ─ train-dispatcher ─ evaluate-final ─ register-dispatchers
```

| Component | 핵심 출력 | Resource request/limit |
|---|---|---|
| `prepare-dataset-bundle` | `DatasetBundle` | 2 CPU / 4 GiB |
| `export-expert-bundle` | `ExpertBundle` | 4 CPU / 8 GiB |
| `validate-run` | normalized contract, split | 4 CPU / 6 GiB |
| `build-routing-dataset` | train/search/final cache | 4 CPU / 12 GiB |
| `train-dispatcher` | search result, Pareto checkpoints | 4 CPU / 8 GiB |
| `evaluate-final` | final report, evaluated bundle | 2 CPU / 4 GiB |
| `register-dispatchers` | report, SQLite registry bundle | 1 CPU / 2 GiB |

Search input에는 final cache가 없다. Final은 search가 보존한 checkpoint 전체를 재학습 없이
복원하며 final metric으로 제거하거나 재선정하지 않는다. Registration은 batch transaction과
idempotency key를 사용하지만 DB는 run artifact이지 중앙 Registry가 아니다.

모든 component image는 `linux/amd64` manifest digest로 고정한다. Mutable tag는 compile 단계에서
거부한다. Production 기본값은 task cache off이고 registration cache는 항상 off다. Immutable input
URI+SHA와 image digest가 유지되는 실패 복구에서만 compile/submit 양쪽에 cache를 켠다.

## Artifact와 lineage

Pipeline의 외부 입력은 CIFAR-10 archive, expert checkpoint 3개와 RunConfig이며 URI와 SHA-256을
쌍으로 전달한다. Component는 다음 세 층을 독립적으로 검증한다.

1. 객체 byte SHA-256: 전송 또는 동일 URI overwrite 감지;
2. PERTINENCE fingerprint: contract, split, expert order와 runtime identity 검증;
3. KFP task cache key: immutable input을 사용한 stage 재사용.

Run root에는 IR, execution manifest, KFP artifact, Pareto checkpoint, final/registration report와
run-scoped SQLite DB가 남는다. 실패 run의 workflow, Pod, event와 log inventory도 별도 diagnostics
prefix에 보존한다.

## 인증과 네트워크

- JSON service-account key를 Pod에 mount하지 않는다.
- GKE node service account는 node 기본 역할과 Artifact Registry reader만 가진다.
- `kubeflow/pipeline-runner` KSA는 Workload Identity로 experiment GSA를 사용한다.
- Raw bucket은 object viewer, Run bucket은 object user 수준으로 분리한다.
- KFP 2.17 launcher의 bucket-level list 요청을 허용하되 object access는 experiment prefix condition으로
  제한한다.
- KFP API/UI는 public LoadBalancer나 Ingress 없이 local port-forward로만 접근한다.

`cloud/kfp/service-account.yaml`은 새 빈 SA를 만들어 KFP RBAC를 대체하기 위한 manifest가 아니다.
Upstream KFP가 제공한 `pipeline-runner`의 이름/namespace 계약을 검증하는 참조 manifest다.

## Terraform 소유 경계

`cloud/iac/`의 base state는 cluster, node service account/IAM, system/pipeline pool과 deletion
protection을 관리한다. KFP addon, Kubernetes resource/PVC, experiment bucket/IAM은 lifecycle이 달라
이 state에 넣지 않는다.

현재 cluster는 non-destructive import 후 remote state에 연결됐고 마지막 확인 plan은 no-op이었다.
Cluster deletion protection과 node-pool `PREVENT` policy를 유지한다. 상세 workflow는
[IaC guide](../iac/README.md)와 [cloud runbook](runbook.ko.md)을 따른다.

## 검증된 경계

Production run에서 다음을 확인했다.

- executor 7개는 `pipeline-pool`, importer/driver 13개는 `system-pool`에 배치;
- pipeline pool의 0→1→0 autoscaling;
- 20 workflow Pod 성공, restart/OOM/warning 없음;
- 50,000/8,000/2,000 split, 2,500 search evaluations와 233 checkpoint 복원;
- final artifact hash와 233 registration row의 일치;
- public LoadBalancer, Ingress, forwarding rule과 reserved address 없음.

Immutable 식별자와 결과 수치는 [실행 기록](runs/2026-09-20-dispatcher-gke.ko.md)에서 확인한다.
