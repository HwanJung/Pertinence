# PERTINENCE GKE cloud environment

이 디렉터리는 실제 GKE에서 완료한 CIFAR-10 dispatcher 학습 환경의 cloud 정본이다. 현행 범위는
GKE Standard 위의 OSS Kubeflow Pipelines(KFP) 2.17.0과 CPU component 실행이다.

KServe, online router, serving/loadgen pool과 latency benchmark는 아직 배포되지 않았다. 구현된
배포 자산과 benchmark profile의 실제 실행 절차는
[KServe benchmark runbook](docs/kserve-benchmark-runbook.ko.md), 남은 보안·운영 완료 조건은
[roadmap](docs/roadmap.ko.md)에 기록한다.

## 2026-09-20 확인 상태

| 항목 | 실제 상태 |
|---|---|
| Project | `pertinence-509116` |
| Cluster | `pertinence-dispatcher-260919-1708`, zonal Standard, RUNNING |
| Location | `asia-northeast3-a` |
| GKE/Kubernetes | `1.35.8-gke.1036000` |
| Workload Identity pool | `pertinence-509116.svc.id.goog` |
| `system-pool` | `e2-standard-4`, 1 node, `pd-standard` 50 GiB |
| `pipeline-pool` | `n2-standard-8`, autoscaling 0–1, `pd-standard` 50 GiB |
| KFP | 14 deployments Available in `kubeflow` |
| Persistent storage | MySQL 20 GiB + SeaweedFS 20 GiB PVC, both Bound |
| External exposure | 없음; KFP API/UI는 port-forward만 사용 |
| Serving resources | 미배포 |

`pipeline-pool`은 유휴 시 0대로 축소된다. `system-pool`, control plane, PVC와 storage 비용은
cluster를 보존하는 동안 계속 발생할 수 있다.

## 구조

```text
cloud/
├── README.md
├── docs/
│   ├── architecture.ko.md                 # 실제 배포 경계와 데이터 흐름
│   ├── runbook.ko.md                      # 반복 실행과 보존 운영 절차
│   ├── kserve-benchmark-runbook.ko.md     # 미실행 online benchmark 절차
│   ├── roadmap.ko.md                      # 미배포 목표와 남은 완료 조건
│   └── runs/2026-09-20-dispatcher-gke.ko.md
├── iac/                                   # GKE Terraform state와 reusable module
└── kfp/                                   # launcher ConfigMap과 KFP runner 참조 manifest
```

## 문서 우선순위

1. 현재 리소스 구조와 경계: [architecture](docs/architecture.ko.md)
2. 반복 실행과 운영: [runbook](docs/runbook.ko.md)
3. 실제 실행 결과와 immutable ID: [run record](docs/runs/2026-09-20-dispatcher-gke.ko.md)
4. Terraform state와 변경 안전성: [IaC guide](iac/README.md)
5. KServe benchmark 실행 설계: [benchmark runbook](docs/kserve-benchmark-runbook.ko.md)
6. 준비 완료 후 20단계: [실제 실행 절차](docs/kserve-benchmark-execution.ko.md)
7. 미배포 목표와 남은 완료 조건: [roadmap](docs/roadmap.ko.md)

Run record의 version과 상태는 해당 실행 시점의 증거다. 반복 실행 전에는 runbook의 read-only
inventory 절차로 현재 상태를 다시 확인한다. Architecture와 runbook이 roadmap과 충돌하면 전자가
우선한다.

## 소유 경계

- Terraform은 GKE cluster, node service account/IAM, system/pipeline node pool과 보호 설정을
  관리한다.
- KFP 설치본, Kubernetes workload, PVC, 실험별 bucket IAM은 base GKE Terraform state가
  소유하지 않는다.
- `cloud/kfp/`에는 upstream KFP 전체 manifest를 vendoring하지 않고 저장소가 변경하는 launcher
  설정과 existing `pipeline-runner` 계약만 둔다.
- Raw input과 run artifact는 GCS에, KFP metadata/MySQL/MLMD는 cluster PVC에 남는다.
- Cluster 삭제는 bucket, Artifact Registry image 또는 GCS run artifact 삭제를 뜻하지 않는다.

현재 cluster와 PVC는 사용자 보존 지시에 따라 유지한다. 명시적인 종료 요청 없이는 teardown이나
Terraform destroy를 수행하지 않는다.
