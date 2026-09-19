# Vertex AI Pipelines CPU 실행 구성

## 1. 결론

CPU만 사용할 첫 클라우드 실행은 GKE에 KFP를 직접 설치하지 않고 **Vertex AI Pipelines**에서
실행한다. Pipeline 코드는 KFP 2.17.0 SDK로 compile하고, 실행 backend와 UI, metadata는 Vertex
AI가 관리한다.

이 선택으로 다음 항목이 모두 사라진다.

- GKE cluster와 system/GPU node pool
- NVIDIA driver, GPU quota, zone capacity 확인
- KFP backend, Argo, MySQL, SeaweedFS 설치와 upgrade
- Kubernetes PVC, KSA, Workload Identity, port-forward
- CUDA/PyTorch/ONNX Runtime GPU 조합 검증

단, Vertex AI의 관리형 backend 버전은 사용자가 `2.17.0`으로 고정하지 못한다. backend까지
정확히 KFP 2.17.0이어야 하는 요구가 생길 때만 GKE standalone 구성을 사용한다. 현재 목표는
가장 단순한 단일 사용자 CPU 실험이므로 SDK와 IR만 2.17.0에 고정한다.

Vertex AI Pipelines는 KFP로 정의한 pipeline을 serverless 방식으로 실행하고, pipeline artifact는
GCS에 저장한다.

- [Vertex AI Pipelines 소개](https://cloud.google.com/vertex-ai/docs/pipelines/introduction)
- [KFP pipeline 빌드](https://cloud.google.com/vertex-ai/docs/pipelines/build-pipeline)
- [프로젝트와 service account 구성](https://cloud.google.com/vertex-ai/docs/pipelines/configure-project)

## 2. 최소 구성

| 항목 | 값 |
|---|---|
| 리전 | `asia-northeast3` |
| 실행기 | Vertex AI Pipelines |
| KFP compiler SDK | `2.17.0` |
| 실행 장치 | CPU only |
| 기본 CPU 병렬도 | 4 threads |
| object storage | regional GCS bucket 1개 |
| container storage | Artifact Registry Docker repository 1개 |
| 실행 identity | user-managed service account 1개 |
| Registry 정본 | GCS의 `registry_bundle` output artifact |
| 첫 정식 run cache | off |

```text
GCS bucket
├─ raw/
│  ├─ cifar10/cifar-10-python.tar.gz
│  └─ checkpoints/{shufflenet,mobilenet,resnet56}.pt
├─ configs/pipeline-run.yaml
└─ pipeline-root/
       └─ Vertex/KFP run artifacts

Artifact Registry
└─ pertinence/components-cpu@sha256:<digest>

Vertex AI Pipelines
└─ prepare → export → validate → build → train → evaluate → register
```

입력과 출력 bucket을 나누지 않는다. 하나의 실험 프로젝트와 하나의 pipeline service account를
사용하므로 bucket 하나를 prefix로 구분하는 편이 가장 단순하다. 민감도나 보존 정책이 달라지는
시점에만 bucket을 분리한다.

## 3. Pipeline 범위

클라우드 DAG는 다음 일곱 단계를 모두 CPU에서 실행한다.

```text
raw CIFAR-10 archive ─→ prepare-dataset-bundle ─┐
                                                  ├→ validate-run
3 raw checkpoints ───→ export-expert-bundle ─┘       │
                                                          ▼
                                               build-routing-dataset
                                                          │
                                                          ▼
                                                 train-dispatcher
                                                          │
                                                          ▼
                                                   evaluate-final
                                                          │
                                                          ▼
                                                register-dispatchers
```

학습 대상은 expert CNN이 아니라 `Linear(1024, 3)` dispatcher head다. Expert checkpoint는
`export-expert-bundle`에서 ONNX로 한 번 변환한 뒤 frozen inference에만 사용한다. Validation은
다음을 확인해야 한다.

```text
K = 3
D = 1024
expert order =
  shufflenetv2_x0_5
  mobilenetv2_x0_5
  resnet56
dispatcher weight shape = [3, 1024]
```

`register-dispatchers`는 pod-local `/registry/registry.db`를 사용하지 않는다. SQLite DB와 등록
report를 `registry_bundle` output artifact로 묶으므로 Vertex가 GCS pipeline root에 보존한다.

## 4. 프로젝트 준비

먼저 값 세 개만 정한다.

```bash
PROJECT_ID=<project-id>
REGION=asia-northeast3
BUCKET=${PROJECT_ID}-pertinence
```

필요한 API는 다음 네 개다.

```bash
gcloud services enable \
  aiplatform.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com \
  storage.googleapis.com \
  --project "${PROJECT_ID}"
```

Bucket과 Docker repository를 만든다.

```bash
gcloud storage buckets create "gs://${BUCKET}" \
  --project "${PROJECT_ID}" \
  --location "${REGION}" \
  --uniform-bucket-level-access

gcloud artifacts repositories create pertinence \
  --project "${PROJECT_ID}" \
  --location "${REGION}" \
  --repository-format docker
```

## 5. IAM

Pipeline용 service account 하나만 만든다.

```bash
gcloud iam service-accounts create pertinence-pipeline \
  --project "${PROJECT_ID}" \
  --display-name "PERTINENCE Vertex pipeline"

PIPELINE_SA=pertinence-pipeline@${PROJECT_ID}.iam.gserviceaccount.com

gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member "serviceAccount:${PIPELINE_SA}" \
  --role roles/aiplatform.user

gcloud storage buckets add-iam-policy-binding "gs://${BUCKET}" \
  --member "serviceAccount:${PIPELINE_SA}" \
  --role roles/storage.objectUser

gcloud artifacts repositories add-iam-policy-binding pertinence \
  --project "${PROJECT_ID}" \
  --location "${REGION}" \
  --member "serviceAccount:${PIPELINE_SA}" \
  --role roles/artifactregistry.reader
```

제출자는 `PIPELINE_SA`에 대한 Service Account User 권한이 필요하다. Image를 build하는 사용자나
Cloud Build identity에는 Docker repository의 Writer 권한을 별도로 부여한다. JSON service
account key는 만들지 않는다.

## 6. CPU image와 입력 업로드

공용 image는 `onnxruntime`과 PyTorch CPU wheel만 설치한다. `onnxruntime-gpu`, CUDA base image,
NVIDIA package는 사용하지 않는다.

```bash
IMAGE=${REGION}-docker.pkg.dev/${PROJECT_ID}/pertinence/components-cpu

gcloud builds submit . \
  --project "${PROJECT_ID}" \
  --region "${REGION}" \
  --config pipelines/pertinence/cloudbuild.yaml \
  --ignore-file pipelines/pertinence/Dockerfile.dockerignore \
  --substitutions "_IMAGE=${IMAGE}:v1"
```

Build 뒤 digest를 조회하고 이후 compile에는 tag가 아니라 digest URI만 사용한다.

```bash
gcloud artifacts docker images describe "${IMAGE}:v1" \
  --project "${PROJECT_ID}" \
  --format 'value(image_summary.digest)'
```

Raw asset과 config는 immutable 경로에 올리고 SHA-256을 함께 기록한다.

```bash
gcloud storage cp artifacts/cifar10/datasets/cifar-10-python.tar.gz \
  "gs://${BUCKET}/raw/cifar10/"
gcloud storage cp artifacts/cifar10/checkpoints/*.pt \
  "gs://${BUCKET}/raw/checkpoints/"
gcloud storage cp experiments/cifar10/configs/pipeline-run.yaml \
  "gs://${BUCKET}/configs/"
```

## 7. Compile과 제출

Compiler 의존성은 component image와 분리한다.

```bash
.venv/bin/python -m pip install -r pipelines/pertinence/requirements-kfp.lock

.venv/bin/python -m pipelines.pertinence.pipeline \
  --image "${IMAGE}@sha256:<digest>" \
  --pipeline-root "gs://${BUCKET}/pipeline-root" \
  --cpu-threads 4 \
  --disable-cache \
  --output artifacts/cifar10/portable/pipeline.yaml
```

`--disable-cache`가 기본값이지만 정식 첫 run에서는 의도를 드러내기 위해 명시한다. 입력 hash가
같은 준비·검증·cache build 결과를 재사용하는 개발/복구 run만 `--enable-cache`로 compile한다.
Vertex `PipelineJob(enable_caching=False)`를 함께 사용하면 run 제출 시에도 전체 cache를 확실히
비활성화할 수 있다.

Vertex AI Console의 Pipelines 화면에서 compile된 YAML을 제출하고 다음 parameter를 입력한다.

- CIFAR-10 archive GCS URI와 SHA-256
- ShuffleNet, MobileNet, ResNet56 checkpoint GCS URI와 각 SHA-256
- `pipeline-run.yaml` GCS URI와 SHA-256

실행 service account는 `PIPELINE_SA`, region은 `asia-northeast3`, pipeline root는 compile 때와
같은 `gs://${BUCKET}/pipeline-root`를 사용한다.

## 8. Smoke gate와 정식 실행

GPU gate는 전부 제거한다. 첫 실행 전 확인할 것은 다음뿐이다.

1. Pipeline service account가 bucket object를 읽고 쓸 수 있다.
2. Vertex task가 digest-pinned CPU image를 pull한다.
3. `prepare`가 50,000/10,000 DatasetBundle을 만든다.
4. `export`가 세 ONNX의 batch 1/2 결과를 PyTorch와 비교한다.
5. `validate`가 `K=3`, `D=1024`, 고정 expert 순서를 보고한다.
6. population/generation/epoch를 작게 줄인 config로 7단계가 끝까지 성공한다.
7. `registry_bundle`과 final report가 GCS에 남는다.

이후 `experiments/cifar10/configs/pipeline-run.yaml`의 50×50 search를 cache off로 제출한다. CPU
실행은 GPU보다 오래 걸릴 수 있으므로 먼저 축소 run의 task별 시간을 기록한다. 정식 run에서
`train-dispatcher`가 `Running`이 되면 학습 시작 milestone으로 본다.

## 9. 삭제된 작업 목록

이 구성에서는 다음 작업을 하지 않는다.

- Terraform으로 VPC/GKE/node pool 생성
- T4 quota와 `asia-northeast3-b/c` capacity 확인
- KFP 2.17 standalone manifest 설치
- GCS Launcher provider와 SeaweedFS 설정
- KFP UI port-forward
- GPU driver, `nvidia-smi`, CUDAExecutionProvider smoke
- Registry PVC 생성
- KServe 설치

필요한 cloud resource는 bucket, Docker repository, pipeline service account 세 종류뿐이다.

## 10. KFP backend 2.17 pin이 필요한 경우

Vertex 관리형 backend 대신 KFP backend 버전을 정확히 2.17.0으로 고정해야 한다면
[`gke-cpu.ko.md`](gke-cpu.ko.md)의 OSS KFP fallback을 사용한다. Fallback에서도 CPU
node pool만 사용하고 accelerator request, GPU driver, NVIDIA taint는 추가하지 않는다.
KFP, MySQL, object store, PVC의 설치와 운영이 다시 필요하므로 기본 권장은 아니다.
