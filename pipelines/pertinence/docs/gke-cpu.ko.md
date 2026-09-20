# GKE CPU + GCS KFP 실행 가이드

이 구성은 GKE 위의 KFP v2 호환 OSS Kubeflow Pipelines backend를 대상으로 한다. Compile과
제출에는 `kfp==2.17.0` SDK를 사용한다. Raw input과 run output은 GCS에 두고, task Pod는 GPU
없이 CPU만 사용한다. KFP API server의 IR 저장소와 run artifact의 pipeline root는 서로 다른
설정이다. API server는 배포판 기본 저장소를 유지하고 KFP launcher의 pipeline root만 GCS로
설정한다.

## 1. 실행 흐름

```text
GCS raw objects
  ├── cifar-10-python.tar.gz
  ├── cifar10_shufflenetv2_x0_5-1308b4e9.pt
  ├── cifar10_mobilenetv2_x0_5-ca14ced9.pt
  ├── cifar10_resnet56-187c023a.pt
  └── pipeline-run.yaml
          │
          ▼
prepare-dataset-bundle ─┐
export-expert-bundle ───┼─ validate-run → build-routing-dataset
pipeline-run.yaml ──────┘                         │
                                                  ▼
                               train-dispatcher → evaluate-final
                                                        │
                                                        ▼
                                              register-dispatchers
```

모든 output artifact는 `gs://<run-bucket>/pertinence/runs` 아래 KFP run 경로에 저장된다.
`register-dispatchers`는 run-scoped `registry.sqlite3`와 registration report를 하나의 KFP output
artifact로 내보낸다. 이는 영속 결과지만 여러 run이 공유하는 중앙 Model Registry는 아니다.

## 2. GCS와 Workload Identity

GKE cluster에는 Workload Identity Federation을 활성화한다. 아래 변수는 환경에 맞게 지정한다.

```bash
PROJECT_ID="your-project"
PROJECT_NUMBER="$(gcloud projects describe "${PROJECT_ID}" --format='value(projectNumber)')"
RAW_BUCKET="your-pertinence-raw"
RUN_BUCKET="your-pertinence-runs"
NAMESPACE="kubeflow"
KSA="pipeline-runner"

gcloud storage buckets create "gs://${RAW_BUCKET}" --location=asia-northeast3
gcloud storage buckets create "gs://${RUN_BUCKET}" --location=asia-northeast3

# KFP 설치가 실행 RBAC와 함께 만든 KSA와 RoleBinding을 확인한다.
kubectl get serviceaccount "${KSA}" -n "${NAMESPACE}"
kubectl get rolebinding pipeline-runner-binding -n "${NAMESPACE}"

KSA_PRINCIPAL="principal://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${PROJECT_ID}.svc.id.goog/subject/ns/${NAMESPACE}/sa/${KSA}"
gcloud storage buckets add-iam-policy-binding "gs://${RAW_BUCKET}" \
  --member="${KSA_PRINCIPAL}" \
  --role=roles/storage.objectViewer
gcloud storage buckets add-iam-policy-binding "gs://${RUN_BUCKET}" \
  --member="${KSA_PRINCIPAL}" \
  --role=roles/storage.objectUser
```

서비스 계정 key JSON을 Pod에 넣지 않는다. GKE metadata server를 통한 Workload Identity를
사용한다. Standard cluster라면 해당 node pool에도 GKE metadata server가 활성화되어 있어야 한다.
새 KSA 대신 KFP가 실행 RBAC와 함께 제공하는 `pipeline-runner`를 사용하고, 이 KSA에 bucket
권한만 추가한다. 계정이나 RoleBinding이 없다면 빈 KSA만 만들지 말고 사용 중인 KFP release의
runner RBAC를 복구한다. `gke/service-account.yaml`은 기대하는 이름/namespace를 확인하기 위한
참조 manifest다.

`pipelines/pertinence/gke/kfp-launcher-configmap.yaml`의 bucket placeholder를 바꾼 뒤 적용한다.
이미 같은 namespace에 `kfp-launcher` ConfigMap이 있으면 값을 먼저 백업한다.

```bash
kubectl get configmap kfp-launcher -n "${NAMESPACE}" -o yaml > /tmp/kfp-launcher.backup.yaml
sed "s/REPLACE_WITH_RUN_BUCKET/${RUN_BUCKET}/" \
  pipelines/pertinence/gke/kfp-launcher-configmap.yaml | kubectl apply -f -
```

## 3. Raw asset 업로드

현재 검증된 SHA-256은 다음과 같다.

```text
CIFAR-10   6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce
ShuffleNet 1308b4e9c4f88656a3b09775c273fc5ffe8fca9c84044123cf3d138c8dec10bd
MobileNet  ca14ced9679a365262eb6ec98fd347deafcc06af2d0457d8b80424cfe8caaac8
ResNet56   187c023aee0c9cf3093a682d9447a538cbaf5489f7ae78a7df4edf2246ce380b
```

```bash
gcloud storage cp artifacts/cifar10/datasets/cifar-10-python.tar.gz \
  "gs://${RAW_BUCKET}/pertinence/raw/cifar-10-python.tar.gz"
gcloud storage cp artifacts/cifar10/checkpoints/cifar10_shufflenetv2_x0_5-1308b4e9.pt \
  "gs://${RAW_BUCKET}/pertinence/raw/cifar10_shufflenetv2_x0_5.pt"
gcloud storage cp artifacts/cifar10/checkpoints/cifar10_mobilenetv2_x0_5-ca14ced9.pt \
  "gs://${RAW_BUCKET}/pertinence/raw/cifar10_mobilenetv2_x0_5.pt"
gcloud storage cp artifacts/cifar10/checkpoints/cifar10_resnet56-187c023a.pt \
  "gs://${RAW_BUCKET}/pertinence/raw/cifar10_resnet56.pt"

RUN_CONFIG="experiments/cifar10/configs/pipeline-run.yaml"
RUN_CONFIG_SHA256="$(.venv/bin/python -m pertinence.pipeline_cli \
  artifact-sha256 "${RUN_CONFIG}")"
gcloud storage cp "${RUN_CONFIG}" \
  "gs://${RAW_BUCKET}/pertinence/config/pipeline-run.yaml"
```

GCS object 이름만으로 immutability를 가정하지 않는다. Pipeline parameter의 SHA-256과 실제로
stage된 바이트가 다르면 첫 단계 또는 validation에서 실패한다. 운영 bucket에는 object versioning과
retention policy를 함께 설정하는 것을 권장한다.

## 4. CPU component image

이미지는 CPU PyTorch와 CPU ONNX Runtime만 포함한다. Artifact Registry에 push한 뒤 tag가 아니라
manifest digest를 사용한다.

```bash
REGION="asia-northeast3"
REPOSITORY="pertinence"
IMAGE_TAG="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/components:cpu-v1"

gcloud artifacts repositories create "${REPOSITORY}" \
  --repository-format=docker --location="${REGION}"
gcloud auth configure-docker "${REGION}-docker.pkg.dev"
docker build --platform linux/amd64 \
  --file pipelines/pertinence/Dockerfile \
  --tag "${IMAGE_TAG}" .
docker push "${IMAGE_TAG}"

IMAGE_DIGEST="$(gcloud artifacts docker images describe "${IMAGE_TAG}" \
  --format='value(image_summary.digest)')"
COMPONENT_IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/components@${IMAGE_DIGEST}"
```

GKE node service account에는 해당 repository의 `roles/artifactregistry.reader`가 필요하다.

## 5. Compile과 제출

Compiler와 제출 client는 `kfp==2.17.0`을 사용한다. Backend는 해당 IR을 지원하는 KFP v2
배포판이어야 하며, SDK와 backend의 release 번호가 같아야 한다는 뜻은 아니다.

```bash
.venv/bin/python -m pip install -r pipelines/pertinence/requirements-kfp.lock
.venv/bin/python -m pipelines.pertinence.pipeline \
  --image "${COMPONENT_IMAGE}" \
  --pipeline-root "gs://${RUN_BUCKET}/pertinence/runs" \
  --cpu-threads 4 \
  --output artifacts/cifar10/portable/pipeline.yaml
```

기본 IR은 KFP task cache를 끈다. 실패 복구에 cache를 쓰려면 compile과 submit 양쪽에
`--enable-cache`를 지정한다. Raw SHA-256과 image digest가 cache key에 포함되지만 같은 GCS URI를
덮어쓰는 운영은 금지한다.

KFP API를 로컬 `8888`에 port-forward한 뒤 제출한다.

```bash
.venv/bin/python -m pipelines.pertinence.submit \
  --host http://127.0.0.1:8888 \
  --pipeline artifacts/cifar10/portable/pipeline.yaml \
  --pipeline-root "gs://${RUN_BUCKET}/pertinence/runs" \
  --namespace "${NAMESPACE}" \
  --service-account "${KSA}" \
  --cifar10-uri "gs://${RAW_BUCKET}/pertinence/raw/cifar-10-python.tar.gz" \
  --cifar10-sha256 6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce \
  --shufflenet-checkpoint-uri "gs://${RAW_BUCKET}/pertinence/raw/cifar10_shufflenetv2_x0_5.pt" \
  --shufflenet-checkpoint-sha256 1308b4e9c4f88656a3b09775c273fc5ffe8fca9c84044123cf3d138c8dec10bd \
  --mobilenet-checkpoint-uri "gs://${RAW_BUCKET}/pertinence/raw/cifar10_mobilenetv2_x0_5.pt" \
  --mobilenet-checkpoint-sha256 ca14ced9679a365262eb6ec98fd347deafcc06af2d0457d8b80424cfe8caaac8 \
  --resnet56-checkpoint-uri "gs://${RAW_BUCKET}/pertinence/raw/cifar10_resnet56.pt" \
  --resnet56-checkpoint-sha256 187c023aee0c9cf3093a682d9447a538cbaf5489f7ae78a7df4edf2246ce380b \
  --run-config-uri "gs://${RAW_BUCKET}/pertinence/config/pipeline-run.yaml" \
  --run-config-sha256 "${RUN_CONFIG_SHA256}"
```

Pipeline은 GPU resource를 요청하지 않는다. 초기 resource limit은 dataset 2 CPU/4 GiB,
expert export 4 CPU/8 GiB, validation 4 CPU/6 GiB, routing cache 4 CPU/12 GiB,
dispatcher search 4 CPU/8 GiB, final evaluation 2 CPU/4 GiB, registration 1 CPU/2 GiB다.

## 6. 배포 전 점검

```bash
.venv/bin/python -m pytest
.venv/bin/ruff check .

kubectl auth can-i create workflows.argoproj.io \
  --as="system:serviceaccount:${NAMESPACE}:ml-pipeline" -n "${NAMESPACE}"
kubectl run gcs-auth-check --rm -it --restart=Never \
  --namespace="${NAMESPACE}" \
  --overrides="{\"spec\":{\"serviceAccountName\":\"${KSA}\"}}" \
  --image=google/cloud-sdk:slim -- \
  gcloud storage ls "gs://${RAW_BUCKET}/pertinence/raw/"
```

첫 smoke run에서는 작은 NSGA-II 설정을 별도 RunConfig로 사용하고, 성공 후 제공된 production
RunConfig로 전환하는 것이 안전하다. CPU-only 전체 run은 세 expert의 60,000장 추론과 2,500회
dispatcher 학습을 수행하므로 node 성능에 따라 오래 걸릴 수 있다.

참고 문서:

- <https://www.kubeflow.org/docs/components/pipelines/concepts/pipeline-root/>
- <https://www.kubeflow.org/docs/components/pipelines/operator-guides/configure-object-store/>
- <https://cloud.google.com/kubernetes-engine/docs/concepts/workload-identity>
