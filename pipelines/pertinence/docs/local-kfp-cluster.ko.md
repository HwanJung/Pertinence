# 로컬 Minikube KFP 환경 구성

이 문서는 `pipelines/pertinence/`의 container component와 pipeline을 로컬 Minikube
클러스터에서 사용할 수 있도록 구성한 절차를 기록한다. 2026-09-10에 설치했고
2026-09-11에 상태를 다시 확인했다.

이 구성은 단일 사용자의 로컬 개발과 smoke test를 위한 것이다. 인증, TLS, 다중 사용자 격리,
외부 object store, image signing을 포함하는 production 구성이 아니다.

## 1. 구성 결과

| 항목 | 값 |
|---|---|
| kube context / Minikube profile | `pertinence` |
| Minikube driver / runtime | `docker` / `docker` |
| Kubernetes | `v1.36.4`, 단일 control-plane node |
| KFP manifest ref | `2.17.0`, standalone `env/dev` |
| KFP mode | single-user, database pipeline store |
| 로컬 compiler SDK | `kfp==2.12.1` |
| component image | `localhost:5000/pertinence-components@sha256:a0af60de9aac29401186e1279d7dfc12c277b008106c32d7e84d6256c706c567` |
| pipeline ID | `fb973808-210c-49ca-859a-19517244174c` |
| pipeline version ID | `5bf90f9d-177f-4a8d-9160-2eaa7a6c1dd6` |
| compiled IR | `artifacts/cifar10/portable/pipeline.yaml` |

KFP core deployment 15개가 `Running`이고 다음 PVC가 `Bound`인 것을 확인했다.

- `mysql-pv-claim`: 20 GiB
- `seaweedfs-pvc`: 20 GiB

KFP `2.17.0`의 `env/dev` manifest는 이 설치 시점에 다수의 KFP image를 `:master`로
참조했다. 따라서 위 manifest ref만으로 모든 runtime image가 immutable하게 고정되지는 않는다.
재현성과 공급망 통제가 필요한 환경에서는 deployment image를 release digest로 별도 고정해야 한다.

로컬 compiler는 저장소 lock에 있는 `kfp==2.12.1`을 그대로 사용했다. KFP 2.17 backend에 IR을
upload하고 다시 조회하는 것까지는 검증했지만 SDK와 backend minor가 일치하는 구성은 아니다.
같은 minor 고정이 필요하면 lock을 2.17 계열로 갱신하고 IR 구조 test를 다시 통과시켜야 한다.

## 2. 사전 확인

저장소 루트에서 실행한다.

```bash
kubectl config current-context
minikube status -p pertinence
kubectl cluster-info
kubectl get nodes -o wide
```

예상 context는 `pertinence`다. 다른 context가 출력되면 아래 설치 명령을 실행하지 않는다.
로컬 compiler 환경은 저장소 lock을 사용한다.

```bash
.venv/bin/python -m pip install -r pipelines/pertinence/requirements-kfp.lock
.venv/bin/python -m pip install -e . --no-deps
.venv/bin/python -m pytest pipelines/pertinence/tests -q
```

구성 당시 pipeline test 결과는 `50 passed`였다.

## 3. KFP standalone 설치

공식 standalone 개발 overlay를 `2.17.0` ref로 적용했다.

```bash
KFP_VERSION=2.17.0

kubectl apply -k \
  "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=${KFP_VERSION}"

kubectl wait \
  --for=condition=established \
  --timeout=120s \
  crd/applications.app.k8s.io

kubectl apply -k \
  "github.com/kubeflow/pipelines/manifests/kustomize/env/dev?ref=${KFP_VERSION}"
```

이 overlay는 `kubeflow` namespace와 Argo Workflow CRD, KFP API/UI, MySQL, ML Metadata,
SeaweedFS, cache server 등을 생성한다. 최초 실행은 image pull 때문에 수 분 걸릴 수 있다.

### 3.1 Minikube Pod Security 대응

현재 Minikube의 Pod Security `baseline` 정책은 `hostNetwork: true`인
`proxy-agent` Pod 생성을 거부한다. `proxy-agent`는 이 standalone 실행에 필요하지 않은
webhook/debug proxy이므로 namespace 전체를 `privileged`로 낮추지 않고 deployment만 껐다.

```bash
kubectl scale deployment/proxy-agent -n kubeflow --replicas=0
kubectl rollout status deployment/proxy-agent -n kubeflow --timeout=60s
```

KFP manifest를 다시 apply하면 replica가 다시 1이 될 수 있다. 다음 오류가 보이면 위 명령을
다시 실행한다.

```text
violates PodSecurity "baseline:latest": host namespaces (hostNetwork=true)
```

나머지 deployment의 준비 상태를 기다린다.

```bash
kubectl wait \
  --for=condition=available \
  --timeout=600s \
  deployment \
  --all \
  -n kubeflow

kubectl get pods,pvc,svc -n kubeflow -o wide
```

## 4. Minikube registry와 component image

`components.py`는 image tag를 허용하지 않고
`registry/name@sha256:<64-hex-digest>`만 허용한다. 외부 registry 없이 이 계약을 지키기 위해
Minikube registry addon을 사용했다.

```bash
minikube addons enable registry -p pertinence
kubectl rollout status deployment/registry -n kube-system --timeout=300s
```

Component 공용 image를 저장소의 Dockerfile로 빌드한다.

```bash
docker build \
  --file pipelines/pertinence/Dockerfile \
  --tag pertinence-components:local \
  .
```

Docker daemon과 Minikube registry가 서로 다른 network namespace에 있을 수 있다. 이 환경에서는
호스트에서 registry로 직접 push하지 않고 image를 Minikube node로 먼저 복사했다.

```bash
minikube image load pertinence-components:local -p pertinence

minikube ssh -p pertinence -- \
  "docker tag pertinence-components:local localhost:5000/pertinence-components:local"

minikube ssh -p pertinence -- \
  "docker push localhost:5000/pertinence-components:local"
```

마지막 명령이 출력한 registry digest를 기록한다. 현재 build의 결과는 다음과 같다.

```text
sha256:a0af60de9aac29401186e1279d7dfc12c277b008106c32d7e84d6256c706c567
```

Node 내부에서 digest 참조와 component entry point를 확인했다.

```bash
COMPONENT_IMAGE="localhost:5000/pertinence-components@sha256:a0af60de9aac29401186e1279d7dfc12c277b008106c32d7e84d6256c706c567"

minikube ssh -p pertinence -- \
  "docker run --rm '${COMPONENT_IMAGE}' --help"
```

Node 관점의 registry 주소는 `localhost:5000`이다. Docker driver가 호스트에 공개하는 port는
profile을 만들 때 달라질 수 있으므로 pipeline image 주소에 호스트 port를 넣지 않는다.

현재 Dockerfile은 CPU pipeline image에도 root `requirements.lock`을 설치한다. 이 lock에는
CUDA PyTorch wheel이 포함되어 image가 약 6 GB의 큰 layer를 갖는다. 첫 build, node load,
registry push가 오래 걸리는 주된 이유다.

## 5. Pipeline IR compile

현재 설치에는 CPU용 IR을 생성했다.

```bash
mkdir -p artifacts/cifar10/portable

COMPONENT_IMAGE="localhost:5000/pertinence-components@sha256:a0af60de9aac29401186e1279d7dfc12c277b008106c32d7e84d6256c706c567"

.venv/bin/python -m pipelines.pertinence.pipeline \
  --image "${COMPONENT_IMAGE}" \
  --output artifacts/cifar10/portable/pipeline.yaml
```

IR에 다섯 component가 동일한 digest image를 사용하는지 확인한다.

```bash
rg -n "image:|pertinence-portable-dispatchers|nvidia.com/gpu" \
  artifacts/cifar10/portable/pipeline.yaml
```

GPU IR은 `--device cuda`로 별도 compile해야 한다. 다만 현재 image에는 CPU용
`onnxruntime`이 설치되어 있으므로 GPU IR만 생성해서는 CUDA 실행 환경이 완성되지 않는다.

## 6. KFP API 연결과 pipeline upload

KFP API를 로컬 port에 연결한다. 이 명령은 실행 중인 terminal을 점유한다.

```bash
kubectl port-forward -n kubeflow service/ml-pipeline 8888:8888
```

다른 terminal에서 health endpoint와 SDK 연결을 확인한다.

```bash
curl --fail http://127.0.0.1:8888/apis/v2beta1/healthz

.venv/bin/python -c \
  "import kfp; print(kfp.Client(host='http://127.0.0.1:8888').get_kfp_healthz())"
```

Compile한 IR을 upload한다.

```bash
.venv/bin/python - <<'PY'
import kfp

client = kfp.Client(host="http://127.0.0.1:8888")
pipeline = client.upload_pipeline(
    pipeline_package_path="artifacts/cifar10/portable/pipeline.yaml",
    pipeline_name="pertinence-portable-dispatchers",
    description="Portable PERTINENCE dispatcher pipeline; local Minikube CPU environment",
)
print(pipeline)
PY
```

최초 upload에서 생성된 ID는 다음과 같다.

- Pipeline: `fb973808-210c-49ca-859a-19517244174c`
- Version: `5bf90f9d-177f-4a8d-9160-2eaa7a6c1dd6`

새 클러스터에서는 ID가 달라진다. 같은 이름을 다시 upload할 때는 새 pipeline을 만들지,
기존 pipeline의 새 version을 만들지 명시적으로 선택한다.

## 7. UI 접속

UI도 port-forward로만 노출했다. 인증 없는 dev UI를 NodePort나 외부 interface에 공개하지 않았다.

```bash
kubectl port-forward -n kubeflow service/ml-pipeline-ui 8080:80
```

브라우저에서 다음 주소를 연다.

```text
http://127.0.0.1:8080/
```

Port-forward는 process가 종료되거나 Minikube가 재시작되면 끊어진다. 필요할 때 API와 UI
port-forward를 다시 실행한다.

## 8. 최종 검증

전체 workload 상태와 실제 실행 주체의 RBAC를 확인했다.

```bash
kubectl get pods -A -o wide
kubectl get pvc -n kubeflow

kubectl auth can-i \
  --as=system:serviceaccount:kubeflow:ml-pipeline \
  create workflows.argoproj.io \
  -n kubeflow

kubectl auth can-i \
  --as=system:serviceaccount:kubeflow:argo \
  create pods \
  -n kubeflow

kubectl auth can-i \
  --as=system:serviceaccount:kubeflow:pipeline-runner \
  get secrets \
  -n kubeflow
```

세 결과는 모두 `yes`였다. `pipeline-runner`가 Workflow를 직접 생성할 권한은 없어도 된다.
Workflow 생성은 `ml-pipeline`, task Pod 생성은 Argo controller가 담당한다.

## 9. 현재 남은 실행 조건

Pipeline 정의와 runtime은 upload했지만 실제 학습 Run은 제출하지 않았다.

1. `artifacts/`에 기존 CIFAR raw data와 PyTorch checkpoint는 있지만 pipeline 공개 입력인
   `DatasetBundle`, ONNX `ExpertBundle`, `RunConfig`가 아직 없다.
2. 세 입력은 cluster에서 접근할 수 있는 immutable object URI와 SHA-256으로 준비해야 한다.
   KFP에 포함된 SeaweedFS를 artifact store로 사용할 수 있지만 입력 bundle upload 절차는 아직
   이 저장소에 구현되어 있지 않다.
3. `register-pareto-dispatchers`의 기본 SQLite 경로 `/registry/registry.db`에는 PVC가 mount되지
   않는다. 현 상태로 실행하면 registration report는 KFP artifact로 남지만 SQLite DB는 task Pod와
   함께 사라진다.
4. 이 구성은 CPU IR이다. CUDA를 사용하려면 GPU ONNX Runtime을 포함한 별도 immutable image와
   `--device cuda` IR이 필요하다.

따라서 현재 상태는 pipeline compile, image pull, upload, UI/API 개발을 할 수 있는 로컬 환경이며,
end-to-end Run 전에는 입력 bundle 배치와 Registry 영속화를 추가해야 한다.

## 10. 재시작 후 점검

```bash
minikube start -p pertinence
kubectl config use-context pertinence
kubectl get pods -n kubeflow
kubectl get pods -n kube-system -l kubernetes.io/minikube-addons=registry
kubectl get deployment proxy-agent -n kubeflow
```

`proxy-agent`가 다시 replica 1이 되었고 Pod Security 오류가 발생하면 replica 0으로 되돌린다.
MySQL과 SeaweedFS PVC가 기존 volume에 다시 bind되는지도 확인한다.

## 11. 제거

다음 명령은 pipeline metadata와 로컬 artifact store를 포함한 KFP 리소스를 제거하므로 필요한
결과를 먼저 백업한 뒤에만 실행한다.

```bash
KFP_VERSION=2.17.0

kubectl delete -k \
  "github.com/kubeflow/pipelines/manifests/kustomize/env/dev?ref=${KFP_VERSION}"

kubectl delete -k \
  "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=${KFP_VERSION}"

minikube addons disable registry -p pertinence
```

참고: [Kubeflow Pipelines standalone 설치 문서](https://www.kubeflow.org/docs/components/pipelines/operator-guides/installation/)
