# PERTINENCE Kubeflow pipeline

이 디렉터리는 Kubeflow와 클라우드 운영에만 필요한 코드를 한곳에 모은다.
데이터 계약, ONNX runtime, 학습·평가 같은 재사용 가능한 구현은 `src/pertinence/`에 둔다.

```text
pipelines/pertinence/
├── components.py                 # KFP v2 CPU container component 정의
├── pipeline.py                   # GCS 기반 7단계 DAG 및 IR compiler
├── submit.py                     # GKE의 OSS KFP API에 run 제출
├── Dockerfile                    # 일곱 CPU component 공용 이미지
├── Dockerfile.dockerignore
├── cloudbuild.yaml               # Artifact Registry image build
├── gke/                          # GKE OSS KFP backend manifest
├── requirements-components.lock # container runtime 의존성
├── requirements-kfp.lock        # 로컬 KFP compiler 의존성
├── docs/                         # 설계, 최종 계획, 실행 및 cloud 운영 문서
└── tests/                        # component E2E와 KFP IR 구조 테스트
```

로컬 component 사용법은 [docs/local-components.ko.md](docs/local-components.ko.md)를 참고한다.
Minikube에 KFP를 설치하고 image build부터 pipeline upload까지 재현하는 방법은
[docs/local-kfp-cluster.ko.md](docs/local-kfp-cluster.ko.md)를 참고한다.
GKE에서 OSS KFP backend를 실행하는 cloud 구성은
[docs/gke-cpu.ko.md](docs/gke-cpu.ko.md)를 참고한다.
다음 명령은 cluster 접속 없이 KFP IR만 생성한다.

```bash
.venv/bin/python -m pipelines.pertinence.pipeline \
  --image registry/name@sha256:<64-hex-digest> \
  --pipeline-root gs://bucket/pipeline-root \
  --disable-cache \
  --output artifacts/cifar10/portable/pipeline.yaml
```
