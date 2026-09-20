# PERTINENCE portable pipeline

이 package는 dataset-independent component와 GCS 기반 KFP DAG를 정의한다. 재사용 가능한
contract, ONNX runtime, cache, 학습과 평가는 `src/pertinence/`에 있고, 이 디렉터리는 container
interface, orchestration, image build와 compile/submit client만 소유한다.

```text
pipelines/pertinence/
├── components.py                 # 일곱 KFP v2 container component
├── pipeline.py                   # 고정 7단계 DAG와 IR compiler
├── submit.py                     # OSS KFP API submit client
├── Dockerfile                    # 공용 CPU component image
├── cloudbuild.yaml               # Artifact Registry build
├── requirements-components.lock # image runtime pins
├── requirements-kfp.lock        # compiler/client pins
└── tests/                        # component E2E와 IR 구조 검사
```

GKE, Workload Identity, launcher ConfigMap과 production 운영 절차는
[cloud runbook](../../cloud/docs/runbook.ko.md)이 정본이다. Repository-owned Kubernetes manifest는
`cloud/kfp/`에 둔다.

## 설치와 검증

Python 3.11을 기준으로 pipeline/KFP 선택 의존성을 설치한다.

```bash
.venv/bin/python -m pip install -r pipelines/pertinence/requirements-kfp.lock
.venv/bin/python -m pip install -e . --no-deps
.venv/bin/python -m pytest pipelines/pertinence/tests
.venv/bin/ruff check .
```

Component image는 CPU PyTorch와 CPU ONNX Runtime을 사용하고 KFP SDK를 포함하지 않는다. Unit
test는 다운로드 없이 synthetic PNG/Parquet와 실제 ONNX model을 만들어 class 3/10/100,
expert 2/3/5, 여러 feature dimension에서 validation부터 registration까지 검사한다. 이 test는
실제 CIFAR 결과를 대신하지 않는다.

## 입력 계약

Portable workflow는 세 immutable 입력과 각각의 SHA-256을 받는다.

| 입력 | 내용 |
|---|---|
| `DatasetBundle` | manifest와 train/evaluation Parquet shard |
| `ExpertBundle` | cost-sorted ONNX expert와 입출력·전처리 metadata |
| `RunConfig` | expert 선택, split, dispatcher, NSGA-II 설정 |

Dataset row는 안정적인 `sample_id`, encoded image와 integer target을 가진다. 모든 expert는 같은
class index를 사용하고 동적 batch를 지원해야 한다. Feature extractor에는 `[batch, D]` feature
출력이 필요하다. ONNX external data, custom operator domain, 임의 Python source와 remote model
endpoint는 지원하지 않는다.

Directory digest는 정렬된 `{relative POSIX path: file SHA-256}` mapping의 canonical JSON digest다.
Package와 validation 단계는 symlink, path escape, hash mismatch, 중복 sample ID, split overlap,
shape/class/cost 불일치를 거부한다.

## 로컬 component 실행

먼저 bundle을 만들고 출력된 digest를 기록한다.

```bash
.venv/bin/pertinence package-dataset \
  --source artifacts/cifar10/portable/dataset-source \
  --output artifacts/cifar10/portable/dataset-bundle

.venv/bin/pertinence package-experts \
  --source artifacts/cifar10/portable/expert-source \
  --output artifacts/cifar10/portable/expert-bundle

.venv/bin/pertinence artifact-sha256 artifacts/cifar10/portable/run.yaml
```

처리 단계는 명시적 file/directory input과 output만 사용한다.

```bash
.venv/bin/pertinence validate-run \
  --dataset-bundle-uri artifacts/cifar10/portable/dataset-bundle \
  --dataset-bundle-sha256 "$DATASET_SHA256" \
  --expert-bundle-uri artifacts/cifar10/portable/expert-bundle \
  --expert-bundle-sha256 "$EXPERT_SHA256" \
  --run-config-uri artifacts/cifar10/portable/run.yaml \
  --run-config-sha256 "$RUN_CONFIG_SHA256" \
  --normalized-contract artifacts/cifar10/portable/contract.json \
  --split-manifest artifacts/cifar10/portable/split.json \
  --validation-report artifacts/cifar10/portable/validation.json

.venv/bin/pertinence build-routing-dataset \
  --dataset-bundle artifacts/cifar10/portable/dataset-bundle \
  --expert-bundle artifacts/cifar10/portable/expert-bundle \
  --normalized-contract artifacts/cifar10/portable/contract.json \
  --split-manifest artifacts/cifar10/portable/split.json \
  --train-cache artifacts/cifar10/portable/train-cache \
  --search-cache artifacts/cifar10/portable/search-cache \
  --final-cache artifacts/cifar10/portable/final-cache

.venv/bin/pertinence train-dispatcher \
  --train-cache artifacts/cifar10/portable/train-cache \
  --search-cache artifacts/cifar10/portable/search-cache \
  --normalized-contract artifacts/cifar10/portable/contract.json \
  --search-result artifacts/cifar10/portable/search-result.json \
  --pareto-bundle artifacts/cifar10/portable/pareto

.venv/bin/pertinence evaluate-final \
  --final-cache artifacts/cifar10/portable/final-cache \
  --search-result artifacts/cifar10/portable/search-result.json \
  --pareto-bundle artifacts/cifar10/portable/pareto \
  --normalized-contract artifacts/cifar10/portable/contract.json \
  --final-report artifacts/cifar10/portable/final-report.json \
  --evaluated-bundle artifacts/cifar10/portable/evaluated

.venv/bin/pertinence register-pareto-dispatchers \
  --evaluated-bundle artifacts/cifar10/portable/evaluated \
  --registry-db artifacts/cifar10/portable/registry.sqlite3 \
  --registration-report artifacts/cifar10/portable/registration.json
```

Outputs are written atomically where the interface supports it and existing non-empty bundle directories are
not overwritten. A failed multi-output stage must be retried in a new output directory. The registry is a
run-scoped SQLite artifact with idempotent batch registration; it is not a central model registry.

Search receives only train/search cache and normalized contract. Final cache is absent from its interface.
Final restores every retained checkpoint without retraining or final-metric reselection.

## KFP DAG

The production DAG has seven container tasks:

```text
prepare-dataset-bundle ─┐
export-expert-bundle ───┴─ validate-run ─ build-routing-dataset
  ─ train-dispatcher ─ evaluate-final ─ register-dispatchers
```

Five KFP importer tasks register the external GCS inputs. The seven container executors use
`pipeline-pool` selector+toleration; importer and driver/root Pods may remain on `system-pool`.
All tasks are CPU-only. Search has no retry, final has one retry, and registration caching is always disabled.

Compile without cluster access:

```bash
.venv/bin/python -m pipelines.pertinence.pipeline \
  --image "registry/name@sha256:<64-hex-digest>" \
  --pipeline-root gs://bucket/pertinence/runs \
  --output artifacts/cifar10/portable/pipeline.yaml
```

Image tags are rejected; an immutable manifest digest is mandatory. KFP task caching defaults to off and may
be enabled only for a recovery IR whose input object bytes and image digest remain immutable.

Submit through a local port-forward to the single-user OSS KFP API:

```bash
.venv/bin/python -m pipelines.pertinence.submit \
  --host http://127.0.0.1:8888 \
  --pipeline artifacts/cifar10/portable/pipeline.yaml \
  --pipeline-root gs://RUN_BUCKET/pertinence/runs \
  --namespace kubeflow \
  --service-account pipeline-runner \
  --cifar10-uri gs://RAW_BUCKET/pertinence/raw/cifar-10-python.tar.gz \
  --cifar10-sha256 "$CIFAR10_SHA256" \
  --shufflenet-checkpoint-uri gs://RAW_BUCKET/pertinence/raw/cifar10_shufflenetv2_x0_5.pt \
  --shufflenet-checkpoint-sha256 "$SHUFFLENET_SHA256" \
  --mobilenet-checkpoint-uri gs://RAW_BUCKET/pertinence/raw/cifar10_mobilenetv2_x0_5.pt \
  --mobilenet-checkpoint-sha256 "$MOBILENET_SHA256" \
  --resnet56-checkpoint-uri gs://RAW_BUCKET/pertinence/raw/cifar10_resnet56.pt \
  --resnet56-checkpoint-sha256 "$RESNET56_SHA256" \
  --run-config-uri gs://RAW_BUCKET/pertinence/config/pipeline-run.yaml \
  --run-config-sha256 "$RUN_CONFIG_SHA256"
```

For project IDs, bucket IAM, image build, smoke gates, preservation and incident handling, follow the
[GKE runbook](../../cloud/docs/runbook.ko.md).
