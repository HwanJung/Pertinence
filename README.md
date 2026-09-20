# PERTINENCE CIFAR experiments

This repository implements a reproducible CIFAR adaptation of **PERTINENCE:
Input-based Opportunistic Neural Network Dynamic Execution**. The current
dispatcher routes each input among three frozen experts while preserving the
ten-model CIFAR-10 accuracy–cost boundary as a static comparison.

Two execution paths are intentionally separate:

- the local CUDA workflow builds native PyTorch caches and runs the original CLI stages;
- the verified cloud workflow runs the portable seven-component pipeline on GKE and OSS
  Kubeflow Pipelines (KFP).

The cloud documentation describes what was actually deployed and exercised. Future KServe
and online-serving work is isolated in a roadmap and is not presented as current capability.

## Repository layout

```text
src/pertinence/          # shared contracts, routing, training, metrics, and runtime code
scripts/                 # shared asset, cache, preflight, and Terraform helpers
experiments/
├── method.ko.md         # paper method and project-wide reproduction assumptions
├── cifar10/             # runnable CIFAR-10 configuration, results, figures, and scripts
└── cifar100/            # staged CIFAR-100 catalog work
pipelines/pertinence/    # portable components, KFP DAG, image build, and tests
cloud/
├── docs/                # verified architecture, runbook, run records, and future roadmap
├── iac/                 # protected GKE Terraform state and reusable module
└── kfp/                 # repository-owned KFP launcher and service-account manifests
artifacts/<dataset>/     # downloaded assets, caches, compiled IR, and run outputs (ignored)
```

Dataset-specific generated or versioned material stays below its experiment directory.
Do not recreate root-level `configs/`, `data/`, `docs/`, or `figures/` directories.

## Verified state

The current production configuration uses these routing experts:

1. `shufflenetv2_x0_5` — also the frozen feature extractor and fallback;
2. `mobilenetv2_x0_5`;
3. `resnet56`.

The successful GKE run completed 50 generations × 50 individuals, evaluated 233 retained
dispatchers on the final split, and registered all 233 results. Its immutable identifiers,
metrics, incidents, and preserved resources are recorded in the
[2026-09-20 GKE run report](cloud/docs/runs/2026-09-20-dispatcher-gke.ko.md).

The versioned local result table and final figure predate the three-expert configuration and
belong to a completed four-expert run. They are retained as an explicitly historical result in
[the local four-expert report](experiments/cifar10/docs/local-4expert-results.ko.md), not as the
output of the current config.

## Environment

Python 3.11 is canonical; Python 3.12 is also supported.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock
.venv/bin/python -m pip install -e . --no-deps
```

Downloaded datasets, source archives, checkpoints, caches, and raw run output belong under
`artifacts/<dataset>/`. Network access occurs only during explicit asset preparation or cloud
operations; model loading never downloads code or checkpoints implicitly.

## Local CIFAR-10 reproduction

The local execution contract, uninterrupted command sequence, acceptance gates, retry rules,
and final-holdout boundary are in the
[CIFAR-10 local reproduction guide](experiments/cifar10/docs/local-reproduction.ko.md).

The short path is:

```bash
python scripts/prepare_assets.py \
  --manifest experiments/cifar10/configs/assets.yaml \
  --artifacts-dir artifacts/cifar10

python scripts/preflight.py \
  --config experiments/cifar10/configs/experiment.yaml \
  --manifest experiments/cifar10/configs/assets.yaml \
  --artifacts-dir artifacts/cifar10 \
  --dataset-dir artifacts/cifar10/datasets \
  --source-dir artifacts/cifar10/sources/pytorch-cifar-models-786c16252c0fc58ee9adac063f8337cc4a7a497a \
  --cache-dir artifacts/cifar10/cache/pareto
```

Continue with cache generation, baseline, fixed dispatcher, search, and final evaluation only
after the preflight passes on the configured CUDA device.

## Portable pipeline and GKE

- [Pipeline package README](pipelines/pertinence/README.md): bundle contracts, local component
  execution, IR compilation, and submission interface.
- [Cloud architecture](cloud/docs/architecture.ko.md): deployed boundaries and scheduling.
- [Cloud runbook](cloud/docs/runbook.ko.md): Terraform safety, GCS preparation, image build,
  compile, submit, verification, and preservation procedure.
- [Cloud roadmap](cloud/docs/roadmap.ko.md): explicitly unimplemented KServe and benchmark work.
- [Terraform guide](cloud/iac/README.md): state bootstrap, adoption, plan guards, and profiles.

## Development checks

```bash
.venv/bin/python -m pytest
.venv/bin/ruff check .
terraform fmt -check -recursive cloud/iac
terraform -chdir=cloud/iac/modules/gke test
```

Ordinary unit tests do not require network access or downloaded assets. Tests marked
`integration` require the documented assets and PyTorch environment.
