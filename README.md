# PERTINENCE CIFAR experiments

This repository implements the method in **PERTINENCE: Input-based
Opportunistic Neural Network Dynamic Execution** (arXiv:2507.01695v3) for
CIFAR-10. It retains the ten non-dominated checkpoints from the current
`chenyaofo/pytorch-cifar-models` CIFAR-10 catalog as the static accuracy-cost
boundary, while the dynamic dispatcher routes among four representative
experts.

The completed CIFAR-10 experiment and the new CIFAR-100 work are isolated under
`experiments/cifar10/` and `experiments/cifar100/`. Shared PERTINENCE code stays
under `src/pertinence/`; generated assets, caches, and runs use the matching
`artifacts/<dataset>/` directory.

## Repository layout

```text
experiments/
├── cifar10/             # config, catalog, experiment docs, figures, analysis scripts
└── cifar100/            # CIFAR-100 catalog and all new experiment-specific work
src/pertinence/          # shared routing, training, optimization, cache, and metrics code
scripts/                 # shared command-line entry points
artifacts/
├── cifar10/             # downloaded assets, caches, and completed runs (git-ignored)
└── cifar100/            # reserved for CIFAR-100 runtime outputs (git-ignored)
```

Dataset-specific files must not be added back to root-level `configs/`,
`data/`, `docs/`, or `figures/` directories.

## Scope

- Executable reproduction pipeline: CIFAR-10. CIFAR-100 adaptation is staged in
  its own experiment directory.
- Static Pareto boundary: the ten cost-sorted models in
  `experiments/cifar10/data/model_catalog.csv` with `pareto=True`.
- Routing experts: `shufflenetv2_x0_5`, `mobilenetv2_x0_5`, `resnet56`, and
  `repvgg_a1`.
- Dispatcher: frozen `shufflenetv2_x0_5` features followed by one trainable
  linear layer.
- Labels: cheapest expert that classifies each sample correctly; expert 0 is
  the no-correct fallback.
- Loss: the literal argmax-indexed asymmetric penalty loss from Eq. (6)-(7),
  with INS, ISNS, or ENS weighting.
- Search: NSGA-II with the paper's population, generation, SBX, mutation, and
  per-individual epoch defaults.
- Cost: catalog MAdds converted with the single convention `1 MAC = 2 FLOPs`;
  dispatcher overhead is included and extractor/expert computation is shared.

This is a method-faithful adaptation, not a numeric reproduction of Figure
6(c). The paper used unavailable ResNet8/14 checkpoints and dominated VGG
models; the requested current Pareto front is a different expert pool.

## Environment

Python 3.11 is the canonical interpreter. The pinned package set also supports
Python 3.12.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.lock
.venv/bin/python -m pip install -e . --no-deps
source .venv/bin/activate
```

Downloaded files and run outputs live under `artifacts/<dataset>/` and are
intentionally git-ignored. Network access occurs only in the explicit asset
preparation command; model loading never calls PyTorch Hub or a URL.

## Staged workflow

The dataset-independent Kubeflow components also run locally with Parquet/ONNX
bundles. They provide contract validation, separate routing caches, dispatcher
search, final evaluation, and transactional SQLite registration. See the
[local component guide](pipelines/pertinence/docs/local-components.ko.md)
for packaging, CLI commands, and optional KFP IR compilation. No cluster setup
is required for these local components.

Run the read-only preflight before GPU inference. It verifies pinned package
versions, every asset hash, extracted source bytes, CIFAR-10 sizes, free disk,
the configured device, all four checkpoint loads, tiny real-model forwards,
and the planned cache fingerprints. Do not use a CPU override for a production
run: it is only useful for local asset smoke testing.

```bash
# 1. Download or verify CIFAR-10, pinned source, and the boundary-model assets.
python scripts/prepare_assets.py --manifest experiments/cifar10/configs/assets.yaml --artifacts-dir artifacts/cifar10
python scripts/prepare_assets.py --manifest experiments/cifar10/configs/assets.yaml --artifacts-dir artifacts/cifar10 --verify-only

# Extract only after the archive hashes have passed verification.
tar -xzf artifacts/cifar10/datasets/cifar-10-python.tar.gz -C artifacts/cifar10/datasets
tar -xzf artifacts/cifar10/sources/pytorch-cifar-models-786c16252c0fc58ee9adac063f8337cc4a7a497a.tar.gz -C artifacts/cifar10/sources

# 2. Read-only production preflight. This must report status=passed on CUDA.
python scripts/preflight.py \
  --config experiments/cifar10/configs/experiment.yaml \
  --manifest experiments/cifar10/configs/assets.yaml \
  --artifacts-dir artifacts/cifar10 \
  --dataset-dir artifacts/cifar10/datasets \
  --source-dir artifacts/cifar10/sources/pytorch-cifar-models-786c16252c0fc58ee9adac063f8337cc4a7a497a \
  --cache-dir artifacts/cifar10/cache/pareto

# 3. GPU inference: validate the four routing experts and build immutable caches.
python scripts/build_cache.py \
  --config experiments/cifar10/configs/experiment.yaml \
  --manifest experiments/cifar10/configs/assets.yaml \
  --artifacts-dir artifacts/cifar10 \
  --dataset-dir artifacts/cifar10/datasets \
  --source-dir artifacts/cifar10/sources/pytorch-cifar-models-786c16252c0fc58ee9adac063f8337cc4a7a497a \
  --cache-dir artifacts/cifar10/cache/pareto

# 4. Check train/search baselines and oracle without opening final holdout.
pertinence baseline \
  --config experiments/cifar10/configs/experiment.yaml \
  --cache-dir artifacts/cifar10/cache/pareto \
  --output artifacts/cifar10/runs/pareto/baselines.json

# 5. Train/evaluate one paper-style fixed directional penalty.
pertinence fixed \
  --config experiments/cifar10/configs/experiment.yaml \
  --cache-dir artifacts/cifar10/cache/pareto \
  --output artifacts/cifar10/runs/pareto/fixed.json

# 6. Run the full NSGA-II search (50 x 50 individuals, 20 FC epochs each).
pertinence search \
  --config experiments/cifar10/configs/experiment.yaml \
  --cache-dir artifacts/cifar10/cache/pareto \
  --output artifacts/cifar10/runs/pareto/search.json

# 7. Evaluate only the retained front on the untouched 2,000-image split.
pertinence final \
  --config experiments/cifar10/configs/experiment.yaml \
  --cache-dir artifacts/cifar10/cache/pareto \
  --search-result artifacts/cifar10/runs/pareto/search.json \
  --output artifacts/cifar10/runs/pareto/final.json
```

Cache inference compares every routing expert against its catalog Top-1 on the
complete official 10,000-image test set. The accepted delta is -0.25 to +0.75
percentage points because the catalog records a best-validation training-log
value, while pinned-checkpoint reevaluation can differ slightly by runtime. A
failure occurs before any new cache is committed. This fixed-expert integrity
check is not exposed to NSGA-II; `baseline`, `fixed`, and `search`
load only dispatcher-train and GA-search caches. `final` is the first command
that evaluates a selected dispatcher on the 2,000-image holdout.

For the exact uninterrupted command block, pass/fail gates, rerun behavior,
and long-running search precautions, see
`experiments/cifar10/docs/runbook.ko.md`.

The default split uses all 50,000 official training images for dispatcher
training, a deterministic 8,000-image portion of the official test set for GA
fitness, and the remaining 2,000 images only for final evaluation. The paper
does not publish exact sample IDs, so the generated split manifest and its hash
are part of every cache/run fingerprint.

See `experiments/cifar10/docs/reproduction-plan.ko.md` for the detailed protocol, assumptions,
acceptance gates, and expected artifacts.

The completed experiment analysis, same-split expert comparisons, limitations,
and regenerated final figures are documented in
`experiments/cifar10/docs/results.ko.md`.
