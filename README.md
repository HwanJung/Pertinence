# PERTINENCE CIFAR-10 reproduction

This repository implements the method in **PERTINENCE: Input-based
Opportunistic Neural Network Dynamic Execution** (arXiv:2507.01695v3) for
CIFAR-10. It retains the ten non-dominated checkpoints from the current
`chenyaofo/pytorch-cifar-models` CIFAR-10 catalog as the static accuracy-cost
boundary, while the dynamic dispatcher routes among four representative
experts.

The experiment itself has deliberately **not** been run. Dataset/model
preparation is separate from GPU inference, dispatcher training, and NSGA-II.

## Scope

- Dataset: CIFAR-10 only.
- Static Pareto boundary: the ten cost-sorted models in
  `data/chenyaofo_cifar10_models.csv` with `pareto=True`.
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

Downloaded files and run outputs live under `artifacts/` and are intentionally
git-ignored. Network access occurs only in the explicit asset preparation
command; model loading never calls PyTorch Hub or a URL.

## Staged workflow

The following commands document the intended order. Do not run stages 3-6 if
you only want to inspect the prepared experiment.

```bash
# 1. Download or verify CIFAR-10, pinned source, and the boundary-model assets.
python scripts/prepare_assets.py --manifest configs/assets.yaml --artifacts-dir artifacts
python scripts/prepare_assets.py --manifest configs/assets.yaml --artifacts-dir artifacts --verify-only

# Extract only after the archive hashes have passed verification.
tar -xzf artifacts/datasets/cifar-10-python.tar.gz -C artifacts/datasets
tar -xzf artifacts/sources/pytorch-cifar-models-786c16252c0fc58ee9adac063f8337cc4a7a497a.tar.gz -C artifacts/sources

# 2. Preview cache work without inference.
python scripts/build_cache.py \
  --config configs/cifar10_pareto.yaml \
  --manifest configs/assets.yaml \
  --artifacts-dir artifacts \
  --dataset-dir artifacts/datasets \
  --source-dir artifacts/sources/pytorch-cifar-models-786c16252c0fc58ee9adac063f8337cc4a7a497a \
  --cache-dir artifacts/cache/cifar10_pareto \
  --dry-run

# 3. GPU inference: validate the four routing experts and build immutable caches.
python scripts/build_cache.py \
  --config configs/cifar10_pareto.yaml \
  --manifest configs/assets.yaml \
  --artifacts-dir artifacts \
  --dataset-dir artifacts/datasets \
  --source-dir artifacts/sources/pytorch-cifar-models-786c16252c0fc58ee9adac063f8337cc4a7a497a \
  --cache-dir artifacts/cache/cifar10_pareto

# 4. Check train/search baselines and oracle without opening final holdout.
pertinence baseline \
  --config configs/cifar10_pareto.yaml \
  --cache-dir artifacts/cache/cifar10_pareto \
  --output artifacts/runs/cifar10_pareto/baselines.json

# 5. Train/evaluate one paper-style fixed directional penalty.
pertinence fixed \
  --config configs/cifar10_pareto.yaml \
  --cache-dir artifacts/cache/cifar10_pareto \
  --output artifacts/runs/cifar10_pareto/fixed.json

# 6. Run the full NSGA-II search (50 x 50 individuals, 20 FC epochs each).
pertinence search \
  --config configs/cifar10_pareto.yaml \
  --cache-dir artifacts/cache/cifar10_pareto \
  --output artifacts/runs/cifar10_pareto/search.json

# 7. Evaluate only the retained front on the untouched 2,000-image split.
pertinence final \
  --config configs/cifar10_pareto.yaml \
  --cache-dir artifacts/cache/cifar10_pareto \
  --search-result artifacts/runs/cifar10_pareto/search.json \
  --output artifacts/runs/cifar10_pareto/final.json
```

The default split uses all 50,000 official training images for dispatcher
training, a deterministic 8,000-image portion of the official test set for GA
fitness, and the remaining 2,000 images only for final evaluation. The paper
does not publish exact sample IDs, so the generated split manifest and its hash
are part of every cache/run fingerprint.

See `docs/reproduction-plan.ko.md` for the detailed protocol, assumptions,
acceptance gates, and expected artifacts.
