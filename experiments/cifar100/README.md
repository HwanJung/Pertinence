# CIFAR-100 experiment

This directory is the exclusive home of the new CIFAR-100 experiment.

The model-zoo catalog, Pareto calculation, and initial figures are available
now. Dataset configuration, asset manifests, runbooks, compact results, and
additional analysis should be added to the corresponding subdirectories here
as the CIFAR-100 pipeline is implemented.

Runtime files must use `artifacts/cifar100/`; CIFAR-10 assets and caches under
`artifacts/cifar10/` are never valid inputs for this experiment.

Regenerate the current catalog analysis from the repository root:

```bash
python3 experiments/cifar100/scripts/build_model_pareto.py
```
