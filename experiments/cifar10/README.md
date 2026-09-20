# CIFAR-10 experiment

This directory contains the current three-expert CIFAR-10 reproduction bundle and an explicitly
labelled result from the earlier four-expert local run.

- `configs/`: runnable experiment and asset manifests
- `data/`: the model-zoo catalog and run-labelled compact result tables
- `docs/`: the current local protocol, Pareto analysis, and run-labelled results
- `figures/`: catalog figures and run-labelled experiment figures
- `scripts/`: CIFAR-10-specific catalog and final-result analysis

Runtime files are isolated under `artifacts/cifar10/`. Start with
[`docs/local-reproduction.ko.md`](docs/local-reproduction.ko.md) for the current local workflow.
The verified three-expert cloud result is documented separately in the
[GKE run record](../../cloud/docs/runs/2026-09-20-dispatcher-gke.ko.md).
