# CIFAR-100 configuration staging

Place the runnable CIFAR-100 `experiment.yaml` and immutable `assets.yaml` in
this directory after the shared loader, expert registry, and preflight code
support CIFAR-100.

Both files must point exclusively at `artifacts/cifar100/`. Copying the
CIFAR-10 manifest or referencing `artifacts/cifar10/` is not allowed.
