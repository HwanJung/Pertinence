# Pertinence GKE module

This module provisions a zonal Standard GKE cluster on an existing VPC/subnet, a dedicated node
service account, additive node IAM bindings, and up to four separately managed node pools. It does
not install KFP/KServe, manage Kubernetes resources, or own the VPC.

The defaults create `system-pool` and `pipeline-pool` only. `serving-pool` and `loadgen-pool` are
opt-in. Workload pools receive dedicated `NoSchedule` taints and may scale to zero; the untainted
system pool is fixed at one node. Imported pools expose their existing taints only through the
provider's computed `effective_taints`, so the module ignores the write-only taint representation
but fails its resource postcondition if the required effective workload taint is missing.

```hcl
module "gke" {
  source = "../../modules/gke"

  project_id                  = "example-project"
  region                      = "asia-northeast3"
  zone                        = "asia-northeast3-a"
  cluster_name                = "pertinence-dispatcher"
  network                     = "default"
  subnetwork                  = "default"
  artifact_registry_repository = "pertinence"

  node_pools = {
    loadgen-pool = { enabled = true }
  }

  # system(4) + pipeline(8) + loadgen(4)
  vcpu_quota_budget = 16
}
```

To omit `system-pool`, set both its `enabled` override and `allow_without_system_pool = true`.
Disabling every pool additionally requires `allow_control_plane_only = true`. Such configurations
are intentionally reported as unsuitable for KFP/KServe add-ons.

Clusters default to deletion protection and node pools default to `deletion_policy = "PREVENT"`.
To remove a pool, first apply `node_pool_deletion_policy = "DELETE"` while the pool is still enabled,
then disable it in a second reviewed apply. A cluster destroy similarly requires a separate apply
with `deletion_protection = false`.

Existing resources must be imported before the first apply. In particular, import the cluster,
enabled node pools, service account, every managed API and additive IAM member, then require a
zero-change saved plan before changing protection settings.

One-off maintenance exclusions are experiment runtime state rather than base-cluster state. The
Google provider requires a recurring or daily window when exclusions are declared, while the
preserved cluster has only a temporary exclusion, so this module deliberately leaves it untouched.
