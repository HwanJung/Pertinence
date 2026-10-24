# Preserved cluster plus the two benchmark-only pools. Apply only through a saved plan accepted by
# terraform_plan_guard.py --mode benchmark-pools-create. KFP remains installed; do not run KFP
# workflows while benchmarking, and require pipeline-pool to be scaled to zero before measurement.
vcpu_quota_budget = 24

node_pools = {
  system-pool = {
    enabled            = true
    machine_type       = "e2-standard-4"
    min_count          = 1
    max_count          = 1
    initial_node_count = 1
    vcpu_per_node      = 4
  }
  pipeline-pool = {
    enabled            = true
    machine_type       = "n2-standard-8"
    min_count          = 0
    max_count          = 1
    initial_node_count = 0
    vcpu_per_node      = 8
  }
  serving-pool = {
    enabled            = true
    machine_type       = "n2-standard-8"
    min_count          = 0
    max_count          = 1
    initial_node_count = 0
    vcpu_per_node      = 8
  }
  loadgen-pool = {
    enabled            = true
    machine_type       = "n2-standard-4"
    min_count          = 0
    max_count          = 1
    initial_node_count = 0
    vcpu_per_node      = 4
  }
}

deletion_protection       = true
node_pool_deletion_policy = "PREVENT"
