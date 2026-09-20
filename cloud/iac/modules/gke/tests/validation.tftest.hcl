mock_provider "google" {}

variables {
  project_id                   = "example-project"
  region                       = "asia-northeast3"
  zone                         = "asia-northeast3-a"
  cluster_name                 = "pertinence-test"
  network                      = "default"
  subnetwork                   = "default"
  artifact_registry_repository = "pertinence"
}

run "system_requires_acknowledgement" {
  command = plan

  variables {
    node_pools = {
      system-pool = { enabled = false }
    }
  }

  expect_failures = [google_container_cluster.this]
}

run "control_plane_only_requires_acknowledgement" {
  command = plan

  variables {
    allow_without_system_pool = true
    node_pools = {
      system-pool   = { enabled = false }
      pipeline-pool = { enabled = false }
      serving-pool  = { enabled = false }
      loadgen-pool  = { enabled = false }
    }
  }

  expect_failures = [google_container_cluster.this]
}

run "invalid_node_count_range" {
  command = plan

  variables {
    node_pools = {
      pipeline-pool = {
        min_count          = 1
        initial_node_count = 0
        max_count          = 1
      }
    }
  }

  expect_failures = [google_container_cluster.this]
}

run "vcpu_budget_is_enforced" {
  command = plan

  variables {
    node_pools = {
      loadgen-pool = { enabled = true }
    }
  }

  expect_failures = [google_container_cluster.this]
}

run "unknown_pool_is_rejected" {
  command = plan

  variables {
    node_pools = {
      gpu-pool = { enabled = true }
    }
  }

  expect_failures = [var.node_pools]
}

run "system_pool_size_is_fixed" {
  command = plan

  variables {
    node_pools = {
      system-pool = {
        min_count          = 0
        initial_node_count = 0
      }
    }
  }

  expect_failures = [google_container_cluster.this]
}

run "explicit_control_plane_only" {
  command = plan

  variables {
    allow_without_system_pool = true
    allow_control_plane_only  = true
    node_pools = {
      system-pool   = { enabled = false }
      pipeline-pool = { enabled = false }
      serving-pool  = { enabled = false }
      loadgen-pool  = { enabled = false }
    }
  }

  assert {
    condition     = length(google_container_node_pool.this) == 0
    error_message = "An explicitly acknowledged control-plane-only cluster must have no node pools."
  }
}
