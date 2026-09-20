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

run "training_defaults" {
  command = plan

  assert {
    condition = (
      length(google_container_node_pool.this) == 2 &&
      contains(keys(google_container_node_pool.this), "system-pool") &&
      contains(keys(google_container_node_pool.this), "pipeline-pool")
    )
    error_message = "Defaults must create only system-pool and pipeline-pool."
  }

  assert {
    condition     = output.aggregate_max_vcpus == 12
    error_message = "Default pools must fit the 12-vCPU budget exactly."
  }

  assert {
    condition = (
      google_container_node_pool.this["system-pool"].node_config[0].disk_type == "pd-standard" &&
      google_container_node_pool.this["system-pool"].node_config[0].disk_size_gb == 50 &&
      length(google_container_node_pool.this["system-pool"].autoscaling) == 0 &&
      google_container_node_pool.this["pipeline-pool"].node_config[0]
      .workload_metadata_config[0].mode == "GKE_METADATA"
    )
    error_message = "Node pools must retain the reproducible disk and metadata settings."
  }

  assert {
    condition = (
      google_container_cluster.this.deletion_protection &&
      google_container_cluster.this.enable_shielded_nodes &&
      google_container_cluster.this.release_channel[0].channel == "REGULAR"
    )
    error_message = "Cluster safety and release-channel defaults changed."
  }
}

run "loadgen_enabled" {
  command = plan

  variables {
    vcpu_quota_budget = 16
    node_pools = {
      loadgen-pool = { enabled = true }
    }
  }

  assert {
    condition = (
      google_container_node_pool.this["loadgen-pool"]
      .node_config[0].machine_type == "n2-standard-4" &&
      google_container_node_pool.this["loadgen-pool"].autoscaling[0].min_node_count == 0 &&
      google_container_node_pool.this["loadgen-pool"].autoscaling[0].max_node_count == 1 &&
      google_container_node_pool.this["loadgen-pool"]
      .node_config[0].taint[0].value == "pertinence-loadgen"
    )
    error_message = "loadgen-pool must have its documented size, scale range, and taint."
  }
}

run "system_and_loadgen_omitted" {
  command = plan

  variables {
    allow_without_system_pool = true
    node_pools = {
      system-pool  = { enabled = false }
      loadgen-pool = { enabled = false }
    }
  }

  assert {
    condition = (
      !contains(keys(google_container_node_pool.this), "system-pool") &&
      !contains(keys(google_container_node_pool.this), "loadgen-pool") &&
      contains(keys(google_container_node_pool.this), "pipeline-pool") &&
      !output.kfp_addons_supported
    )
    error_message = "system-pool and loadgen-pool must be independently omittable."
  }
}

run "missing_effective_taint_is_rejected" {
  command = apply

  override_resource {
    target = google_container_node_pool.this["pipeline-pool"]
    values = {
      node_config = {
        effective_taints = []
      }
    }
  }

  expect_failures = [
    google_container_node_pool.this["pipeline-pool"],
  ]
}
