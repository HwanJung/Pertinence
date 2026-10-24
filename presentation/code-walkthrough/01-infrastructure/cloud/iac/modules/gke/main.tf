locals {
  default_node_pools = {
    system-pool = {
      enabled            = true
      machine_type       = "e2-standard-4"
      min_count          = 1
      max_count          = 1
      initial_node_count = 1
      disk_size_gb       = 50
      disk_type          = "pd-standard"
      vcpu_per_node      = 4
    }
    pipeline-pool = {
      enabled            = true
      machine_type       = "n2-standard-8"
      min_count          = 0
      max_count          = 1
      initial_node_count = 0
      disk_size_gb       = 50
      disk_type          = "pd-standard"
      vcpu_per_node      = 8
    }
    serving-pool = {
      enabled            = false
      machine_type       = "n2-standard-8"
      min_count          = 0
      max_count          = 3
      initial_node_count = 0
      disk_size_gb       = 50
      disk_type          = "pd-standard"
      vcpu_per_node      = 8
    }
    loadgen-pool = {
      enabled            = false
      machine_type       = "n2-standard-4"
      min_count          = 0
      max_count          = 1
      initial_node_count = 0
      disk_size_gb       = 50
      disk_type          = "pd-standard"
      vcpu_per_node      = 4
    }
  }

  node_pools = {
    for name, defaults in local.default_node_pools : name => {
      enabled = coalesce(try(var.node_pools[name].enabled, null), defaults.enabled)
      machine_type = coalesce(
        try(var.node_pools[name].machine_type, null),
        defaults.machine_type,
      )
      min_count = coalesce(try(var.node_pools[name].min_count, null), defaults.min_count)
      max_count = coalesce(try(var.node_pools[name].max_count, null), defaults.max_count)
      initial_node_count = coalesce(
        try(var.node_pools[name].initial_node_count, null),
        defaults.initial_node_count,
      )
      disk_size_gb = coalesce(
        try(var.node_pools[name].disk_size_gb, null),
        defaults.disk_size_gb,
      )
      disk_type = coalesce(try(var.node_pools[name].disk_type, null), defaults.disk_type)
      vcpu_per_node = coalesce(
        try(var.node_pools[name].vcpu_per_node, null),
        defaults.vcpu_per_node,
      )
    }
  }

  active_node_pools = {
    for name, pool in local.node_pools : name => pool if pool.enabled
  }

  pool_taints = {
    system-pool   = null
    pipeline-pool = { key = "workload", value = "pertinence-pipeline", effect = "NO_SCHEDULE" }
    serving-pool  = { key = "workload", value = "pertinence-serving", effect = "NO_SCHEDULE" }
    loadgen-pool  = { key = "workload", value = "pertinence-loadgen", effect = "NO_SCHEDULE" }
  }

  aggregate_max_vcpus = sum(concat(
    [0],
    [for pool in values(local.active_node_pools) : pool.max_count * pool.vcpu_per_node],
  ))
  system_pool_enabled = local.node_pools["system-pool"].enabled
  artifact_registry_location = coalesce(
    var.artifact_registry_repository_location,
    var.region,
  )
}

resource "google_project_service" "required" {
  for_each = var.enable_apis ? var.required_apis : toset([])

  project                    = var.project_id
  service                    = each.value
  disable_on_destroy         = false
  disable_dependent_services = false
}

data "google_compute_network" "selected" {
  name    = var.network
  project = var.project_id
}

data "google_compute_subnetwork" "selected" {
  name    = var.subnetwork
  project = var.project_id
  region  = var.region
}

resource "google_service_account" "nodes" {
  project      = var.project_id
  account_id   = var.node_service_account_id
  display_name = var.node_service_account_display_name

  depends_on = [google_project_service.required]
}

resource "google_project_iam_member" "node" {
  for_each = var.node_project_roles

  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.nodes.email}"
}

resource "google_artifact_registry_repository_iam_member" "node_reader" {
  project    = var.project_id
  location   = local.artifact_registry_location
  repository = var.artifact_registry_repository
  role       = "roles/artifactregistry.reader"
  member     = "serviceAccount:${google_service_account.nodes.email}"

  depends_on = [google_project_service.required]
}

resource "google_container_cluster" "this" {
  project  = var.project_id
  name     = var.cluster_name
  location = var.zone

  network    = data.google_compute_network.selected.self_link
  subnetwork = data.google_compute_subnetwork.selected.self_link

  networking_mode          = "VPC_NATIVE"
  remove_default_node_pool = true
  initial_node_count       = 1
  deletion_protection      = var.deletion_protection
  enable_shielded_nodes    = true
  resource_labels          = var.resource_labels

  release_channel {
    channel = "REGULAR"
  }

  workload_identity_config {
    workload_pool = "${var.project_id}.svc.id.goog"
  }

  ip_allocation_policy {
    cluster_secondary_range_name  = var.pod_secondary_range_name
    services_secondary_range_name = var.service_secondary_range_name
  }

  addons_config {
    dns_cache_config {
      enabled = true
    }

    gce_persistent_disk_csi_driver_config {
      enabled = true
    }
  }

  logging_config {
    enable_components = ["SYSTEM_COMPONENTS", "WORKLOADS"]
  }

  monitoring_config {
    enable_components = [
      "SYSTEM_COMPONENTS",
      "STORAGE",
      "HPA",
      "POD",
      "DAEMONSET",
      "DEPLOYMENT",
      "STATEFULSET",
      "CADVISOR",
      "KUBELET",
      "DCGM",
      "JOBSET",
    ]

    managed_prometheus {
      enabled = true
    }
  }

  lifecycle {
    # GKE reports zero after the temporary default pool has been removed. The value is used only
    # while creating a fresh cluster and must never force replacement of an imported cluster.
    # One-off experiment maintenance exclusions are also intentionally outside this base state.
    ignore_changes = [initial_node_count, remove_default_node_pool, maintenance_policy]

    precondition {
      condition = local.system_pool_enabled || var.allow_without_system_pool
      error_message = (
        "Disabling system-pool requires allow_without_system_pool=true."
      )
    }

    precondition {
      condition = length(local.active_node_pools) > 0 || var.allow_control_plane_only
      error_message = (
        "Disabling every node pool requires allow_control_plane_only=true."
      )
    }

    precondition {
      condition = alltrue([
        for pool in values(local.node_pools) :
        pool.min_count <= pool.initial_node_count && pool.initial_node_count <= pool.max_count
      ])
      error_message = "Each node pool must satisfy min_count <= initial_node_count <= max_count."
    }

    precondition {
      condition = (
        local.node_pools["system-pool"].min_count == 1 &&
        local.node_pools["system-pool"].max_count == 1 &&
        local.node_pools["system-pool"].initial_node_count == 1
      )
      error_message = "system-pool is fixed at exactly one node, even when it is disabled."
    }

    precondition {
      condition     = local.aggregate_max_vcpus <= var.vcpu_quota_budget
      error_message = "Enabled pools exceed vcpu_quota_budget at their configured maximum sizes."
    }
  }

  depends_on = [
    google_project_service.required,
    google_project_iam_member.node,
  ]
}

resource "google_container_node_pool" "this" {
  for_each = local.active_node_pools

  project            = var.project_id
  name               = each.key
  location           = var.zone
  cluster            = google_container_cluster.this.name
  node_count         = each.key == "system-pool" ? 1 : null
  initial_node_count = each.key == "system-pool" ? null : each.value.initial_node_count
  deletion_policy    = var.node_pool_deletion_policy

  dynamic "autoscaling" {
    for_each = each.key == "system-pool" ? [] : [each.value]

    content {
      min_node_count  = autoscaling.value.min_count
      max_node_count  = autoscaling.value.max_count
      location_policy = "BALANCED"
    }
  }

  management {
    auto_repair  = true
    auto_upgrade = true
  }

  node_config {
    machine_type    = each.value.machine_type
    disk_size_gb    = each.value.disk_size_gb
    disk_type       = each.value.disk_type
    image_type      = "COS_CONTAINERD"
    service_account = google_service_account.nodes.email
    oauth_scopes    = ["https://www.googleapis.com/auth/cloud-platform"]
    spot            = false

    workload_metadata_config {
      mode = "GKE_METADATA"
    }

    dynamic "taint" {
      for_each = local.pool_taints[each.key] == null ? [] : [local.pool_taints[each.key]]

      content {
        key    = taint.value.key
        value  = taint.value.value
        effect = taint.value.effect
      }
    }
  }

  lifecycle {
    # GKE/provider imports expose existing taints through effective_taints but leave taint empty.
    # Fresh pools still receive the declared taint at creation. The postcondition checks the
    # provider's effective live value so an imported pool cannot silently lose its workload taint.
    ignore_changes = [node_config[0].taint]

    postcondition {
      condition = local.pool_taints[each.key] == null ? true : contains(
        [
          for taint in self.node_config[0].effective_taints :
          "${taint.key}:${taint.value}:${taint.effect}"
        ],
        join(":", [
          local.pool_taints[each.key].key,
          local.pool_taints[each.key].value,
          local.pool_taints[each.key].effect,
        ]),
      )
      error_message = "${each.key} is missing its required effective workload taint."
    }
  }
}
