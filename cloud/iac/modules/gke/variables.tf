variable "project_id" {
  description = "Google Cloud project that owns the cluster and project-level IAM bindings."
  type        = string

  validation {
    condition     = length(trimspace(var.project_id)) > 0
    error_message = "project_id must not be empty."
  }
}

variable "region" {
  description = "Region containing the zonal cluster, subnet, and Artifact Registry repository."
  type        = string
}

variable "zone" {
  description = "Zone in which to create the Standard GKE cluster."
  type        = string

  validation {
    condition     = startswith(var.zone, "${var.region}-")
    error_message = "zone must be located in region."
  }
}

variable "cluster_name" {
  description = "GKE cluster name."
  type        = string

  validation {
    condition     = can(regex("^[a-z][a-z0-9-]{0,38}[a-z0-9]$", var.cluster_name))
    error_message = "cluster_name must be a valid GKE name between 2 and 40 characters."
  }
}

variable "network" {
  description = "Name of an existing VPC network. The module never creates a network."
  type        = string
}

variable "subnetwork" {
  description = "Name of an existing regional subnetwork. The module never creates a subnet."
  type        = string
}

variable "pod_secondary_range_name" {
  description = "Optional existing subnet secondary range used for Pod IP addresses."
  type        = string
  default     = null
  nullable    = true
}

variable "service_secondary_range_name" {
  description = "Optional existing subnet secondary range used for Service IP addresses."
  type        = string
  default     = null
  nullable    = true
}

variable "artifact_registry_repository" {
  description = "Artifact Registry repository from which GKE nodes may pull images."
  type        = string
}

variable "artifact_registry_repository_location" {
  description = "Artifact Registry repository location; null uses region."
  type        = string
  default     = null
  nullable    = true
}

variable "node_service_account_id" {
  description = "Account ID of the dedicated GKE node service account."
  type        = string
  default     = "pertinence-gke-nodes"

  validation {
    condition     = can(regex("^[a-z][a-z0-9-]{4,28}[a-z0-9]$", var.node_service_account_id))
    error_message = "node_service_account_id must be a valid IAM service account ID."
  }
}

variable "node_service_account_display_name" {
  description = "Display name of the dedicated GKE node service account."
  type        = string
  default     = "PERTINENCE GKE node service account"
}

variable "node_project_roles" {
  description = "Additive project roles granted to the node service account."
  type        = set(string)
  default = [
    "roles/container.defaultNodeServiceAccount",
  ]

  validation {
    condition     = alltrue([for role in var.node_project_roles : startswith(role, "roles/")])
    error_message = "Every node_project_roles entry must be a predefined roles/... role."
  }
}

variable "enable_apis" {
  description = "Whether this module manages the required project API services."
  type        = bool
  default     = true
}

variable "required_apis" {
  description = "Project APIs managed when enable_apis is true. APIs remain enabled on destroy."
  type        = set(string)
  default = [
    "artifactregistry.googleapis.com",
    "compute.googleapis.com",
    "container.googleapis.com",
    "iam.googleapis.com",
    "iamcredentials.googleapis.com",
    "serviceusage.googleapis.com",
  ]
}

variable "node_pools" {
  description = <<-EOT
    Overrides for the four fixed pools. Omitted attributes retain the module defaults. The only
    valid keys are system-pool, pipeline-pool, serving-pool, and loadgen-pool.
  EOT
  type = map(object({
    enabled            = optional(bool)
    machine_type       = optional(string)
    min_count          = optional(number)
    max_count          = optional(number)
    initial_node_count = optional(number)
    disk_size_gb       = optional(number)
    disk_type          = optional(string)
    vcpu_per_node      = optional(number)
  }))
  default = {}

  validation {
    condition = length(setsubtract(
      toset(keys(var.node_pools)),
      toset(["system-pool", "pipeline-pool", "serving-pool", "loadgen-pool"]),
    )) == 0
    error_message = "node_pools may only contain system-pool, pipeline-pool, serving-pool, and loadgen-pool."
  }

  validation {
    condition = alltrue([
      for pool in values(var.node_pools) :
      (pool.min_count == null || pool.min_count >= 0) &&
      (pool.min_count == null || pool.min_count == floor(pool.min_count)) &&
      (pool.max_count == null || pool.max_count >= 0) &&
      (pool.max_count == null || pool.max_count == floor(pool.max_count)) &&
      (pool.initial_node_count == null || pool.initial_node_count >= 0) &&
      (pool.initial_node_count == null || pool.initial_node_count == floor(pool.initial_node_count)) &&
      (pool.disk_size_gb == null || pool.disk_size_gb >= 10) &&
      (pool.disk_size_gb == null || pool.disk_size_gb == floor(pool.disk_size_gb)) &&
      (pool.vcpu_per_node == null || pool.vcpu_per_node > 0) &&
      (pool.vcpu_per_node == null || pool.vcpu_per_node == floor(pool.vcpu_per_node))
    ])
    error_message = "Node counts must be non-negative integers, disk_size_gb an integer of at least 10, and vcpu_per_node a positive integer."
  }
}

variable "allow_without_system_pool" {
  description = "Explicit acknowledgement that system-pool may be disabled."
  type        = bool
  default     = false
}

variable "allow_control_plane_only" {
  description = "Explicit acknowledgement that all node pools may be disabled."
  type        = bool
  default     = false
}

variable "vcpu_quota_budget" {
  description = "Maximum aggregate vCPU count implied by enabled pools at their maximum size."
  type        = number
  default     = 12

  validation {
    condition     = var.vcpu_quota_budget > 0
    error_message = "vcpu_quota_budget must be positive."
  }
}

variable "deletion_protection" {
  description = "Protect the GKE cluster from Terraform deletion. Disable in a separate apply before destroy."
  type        = bool
  default     = true
}

variable "node_pool_deletion_policy" {
  description = "Terraform deletion policy for every managed node pool. Set DELETE in a separate apply before removal."
  type        = string
  default     = "PREVENT"

  validation {
    condition     = contains(["PREVENT", "DELETE"], var.node_pool_deletion_policy)
    error_message = "node_pool_deletion_policy must be PREVENT or DELETE."
  }
}

variable "resource_labels" {
  description = "Non-authoritative labels managed on the GKE cluster."
  type        = map(string)
  default     = {}
}
