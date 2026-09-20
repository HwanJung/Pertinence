variable "project_id" {
  description = "Google Cloud project containing the preserved dispatcher cluster."
  type        = string
  default     = "pertinence-509116"
}

variable "region" {
  description = "Region containing the selected subnetwork and Artifact Registry."
  type        = string
  default     = "asia-northeast3"
}

variable "zone" {
  description = "Zone containing the Standard GKE control plane."
  type        = string
  default     = "asia-northeast3-a"
}

variable "cluster_name" {
  description = "Name of the preserved dispatcher cluster."
  type        = string
  default     = "pertinence-dispatcher-260919-1708"
}

variable "network" {
  description = "Existing VPC network name."
  type        = string
  default     = "default"
}

variable "subnetwork" {
  description = "Existing subnetwork name."
  type        = string
  default     = "default"
}

variable "artifact_registry_repository" {
  description = "Existing Docker repository from which GKE nodes pull images."
  type        = string
  default     = "pertinence"
}

variable "node_pools" {
  description = "Four fixed node-pool roles and their sizing for this environment."
  type = map(object({
    # Keep omitted values null so the child module can inherit its per-pool defaults. In
    # particular, a sizing-only override must never implicitly disable an existing pool.
    enabled            = optional(bool)
    machine_type       = optional(string)
    min_count          = optional(number)
    max_count          = optional(number)
    initial_node_count = optional(number)
    disk_size_gb       = optional(number)
    disk_type          = optional(string)
    vcpu_per_node      = optional(number)
  }))
  default = {
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
      enabled            = false
      machine_type       = "n2-standard-8"
      min_count          = 0
      max_count          = 3
      initial_node_count = 0
      vcpu_per_node      = 8
    }
    loadgen-pool = {
      enabled            = false
      machine_type       = "n2-standard-4"
      min_count          = 0
      max_count          = 1
      initial_node_count = 0
      vcpu_per_node      = 4
    }
  }
}

variable "allow_without_system_pool" {
  description = "Explicit acknowledgement for clusters without an untainted system pool."
  type        = bool
  default     = false
}

variable "allow_control_plane_only" {
  description = "Explicit acknowledgement for a Standard cluster with no enabled node pools."
  type        = bool
  default     = false
}

variable "vcpu_quota_budget" {
  description = "Maximum declared vCPU capacity allowed by validation."
  type        = number
  default     = 12
}

variable "deletion_protection" {
  description = "Protect the GKE cluster from Terraform deletion."
  type        = bool
  default     = true
}

variable "node_pool_deletion_policy" {
  description = "Terraform provider deletion policy for every enabled node pool."
  type        = string
  default     = "PREVENT"
}

variable "resource_labels" {
  description = "Labels applied to newly created or deliberately converged cluster resources."
  type        = map(string)
  default     = {}
}
