output "cluster_name" {
  description = "GKE cluster name for gcloud and addon workflows."
  value       = module.gke.cluster_name
}

output "cluster_location" {
  description = "Zonal GKE control-plane location."
  value       = module.gke.cluster_location
}

output "cluster_master_version" {
  description = "Actual GKE control-plane version selected by the Regular release channel."
  value       = module.gke.cluster_master_version
}

output "workload_identity_pool" {
  description = "Workload Identity pool used by Kubernetes service accounts."
  value       = module.gke.workload_identity_pool
}

output "active_node_pools" {
  description = "Enabled node pools in this state."
  value       = module.gke.active_node_pools
}

output "node_pool_versions" {
  description = "Actual Kubernetes version of every enabled node pool."
  value       = module.gke.node_pool_versions
}

output "network" {
  description = "Existing VPC network self-link used by the cluster."
  value       = module.gke.network
}

output "subnetwork" {
  description = "Existing regional subnetwork self-link used by the cluster."
  value       = module.gke.subnetwork
}

output "pod_ipv4_cidr" {
  description = "Effective Pod IPv4 range."
  value       = module.gke.pod_ipv4_cidr
}

output "service_ipv4_cidr" {
  description = "Effective Service IPv4 range."
  value       = module.gke.service_ipv4_cidr
}

output "node_service_account_email" {
  description = "Dedicated GKE node service account."
  value       = module.gke.node_service_account_email
}

output "node_scheduling" {
  description = "Node selectors and tolerations for addon and workload manifests."
  value       = module.gke.node_scheduling
}

output "kfp_addons_supported" {
  description = "Whether an untainted system pool exists for KFP/KServe controllers."
  value       = module.gke.kfp_addons_supported
}
