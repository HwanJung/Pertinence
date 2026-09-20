output "cluster_id" {
  description = "Fully qualified GKE cluster resource ID."
  value       = google_container_cluster.this.id
}

output "cluster_name" {
  description = "GKE cluster name."
  value       = google_container_cluster.this.name
}

output "cluster_location" {
  description = "GKE cluster zone."
  value       = google_container_cluster.this.location
}

output "cluster_master_version" {
  description = "Actual Kubernetes control-plane patch version selected by the Regular channel."
  value       = google_container_cluster.this.master_version
}

output "cluster_endpoint" {
  description = "GKE control-plane endpoint."
  value       = google_container_cluster.this.endpoint
  sensitive   = true
}

output "cluster_ca_certificate" {
  description = "Base64-encoded GKE cluster CA certificate."
  value       = try(google_container_cluster.this.master_auth[0].cluster_ca_certificate, null)
  sensitive   = true
}

output "workload_identity_pool" {
  description = "Workload Identity Federation pool configured for the cluster."
  value       = google_container_cluster.this.workload_identity_config[0].workload_pool
}

output "network" {
  description = "Self-link of the existing VPC network."
  value       = data.google_compute_network.selected.self_link
}

output "subnetwork" {
  description = "Self-link of the existing regional subnetwork."
  value       = data.google_compute_subnetwork.selected.self_link
}

output "pod_ipv4_cidr" {
  description = "Effective primary Pod IPv4 CIDR allocated by GKE."
  value       = google_container_cluster.this.cluster_ipv4_cidr
}

output "service_ipv4_cidr" {
  description = "Effective Service IPv4 CIDR allocated by GKE."
  value       = google_container_cluster.this.services_ipv4_cidr
}

output "node_service_account_email" {
  description = "Email address of the dedicated GKE node service account."
  value       = google_service_account.nodes.email
}

output "active_node_pools" {
  description = "Effective configuration of node pools managed in this environment."
  value       = local.active_node_pools
}

output "node_pool_versions" {
  description = "Actual Kubernetes patch version reported by every enabled node pool."
  value = {
    for name, pool in google_container_node_pool.this : name => pool.version
  }
}

output "node_scheduling" {
  description = "Node selectors and tolerations that workload manifests must use per pool."
  value = {
    for name in keys(local.active_node_pools) : name => {
      node_selector = {
        "cloud.google.com/gke-nodepool" = name
      }
      tolerations = local.pool_taints[name] == null ? [] : [{
        key      = local.pool_taints[name].key
        operator = "Equal"
        value    = local.pool_taints[name].value
        effect   = "NoSchedule"
      }]
    }
  }
}

output "kfp_addons_supported" {
  description = "Whether the required untainted system pool exists for KFP/KServe controllers."
  value       = local.system_pool_enabled
}

output "aggregate_max_vcpus" {
  description = "Maximum vCPUs across all enabled node pools."
  value       = local.aggregate_max_vcpus
}
