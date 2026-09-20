module "gke" {
  source = "../../modules/gke"

  project_id                            = var.project_id
  region                                = var.region
  zone                                  = var.zone
  cluster_name                          = var.cluster_name
  network                               = var.network
  subnetwork                            = var.subnetwork
  artifact_registry_repository          = var.artifact_registry_repository
  artifact_registry_repository_location = var.region
  node_pools                            = var.node_pools
  allow_without_system_pool             = var.allow_without_system_pool
  allow_control_plane_only              = var.allow_control_plane_only
  vcpu_quota_budget                     = var.vcpu_quota_budget
  deletion_protection                   = var.deletion_protection
  node_pool_deletion_policy             = var.node_pool_deletion_policy
  resource_labels                       = var.resource_labels
}
