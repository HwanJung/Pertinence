# These import blocks adopt the preserved cluster without creating a duplicate.
# Run the first plan with profiles/import-baseline.tfvars and reject every mutation.

import {
  to = module.gke.google_project_service.required["artifactregistry.googleapis.com"]
  id = "pertinence-509116/artifactregistry.googleapis.com"
}

import {
  to = module.gke.google_project_service.required["compute.googleapis.com"]
  id = "pertinence-509116/compute.googleapis.com"
}

import {
  to = module.gke.google_project_service.required["container.googleapis.com"]
  id = "pertinence-509116/container.googleapis.com"
}

import {
  to = module.gke.google_project_service.required["iam.googleapis.com"]
  id = "pertinence-509116/iam.googleapis.com"
}

import {
  to = module.gke.google_project_service.required["iamcredentials.googleapis.com"]
  id = "pertinence-509116/iamcredentials.googleapis.com"
}

import {
  to = module.gke.google_project_service.required["serviceusage.googleapis.com"]
  id = "pertinence-509116/serviceusage.googleapis.com"
}

import {
  to = module.gke.google_service_account.nodes
  id = "projects/pertinence-509116/serviceAccounts/pertinence-gke-nodes@pertinence-509116.iam.gserviceaccount.com"
}

import {
  to = module.gke.google_project_iam_member.node["roles/container.defaultNodeServiceAccount"]
  id = "pertinence-509116 roles/container.defaultNodeServiceAccount serviceAccount:pertinence-gke-nodes@pertinence-509116.iam.gserviceaccount.com"
}

import {
  to = module.gke.google_artifact_registry_repository_iam_member.node_reader
  id = "projects/pertinence-509116/locations/asia-northeast3/repositories/pertinence roles/artifactregistry.reader serviceAccount:pertinence-gke-nodes@pertinence-509116.iam.gserviceaccount.com"
}

import {
  to = module.gke.google_container_cluster.this
  id = "projects/pertinence-509116/locations/asia-northeast3-a/clusters/pertinence-dispatcher-260919-1708"
}

import {
  to = module.gke.google_container_node_pool.this["system-pool"]
  id = "pertinence-509116/asia-northeast3-a/pertinence-dispatcher-260919-1708/system-pool"
}

import {
  to = module.gke.google_container_node_pool.this["pipeline-pool"]
  id = "pertinence-509116/asia-northeast3-a/pertinence-dispatcher-260919-1708/pipeline-pool"
}
