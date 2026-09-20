output "state_bucket_name" {
  description = "Name of the GCS bucket that stores Terraform state."
  value       = google_storage_bucket.terraform_state.name
}

output "state_bucket_url" {
  description = "Canonical URL of the GCS state bucket."
  value       = google_storage_bucket.terraform_state.url
}

output "bootstrap_state_prefix" {
  description = "GCS object prefix used by the bootstrap root after migration."
  value       = "bootstrap"
}
