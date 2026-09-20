resource "google_storage_bucket" "terraform_state" {
  name     = var.state_bucket_name
  project  = var.project_id
  location = var.location

  storage_class               = "STANDARD"
  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"
  force_destroy               = false
  deletion_policy             = "PREVENT"

  versioning {
    enabled = true
  }

  labels = merge(
    {
      managed-by = "terraform"
      purpose    = "terraform-state"
    },
    var.labels,
  )

  lifecycle {
    prevent_destroy = true
  }
}
