terraform {
  required_version = "= 1.16.3"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "= 7.42.0"
    }
  }
}

provider "google" {
  project = var.project_id
}
