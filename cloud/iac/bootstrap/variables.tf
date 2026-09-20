variable "project_id" {
  description = "Google Cloud project that owns the Terraform state bucket."
  type        = string
  default     = "pertinence-509116"

  validation {
    condition     = length(trimspace(var.project_id)) > 0
    error_message = "project_id must not be empty."
  }
}

variable "state_bucket_name" {
  description = "Globally unique GCS bucket used for Terraform remote state."
  type        = string
  default     = "pertinence-509116-pertinence-iac-state"

  validation {
    condition     = can(regex("^[a-z0-9][a-z0-9._-]{1,61}[a-z0-9]$", var.state_bucket_name))
    error_message = "state_bucket_name must be a valid 3-63 character GCS bucket name."
  }
}

variable "location" {
  description = "GCS location for the Terraform state bucket."
  type        = string
  default     = "asia-northeast3"

  validation {
    condition     = var.location == "asia-northeast3"
    error_message = "The dispatcher state bucket must remain in asia-northeast3."
  }
}

variable "labels" {
  description = "Additional labels to merge into the state bucket labels."
  type        = map(string)
  default     = {}
}
