variable "project_id" {
  description = "The GCP project ID"
  type        = string
  default     = "maft-465719"
}

variable "region" {
  description = "The GCP region"
  type        = string
  default     = "us-central1"
}

variable "zone" {
  description = "The GCP zone"
  type        = string
  default     = "us-central1-b"
}

variable "machine_type" {
  description = "The machine type for the compute instance"
  type        = string
  default     = "n1-standard-4"
}

variable "gpu_type" {
  description = "The GPU type to attach to the instance"
  type        = string
  default     = "nvidia-tesla-t4"
}

variable "gpu_count" {
  description = "Number of GPUs to attach"
  type        = number
  default     = 1
}

variable "instance_name" {
  description = "Name of the compute instance"
  type        = string
  default     = "maft-training-instance"
}

variable "disk_size_gb" {
  description = "Size of the boot disk in GB"
  type        = number
  default     = 100
} 