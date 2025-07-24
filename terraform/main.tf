terraform {
  required_version = ">= 1.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
    google-beta = {
      source  = "hashicorp/google-beta"
      version = "~> 5.0"
    }
  }
}

# Configure the Google Provider
provider "google" {
  project = var.project_id
  region  = var.region
  zone    = var.zone
}

provider "google-beta" {
  project = var.project_id
  region  = var.region
  zone    = var.zone
}

# Enable required APIs
resource "google_project_service" "required_apis" {
  for_each = toset([
    "compute.googleapis.com",
    "storage.googleapis.com",
    "aiplatform.googleapis.com",
    "cloudbuild.googleapis.com",
    "containerregistry.googleapis.com",
    "logging.googleapis.com",
    "monitoring.googleapis.com"
  ])
  
  project = var.project_id
  service = each.value
  
  disable_dependent_services = true
  disable_on_destroy         = false
}

# Create VPC network
resource "google_compute_network" "maft_network" {
  name                    = "maft-network"
  auto_create_subnetworks = false
  depends_on             = [google_project_service.required_apis]
}

# Create subnet
resource "google_compute_subnetwork" "maft_subnet" {
  name          = "maft-subnet"
  ip_cidr_range = "10.0.0.0/24"
  network       = google_compute_network.maft_network.id
  region        = var.region
}

# Create firewall rules
resource "google_compute_firewall" "maft_firewall" {
  name    = "maft-firewall"
  network = google_compute_network.maft_network.id

  allow {
    protocol = "tcp"
    ports    = ["22", "80", "443", "8080", "8888"]
  }

  source_ranges = ["0.0.0.0/0"]
  target_tags   = ["maft-instance"]
}

# Create Cloud Storage bucket for datasets and results
resource "google_storage_bucket" "maft_bucket" {
  name          = "${var.project_id}-maft-data"
  location      = var.region
  force_destroy = true

  uniform_bucket_level_access = true

  lifecycle_rule {
    condition {
      age = 30
    }
    action {
      type = "Delete"
    }
  }
}

# Create service account for the compute instance
resource "google_service_account" "maft_service_account" {
  account_id   = "maft-service-account"
  display_name = "MAFT Service Account"
  description  = "Service account for MAFT compute instance"
}

# Grant necessary roles to service account
resource "google_project_iam_member" "service_account_roles" {
  for_each = toset([
    "roles/storage.admin",
    "roles/logging.logWriter",
    "roles/monitoring.metricWriter",
    "roles/aiplatform.user"
  ])
  
  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.maft_service_account.email}"
}

# Create compute instance template
resource "google_compute_instance_template" "maft_template" {
  name_prefix  = "maft-template-"
  machine_type = var.machine_type
  region       = var.region

  disk {
    source_image = "debian-cloud/debian-11"
    auto_delete  = false
    boot         = true
    disk_size_gb = 1024
    disk_name    = "maft-persistent-disk"
  }

  # Temporarily commented out GPU requirement due to quota issues
  # guest_accelerator {
  #   type  = var.gpu_type
  #   count = var.gpu_count
  # }

  network_interface {
    subnetwork = google_compute_subnetwork.maft_subnet.id
    access_config {
      // Ephemeral public IP
    }
  }

  service_account {
    email  = google_service_account.maft_service_account.email
    scopes = ["cloud-platform"]
  }

  metadata = {
    startup-script = templatefile("${path.module}/startup-script.sh", {
      project_id = var.project_id
      bucket_name = google_storage_bucket.maft_bucket.name
    })
    "install-nvidia-driver" = "true"
  }

  scheduling {
    on_host_maintenance = "TERMINATE"
    automatic_restart   = true
    preemptible         = false
    provisioning_model  = "STANDARD"
  }

  tags = ["maft-instance"]

  lifecycle {
    create_before_destroy = true
  }
}

# Create managed instance group
resource "google_compute_instance_group_manager" "maft_group" {
  name = "maft-instance-group"

  base_instance_name = "maft-instance"
  zone               = var.zone

  version {
    instance_template = google_compute_instance_template.maft_template.id
  }

  target_size = 1

  named_port {
    name = "http"
    port = 8080
  }

  auto_healing_policies {
    health_check      = google_compute_health_check.maft_health_check.id
    initial_delay_sec = 300
  }
}

# Health check for the instance group
resource "google_compute_health_check" "maft_health_check" {
  name = "maft-health-check"

  tcp_health_check {
    port = 8080
  }
}
