output "project_id" {
  description = "The GCP project ID"
  value       = var.project_id
}

output "bucket_name" {
  description = "The name of the Cloud Storage bucket"
  value       = google_storage_bucket.maft_bucket.name
}

output "bucket_url" {
  description = "The URL of the Cloud Storage bucket"
  value       = "gs://${google_storage_bucket.maft_bucket.name}"
}

output "instance_group_name" {
  description = "The name of the managed instance group"
  value       = google_compute_instance_group_manager.maft_group.name
}

output "network_name" {
  description = "The name of the VPC network"
  value       = google_compute_network.maft_network.name
}

output "service_account_email" {
  description = "The email of the service account"
  value       = google_service_account.maft_service_account.email
}

output "health_check_url" {
  description = "The health check URL for the instance"
  value       = "http://[INSTANCE_IP]:8080/health"
}

output "gcs_commands" {
  description = "Useful GCS commands for managing data"
  value = {
    upload_data    = "gsutil -m cp -r /path/to/data gs://${google_storage_bucket.maft_bucket.name}/data/"
    download_data  = "gsutil -m cp -r gs://${google_storage_bucket.maft_bucket.name}/data/ /path/to/local/"
    list_bucket    = "gsutil ls gs://${google_storage_bucket.maft_bucket.name}/"
    sync_bucket    = "gsutil -m rsync -r /path/to/local gs://${google_storage_bucket.maft_bucket.name}/"
  }
}

output "monitoring_commands" {
  description = "Commands to monitor the deployment"
  value = {
    list_instances = "gcloud compute instances list --filter='name~maft'"
    view_logs      = "gcloud logging read 'resource.type=gce_instance AND resource.labels.instance_name~maft' --limit=50"
    ssh_instance   = "gcloud compute ssh [INSTANCE_NAME] --zone=${var.zone}"
    check_health   = "curl http://[INSTANCE_IP]:8080/health"
  }
} 