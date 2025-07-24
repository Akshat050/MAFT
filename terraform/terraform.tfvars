# MAFT GCP Configuration
project_id   = "maft-465719"
region       = "us-central1"
zone         = "us-central1-a"

# Instance configuration
machine_type = "n1-standard-4"  # 4 vCPUs, 15 GB RAM
gpu_type     = "nvidia-tesla-t4"
gpu_count    = 1

# Storage configuration
disk_size_gb = 100

# Instance naming
instance_name = "maft-training-instance" 