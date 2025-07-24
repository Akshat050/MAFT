#!/usr/bin/env python3
"""
Complete GCP Setup for MAFT

This script sets up the entire MAFT infrastructure on GCP including:
- GCS buckets for data storage
- Compute Engine instances with GPUs
- Persistent disks for datasets
- Automated training pipeline
"""

import os
import sys
import json
import subprocess
import argparse
import time
from pathlib import Path
from datetime import datetime
import google.auth
from google.cloud import storage, compute_v1
from google.auth.exceptions import DefaultCredentialsError

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class GCPMAFTSetup:
    def __init__(self, project_id, zone="us-central1-a", region="us-central1"):
        self.project_id = project_id
        self.zone = zone
        self.region = region
        self.instance_name = "maft-training"
        self.bucket_name = f"maft-{project_id}-data"
        self.disk_name = "maft-dataset-disk"
        
        # Initialize clients
        self.storage_client = storage.Client(project=project_id)
        self.compute_client = compute_v1.InstancesClient()
        self.disks_client = compute_v1.DisksClient()
        
        print(f"Initializing MAFT GCP Setup for project: {project_id}")
        print(f"Zone: {zone}, Region: {region}")
    
    def check_credentials(self):
        """Check if GCP credentials are properly configured."""
        try:
            credentials, project = google.auth.default()
            print(f"GCP credentials verified for project: {project}")
            return True
        except DefaultCredentialsError:
            print("GCP credentials not found. Please run:")
            print("   gcloud auth application-default login")
            return False
    
    def enable_apis(self):
        """Enable required GCP APIs."""
        print("Enabling required GCP APIs...")
        
        apis = [
            "compute.googleapis.com",
            "storage.googleapis.com",
            "aiplatform.googleapis.com",
            "cloudresourcemanager.googleapis.com"
        ]
        
        for api in apis:
            try:
                subprocess.run([
                    "gcloud", "services", "enable", api,
                    "--project", self.project_id
                ], check=True, capture_output=True)
                print(f"Enabled {api}")
            except subprocess.CalledProcessError as e:
                print(f"Failed to enable {api}: {e}")
    
    def create_gcs_bucket(self):
        """Create GCS bucket for data storage."""
        print(f"Creating GCS bucket: {self.bucket_name}")
        
        try:
            bucket = self.storage_client.create_bucket(
                self.bucket_name,
                location=self.region
            )
            print(f"Created bucket: gs://{self.bucket_name}")
            return bucket
        except Exception as e:
            if "already exists" in str(e):
                print(f"Bucket already exists: gs://{self.bucket_name}")
                return self.storage_client.bucket(self.bucket_name)
            else:
                print(f"Failed to create bucket: {e}")
                return None
    
    def create_persistent_disk(self, size_gb=100):
        """Create persistent disk for dataset storage."""
        print(f"Creating persistent disk: {self.disk_name}")
        
        disk_resource = {
            "name": self.disk_name,
            "size_gb": str(size_gb),
            "zone": f"projects/{self.project_id}/zones/{self.zone}",
            "type": f"projects/{self.project_id}/zones/{self.zone}/diskTypes/pd-standard"
        }
        
        try:
            operation = self.disks_client.insert(
                project=self.project_id,
                zone=self.zone,
                disk_resource=disk_resource
            )
            
            # Wait for operation to complete
            self._wait_for_operation(operation.name, "disk")
            print(f"Created disk: {self.disk_name}")
            return True
        except Exception as e:
            print(f"Failed to create disk: {e}")
            return False
    
    def create_compute_instance(self, machine_type="n1-standard-4", gpu_type="nvidia-tesla-t4"):
        """Create Compute Engine instance with GPU."""
        print(f"Creating Compute Engine instance: {self.instance_name}")
        
        # Create startup script
        startup_script = self._create_startup_script()
        
        # Instance configuration
        instance_config = {
            "name": self.instance_name,
            "machine_type": f"zones/{self.zone}/machineTypes/{machine_type}",
            "disks": [
                {
                    "boot": True,
                    "auto_delete": True,
                    "initialize_params": {
                        "source_image": "projects/debian-cloud/global/images/family/debian-11"
                    }
                },
                {
                    "boot": False,
                    "auto_delete": False,
                    "source": f"projects/{self.project_id}/zones/{self.zone}/disks/{self.disk_name}"
                }
            ],
            "network_interfaces": [
                {
                    "network": "global/networks/default",
                    "access_configs": [
                        {
                            "name": "External NAT",
                            "type": "ONE_TO_ONE_NAT"
                        }
                    ]
                }
            ],
            "metadata": {
                "items": [
                    {
                        "key": "startup-script",
                        "value": startup_script
                    },
                    {
                        "key": "GCS_BUCKET",
                        "value": self.bucket_name
                    },
                    {
                        "key": "AUTO_SHUTDOWN",
                        "value": "true"
                    }
                ]
            },
            "scheduling": {
                "preemptible": False,
                "on_host_maintenance": "TERMINATE"
            },
            "service_accounts": [
                {
                    "email": "default",
                    "scopes": ["https://www.googleapis.com/auth/cloud-platform"]
                }
            ]
        }
        
        # Add GPU if specified
        if gpu_type:
            instance_config["guest_accelerators"] = [
                {
                    "accelerator_type": f"zones/{self.zone}/acceleratorTypes/{gpu_type}",
                    "accelerator_count": 1
                }
            ]
        
        try:
            operation = self.compute_client.insert(
                project=self.project_id,
                zone=self.zone,
                instance_resource=instance_config
            )
            
            # Wait for operation to complete
            self._wait_for_operation(operation.name, "instance")
            print(f"Created instance: {self.instance_name}")
            return True
        except Exception as e:
            print(f"Failed to create instance: {e}")
            return False
    
    def _create_startup_script(self):
        """Create enhanced startup script for GCP instance."""
        script = f"""#!/bin/bash
# Enhanced MAFT GCP Startup Script

set -e

echo "Starting MAFT GCP deployment..."

# Update system
echo "Updating system packages..."
sudo apt-get update
sudo apt-get install -y git wget curl unzip ffmpeg libsm6 libxext6 python3-pip

# Install Google Cloud SDK
echo "Installing Google Cloud SDK..."
curl https://sdk.cloud.google.com | bash
export PATH=$PATH:$HOME/google-cloud-sdk/bin
gcloud auth activate-service-account --key-file=/tmp/service-account-key.json

# Setup MAFT repository
echo "Setting up MAFT repository..."
cd /home
mkdir -p maft
cd maft

# Create necessary directories
echo "Creating directories..."
mkdir -p data/mosei experiments logs

# Mount persistent disk
echo "Mounting persistent disk..."
sudo mkdir -p /mnt/maft-data
sudo mount /dev/sdb /mnt/maft-data
sudo chown $USER:$USER /mnt/maft-data

# Download dataset to GCS if not exists
echo "Checking dataset in GCS..."
if ! gsutil ls gs://{self.bucket_name}/datasets/mosei/ &>/dev/null; then
    echo "Downloading dataset to GCS..."
    python scripts/prepare_mosei.py --output_dir /tmp/mosei --use_mock
    gsutil -m cp -r /tmp/mosei gs://{self.bucket_name}/datasets/
fi

# Download dataset from GCS to persistent disk
echo "Downloading dataset from GCS..."
gsutil -m cp -r gs://{self.bucket_name}/datasets/mosei /mnt/maft-data/
ln -s /mnt/maft-data/mosei data/mosei

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Set environment variables
export PYTHONPATH=/home/maft
export CUDA_VISIBLE_DEVICES=0

# Run training
echo "Starting MAFT training..."
python scripts/deploy_gcp.py \\
    --data_dir data/mosei \\
    --output_dir experiments/mosei_gcp \\
    --gcs_bucket {self.bucket_name} \\
    --gcs_path maft_results

# Upload results to GCS
echo "Uploading results to GCS..."
gsutil -m cp -r experiments/mosei_gcp gs://{self.bucket_name}/maft_results/

echo "MAFT deployment completed!"

# Shutdown instance
echo "Shutting down instance..."
sudo shutdown -h now
"""
        return script
    
    def _wait_for_operation(self, operation_name, operation_type):
        """Wait for GCP operation to complete."""
        print(f"Waiting for {operation_type} operation to complete...")
        
        if operation_type == "disk":
            operations_client = compute_v1.ZoneOperationsClient()
            operation = operations_client.wait(
                project=self.project_id,
                zone=self.zone,
                operation=operation_name
            )
        elif operation_type == "instance":
            operations_client = compute_v1.ZoneOperationsClient()
            operation = operations_client.wait(
                project=self.project_id,
                zone=self.zone,
                operation=operation_name
            )
        
        if operation.status == "DONE":
            print(f"Completed {operation_type} operation")
        else:
            print(f"Failed {operation_type} operation")
    
    def get_instance_ip(self):
        """Get the external IP of the instance."""
        try:
            instance = self.compute_client.get(
                project=self.project_id,
                zone=self.zone,
                instance=self.instance_name
            )
            
            for network_interface in instance.network_interfaces:
                for access_config in network_interface.access_configs:
                    if access_config.name == "External NAT":
                        return access_config.nat_i_p
            
            return None
        except Exception as e:
            print(f"Failed to get instance IP: {e}")
            return None
    
    def monitor_instance(self):
        """Monitor instance status and logs."""
        print("Monitoring instance...")
        
        # Get instance status
        try:
            instance = self.compute_client.get(
                project=self.project_id,
                zone=self.zone,
                instance=self.instance_name
            )
            print(f"Instance status: {instance.status}")
        except Exception as e:
            print(f"Failed to get instance status: {e}")
            return
        
        # Get external IP
        ip = self.get_instance_ip()
        if ip:
            print(f"External IP: {ip}")
            print(f"SSH command: gcloud compute ssh {self.instance_name} --zone={self.zone}")
    
    def setup_complete(self):
        """Run complete setup process."""
        print("Starting complete MAFT GCP setup...")
        
        # Check credentials
        if not self.check_credentials():
            return False
        
        # Enable APIs
        self.enable_apis()
        
        # Create GCS bucket
        bucket = self.create_gcs_bucket()
        if not bucket:
            return False
        
        # Create persistent disk
        if not self.create_persistent_disk():
            return False
        
        # Create compute instance
        if not self.create_compute_instance():
            return False
        
        # Monitor instance
        self.monitor_instance()
        
        print("\nMAFT GCP setup completed successfully!")
        print(f"Data bucket: gs://{self.bucket_name}")
        print(f"Instance: {self.instance_name}")
        print(f"Persistent disk: {self.disk_name}")
        print(f"Zone: {self.zone}")
        
        return True


def main():
    parser = argparse.ArgumentParser(description='Complete MAFT GCP Setup')
    parser.add_argument('--project_id', required=True, help='GCP Project ID')
    parser.add_argument('--zone', default='us-central1-a', help='GCP Zone')
    parser.add_argument('--region', default='us-central1', help='GCP Region')
    parser.add_argument('--machine_type', default='n1-standard-4', help='Machine type')
    parser.add_argument('--gpu_type', default='nvidia-tesla-t4', help='GPU type')
    
    args = parser.parse_args()
    
    # Create setup instance
    setup = GCPMAFTSetup(args.project_id, args.zone, args.region)
    
    # Run complete setup
    success = setup.setup_complete()
    
    if success:
        print("\nNext steps:")
        print("1. Wait for instance to start (5-10 minutes)")
        print("2. Monitor progress: gcloud compute ssh maft-training --zone=us-central1-a")
        print("3. View logs: gcloud logging read 'resource.type=gce_instance'")
        print("4. Download results: gsutil -m cp -r gs://maft-{args.project_id}-data/maft_results/ ./results/")
    else:
        print("\nSetup failed. Please check the errors above.")


if __name__ == '__main__':
    main() 