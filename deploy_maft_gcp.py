#!/usr/bin/env python3
"""
MAFT GCP Deployment Script using Terraform
This script deploys the MAFT infrastructure on Google Cloud Platform
"""

import os
import sys
import subprocess
import time
import json
from pathlib import Path

class MAFTGCPDeployer:
    def __init__(self, project_id="maft-465719"):
        self.project_id = project_id
        self.terraform_dir = Path("terraform")
        self.working_dir = Path.cwd()
        
    def log(self, message):
        """Log a message with timestamp"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {message}")
        
    def run_command(self, command, description, check=True, cwd=None):
        """Run a command and handle the result"""
        self.log(f"Running: {description}")
        self.log(f"Command: {command}")
        
        try:
            result = subprocess.run(
                command,
                shell=True,
                check=check,
                capture_output=True,
                text=True,
                cwd=cwd or self.working_dir
            )
            
            if result.stdout:
                self.log(f"Output: {result.stdout.strip()}")
                
            if result.returncode == 0:
                self.log(f"Success: {description}")
                return True
            else:
                self.log(f"Error: {description}")
                if result.stderr:
                    self.log(f"Error output: {result.stderr.strip()}")
                return False
                
        except subprocess.CalledProcessError as e:
            self.log(f"Error: {description}")
            if e.stderr:
                self.log(f"Error output: {e.stderr.strip()}")
            return False
            
    def check_prerequisites(self):
        """Check if required tools are installed"""
        self.log("Checking prerequisites...")
        
        # Check gcloud
        if not self.run_command("gcloud --version", "Checking gcloud", check=False):
            self.log("ERROR: gcloud CLI not found. Please install Google Cloud SDK.")
            return False
            
        # Check terraform
        if not self.run_command("terraform --version", "Checking Terraform", check=False):
            self.log("ERROR: Terraform not found. Please install Terraform.")
            return False
            
        # Check authentication
        if not self.run_command("gcloud auth list", "Checking GCP authentication", check=False):
            self.log("ERROR: Not authenticated with GCP. Please run 'gcloud auth login'")
            return False
            
        self.log("All prerequisites satisfied!")
        return True
        
    def setup_terraform(self):
        """Initialize and plan Terraform deployment"""
        self.log("Setting up Terraform...")
        
        # Change to terraform directory
        os.chdir(self.terraform_dir)
        
        # Initialize Terraform
        if not self.run_command("terraform init", "Initializing Terraform"):
            return False
            
        # Validate Terraform configuration
        if not self.run_command("terraform validate", "Validating Terraform configuration"):
            return False
            
        # Plan Terraform deployment
        if not self.run_command("terraform plan -out=tfplan", "Planning Terraform deployment"):
            return False
            
        return True
        
    def deploy_infrastructure(self):
        """Deploy the infrastructure using Terraform"""
        self.log("Deploying infrastructure...")
        
        # Apply Terraform plan
        if not self.run_command("terraform apply tfplan", "Applying Terraform plan"):
            return False
            
        # Get outputs
        if not self.run_command("terraform output -json", "Getting Terraform outputs", check=False):
            self.log("Warning: Could not get Terraform outputs")
            return True
            
        return True
        
    def upload_maft_code(self):
        """Upload MAFT code to Cloud Storage"""
        self.log("Uploading MAFT code to Cloud Storage...")
        
        # Get bucket name from Terraform output
        result = subprocess.run(
            "terraform output -raw bucket_name",
            shell=True,
            capture_output=True,
            text=True,
            cwd=self.terraform_dir
        )
        
        if result.returncode != 0:
            self.log("Warning: Could not get bucket name from Terraform output")
            return False
            
        bucket_name = result.stdout.strip()
        
        # Create a temporary archive of the MAFT code
        self.log("Creating code archive...")
        os.chdir(self.working_dir)
        
        # Exclude terraform and other unnecessary files
        exclude_patterns = [
            "--exclude=terraform",
            "--exclude=__pycache__",
            "--exclude=*.pyc",
            "--exclude=.git",
            "--exclude=node_modules",
            "--exclude=venv"
        ]
        
        tar_command = f"tar -czf maft-code.tar.gz {' '.join(exclude_patterns)} ."
        if not self.run_command(tar_command, "Creating code archive"):
            return False
            
        # Upload to Cloud Storage
        upload_command = f"gsutil cp maft-code.tar.gz gs://{bucket_name}/code/"
        if not self.run_command(upload_command, "Uploading code to Cloud Storage"):
            return False
            
        # Clean up
        self.run_command("rm maft-code.tar.gz", "Cleaning up temporary files", check=False)
        
        return True
        
    def monitor_deployment(self):
        """Monitor the deployment status"""
        self.log("Monitoring deployment...")
        
        # Get instance information
        self.log("Getting instance information...")
        list_command = "gcloud compute instances list --filter='name~maft' --format='table(name,status,zone,externalIP)'"
        self.run_command(list_command, "Listing MAFT instances", check=False)
        
        # Check health status
        self.log("Checking health status...")
        # Note: You'll need to get the instance IP from the list command above
        # health_command = f"curl -s http://[INSTANCE_IP]:8080/health"
        # self.run_command(health_command, "Checking instance health", check=False)
        
        self.log("Deployment monitoring completed!")
        
    def cleanup(self):
        """Clean up temporary files"""
        self.log("Cleaning up...")
        os.chdir(self.terraform_dir)
        self.run_command("rm -f tfplan", "Removing Terraform plan file", check=False)
        os.chdir(self.working_dir)
        
    def deploy(self):
        """Main deployment function"""
        self.log("Starting MAFT GCP deployment...")
        self.log(f"Project ID: {self.project_id}")
        
        try:
            # Check prerequisites
            if not self.check_prerequisites():
                return False
                
            # Setup Terraform
            if not self.setup_terraform():
                return False
                
            # Deploy infrastructure
            if not self.deploy_infrastructure():
                return False
                
            # Upload MAFT code
            if not self.upload_maft_code():
                self.log("Warning: Could not upload MAFT code")
                
            # Monitor deployment
            self.monitor_deployment()
            
            self.log("MAFT GCP deployment completed successfully!")
            self.log("Next steps:")
            self.log("1. Check the instance status: gcloud compute instances list")
            self.log("2. SSH into the instance: gcloud compute ssh [INSTANCE_NAME] --zone=us-central1-a")
            self.log("3. Monitor logs: gcloud logging read 'resource.type=gce_instance' --limit=50")
            self.log("4. Check Cloud Storage bucket for results")
            
            return True
            
        except Exception as e:
            self.log(f"Error during deployment: {str(e)}")
            return False
        finally:
            self.cleanup()

def main():
    """Main function"""
    print("MAFT GCP Deployment Script")
    print("=" * 50)
    
    # Get project ID from user or use default
    project_id = input("Enter GCP Project ID (default: maft-465719): ").strip()
    if not project_id:
        project_id = "maft-465719"
        
    # Create deployer and run deployment
    deployer = MAFTGCPDeployer(project_id)
    success = deployer.deploy()
    
    if success:
        print("\nDeployment completed successfully!")
        sys.exit(0)
    else:
        print("\nDeployment failed!")
        sys.exit(1)

if __name__ == "__main__":
    main() 