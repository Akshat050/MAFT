#!/usr/bin/env python3
"""
Automated MAFT GCP Setup for Project: maft-465719

This script automatically sets up the complete MAFT infrastructure on GCP.
"""

import os
import sys
import subprocess
import time
from pathlib import Path

# GCP Configuration
PROJECT_ID = "maft-465719"
ZONE = "us-central1-a"
REGION = "us-central1"

def run_command(command, description, check=True):
    """Run a command with error handling."""
    print(f"\n🔄 {description}")
    print(f"📝 Command: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=check, capture_output=True, text=True)
        if result.stdout:
            print(f"✅ Success: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e.stderr}")
        if check:
            raise
        return False

def main():
    print("MAFT GCP Automated Setup")
    print("============================")
    print(f"Project ID: {PROJECT_ID}")
    print(f"Zone: {ZONE}")
    print(f"Budget: $415.46 GCP Credits")
    
    # Check if gcloud is installed
    if not run_command("gcloud --version", "Checking Google Cloud SDK", check=False):
        print("❌ Google Cloud SDK not found. Please install it first:")
        print("   https://cloud.google.com/sdk/docs/install")
        return
    
    # Set project
    print(f"\nSetting GCP project to: {PROJECT_ID}")
    if not run_command(f"gcloud config set project {PROJECT_ID}", "Setting GCP project"):
        return
    
    # Check authentication
    print("\nChecking authentication...")
    if not run_command("gcloud auth list", "Checking current authentication", check=False):
        print("⚠️ Not authenticated. Please run:")
        print("   gcloud auth login")
        print("   gcloud auth application-default login")
        return
    
    # Install Python dependencies
    print("\nInstalling Python dependencies...")
    if not run_command("pip install -r requirements.txt", "Installing dependencies"):
        return
    
    # Run the full setup
    setup_script = "scripts/gcp_full_setup.py"
    if not Path(setup_script).exists():
        print(f"❌ Setup script not found: {setup_script}")
        return
    
    print(f"\nRunning complete GCP setup for project: {PROJECT_ID}")
    print("This will take 5-10 minutes to complete...")
    
    # Run the setup with cost-optimized settings
    setup_command = f"python {setup_script} --project_id {PROJECT_ID} --zone {ZONE} --region {REGION}"
    
    if not run_command(setup_command, "Running GCP infrastructure setup"):
        return
    
    print("\nSetup completed successfully!")
    print("\nNext steps:")
    print("1. Wait for the instance to start (5-10 minutes)")
    print("2. Monitor progress:")
    print(f"   gcloud compute ssh maft-training --zone={ZONE}")
    print("3. View logs:")
    print("   gcloud logging read 'resource.type=gce_instance'")
    print("4. Download results when complete:")
    print(f"   gsutil -m cp -r gs://maft-{PROJECT_ID}-data/maft_results/ ./results/")
    print(f"\nEstimated cost: $10-25 (out of $415.46 credits)")
    print("Instance will auto-shutdown after training to save costs")

if __name__ == '__main__':
    main() 