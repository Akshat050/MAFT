#!/usr/bin/env python3
"""
Simple GCP Setup for MAFT

This script prompts for your GCP credentials and sets up everything automatically.
"""

import os
import sys
import subprocess
from pathlib import Path

def run_command(command, description):
    """Run a command with error handling."""
    print(f"\n🔄 {description}")
    print(f"📝 Command: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        if result.stdout:
            print(f"✅ Success: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e.stderr}")
        return False

def main():
    print("🚀 MAFT GCP Setup")
    print("==================")
    
    # Get GCP Project ID
    project_id = input("📋 Enter your GCP Project ID: ").strip()
    if not project_id:
        print("❌ Project ID is required!")
        return
    
    # Get preferred zone
    zone = input("📍 Enter preferred zone (default: us-central1-a): ").strip()
    if not zone:
        zone = "us-central1-a"
    
    # Get region from zone
    region = zone.rsplit('-', 1)[0]
    
    print(f"\n📊 Setup Configuration:")
    print(f"Project ID: {project_id}")
    print(f"Zone: {zone}")
    print(f"Region: {region}")
    
    # Confirm setup
    confirm = input("\n🤔 Proceed with setup? (y/N): ").strip().lower()
    if confirm not in ['y', 'yes']:
        print("❌ Setup cancelled.")
        return
    
    # Check if gcloud is installed
    if not run_command("gcloud --version", "Checking Google Cloud SDK"):
        print("❌ Google Cloud SDK not found. Please install it first:")
        print("   https://cloud.google.com/sdk/docs/install")
        return
    
    # Set project
    if not run_command(f"gcloud config set project {project_id}", "Setting GCP project"):
        return
    
    # Authenticate
    print("\n🔐 Authentication required...")
    if not run_command("gcloud auth login", "Authenticating with GCP"):
        return
    
    # Set application default credentials
    if not run_command("gcloud auth application-default login", "Setting application default credentials"):
        return
    
    # Install Python dependencies
    if not run_command("pip install -r requirements.txt", "Installing Python dependencies"):
        return
    
    # Run the full setup
    setup_script = "scripts/gcp_full_setup.py"
    if not Path(setup_script).exists():
        print(f"❌ Setup script not found: {setup_script}")
        return
    
    print("\n🎯 Running complete GCP setup...")
    if not run_command(f"python {setup_script} --project_id {project_id} --zone {zone} --region {region}", 
                      "Running GCP setup"):
        return
    
    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Wait for the instance to start (5-10 minutes)")
    print("2. Monitor progress:")
    print(f"   gcloud compute ssh maft-training --zone={zone}")
    print("3. View logs:")
    print("   gcloud logging read 'resource.type=gce_instance'")
    print("4. Download results when complete:")
    print(f"   gsutil -m cp -r gs://maft-{project_id}-data/maft_results/ ./results/")

if __name__ == '__main__':
    main() 