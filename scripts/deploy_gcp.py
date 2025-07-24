#!/usr/bin/env python3
"""
GCP Deployment Script for MAFT

This script automates the deployment of the MAFT project on Google Cloud Platform.
It handles environment setup, data preparation, and training execution.
"""

import os
import sys
import subprocess
import argparse
import json
import time
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_command(command, description, check=True):
    """Run a shell command with error handling."""
    print(f"\n🔄 {description}")
    print(f"📝 Command: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=check, capture_output=True, text=True)
        if result.stdout:
            print(f"✅ Output: {result.stdout}")
        return result
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        print(f"📄 Error output: {e.stderr}")
        if check:
            raise
        return e


def setup_gcp_environment():
    """Set up GCP environment and dependencies."""
    print("🚀 Setting up GCP environment...")
    
    # Update system packages
    run_command("sudo apt-get update", "Updating system packages")
    run_command("sudo apt-get install -y git wget curl unzip", "Installing basic tools")
    
    # Install Python dependencies
    run_command("pip install --upgrade pip", "Upgrading pip")
    run_command("pip install -r requirements.txt", "Installing Python dependencies")
    
    # Install additional system dependencies for audio/visual processing
    run_command("sudo apt-get install -y ffmpeg libsm6 libxext6", "Installing audio/visual dependencies")
    
    print("✅ GCP environment setup completed")


def download_and_prepare_data(data_dir="data/mosei", use_mock=False):
    """Download and prepare the CMU-MOSEI dataset."""
    print("📥 Preparing dataset...")
    
    # Create data directory
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    
    # Run dataset preparation script
    mock_flag = "--use_mock" if use_mock else ""
    run_command(f"python scripts/prepare_mosei.py --output_dir {data_dir} {mock_flag}", 
                "Preparing CMU-MOSEI dataset")
    
    print("✅ Dataset preparation completed")


def run_training(config_path="configs/mosei_config.yaml", output_dir="experiments/mosei_gcp"):
    """Run the MAFT training pipeline."""
    print("🎯 Starting MAFT training...")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Run training
    run_command(f"python train.py --config {config_path} --output_dir {output_dir}", 
                "Running MAFT training")
    
    print("✅ Training completed")


def run_evaluation(config_path="configs/mosei_config.yaml", model_path=None):
    """Run evaluation on the trained model."""
    print("📊 Running evaluation...")
    
    model_arg = f"--model_path {model_path}" if model_path else ""
    run_command(f"python evaluate.py --config {config_path} {model_arg}", 
                "Running evaluation")
    
    print("✅ Evaluation completed")


def run_attention_analysis(config_path="configs/mosei_config.yaml", model_path=None):
    """Run attention analysis on the trained model."""
    print("🔍 Running attention analysis...")
    
    model_arg = f"--model_path {model_path}" if model_path else ""
    run_command(f"python scripts/analyze_attention.py --config {config_path} {model_arg}", 
                "Running attention analysis")
    
    print("✅ Attention analysis completed")


def run_efficiency_analysis(config_path="configs/mosei_config.yaml"):
    """Run efficiency analysis."""
    print("⚡ Running efficiency analysis...")
    
    run_command(f"python scripts/efficiency_analysis.py --config {config_path}", 
                "Running efficiency analysis")
    
    print("✅ Efficiency analysis completed")


def upload_to_gcs(local_path, gcs_bucket, gcs_path):
    """Upload files to Google Cloud Storage."""
    print(f"☁️ Uploading {local_path} to GCS...")
    
    run_command(f"gsutil -m cp -r {local_path} gs://{gcs_bucket}/{gcs_path}", 
                f"Uploading to GCS: gs://{gcs_bucket}/{gcs_path}")
    
    print(f"✅ Upload completed: gs://{gcs_bucket}/{gcs_path}")


def create_gcp_config():
    """Create GCP-specific configuration."""
    print("⚙️ Creating GCP configuration...")
    
    # Read base config
    with open("configs/mosei_config.yaml", 'r') as f:
        config = f.read()
    
    # Add GCP-specific settings
    gcp_config = config + """
# GCP-specific settings
gcp:
  use_tpu: false
  num_gpus: 1
  batch_size: 16  # Increased for GCP
  num_workers: 8  # Increased for GCP
  mixed_precision: true
  gradient_accumulation_steps: 2
"""
    
    # Save GCP config
    with open("configs/mosei_gcp_config.yaml", 'w') as f:
        f.write(gcp_config)
    
    print("✅ GCP configuration created")


def monitor_resources():
    """Monitor system resources during training."""
    print("📊 Monitoring system resources...")
    
    # Get GPU info
    try:
        result = run_command("nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv", 
                           "Getting GPU information", check=False)
        if result.returncode == 0:
            print(f"🖥️ GPU Info:\n{result.stdout}")
    except:
        print("⚠️ No GPU detected or nvidia-smi not available")
    
    # Get memory info
    try:
        result = run_command("free -h", "Getting memory information", check=False)
        if result.returncode == 0:
            print(f"💾 Memory Info:\n{result.stdout}")
    except:
        print("⚠️ Could not get memory information")
    
    # Get disk space
    try:
        result = run_command("df -h", "Getting disk space information", check=False)
        if result.returncode == 0:
            print(f"💿 Disk Space:\n{result.stdout}")
    except:
        print("⚠️ Could not get disk space information")


def main():
    parser = argparse.ArgumentParser(description='Deploy MAFT on GCP')
    parser.add_argument('--setup_only', action='store_true',
                       help='Only set up the environment, skip training')
    parser.add_argument('--use_mock_data', action='store_true',
                       help='Use mock data instead of downloading real CMU-MOSEI')
    parser.add_argument('--data_dir', type=str, default='data/mosei',
                       help='Directory for dataset')
    parser.add_argument('--output_dir', type=str, default='experiments/mosei_gcp',
                       help='Output directory for experiments')
    parser.add_argument('--gcs_bucket', type=str, default=None,
                       help='GCS bucket for uploading results')
    parser.add_argument('--gcs_path', type=str, default='maft_results',
                       help='GCS path for uploading results')
    parser.add_argument('--skip_analysis', action='store_true',
                       help='Skip attention and efficiency analysis')
    
    args = parser.parse_args()
    
    print("🚀 Starting MAFT GCP Deployment")
    print(f"📅 Start time: {datetime.now()}")
    
    try:
        # Step 1: Environment setup
        setup_gcp_environment()
        
        # Step 2: Create GCP-specific config
        create_gcp_config()
        
        if args.setup_only:
            print("✅ Environment setup completed. Exiting.")
            return
        
        # Step 3: Download and prepare data
        download_and_prepare_data(args.data_dir, args.use_mock_data)
        
        # Step 4: Monitor initial resources
        monitor_resources()
        
        # Step 5: Run training
        run_training("configs/mosei_gcp_config.yaml", args.output_dir)
        
        # Step 6: Run evaluation
        model_path = f"{args.output_dir}/best_model.pth"
        run_evaluation("configs/mosei_gcp_config.yaml", model_path)
        
        # Step 7: Run analysis (optional)
        if not args.skip_analysis:
            run_attention_analysis("configs/mosei_gcp_config.yaml", model_path)
            run_efficiency_analysis("configs/mosei_gcp_config.yaml")
        
        # Step 8: Upload results to GCS (if specified)
        if args.gcs_bucket:
            upload_to_gcs(args.output_dir, args.gcs_bucket, args.gcs_path)
        
        # Step 9: Final resource monitoring
        monitor_resources()
        
        print(f"\n🎉 MAFT GCP deployment completed successfully!")
        print(f"📅 End time: {datetime.now()}")
        print(f"📁 Results saved to: {args.output_dir}")
        if args.gcs_bucket:
            print(f"☁️ Results uploaded to: gs://{args.gcs_bucket}/{args.gcs_path}")
        
    except Exception as e:
        print(f"❌ Deployment failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main() 