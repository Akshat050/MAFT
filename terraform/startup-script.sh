#!/bin/bash

# MAFT GCP Startup Script
# This script sets up the environment for MAFT training on GCP

set -e

# Variables
PROJECT_ID="${project_id}"
BUCKET_NAME="${bucket_name}"
INSTANCE_NAME="maft-training-instance"

# Log function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

log "Starting MAFT GCP setup..."

# Update system
log "Updating system packages..."
apt-get update
apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    git \
    wget \
    curl \
    unzip \
    software-properties-common \
    apt-transport-https \
    ca-certificates \
    gnupg \
    lsb-release

# Install Docker
log "Installing Docker..."
curl -fsSL https://download.docker.com/linux/debian/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/debian $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null
apt-get update
apt-get install -y docker-ce docker-ce-cli containerd.io

# Install NVIDIA Docker runtime
log "Installing NVIDIA Docker runtime..."
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | tee /etc/apt/sources.list.d/nvidia-docker.list
apt-get update
apt-get install -y nvidia-docker2
systemctl restart docker

# Install Google Cloud SDK
log "Installing Google Cloud SDK..."
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -
echo "deb https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
apt-get update
apt-get install -y google-cloud-sdk

# Authenticate with GCP (using service account)
log "Setting up GCP authentication..."
gcloud config set project $PROJECT_ID

# Create application directory
log "Setting up application directory..."
mkdir -p /opt/maft
cd /opt/maft

# Clone or copy MAFT code (in production, you'd copy from Cloud Storage)
log "Setting up MAFT codebase..."
# For now, we'll create a basic structure
mkdir -p {src,data,models,logs,scripts}

# Create Python virtual environment
log "Setting up Python environment..."
python3 -m venv /opt/maft/venv
source /opt/maft/venv/bin/activate

# Install Python dependencies
log "Installing Python dependencies..."
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets numpy pandas scikit-learn matplotlib seaborn
pip install tensorboard wandb pyyaml tqdm librosa opencv-python
pip install facenet-pytorch pytorch-lightning omegaconf hydra-core
pip install google-cloud-storage google-cloud-compute google-auth google-auth-oauthlib
pip install requests filelock

# Create a simple health check script
cat > /opt/maft/health_check.py << 'EOF'
#!/usr/bin/env python3
import http.server
import socketserver
import os

PORT = 8080

class HealthCheckHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/health':
            self.send_response(200)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(b'OK')
        else:
            self.send_response(404)
            self.end_headers()

if __name__ == '__main__':
    with socketserver.TCPServer(("", PORT), HealthCheckHandler) as httpd:
        print(f"Health check server running on port {PORT}")
        httpd.serve_forever()
EOF

# Create systemd service for health check
cat > /etc/systemd/system/maft-health.service << EOF
[Unit]
Description=MAFT Health Check Service
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/maft
ExecStart=/opt/maft/venv/bin/python /opt/maft/health_check.py
Restart=always

[Install]
WantedBy=multi-user.target
EOF

# Start health check service
systemctl daemon-reload
systemctl enable maft-health
systemctl start maft-health

# Create training script
cat > /opt/maft/train_maft.py << 'EOF'
#!/usr/bin/env python3
"""
MAFT Training Script for GCP
This script runs the MAFT training pipeline on GCP
"""

import os
import sys
import logging
import subprocess
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/opt/maft/logs/training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

def run_command(command, description):
    """Run a command and log the result"""
    logging.info(f"Running: {description}")
    logging.info(f"Command: {command}")
    
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            check=True, 
            capture_output=True, 
            text=True
        )
        logging.info(f"Success: {description}")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Error: {description}")
        logging.error(f"Error output: {e.stderr}")
        return False

def main():
    """Main training function"""
    logging.info("Starting MAFT training on GCP")
    
    # Set environment variables
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['PYTHONPATH'] = '/opt/maft/src'
    
    # Create necessary directories
    os.makedirs('/opt/maft/data', exist_ok=True)
    os.makedirs('/opt/maft/models', exist_ok=True)
    os.makedirs('/opt/maft/logs', exist_ok=True)
    
    # Download dataset (placeholder - you'll need to implement this)
    logging.info("Setting up dataset...")
    # run_command("python /opt/maft/src/data/prepare_dataset.py", "Dataset preparation")
    
    # Run training (placeholder - you'll need to implement this)
    logging.info("Starting training...")
    # run_command("python /opt/maft/src/train.py", "Training")
    
    # Upload results to GCS
    logging.info("Uploading results to GCS...")
    bucket_name = os.environ.get('BUCKET_NAME', 'maft-465719-maft-data')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    run_command(
        f"gsutil -m cp -r /opt/maft/models gs://{bucket_name}/models/{timestamp}/",
        "Uploading models to GCS"
    )
    
    run_command(
        f"gsutil -m cp -r /opt/maft/logs gs://{bucket_name}/logs/{timestamp}/",
        "Uploading logs to GCS"
    )
    
    logging.info("MAFT training completed successfully")

if __name__ == "__main__":
    main()
EOF

# Make scripts executable
chmod +x /opt/maft/train_maft.py
chmod +x /opt/maft/health_check.py

# Create a simple training launcher
cat > /opt/maft/launch_training.sh << 'EOF'
#!/bin/bash
cd /opt/maft
source venv/bin/activate
python train_maft.py
EOF

chmod +x /opt/maft/launch_training.sh

# Set up automatic shutdown after training (optional)
cat > /opt/maft/schedule_shutdown.sh << 'EOF'
#!/bin/bash
# Schedule shutdown after 8 hours (adjust as needed)
sleep 28800  # 8 hours
sudo shutdown -h now
EOF

chmod +x /opt/maft/schedule_shutdown.sh

# Start training in background
log "Launching MAFT training..."
nohup /opt/maft/launch_training.sh > /opt/maft/logs/launch.log 2>&1 &

# Schedule shutdown
nohup /opt/maft/schedule_shutdown.sh > /opt/maft/logs/shutdown.log 2>&1 &

log "MAFT GCP setup completed successfully!"
log "Training started in background"
log "Instance will auto-shutdown after 8 hours"
log "Check logs at: /opt/maft/logs/" 