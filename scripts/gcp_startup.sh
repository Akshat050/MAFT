#!/bin/bash
# GCP Startup Script for MAFT
# This script runs automatically when a GCP Compute Engine instance starts

set -e  # Exit on any error

echo "🚀 Starting MAFT GCP deployment..."

# Update system
echo "📦 Updating system packages..."
sudo apt-get update
sudo apt-get install -y git wget curl unzip ffmpeg libsm6 libxext6

# Install Docker if not present
if ! command -v docker &> /dev/null; then
    echo "🐳 Installing Docker..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker $USER
fi

# Install Docker Compose if not present
if ! command -v docker-compose &> /dev/null; then
    echo "🐳 Installing Docker Compose..."
    sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
fi

# Setup MAFT repository (if not already present)
if [ ! -d "/app/maft" ]; then
    echo "Setting up MAFT repository..."
    sudo mkdir -p /app
    sudo chown $USER:$USER /app
    cd /app
    mkdir -p maft
    cd maft
else
    echo "MAFT repository already exists..."
    cd /app/maft
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p data/mosei experiments logs

# Set environment variables
export PYTHONPATH=/app/maft
export CUDA_VISIBLE_DEVICES=0

# Check if GPU is available
if command -v nvidia-smi &> /dev/null; then
    echo "🖥️ GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
else
    echo "⚠️ No GPU detected, will use CPU"
fi

# Run MAFT deployment
echo "🎯 Starting MAFT training..."
python scripts/deploy_gcp.py \
    --use_mock_data \
    --data_dir data/mosei \
    --output_dir experiments/mosei_gcp \
    --skip_analysis

echo "✅ MAFT deployment completed!"

# Optional: Upload results to GCS if bucket is specified
if [ ! -z "$GCS_BUCKET" ]; then
    echo "☁️ Uploading results to GCS..."
    gsutil -m cp -r experiments/mosei_gcp gs://$GCS_BUCKET/maft_results/
    echo "✅ Results uploaded to gs://$GCS_BUCKET/maft_results/"
fi

# Shutdown instance if specified
if [ "$AUTO_SHUTDOWN" = "true" ]; then
    echo "🔄 Shutting down instance..."
    sudo shutdown -h now
fi 