# MAFT GCP Deployment Guide

This guide provides comprehensive instructions for deploying the MAFT (Multimodal Attention Fusion Transformer) project on Google Cloud Platform (GCP).

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Deployment Options](#deployment-options)
3. [Option 1: Compute Engine with Startup Script](#option-1-compute-engine-with-startup-script)
4. [Option 2: Docker Container Deployment](#option-2-docker-container-deployment)
5. [Option 3: Cloud Run Deployment](#option-3-cloud-run-deployment)
6. [Option 4: Vertex AI Training](#option-4-vertex-ai-training)
7. [Data Management](#data-management)
8. [Monitoring and Logging](#monitoring-and-logging)
9. [Cost Optimization](#cost-optimization)
10. [Troubleshooting](#troubleshooting)

## Prerequisites

### 1. GCP Account Setup
- Create a GCP account and enable billing
- Install and configure [Google Cloud SDK](https://cloud.google.com/sdk/docs/install)
- Enable required APIs:
  ```bash
  gcloud services enable compute.googleapis.com
  gcloud services enable storage.googleapis.com
  gcloud services enable aiplatform.googleapis.com  # For Vertex AI
  gcloud services enable run.googleapis.com  # For Cloud Run
  ```

### 2. Authentication
```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
```

### 3. Repository Setup
```bash
git clone https://github.com/your-username/maft.git
cd maft
```

## Deployment Options

### Option 1: Compute Engine with Startup Script (Recommended)

**Best for**: Full training runs, custom configurations, cost-effective

#### Step 1: Create Compute Engine Instance

```bash
# Create instance with GPU
gcloud compute instances create maft-training \
    --zone=us-central1-a \
    --machine-type=n1-standard-4 \
    --accelerator="type=nvidia-tesla-t4,count=1" \
    --maintenance-policy=TERMINATE \
    --restart-on-failure \
    --image-family=debian-11 \
    --image-project=debian-cloud \
    --boot-disk-size=50GB \
    --boot-disk-type=pd-standard \
    --metadata-from-file startup-script=scripts/gcp_startup.sh \
    --metadata GCS_BUCKET=your-bucket-name,AUTO_SHUTDOWN=true
```

#### Step 2: Monitor Training

```bash
# SSH into instance
gcloud compute ssh maft-training --zone=us-central1-a

# Check logs
tail -f /var/log/syslog | grep maft

# Monitor GPU usage
watch nvidia-smi
```

#### Step 3: Download Results

```bash
# Download results from GCS
gsutil -m cp -r gs://your-bucket-name/maft_results/ ./results/
```

### Option 2: Docker Container Deployment

**Best for**: Reproducible environments, easy scaling

#### Step 1: Build and Push Docker Image

```bash
# Build image
docker build -t gcr.io/YOUR_PROJECT_ID/maft:latest .

# Push to Google Container Registry
docker push gcr.io/YOUR_PROJECT_ID/maft:latest
```

#### Step 2: Deploy with Docker Compose

```bash
# Run full training pipeline
docker-compose up maft

# Run only data preparation
docker-compose --profile data-only up maft-data

# Run only evaluation
docker-compose --profile eval-only up maft-eval

# Run analysis
docker-compose --profile analysis-only up maft-attention maft-efficiency
```

### Option 3: Cloud Run Deployment

**Best for**: Serverless inference, API endpoints

#### Step 1: Create Inference Service

```bash
# Deploy to Cloud Run
gcloud run deploy maft-inference \
    --image gcr.io/YOUR_PROJECT_ID/maft:latest \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated \
    --memory 4Gi \
    --cpu 2 \
    --timeout 3600
```

### Option 4: Vertex AI Training

**Best for**: Managed training, hyperparameter tuning

#### Step 1: Create Training Job

```bash
# Submit training job
gcloud ai custom-jobs create \
    --region=us-central1 \
    --display-name=maft-training \
    --worker-pool-spec=machine-type=n1-standard-4,replica-count=1,container-image-uri=gcr.io/YOUR_PROJECT_ID/maft:latest,accelerator-type=NVIDIA_TESLA_T4,accelerator-count=1
```

## Data Management

### 1. CMU-MOSEI Dataset

#### Automatic Download (Recommended)
```bash
# Download and prepare dataset
python scripts/prepare_mosei.py --output_dir data/mosei
```

#### Manual Download
1. Visit [CMU-MOSEI website](http://immortal.multicomp.cs.cmu.edu/CMU-MOSEI/)
2. Download the dataset
3. Extract features using CMU Multimodal SDK
4. Place in `data/mosei/` directory

### 2. GCS Storage Setup

```bash
# Create bucket
gsutil mb gs://your-maft-bucket

# Upload dataset (if needed)
gsutil -m cp -r data/mosei gs://your-maft-bucket/datasets/

# Download dataset to instance
gsutil -m cp -r gs://your-maft-bucket/datasets/mosei ./data/
```

### 3. Persistent Storage

```bash
# Create persistent disk
gcloud compute disks create maft-data \
    --size=100GB \
    --zone=us-central1-a \
    --type=pd-standard

# Attach to instance
gcloud compute instances attach-disk maft-training \
    --disk=maft-data \
    --zone=us-central1-a
```

## Monitoring and Logging

### 1. Cloud Logging

```bash
# View logs
gcloud logging read "resource.type=gce_instance AND resource.labels.instance_name=maft-training" --limit=50
```

### 2. TensorBoard

```bash
# Start TensorBoard
tensorboard --logdir=experiments/mosei_gcp/logs --port=8080

# Access via SSH tunnel
gcloud compute ssh maft-training --zone=us-central1-a -- -L 8080:localhost:8080
```

### 3. Resource Monitoring

```bash
# Monitor instance
gcloud compute instances describe maft-training --zone=us-central1-a

# Check quotas
gcloud compute regions describe us-central1
```

## Cost Optimization

### 1. Preemptible Instances
```bash
# Use preemptible instances for cost savings
gcloud compute instances create maft-training-preemptible \
    --preemptible \
    --zone=us-central1-a \
    --machine-type=n1-standard-4 \
    --accelerator="type=nvidia-tesla-t4,count=1"
```

### 2. Spot Instances
```bash
# Use spot instances for even more savings
gcloud compute instances create maft-training-spot \
    --provisioning-model=SPOT \
    --zone=us-central1-a \
    --machine-type=n1-standard-4 \
    --accelerator="type=nvidia-tesla-t4,count=1"
```

### 3. Auto-shutdown
```bash
# Set auto-shutdown after training
export AUTO_SHUTDOWN=true
```

## Configuration Files

### 1. GCP-Specific Config

The deployment script automatically creates `configs/mosei_gcp_config.yaml` with optimized settings:

```yaml
# GCP-specific settings
gcp:
  use_tpu: false
  num_gpus: 1
  batch_size: 16  # Increased for GCP
  num_workers: 8  # Increased for GCP
  mixed_precision: true
  gradient_accumulation_steps: 2
```

### 2. Environment Variables

```bash
# Set in startup script or instance metadata
export PYTHONPATH=/app
export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=disabled  # For containerized training
```

## Troubleshooting

### Common Issues

#### 1. GPU Not Detected
```bash
# Check GPU drivers
nvidia-smi

# Install drivers if needed
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
```

#### 2. Out of Memory
```bash
# Reduce batch size in config
batch_size: 8  # Instead of 16

# Use gradient accumulation
gradient_accumulation_steps: 4
```

#### 3. Dataset Download Issues
```bash
# Use mock data for testing
python scripts/prepare_mosei.py --use_mock --output_dir data/mosei

# Check internet connectivity
curl -I https://github.com/A2Zadeh/CMU-MultimodalSDK
```

#### 4. Docker Issues
```bash
# Check Docker service
sudo systemctl status docker

# Restart Docker
sudo systemctl restart docker
```

### Debug Commands

```bash
# Check system resources
htop
nvidia-smi
df -h
free -h

# Check Python environment
python -c "import torch; print(torch.cuda.is_available())"
python -c "import transformers; print(transformers.__version__)"

# Check logs
tail -f experiments/mosei_gcp/training.log
```

## Performance Optimization

### 1. Mixed Precision Training
```yaml
# Enable in config
mixed_precision: true
```

### 2. Data Loading Optimization
```yaml
# Increase workers for faster data loading
num_workers: 8
pin_memory: true
```

### 3. Model Optimization
```yaml
# Use gradient checkpointing
gradient_checkpointing: true
```

## Security Considerations

### 1. IAM Permissions
```bash
# Create service account
gcloud iam service-accounts create maft-service \
    --display-name="MAFT Service Account"

# Grant necessary permissions
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:maft-service@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/storage.admin"
```

### 2. Network Security
```bash
# Create firewall rules
gcloud compute firewall-rules create maft-allow-ssh \
    --allow tcp:22 \
    --source-ranges 0.0.0.0/0 \
    --target-tags maft-training
```

## Support

For issues and questions:
1. Check the [troubleshooting section](#troubleshooting)
2. Review logs in Cloud Logging
3. Open an issue on the GitHub repository
4. Contact the development team

## Cost Estimation

### Typical Costs (US Central 1)
- **Compute Engine (n1-standard-4 + T4)**: ~$0.50/hour
- **Storage (50GB)**: ~$0.02/hour
- **Network**: ~$0.10/GB

**Total for 24-hour training**: ~$12-15

### Cost Reduction Tips
1. Use preemptible instances (60-80% savings)
2. Use spot instances (90% savings)
3. Auto-shutdown after training
4. Use smaller instances for development
5. Store data in cheaper storage classes 