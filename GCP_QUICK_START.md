# MAFT GCP Quick Start Guide

This guide will help you deploy the MAFT project on Google Cloud Platform with minimal setup.

## What You Need to Provide

### 1. GCP Project ID
- Your GCP project identifier (e.g., `my-maft-project-123`)
- You can find this in the GCP Console or run: `gcloud config get-value project`

### 2. GCP Authentication
- Google Cloud SDK installed and authenticated
- Or a service account key file (optional)

### 3. Budget Preferences
- Standard instances: ~$0.50/hour
- Preemptible instances: ~$0.10/hour (80% savings)
- Spot instances: ~$0.05/hour (90% savings)

## Quick Setup (Recommended)

### Step 1: Install Google Cloud SDK
```bash
# Download and install from:
# https://cloud.google.com/sdk/docs/install

# Or use package manager:
# macOS: brew install google-cloud-sdk
# Ubuntu: sudo apt-get install google-cloud-sdk
```

### Step 2: Run the Setup Script
```bash
# Run the automated setup
python setup_gcp.py
```

The script will prompt you for:
- GCP Project ID
- Preferred zone (default: us-central1-a)
- Confirmation to proceed

### Step 3: Wait for Setup
The script will automatically:
1. ✅ Enable required GCP APIs
2. ✅ Create GCS bucket for data storage
3. ✅ Create persistent disk for datasets
4. ✅ Create Compute Engine instance with GPU
5. ✅ Download CMU-MOSEI dataset to GCS
6. ✅ Start training pipeline
7. ✅ Upload results to GCS

## What Gets Created

### 1. GCS Bucket
- **Name**: `maft-{project-id}-data`
- **Purpose**: Store datasets, models, and results
- **Location**: Your chosen region

### 2. Persistent Disk
- **Name**: `maft-dataset-disk`
- **Size**: 100GB
- **Purpose**: Store CMU-MOSEI dataset locally on instance

### 3. Compute Engine Instance
- **Name**: `maft-training`
- **Machine Type**: n1-standard-4 (4 vCPU, 15GB RAM)
- **GPU**: NVIDIA Tesla T4
- **Disk**: 50GB boot disk + 100GB data disk
- **Auto-shutdown**: Yes (after training completes)

## Monitoring Your Deployment

### Check Instance Status
```bash
# Get instance details
gcloud compute instances describe maft-training --zone=us-central1-a

# Get external IP
gcloud compute instances list --filter="name=maft-training"
```

### SSH into Instance
```bash
# Connect to instance
gcloud compute ssh maft-training --zone=us-central1-a

# Check training progress
tail -f /var/log/syslog | grep maft

# Monitor GPU usage
watch nvidia-smi
```

### View Logs
```bash
# View startup logs
gcloud logging read "resource.type=gce_instance AND resource.labels.instance_name=maft-training" --limit=50

# View real-time logs
gcloud logging tail "resource.type=gce_instance AND resource.labels.instance_name=maft-training"
```

## Downloading Results

### When Training Completes
```bash
# Download all results
gsutil -m cp -r gs://maft-{project-id}-data/maft_results/ ./results/

# Download specific files
gsutil cp gs://maft-{project-id}-data/maft_results/best_model.pth ./
gsutil cp gs://maft-{project-id}-data/maft_results/evaluation_results.json ./
```

## Cost Management

### Estimated Costs (US Central 1)
- **Standard Instance**: ~$12-15 for 24 hours
- **Preemptible Instance**: ~$2-3 for 24 hours
- **Storage**: ~$0.50/month for 100GB

### Cost Optimization
```bash
# Use preemptible instances (add to setup)
--preemptible

# Use spot instances (add to setup)
--provisioning-model=SPOT

# Auto-shutdown after training
--metadata AUTO_SHUTDOWN=true
```

## Troubleshooting

### Common Issues

#### 1. Authentication Errors
```bash
# Re-authenticate
gcloud auth login
gcloud auth application-default login

# Check current account
gcloud auth list
```

#### 2. Quota Exceeded
```bash
# Check quotas
gcloud compute regions describe us-central1

# Request quota increase in GCP Console
# https://console.cloud.google.com/iam-admin/quotas
```

#### 3. GPU Not Available
```bash
# Check GPU availability
gcloud compute accelerator-types list --filter="zone:us-central1-a"

# Use different zone
--zone=us-west1-b
```

#### 4. Instance Won't Start
```bash
# Check startup script logs
gcloud compute ssh maft-training --zone=us-central1-a --command='sudo journalctl -u google-startup-scripts.service'

# Check instance logs
gcloud logging read "resource.type=gce_instance AND resource.labels.instance_name=maft-training AND severity>=ERROR"
```

### Manual Recovery
```bash
# Delete and recreate instance
gcloud compute instances delete maft-training --zone=us-central1-a
python scripts/gcp_full_setup.py --project_id YOUR_PROJECT_ID --zone us-central1-a

# Restart training on existing instance
gcloud compute ssh maft-training --zone=us-central1-a --command='cd /home/maft && python scripts/deploy_gcp.py'
```

## Advanced Configuration

### Custom Machine Types
```bash
# Use larger instance
python scripts/gcp_full_setup.py --project_id YOUR_PROJECT_ID --machine_type n1-standard-8

# Use different GPU
python scripts/gcp_full_setup.py --project_id YOUR_PROJECT_ID --gpu_type nvidia-tesla-v100
```

### Multiple Instances
```bash
# Create multiple training instances
python scripts/gcp_full_setup.py --project_id YOUR_PROJECT_ID --instance_name maft-training-1
python scripts/gcp_full_setup.py --project_id YOUR_PROJECT_ID --instance_name maft-training-2
```

### Custom Dataset
```bash
# Upload your own dataset
gsutil -m cp -r your_dataset/ gs://maft-{project-id}-data/datasets/custom/

# Modify startup script to use custom dataset
```

## Support

If you encounter issues:

1. **Check the logs**: Use the monitoring commands above
2. **Review this guide**: Common solutions are listed
3. **Check GCP Console**: Visit https://console.cloud.google.com
4. **Open an issue**: On the GitHub repository

## Next Steps

After successful deployment:

1. **Analyze Results**: Review training logs and metrics
2. **Run Analysis**: Execute attention and efficiency analysis
3. **Deploy Model**: Use the trained model for inference
4. **Scale Up**: Run hyperparameter tuning or larger experiments
5. **Cost Optimization**: Switch to preemptible instances for cost savings 