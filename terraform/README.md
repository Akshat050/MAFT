# MAFT GCP Terraform Deployment

This directory contains the Terraform configuration for deploying the MAFT (Multimodal Attention Fusion Transformer) infrastructure on Google Cloud Platform.

## Prerequisites

1. **Google Cloud SDK** installed and authenticated
2. **Terraform** installed (version >= 1.0)
3. **GCP Project** with billing enabled
4. **Sufficient permissions** to create resources in your GCP project

## Quick Start

### 1. Install Terraform

Download and install Terraform from [terraform.io](https://www.terraform.io/downloads.html)

### 2. Set up GCP Authentication

```bash
# Login to GCP
gcloud auth login

# Set your project
gcloud config set project maft-465719

# Verify authentication
gcloud auth list
```

### 3. Deploy Infrastructure

#### Option A: Using the Python deployment script (Recommended)

```bash
# From the project root directory
python deploy_maft_gcp.py
```

#### Option B: Manual Terraform deployment

```bash
# Navigate to terraform directory
cd terraform

# Initialize Terraform
terraform init

# Validate configuration
terraform validate

# Plan deployment
terraform plan

# Apply deployment
terraform apply
```

## Infrastructure Components

The Terraform configuration creates the following resources:

### 1. **APIs Enabled**
- Compute Engine API
- Cloud Storage API
- AI Platform API
- Cloud Build API
- Container Registry API
- Logging API
- Monitoring API

### 2. **Networking**
- Custom VPC network (`maft-network`)
- Subnet (`maft-subnet`) with CIDR `10.0.0.0/24`
- Firewall rules for SSH, HTTP, and custom ports

### 3. **Storage**
- Cloud Storage bucket for datasets and results
- Lifecycle policy to automatically delete old data after 30 days

### 4. **Compute**
- Service account with necessary permissions
- Instance template with GPU support
- Managed instance group for auto-scaling and health checks

### 5. **Security**
- Service account with minimal required permissions
- Firewall rules for secure access
- IAM roles for storage, logging, and monitoring

## Configuration

### Variables

Edit `terraform.tfvars` to customize the deployment:

```hcl
project_id   = "maft-465719"        # Your GCP project ID
region       = "us-central1"        # GCP region
zone         = "us-central1-a"      # GCP zone
machine_type = "n1-standard-4"      # Instance type (4 vCPUs, 15 GB RAM)
gpu_type     = "nvidia-tesla-t4"    # GPU type
gpu_count    = 1                    # Number of GPUs
disk_size_gb = 100                  # Boot disk size
```

### Cost Optimization

The current configuration uses:
- **n1-standard-4** instance (4 vCPUs, 15 GB RAM) - ~$0.19/hour
- **nvidia-tesla-t4** GPU - ~$0.35/hour
- **Total estimated cost**: ~$0.54/hour (~$13/day)

For cost optimization, consider:
- Using preemptible instances (50% cost reduction)
- Smaller instance types for development
- Auto-shutdown after training completion

## Monitoring and Management

### Check Instance Status

```bash
# List all MAFT instances
gcloud compute instances list --filter='name~maft'

# Check instance details
gcloud compute instances describe [INSTANCE_NAME] --zone=us-central1-a
```

### SSH into Instance

```bash
# SSH into the training instance
gcloud compute ssh [INSTANCE_NAME] --zone=us-central1-a
```

### Monitor Logs

```bash
# View instance logs
gcloud logging read 'resource.type=gce_instance AND resource.labels.instance_name~maft' --limit=50

# View startup script logs
gcloud logging read 'resource.type=gce_instance AND textPayload:"MAFT GCP setup"' --limit=20
```

### Check Health Status

```bash
# Get instance IP
INSTANCE_IP=$(gcloud compute instances list --filter='name~maft' --format='value(EXTERNAL_IP)')

# Check health endpoint
curl http://$INSTANCE_IP:8080/health
```

### Cloud Storage Management

```bash
# List bucket contents
gsutil ls gs://maft-465719-maft-data/

# Upload data
gsutil -m cp -r /path/to/data gs://maft-465719-maft-data/data/

# Download results
gsutil -m cp -r gs://maft-465719-maft-data/models/ /path/to/local/

# Sync local changes
gsutil -m rsync -r /path/to/local gs://maft-465719-maft-data/
```

## Training Pipeline

The startup script automatically:

1. **Installs dependencies** (Python, PyTorch, CUDA, etc.)
2. **Sets up the environment** (virtual environment, directories)
3. **Downloads datasets** (from Cloud Storage)
4. **Runs training** (MAFT model training)
5. **Uploads results** (models, logs, metrics to Cloud Storage)
6. **Auto-shutdown** (after 8 hours to save costs)

### Manual Training Control

SSH into the instance and run:

```bash
# Check training status
tail -f /opt/maft/logs/training.log

# Manually start training
cd /opt/maft
source venv/bin/activate
python train_maft.py

# Check GPU usage
nvidia-smi

# Monitor system resources
htop
```

## Troubleshooting

### Common Issues

1. **Permission Denied**
   ```bash
   # Ensure you have the necessary IAM roles
   gcloud projects add-iam-policy-binding maft-465719 \
     --member="user:your-email@gmail.com" \
     --role="roles/editor"
   ```

2. **GPU Not Available**
   ```bash
   # Check GPU quotas
   gcloud compute regions describe us-central1 --format='value(quotas)'
   
   # Request GPU quota increase if needed
   ```

3. **Instance Won't Start**
   ```bash
   # Check startup script logs
   gcloud logging read 'resource.type=gce_instance AND textPayload:"startup-script"' --limit=20
   ```

4. **Storage Issues**
   ```bash
   # Check bucket permissions
   gsutil iam get gs://maft-465719-maft-data
   ```

### Cleanup

To destroy the infrastructure:

```bash
cd terraform
terraform destroy
```

**Warning**: This will delete all resources including the Cloud Storage bucket and all data.

## Cost Monitoring

Monitor your costs in the GCP Console:
1. Go to [GCP Billing](https://console.cloud.google.com/billing)
2. Select your project
3. View cost breakdown by service

Set up billing alerts to avoid unexpected charges.

## Security Best Practices

1. **Use service accounts** with minimal permissions
2. **Enable VPC firewall rules** to restrict access
3. **Regularly rotate credentials**
4. **Monitor access logs**
5. **Use private subnets** for production deployments

## Support

For issues with this deployment:
1. Check the troubleshooting section above
2. Review GCP documentation
3. Check Terraform logs and GCP console
4. Monitor instance logs for specific errors 