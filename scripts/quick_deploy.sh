#!/bin/bash
# Quick Deploy Script for MAFT on GCP
# This script automates the entire deployment process

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ID=""
ZONE="us-central1-a"
INSTANCE_NAME="maft-training"
MACHINE_TYPE="n1-standard-4"
GPU_TYPE="nvidia-tesla-t4"
GPU_COUNT="1"
DISK_SIZE="50GB"
BUCKET_NAME=""

echo -e "${BLUE}🚀 MAFT Quick Deploy Script${NC}"
echo "=================================="

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo -e "${RED}❌ Google Cloud SDK not found. Please install it first.${NC}"
    echo "Visit: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Get project ID
if [ -z "$PROJECT_ID" ]; then
    echo -e "${YELLOW}📋 Enter your GCP Project ID:${NC}"
    read PROJECT_ID
fi

# Set project
echo -e "${BLUE}🔧 Setting project to: $PROJECT_ID${NC}"
gcloud config set project $PROJECT_ID

# Get bucket name
if [ -z "$BUCKET_NAME" ]; then
    echo -e "${YELLOW}📦 Enter GCS bucket name for results (or press Enter to skip):${NC}"
    read BUCKET_NAME
fi

# Enable required APIs
echo -e "${BLUE}🔌 Enabling required APIs...${NC}"
gcloud services enable compute.googleapis.com
gcloud services enable storage.googleapis.com
gcloud services enable aiplatform.googleapis.com

# Create GCS bucket if specified
if [ ! -z "$BUCKET_NAME" ]; then
    echo -e "${BLUE}🪣 Creating GCS bucket: $BUCKET_NAME${NC}"
    gsutil mb -l us-central1 gs://$BUCKET_NAME 2>/dev/null || echo "Bucket already exists or creation failed"
fi

# Check if instance already exists
if gcloud compute instances describe $INSTANCE_NAME --zone=$ZONE &>/dev/null; then
    echo -e "${YELLOW}⚠️ Instance $INSTANCE_NAME already exists.${NC}"
    echo -e "${YELLOW}Do you want to delete it and create a new one? (y/N)${NC}"
    read -r response
    if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        echo -e "${BLUE}🗑️ Deleting existing instance...${NC}"
        gcloud compute instances delete $INSTANCE_NAME --zone=$ZONE --quiet
    else
        echo -e "${YELLOW}Using existing instance.${NC}"
        echo -e "${GREEN}✅ Instance is ready at:${NC}"
        echo "gcloud compute ssh $INSTANCE_NAME --zone=$ZONE"
        exit 0
    fi
fi

# Create startup script with bucket configuration
STARTUP_SCRIPT="scripts/gcp_startup.sh"
if [ ! -z "$BUCKET_NAME" ]; then
    # Create modified startup script with bucket
    cp $STARTUP_SCRIPT "${STARTUP_SCRIPT}.tmp"
    sed -i "s/export GCS_BUCKET=/export GCS_BUCKET=$BUCKET_NAME/" "${STARTUP_SCRIPT}.tmp"
    STARTUP_SCRIPT="${STARTUP_SCRIPT}.tmp"
fi

# Create Compute Engine instance
echo -e "${BLUE}🖥️ Creating Compute Engine instance...${NC}"
gcloud compute instances create $INSTANCE_NAME \
    --zone=$ZONE \
    --machine-type=$MACHINE_TYPE \
    --accelerator="type=$GPU_TYPE,count=$GPU_COUNT" \
    --maintenance-policy=TERMINATE \
    --restart-on-failure \
    --image-family=debian-11 \
    --image-project=debian-cloud \
    --boot-disk-size=$DISK_SIZE \
    --boot-disk-type=pd-standard \
    --metadata-from-file startup-script=$STARTUP_SCRIPT \
    --metadata AUTO_SHUTDOWN=true \
    --scopes=cloud-platform

# Clean up temporary startup script
if [ -f "${STARTUP_SCRIPT}.tmp" ]; then
    rm "${STARTUP_SCRIPT}.tmp"
fi

echo -e "${GREEN}✅ Instance created successfully!${NC}"
echo ""
echo -e "${BLUE}📊 Instance Details:${NC}"
echo "Name: $INSTANCE_NAME"
echo "Zone: $ZONE"
echo "Machine Type: $MACHINE_TYPE"
echo "GPU: $GPU_COUNT x $GPU_TYPE"
echo "Disk Size: $DISK_SIZE"

echo ""
echo -e "${BLUE}🔍 Monitoring Commands:${NC}"
echo "SSH into instance:"
echo "  gcloud compute ssh $INSTANCE_NAME --zone=$ZONE"
echo ""
echo "View startup logs:"
echo "  gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --command='tail -f /var/log/syslog | grep maft'"
echo ""
echo "Monitor GPU usage:"
echo "  gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --command='watch nvidia-smi'"
echo ""

if [ ! -z "$BUCKET_NAME" ]; then
    echo -e "${BLUE}☁️ Results will be uploaded to:${NC}"
    echo "  gs://$BUCKET_NAME/maft_results/"
    echo ""
    echo "Download results:"
    echo "  gsutil -m cp -r gs://$BUCKET_NAME/maft_results/ ./results/"
fi

echo ""
echo -e "${YELLOW}⏳ The instance will automatically:${NC}"
echo "1. Install all dependencies"
echo "2. Download and prepare CMU-MOSEI dataset"
echo "3. Train the MAFT model"
echo "4. Run evaluation and analysis"
echo "5. Upload results to GCS (if bucket specified)"
echo "6. Shutdown the instance"

echo ""
echo -e "${GREEN}🎉 Deployment initiated! Check the instance logs for progress.${NC}"

# Optional: Wait and show logs
echo ""
echo -e "${YELLOW}Would you like to wait and show the startup logs? (y/N)${NC}"
read -r response
if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
    echo -e "${BLUE}📋 Waiting for instance to start and showing logs...${NC}"
    sleep 30
    gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --command='tail -f /var/log/syslog | grep maft'
fi 