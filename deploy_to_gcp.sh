#!/bin/bash

# Deploy Rhyme Probe Experiments to GCP
# This script creates a GCP instance and sets up the environment

set -e

# Configuration
INSTANCE_NAME="rhyme-probe-experiment"
ZONE="europe-west4-a"  # Netherlands - good for EU users
MACHINE_TYPE="n1-standard-8"
GPU_TYPE="nvidia-tesla-a100"
GPU_COUNT=1
BOOT_DISK_SIZE="100GB"
PROJECT_ID=$(gcloud config get-value project)

echo "🚀 Deploying Rhyme Probe Experiments to GCP"
echo "=============================================="
echo "Project ID: $PROJECT_ID"
echo "Instance: $INSTANCE_NAME"
echo "Zone: $ZONE"
echo "Machine Type: $MACHINE_TYPE"
echo "GPU: $GPU_COUNT x $GPU_TYPE"
echo ""

# Check if gcloud is configured
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
    echo "❌ Error: gcloud not authenticated. Please run 'gcloud auth login' first."
    exit 1
fi

# Check if instance already exists
if gcloud compute instances describe $INSTANCE_NAME --zone=$ZONE --quiet 2>/dev/null; then
    echo "⚠️  Instance $INSTANCE_NAME already exists in zone $ZONE"
    read -p "Do you want to delete it and create a new one? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🗑️  Deleting existing instance..."
        gcloud compute instances delete $INSTANCE_NAME --zone=$ZONE --quiet
    else
        echo "❌ Aborting. Please delete the instance manually or choose a different name."
        exit 1
    fi
fi

# Create the instance with Ubuntu 24.04 LTS
echo "🔧 Creating GCP instance with Ubuntu 24.04 LTS..."
gcloud compute instances create $INSTANCE_NAME \
    --zone=$ZONE \
    --machine-type=$MACHINE_TYPE \
    --maintenance-policy=TERMINATE \
    --accelerator="type=$GPU_TYPE,count=$GPU_COUNT" \
    --boot-disk-size=$BOOT_DISK_SIZE \
    --boot-disk-type=pd-ssd \
    --image-family=ubuntu-2404-lts \
    --image-project=ubuntu-os-cloud \
    --scopes=default,storage-full \
    --metadata="install-nvidia-driver=true" \
    --metadata-from-file=startup-script=setup_gpu.sh

echo "✅ Instance created successfully!"

# Wait for instance to be ready
echo "⏳ Waiting for instance to be ready..."
sleep 30

# Copy project files to instance
echo "📁 Copying project files to instance..."
gcloud compute scp --recurse . $INSTANCE_NAME:~/rhyme-probe --zone=$ZONE

# SSH into instance and run setup
echo "🔧 Setting up environment on instance..."
gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --command="
    cd ~/rhyme-probe
    echo 'Setting up Ubuntu 24.04 LTS environment...'
    
    # Install Python dependencies
    echo 'Installing Python dependencies...'
    pip3 install -r requirements.txt
    
    echo 'Downloading spaCy model...'
    python3 -m spacy download en_core_web_sm
    
    echo 'Downloading NLTK data...'
    python3 -c \"import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger'); nltk.download('maxent_ne_chunker'); nltk.download('words')\"
    
    echo 'Running environment test...'
    python3 test_setup.py
    
    echo 'Setup complete!'
"

echo ""
echo "🎉 Deployment complete!"
echo "======================"
echo "Instance: $INSTANCE_NAME"
echo "Zone: $ZONE"
echo "OS: Ubuntu 24.04 LTS"
echo ""
echo "To connect to your instance:"
echo "  gcloud compute ssh $INSTANCE_NAME --zone=$ZONE"
echo ""
echo "To run experiments:"
echo "  cd ~/rhyme-probe"
echo "  python3 run_experiment.py --phase all"
echo ""
echo "To check instance status:"
echo "  gcloud compute instances describe $INSTANCE_NAME --zone=$ZONE"
echo ""
echo "To delete the instance when done:"
echo "  gcloud compute instances delete $INSTANCE_NAME --zone=$ZONE"
echo ""
echo "Estimated cost: ~$3.00/hour"
echo "Remember to stop/delete the instance when not in use!" 