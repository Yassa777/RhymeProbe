#!/bin/bash

# Simple GCP instance creation with Ubuntu 24.04 LTS
# Run this if you prefer manual control

echo "Creating GCP instance with Ubuntu 24.04 LTS..."

gcloud compute instances create rhyme-probe-experiment \
  --zone=europe-west4-a \
  --machine-type=n1-standard-8 \
  --maintenance-policy=TERMINATE \
  --accelerator="type=nvidia-tesla-a100,count=1" \
  --boot-disk-size=100GB \
  --boot-disk-type=pd-ssd \
  --image-family=ubuntu-2404-lts \
  --image-project=ubuntu-os-cloud \
  --scopes=default,storage-full

echo "Instance created! Connect with:"
echo "gcloud compute ssh rhyme-probe-experiment --zone=europe-west4-a" 