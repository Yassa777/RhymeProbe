#!/bin/bash

# Setup script for Deep Learning VM (PyTorch 2.4 + CUDA 12.4)
# This is much simpler since the ML environment is pre-configured

echo "🚀 Setting up Rhyme Probe Experiments on Deep Learning VM"
echo "=========================================================="

# Check what's already installed
echo "Checking pre-installed packages..."
python3 -c "import torch; print(f'✓ PyTorch {torch.__version__} available')"
python3 -c "import torch; print(f'✓ CUDA available: {torch.cuda.is_available()}')"
nvidia-smi

# Install additional dependencies
echo ""
echo "Installing additional Python dependencies..."
pip3 install -r requirements.txt

# Download NLP models
echo ""
echo "Downloading NLP models..."
python3 -m spacy download en_core_web_sm
python3 -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger'); nltk.download('maxent_ne_chunker'); nltk.download('words')"

# Test the environment
echo ""
echo "Running environment test..."
python3 test_setup.py

echo ""
echo "🎉 Setup complete! Your Deep Learning VM is ready for experiments."
echo ""
echo "To run experiments:"
echo "  python3 run_experiment.py --phase all"
echo ""
echo "To start Jupyter Lab:"
echo "  jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root" 