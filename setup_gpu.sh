#!/bin/bash

# Update system
apt-get update
apt-get install -y wget curl git build-essential

# Install NVIDIA drivers and CUDA
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

apt-get update
apt-get install -y nvidia-container-toolkit

# Install Docker
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null

apt-get update
apt-get install -y docker-ce docker-ce-cli containerd.io

# Add user to docker group
usermod -aG docker $USER

# Install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p /opt/miniconda
echo 'export PATH="/opt/miniconda/bin:$PATH"' >> /home/$USER/.bashrc

# Install Python packages
/opt/miniconda/bin/conda install -y python=3.10
/opt/miniconda/bin/pip install -r /home/$USER/requirements.txt

# Download spaCy model
/opt/miniconda/bin/python -m spacy download en_core_web_sm

# Download NLTK data
/opt/miniconda/bin/python -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger'); nltk.download('maxent_ne_chunker'); nltk.download('words')"

# Set up Jupyter
/opt/miniconda/bin/jupyter lab --generate-config
echo "c.NotebookApp.ip = '0.0.0.0'" >> /home/$USER/.jupyter/jupyter_lab_config.py
echo "c.NotebookApp.allow_root = True" >> /home/$USER/.jupyter/jupyter_lab_config.py
echo "c.NotebookApp.open_browser = False" >> /home/$USER/.jupyter/jupyter_lab_config.py

# Create systemd service for Jupyter
cat > /etc/systemd/system/jupyter.service << EOF
[Unit]
Description=Jupyter Lab
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=/home/$USER
ExecStart=/opt/miniconda/bin/jupyter lab --config=/home/$USER/.jupyter/jupyter_lab_config.py
Restart=always

[Install]
WantedBy=multi-user.target
EOF

systemctl enable jupyter
systemctl start jupyter

echo "Setup complete! Jupyter Lab is running on port 8888" 