#!/bin/bash

set -e

echo "Updating package index..."
sudo apt update

echo "Installing required packages..."
sudo apt install -y \
    ca-certificates \
    curl \
    gnupg \
    lsb-release

echo "Adding Docker's official GPG key..."
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
    sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

echo "Setting up the Docker repository..."
echo \
  "deb [arch=$(dpkg --print-architecture) \
  signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

echo "Updating package index (again)..."
sudo apt update

echo "Installing Docker Engine..."
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

echo "Enabling Docker to start on boot..."
sudo systemctl enable docker
sudo systemctl start docker

echo "Adding current user to docker group..."
sudo usermod -aG docker $USER

echo "Docker installation completed!"
echo "You may need to log out and back in for group changes to take effect."
echo "To verify Docker installation, run: docker --version"
echo "To test Docker, run: docker run hello-world"