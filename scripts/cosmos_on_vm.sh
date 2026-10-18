#!/usr/bin/env bash
# Session A runs this ON THE VM (copy with scp, then: bash ~/cosmos_on_vm.sh). Reviewed script, not curl|bash.
# Starts the two Cosmos NIM containers on the single RTX PRO 6000. Needs NVIDIA_API_KEY (NGC key) exported in the env.
set -euo pipefail
[ -f ~/.env ] && { set -a; source ~/.env; set +a; }
: "${NVIDIA_API_KEY:?NVIDIA_API_KEY missing: scp private/secrets/.env to ~/.env on the VM (chmod 600) first}"
MODE="${1:-pull}"   # pull = install docker + login + pull images only (safe during sim collection); run = start containers

# docker + nvidia runtime (Ubuntu 24.04 CUDA image may already have docker; install if missing)
if ! command -v docker >/dev/null; then
  sudo apt-get update -qq && sudo apt-get install -y -qq docker.io
  sudo usermod -aG docker "$USER"
fi
if ! sudo docker info 2>/dev/null | grep -q nvidia; then
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --batch --yes --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
  sudo apt-get update -qq && sudo apt-get install -y -qq nvidia-container-toolkit
  sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker
fi

echo "$NVIDIA_API_KEY" | sudo docker login nvcr.io --username '$oauthtoken' --password-stdin
mkdir -p ~/.cache/nim
if [ "$MODE" = pull ]; then
  sudo docker pull nvcr.io/nim/nvidia/cosmos-reason2-2b:1.7.0
  sudo docker pull nvcr.io/nim/nvidia/cosmos-transfer2.5-2b:latest
  echo "images pulled; rerun with: bash ~/cosmos_on_vm.sh run"; exit 0
fi

# Cosmos Reason2 2B (VLM NIM) on :8002  — scorer: NVIDIA_BASE_URL=http://localhost:8002/v1 COSMOS_REASON_MODEL=nvidia/cosmos-reason2-2b
sudo docker run -d --name cosmos-reason --runtime=nvidia --gpus all --shm-size=16GB \
  -e NGC_API_KEY="$NVIDIA_API_KEY" -v ~/.cache/nim:/opt/nim/.cache -u "$(id -u)" -p 8002:8000 \
  nvcr.io/nim/nvidia/cosmos-reason2-2b:1.7.0

# Cosmos Transfer2.5 2B (WFM NIM) on :8001
sudo docker run -d --name cosmos-transfer --runtime=nvidia --gpus all --shm-size=16GB --ulimit nofile=65536:65536 \
  -e NGC_API_KEY="$NVIDIA_API_KEY" -v ~/.cache/nim:/opt/nim/.cache -u "$(id -u)" -p 8001:8000 \
  nvcr.io/nim/nvidia/cosmos-transfer2.5-2b:latest

echo "Containers starting (weights download: Reason ~5 GB, Transfer ~18 GB). Poll:"
echo "  curl -s localhost:8002/v1/health/ready ; curl -s localhost:8001/v1/health/ready"
echo "Stop both BEFORE training:  sudo docker stop cosmos-reason cosmos-transfer"
