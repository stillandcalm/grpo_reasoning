#!/bin/bash
# Complete Setup Script for SFT/GRPO Training on RunPod H100 Cluster
# Supports both single-node and multi-node configurations
# No conda - uses Python venv

set -e  # Exit on error

# ========== CLUSTER CONFIGURATION ==========
# Usage for multi-node:
# export IS_CLUSTER=true
# export MASTER_IP="10.65.0.2"
# export SLAVE_IP="10.65.0.3"
# ./setup.sh

MASTER_IP="${MASTER_IP:-10.65.0.2}"
SLAVE_IP="${SLAVE_IP:-10.65.0.3}"
MASTER_PORT="${MASTER_PORT:-29500}"
IS_CLUSTER="${IS_CLUSTER:-false}"  # Set to true for multi-node setup

# Detect current node IP
CURRENT_IP=$(hostname -I | awk '{print $1}')

echo "🚀 Starting Complete Setup for SFT/GRPO Training..."
echo "Current node IP: $CURRENT_IP"
echo "Python version: $(python3 --version)"

if [[ "$IS_CLUSTER" == "true" ]]; then
    echo "📡 Multi-node cluster configuration:"
    echo "   Master: $MASTER_IP"
    echo "   Slave: $SLAVE_IP"
    echo "   Port: $MASTER_PORT"
fi

# ========== SYSTEM PACKAGE INSTALLATION ==========
echo "📦 Installing system packages..."
apt update -y && apt install -y \
    ninja-build \
    iputils-ping \
    net-tools \
    htop \
    nvtop \
    iotop \
    iftop \
    tmux \
    vim \
    git \
    curl \
    wget \
    build-essential \
    python3-dev \
    python3-venv \
    python3-pip \
    openssh-server \
    pdsh \
    libopenmpi-dev \
    openmpi-bin \
    libssl-dev \
    libffi-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libncurses5-dev \
    libgdbm-dev \
    liblzma-dev \
    tk-dev \
    lsof

# ========== SSH CONFIGURATION FOR CLUSTER ==========
if [[ "$IS_CLUSTER" == "true" ]]; then
    echo "🔐 Configuring SSH for cluster communication..."
    
    # Configure SSH for root access (WARNING: Only for private cluster!)
    sed -i 's|^#\?\s*PermitRootLogin\s\+.*|PermitRootLogin yes|' /etc/ssh/sshd_config
    sed -i 's|^#\?\s*PasswordAuthentication\s\+.*|PasswordAuthentication yes|' /etc/ssh/sshd_config
    sed -i 's|^#\?\s*PermitEmptyPasswords\s\+.*|PermitEmptyPasswords no|' /etc/ssh/sshd_config
    sed -i 's|^#\?\s*PubkeyAuthentication\s\+.*|PubkeyAuthentication yes|' /etc/ssh/sshd_config
    service ssh restart
    
    # ========== MASTER NODE SPECIFIC SETUP ==========
    if [[ "$CURRENT_IP" == "$MASTER_IP" ]]; then
        echo "📍 Configuring master node..."
        
        # Generate SSH keys if not exists
        if [ ! -f ~/.ssh/id_ed25519 ]; then
            ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -N ""
        fi
        
        # Add to local authorized keys
        cat ~/.ssh/id_ed25519.pub >> ~/.ssh/authorized_keys
        
        # Set proper permissions
        chmod 700 ~/.ssh
        chmod 600 ~/.ssh/authorized_keys
        chmod 600 ~/.ssh/id_ed25519 ~/.ssh/id_ed25519.pub
        
        echo "⚠️  IMPORTANT: Copy the following public key to the slave node:"
        echo "================================================"
        cat ~/.ssh/id_ed25519.pub
        echo "================================================"
        echo "Run this on slave node: echo 'PASTE_KEY_HERE' >> ~/.ssh/authorized_keys"
        
        # Create hostfile for DeepSpeed
        cat > ~/hostfile << EOF
$MASTER_IP slots=8
$SLAVE_IP slots=8
EOF
        
        # Add known hosts to prevent SSH warnings
        echo "📝 Adding nodes to known hosts..."
        ssh-keyscan -H $MASTER_IP >> ~/.ssh/known_hosts 2>/dev/null || echo "Warning: Could not scan $MASTER_IP"
        ssh-keyscan -H $SLAVE_IP >> ~/.ssh/known_hosts 2>/dev/null || echo "Warning: Could not scan $SLAVE_IP (this is normal if slave is not ready yet)"
        ssh-keyscan -H localhost >> ~/.ssh/known_hosts 2>/dev/null
        ssh-keyscan -H 127.0.0.1 >> ~/.ssh/known_hosts 2>/dev/null
    fi
    
    # ========== SLAVE NODE SPECIFIC SETUP ==========
    if [[ "$CURRENT_IP" == "$SLAVE_IP" ]]; then
        echo "📍 Configuring slave node..."
        echo "⚠️  Add master's public key to ~/.ssh/authorized_keys when prompted"
        
        # Ensure SSH directory exists
        mkdir -p ~/.ssh
        chmod 700 ~/.ssh
        touch ~/.ssh/authorized_keys
        chmod 600 ~/.ssh/authorized_keys
        
        # Add known hosts for master
        echo "📝 Adding master to known hosts..."
        ssh-keyscan -H $MASTER_IP >> ~/.ssh/known_hosts 2>/dev/null || echo "Warning: Could not scan $MASTER_IP"
        ssh-keyscan -H localhost >> ~/.ssh/known_hosts 2>/dev/null
        ssh-keyscan -H 127.0.0.1 >> ~/.ssh/known_hosts 2>/dev/null
    fi
fi

# ========== CREATE PROJECT STRUCTURE ==========
echo "📁 Creating project directories..."
mkdir -p ~/sft_reasoning/{models,checkpoints,logs,data}
mkdir -p ~/grpo_reasoning/{models,checkpoints,logs,data}

# ========== PYTHON ENVIRONMENT SETUP ==========
echo "🐍 Setting up Python environments..."

# Create virtual environment for SFT
cd ~/sft_reasoning
if [ ! -d "sft_env" ]; then
    echo "Creating SFT virtual environment..."
    python3 -m venv ~/sft_env
fi

# Create virtual environment for GRPO
cd ~/grpo_reasoning
if [ ! -d "grpo_env" ]; then
    echo "Creating GRPO virtual environment..."
    python3 -m venv ~/grpo_env
fi

# ========== INSTALL SFT DEPENDENCIES ==========
echo "📦 Installing SFT dependencies..."
source ~/sft_env/bin/activate

# Verify we're in the virtual environment
echo "Python location: $(which python)"
echo "Pip location: $(which pip)"

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA 11.8
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# Install core ML libraries
pip install transformers>=4.36.0 \
    datasets>=2.14.0 \
    accelerate>=0.25.0 \
    deepspeed>=0.12.6

# Install additional dependencies
pip install numpy>=1.24.0 \
    pandas>=2.0.0 \
    scikit-learn>=1.3.0 \
    wandb>=0.16.0 \
    tensorboard>=2.14.0 \
    psutil>=5.9.0 \
    tqdm>=4.66.0 \
    matplotlib>=3.7.0 \
    sentencepiece>=0.1.99 \
    protobuf>=3.20.0 \
    scipy>=1.11.0 \
    h5py>=3.9.0 \
    pyarrow>=13.0.0 \
    python-dotenv>=1.0.0 \
    pyyaml>=6.0.1 \
    jsonlines>=4.0.0 \
    nvidia-ml-py>=12.535.0 \
    pynvml>=11.5.0

# Install Flash Attention for H100 (optional but recommended)
echo "⚡ Installing Flash Attention for H100..."
pip install flash-attn --no-build-isolation || echo "Flash Attention installation failed (optional)"

deactivate

# ========== INSTALL GRPO DEPENDENCIES ==========
echo "📦 Installing GRPO dependencies (if different from SFT)..."
source ~/grpo_env/bin/activate

# Install same dependencies as SFT (modify if GRPO needs different packages)
pip install --upgrade pip setuptools wheel
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install transformers>=4.36.0 datasets>=2.14.0 accelerate>=0.25.0 deepspeed>=0.12.6
pip install numpy pandas scikit-learn wandb tensorboard psutil tqdm matplotlib
pip install sentencepiece protobuf scipy h5py pyarrow python-dotenv pyyaml jsonlines
pip install nvidia-ml-py pynvml
pip install flash-attn --no-build-isolation || echo "Flash Attention installation failed (optional)"

deactivate

# ========== SETUP ENVIRONMENT VARIABLES ==========
echo "🔧 Setting up environment variables..."

# Backup existing .bashrc
cp ~/.bashrc ~/.bashrc.backup

# Add environment variables to .bashrc
cat >> ~/.bashrc << 'EOF'

# ========== SFT/GRPO Training Environment ==========

# NCCL Configuration for H100
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=5
export NCCL_P2P_LEVEL=NVL
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_TC=106
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=^docker0,lo

# CUDA settings
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# Distributed training
export OMP_NUM_THREADS=8

# Default cluster settings
export MASTER_ADDR=${MASTER_ADDR:-localhost}
export MASTER_PORT=${MASTER_PORT:-29500}

# Python environments
export SFT_ENV_PATH=~/sft_env
export GRPO_ENV_PATH=~/grpo_env

# Project paths
export SFT_PROJECT_PATH=~/sft_reasoning
export GRPO_PROJECT_PATH=~/grpo_reasoning
EOF

# Add cluster-specific exports if multi-node
if [[ "$IS_CLUSTER" == "true" ]]; then
    echo "export MASTER_ADDR=$MASTER_IP" >> ~/.bashrc
    echo "export MASTER_PORT=$MASTER_PORT" >> ~/.bashrc
    echo "export SLAVE_ADDR=$SLAVE_IP" >> ~/.bashrc
fi

# ========== CREATE USEFUL ALIASES ==========
echo "💡 Adding useful aliases..."
cat >> ~/.bash_aliases << 'EOF'
# Environment activation
alias sft-env='source ~/sft_env/bin/activate'
alias grpo-env='source ~/grpo_env/bin/activate'

# GPU monitoring
alias gpu='nvidia-smi'
alias gpuw='watch -n 1 nvidia-smi'
alias gpud='nvidia-smi dmon -s u -c 1'

# Network monitoring
alias netstat='ss -tuln'
alias ports='sudo lsof -i -P -n'

# Training shortcuts
alias train-sft='cd ~/sft_reasoning && source ~/sft_env/bin/activate && python sft_cot_trainer.py'
alias train-grpo='cd ~/grpo_reasoning && source ~/grpo_env/bin/activate && bash launch_grpo_training.sh'

# Navigation
alias cdsft='cd ~/sft_reasoning'
alias cdgrpo='cd ~/grpo_reasoning'

# Process monitoring
alias pstree='pstree -p'
alias killgpu='sudo fuser -v /dev/nvidia* |& grep -v kernel | awk "{print \$2}" | xargs -r kill -9'
EOF

# Source the updated configurations
source ~/.bashrc
source ~/.bash_aliases

# ========== COPY TRAINING FILES ==========
echo "📄 Setting up training files..."

# Copy SFT trainer to project directory
if [ -f "sft_cot_trainer.py" ]; then
    cp sft_cot_trainer.py ~/sft_reasoning/
fi

# Copy launch scripts
if [ -f "launch_sft_training.sh" ]; then
    cp launch_sft_training.sh ~/sft_reasoning/
    chmod +x ~/sft_reasoning/launch_sft_training.sh
fi

# ========== CREATE DEEPSPEED CONFIG ==========
echo "⚙️ Creating DeepSpeed configuration..."
cat > ~/sft_reasoning/ds_config_sft.json << 'EOF'
{
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "gradient_accumulation_steps": "auto",
  "gradient_clipping": 1.0,
  "steps_per_print": 50,
  "bf16": {
    "enabled": "auto"
  },
  "communication_data_type": "bf16",
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "allgather_partitions": true,
    "allgather_bucket_size": 2e8,
    "overlap_comm": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 2e8,
    "contiguous_gradients": true
  },
  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": "auto",
      "betas": [0.9, 0.999],
      "eps": 1e-8,
      "weight_decay": "auto"
    }
  },
  "scheduler": {
    "type": "WarmupLR",
    "params": {
      "warmup_min_lr": 0,
      "warmup_max_lr": "auto",
      "warmup_num_steps": "auto"
    }
  },
  "wall_clock_breakdown": false
}
EOF

# ========== CREATE LAUNCH SCRIPT ==========
echo "🚀 Creating launch script..."
cat > ~/sft_reasoning/launch_sft_training.sh << 'EOF'
#!/bin/bash
cd ~/sft_reasoning
source ~/sft_env/bin/activate

# For single GPU
# python sft_cot_trainer.py --num_train_samples 100 --num_epochs 1

# For single node multi-GPU
# deepspeed --num_gpus=8 sft_cot_trainer.py --num_train_samples 1000 --num_epochs 3

# For multi-node (default)
deepspeed --hostfile=~/hostfile \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    sft_cot_trainer.py \
    --num_train_samples 1000 \
    --num_epochs 2 \
    --batch_size 1
EOF
chmod +x ~/sft_reasoning/launch_sft_training.sh

# ========== TEST INSTALLATIONS ==========
echo "✅ Testing installations..."

# Test SFT environment
source ~/sft_env/bin/activate
echo "=== SFT Environment ==="
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Not installed')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'Not installed')"
echo "Number of GPUs: $(python -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null || echo 'Not installed')"
echo "DeepSpeed version: $(python -c 'import deepspeed; print(deepspeed.__version__)' 2>/dev/null || echo 'Not installed')"
deactivate

# Test GRPO environment
source ~/grpo_env/bin/activate
echo -e "\n=== GRPO Environment ==="
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Not installed')"
deactivate

# Test network tools
echo -e "\n🌐 Testing network tools..."
ping -c 2 google.com || echo "Warning: No internet connectivity"

# Show GPU info
echo -e "\n🎮 GPU Information:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# Check InfiniBand (if available)
echo -e "\n🔌 Checking InfiniBand..."
ibstat 2>/dev/null || echo "InfiniBand not detected (normal for single node)"

# Test cluster connectivity if multi-node
if [[ "$IS_CLUSTER" == "true" ]]; then
    echo -e "\n🔗 Testing cluster connectivity..."
    if [[ "$CURRENT_IP" == "$MASTER_IP" ]]; then
        echo "Waiting for manual SSH key setup..."
        echo "After copying SSH key to slave, test with:"
        echo "  ssh root@$SLAVE_IP 'hostname'"
    fi
fi

# ========== FINAL INSTRUCTIONS ==========
echo ""
echo "✅ Setup complete! Environment is ready for SFT/GRPO training."
echo ""
echo "📋 Quick Start Guide:"
echo ""
echo "1. Source the environment:"
echo "   source ~/.bashrc"
echo ""
echo "2. For SFT training:"
echo "   sft-env  # or: source ~/sft_env/bin/activate"
echo "   cd ~/sft_reasoning"
echo "   python sft_cot_trainer.py --num_train_samples 1000 --num_epochs 2"
echo ""
echo "3. For GRPO training (after SFT):"
echo "   grpo-env  # or: source ~/grpo_env/bin/activate"
echo "   cd ~/grpo_reasoning"
echo "   # Update MODEL_PATH in launch script to point to SFT model"
echo "   ./launch_grpo_training.sh"
echo ""
echo "4. Login to HuggingFace (if using gated models):"
echo "   huggingface-cli login"
echo ""

if [[ "$IS_CLUSTER" == "true" ]]; then
    echo "🔑 CLUSTER SETUP STEPS:"
    if [[ "$CURRENT_IP" == "$MASTER_IP" ]]; then
        echo "5. Copy the SSH public key shown above to slave node"
        echo "6. Test SSH: ssh root@$SLAVE_IP 'hostname'"
        echo "7. For multi-node training, use the hostfile at ~/hostfile"
    else
        echo "5. Add master's SSH public key to ~/.ssh/authorized_keys"
        echo "6. Wait for master to initiate training"
    fi
fi

echo ""
echo "📝 Useful commands:"
echo "  sft-env     - Activate SFT environment"
echo "  grpo-env    - Activate GRPO environment"
echo "  gpu         - Show GPU status"
echo "  gpuw        - Watch GPU status"
echo "  train-sft   - Quick start SFT training"
echo "  train-grpo  - Quick start GRPO training"

if [[ "$IS_CLUSTER" == "true" ]]; then
    echo "  pdsh -w $MASTER_IP,$SLAVE_IP 'nvidia-smi' - Check GPUs on all nodes"
fi

echo ""
echo "🎯 Training files location:"
echo "  SFT: ~/sft_reasoning/"
echo "  GRPO: ~/grpo_reasoning/"
echo ""
echo "Happy training! 🚀"
