#!/bin/bash
# Setup script for RunPod H100 cluster environment
# Supports both single-node and multi-node configurations

# ========== CLUSTER CONFIGURATION ==========
# Define cluster IPs (CORRECTED based on actual setup)
# export IS_CLUSTER=true
# export MASTER_IP="10.65.0.2"
# export SLAVE_IP="10.65.0.3"
# ./setup_runpod.sh
#


MASTER_IP="${MASTER_IP:-10.65.0.2}"
SLAVE_IP="${SLAVE_IP:-10.65.0.3}"
MASTER_PORT="${MASTER_PORT:-29500}"
IS_CLUSTER="${IS_CLUSTER:-false}"  # Set to true for multi-node setup

# Detect current node IP
CURRENT_IP=$(hostname -I | awk '{print $1}')

echo "🚀 Setting up RunPod H100 environment..."
echo "Current node IP: $CURRENT_IP"

if [[ "$IS_CLUSTER" == "true" ]]; then
    echo "📡 Multi-node cluster configuration:"
    echo "   Master: $MASTER_IP"
    echo "   Slave: $SLAVE_IP"
    echo "   Port: $MASTER_PORT"
fi

# ========== COMMON SETUP FOR ALL NODES ==========

# Update system packages
apt update -y

# Install system utilities INCLUDING NINJA
echo "📦 Installing system utilities..."
apt install -y \
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
    openssh-server \
    pdsh \
    ninja-build  # ADDED: Required for DeepSpeed compilation

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
        
        # Create hostfile for DeepSpeed with CORRECT IPs
        cat > ~/hostfile << EOF
$MASTER_IP slots=8
$SLAVE_IP slots=8
EOF
        
        # Add known hosts to prevent SSH warnings
        echo "📝 Adding nodes to known hosts..."
        ssh-keyscan -H $MASTER_IP >> ~/.ssh/known_hosts 2>/dev/null || echo "Warning: Could not scan $MASTER_IP"
        ssh-keyscan -H $SLAVE_IP >> ~/.ssh/known_hosts 2>/dev/null || echo "Warning: Could not scan $SLAVE_IP (this is normal if slave is not ready yet)"
        
        # Also add localhost variations
        ssh-keyscan -H localhost >> ~/.ssh/known_hosts 2>/dev/null
        ssh-keyscan -H 127.0.0.1 >> ~/.ssh/known_hosts 2>/dev/null
        
        echo ""
        echo "📋 Final setup steps:"
        echo "1. Wait for slave node to be configured"
        echo "2. Copy the SSH public key above to slave's ~/.ssh/authorized_keys"
        echo "3. Test connection: ssh root@$SLAVE_IP 'echo Slave SSH: OK'"
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

# ========== PYTHON ENVIRONMENT SETUP ==========
echo "🐍 Setting up Python environment..."

# Create virtual environment
cd /root
if [ ! -d "grpo_env" ]; then
    echo "Creating virtual environment..."
    python3 -m venv grpo_env
fi

# ACTIVATE the virtual environment before installing packages!
echo "Activating virtual environment..."
source grpo_env/bin/activate

# Verify we're in the virtual environment
echo "Python location: $(which python)"
echo "Pip location: $(which pip)"

# Install Python requirements INSIDE the virtual environment
echo "📦 Installing Python packages..."
pip install --upgrade pip

# Check if requirements.txt exists in grpo_reasoning directory
if [ -f "grpo_reasoning/requirements.txt" ]; then
    pip install -r grpo_reasoning/requirements.txt
elif [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "⚠️  requirements.txt not found, installing essential packages..."
    # Install with specific PyTorch version for CUDA 11.8 compatibility
    pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
    pip install transformers datasets accelerate deepspeed numpy pandas \
                scikit-learn wandb tensorboard psutil tqdm matplotlib \
                sentencepiece protobuf scipy h5py pyarrow python-dotenv \
                pyyaml jsonlines nvidia-ml-py pynvml
fi

# Install Flash Attention for H100 (optional but recommended)
echo "⚡ Installing Flash Attention for H100..."
pip install flash-attn --no-build-isolation || echo "Flash Attention installation failed (optional)"

# Setup NCCL environment variables
echo "🔧 Setting up NCCL environment..."
cat >> ~/.bashrc << 'EOF'

# NCCL Configuration for H100
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=5
export NCCL_P2P_LEVEL=NVL
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_TC=106
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=^docker0,lo

# CUDA settings - REMOVED PYTORCH_CUDA_ALLOC_CONF to avoid compatibility issues
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# Distributed training
export OMP_NUM_THREADS=8

# Default cluster settings
export MASTER_ADDR=${MASTER_ADDR:-localhost}
export MASTER_PORT=${MASTER_PORT:-29500}
EOF

# Add cluster-specific exports if multi-node with CORRECT IPs
if [[ "$IS_CLUSTER" == "true" ]]; then
    echo "export MASTER_ADDR=$MASTER_IP" >> ~/.bashrc
    echo "export MASTER_PORT=$MASTER_PORT" >> ~/.bashrc
    echo "export SLAVE_ADDR=$SLAVE_IP" >> ~/.bashrc
fi

# Create useful aliases
echo "💡 Adding useful aliases..."
cat >> ~/.bash_aliases << 'EOF'
# GPU monitoring
alias gpu='nvidia-smi'
alias gpuw='watch -n 1 nvidia-smi'
alias gpud='nvidia-smi dmon -s u -c 1'

# Network monitoring
alias netstat='ss -tuln'
alias ports='sudo lsof -i -P -n'

# Training shortcuts
alias train-sft='cd ~/grpo_reasoning && python sft_cot_trainer.py'
alias train-grpo='cd ~/grpo_reasoning && bash launch_grpo_training.sh'
alias monitor='cd ~/grpo_reasoning && python monitor_grpo_runpod.py'

# Navigation
alias cdt='cd ~/grpo_reasoning'

# Environment activation
alias activate='source ~/grpo_env/bin/activate'
EOF

# Source the updated configurations
source ~/.bashrc
source ~/.bash_aliases

# Create project directories
echo "📁 Creating project directories..."
mkdir -p ~/grpo_reasoning/{models,checkpoints,logs,data}

# Test installations
echo "✅ Testing installations..."
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Not installed yet')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'Not installed yet')"
echo "Number of GPUs: $(python -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null || echo 'Not installed yet')"

# Test network tools
echo "🌐 Testing network tools..."
ping -c 2 google.com || echo "Warning: No internet connectivity"

# Show GPU info
echo "🎮 GPU Information:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# Check InfiniBand (if available)
echo "🔌 Checking InfiniBand..."
ibstat 2>/dev/null || echo "InfiniBand not detected (normal for single node)"

# Test cluster connectivity if multi-node
if [[ "$IS_CLUSTER" == "true" ]]; then
    echo "🔗 Testing cluster connectivity..."
    if [[ "$CURRENT_IP" == "$MASTER_IP" ]]; then
        echo "Waiting for manual SSH key setup..."
        echo "After copying SSH key to slave, test with:"
        echo "  ssh root@$SLAVE_IP 'hostname'"
    fi
fi

echo ""
echo "✅ Setup complete! Environment is ready for GRPO training."
echo ""
echo "⚠️  IMPORTANT: Always activate the virtual environment first!"
echo "    source grpo_env/bin/activate"
echo "    OR use the alias: activate"
echo ""
echo "Next steps:"
echo "1. Source the environment: source ~/.bashrc && activate"
echo "2. Login to HuggingFace (if using gated models): huggingface-cli login"
echo "3. Copy your training files to ~/grpo_reasoning/"

if [[ "$IS_CLUSTER" == "true" ]]; then
    echo ""
    echo "🔑 CLUSTER SETUP STEPS:"
    if [[ "$CURRENT_IP" == "$MASTER_IP" ]]; then
        echo "4. Copy the SSH public key shown above to slave node"
        echo "5. Test SSH: ssh root@$SLAVE_IP 'hostname'"
        echo "6. Start training on both nodes simultaneously"
    else
        echo "4. Add master's SSH public key to ~/.ssh/authorized_keys"
        echo "5. Wait for master to initiate training"
    fi
else
    echo "4. Start SFT training: cd ~/grpo_reasoning && python sft_cot_trainer.py"
    echo "5. Then run GRPO: cd ~/grpo_reasoning && bash launch_grpo_training.sh"
fi

# Display helpful commands
echo ""
echo "📝 Useful commands:"
echo "  activate    - Activate Python environment"
echo "  gpu         - Show GPU status"
echo "  gpuw        - Watch GPU status"
echo "  monitor     - Start training monitor"
echo "  train-sft   - Start SFT training"
echo "  train-grpo  - Start GRPO training"
if [[ "$IS_CLUSTER" == "true" ]]; then
    echo "  pdsh -w $MASTER_IP,$SLAVE_IP 'nvidia-smi' - Check GPUs on all nodes"
fi
