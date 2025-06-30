#!/bin/bash

# setup.sh - Complete setup script for GRPO implementations with DeepSpeed
# Supports both single-GPU and distributed training setups

set -e  # Exit on error

echo "=================================================="
echo "GRPO Implementation Setup Script (with DeepSpeed)"
echo "=================================================="

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check Python version
print_status "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
REQUIRED_VERSION="3.8"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then 
    print_error "Python 3.8 or higher is required. Current version: $PYTHON_VERSION"
    exit 1
fi
print_status "Python version $PYTHON_VERSION is compatible"

# Create virtual environment
print_status "Creating virtual environment..."
if [ -d "venv" ]; then
    print_warning "Virtual environment already exists. Removing old environment..."
    rm -rf venv
fi
python3 -m venv venv

# Activate virtual environment
print_status "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support
print_status "Installing PyTorch with CUDA support..."
# Detect CUDA version if available
if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d. -f1,2)
    print_status "Detected CUDA version: $CUDA_VERSION"
    
    # Install appropriate PyTorch version based on CUDA
    if [[ "$CUDA_VERSION" == "11.8" ]]; then
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    elif [[ "$CUDA_VERSION" == "12.1" ]]; then
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    else
        print_warning "CUDA $CUDA_VERSION detected. Installing latest PyTorch with CUDA support..."
        pip install torch torchvision torchaudio
    fi
else
    print_warning "CUDA not detected. Installing CPU-only PyTorch..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Install Transformers and related libraries
print_status "Installing Transformers and core dependencies..."
pip install transformers>=4.36.0
pip install datasets>=2.14.0
pip install accelerate>=0.25.0
pip install sentencepiece>=0.1.99
pip install protobuf>=3.20.0

# Install model-specific dependencies
print_status "Installing model-specific dependencies..."
pip install tokenizers>=0.15.0
pip install safetensors>=0.4.0

# Install reward model dependencies
print_status "Installing reward model dependencies..."
pip install scipy>=1.10.0
pip install scikit-learn>=1.3.0

# Install DeepSpeed and distributed training dependencies
print_status "Installing DeepSpeed and distributed training tools..."
# Check if we have CUDA for DeepSpeed
if command -v nvidia-smi &> /dev/null; then
    print_status "Installing DeepSpeed with CUDA support..."
    # Install system dependencies for DeepSpeed
    print_warning "Note: DeepSpeed may require system dependencies (libaio-dev on Ubuntu/Debian)"
    pip install deepspeed>=0.12.0
    
    # Install MPI for multi-node support
    print_status "Checking for MPI support..."
    if command -v mpirun &> /dev/null; then
        print_status "MPI detected, installing mpi4py..."
        pip install mpi4py
    else
        print_warning "MPI not detected. Install OpenMPI for multi-node training support."
    fi
else
    print_warning "Installing DeepSpeed without CUDA support (CPU mode)..."
    pip install deepspeed>=0.12.0 --global-option="build_ext" --global-option="-j8"
fi

# Install additional optimization libraries
print_status "Installing optimization libraries..."
pip install ninja  # For faster builds
pip install peft>=0.7.0  # For LoRA support
if command -v nvidia-smi &> /dev/null; then
    pip install bitsandbytes>=0.41.0  # For 8-bit quantization (CUDA only)
fi

# Install monitoring and logging tools
print_status "Installing monitoring tools..."
pip install wandb
pip install tensorboard>=2.13.0
pip install tqdm>=4.65.0

# Install data processing libraries
print_status "Installing data processing libraries..."
pip install numpy>=1.24.0
pip install pandas>=2.0.0
pip install pyarrow>=12.0.0  # For faster dataset loading

# Install development tools
print_status "Installing development tools..."
pip install ipython
pip install jupyter
pip install black
pip install flake8
pip install pytest

# Create necessary directories
print_status "Creating project directories..."
mkdir -p outputs
mkdir -p logs
mkdir -p checkpoints
mkdir -p data
mkdir -p configs

# Create DeepSpeed configuration files
print_status "Creating DeepSpeed configuration files..."

# Zero-2 config for medium models
cat > configs/ds_zero2_config.json << 'EOF'
{
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": 1.0,
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
    "fp16": {
        "enabled": true,
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    },
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": "auto",
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01
        }
    },
    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": 0,
            "warmup_max_lr": "auto",
            "warmup_num_steps": "auto"
        }
    }
}
EOF

# Zero-3 config for large models
cat > configs/ds_zero3_config.json << 'EOF'
{
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": 1.0,
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": true
        },
        "overlap_comm": true,
        "contiguous_gradients": true,
        "sub_group_size": 1e9,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_16bit_weights_on_model_save": true
    },
    "bf16": {
        "enabled": true
    },
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": "auto",
            "betas": [0.9, 0.95],
            "eps": 1e-8,
            "weight_decay": 0.1
        }
    },
    "activation_checkpointing": {
        "partition_activations": true,
        "cpu_checkpointing": true,
        "contiguous_memory_optimization": true,
        "number_checkpoints": null,
        "synchronize_checkpoint_boundary": false,
        "profile": false
    }
}
EOF

# Create run scripts
print_status "Creating run scripts..."

# Single GPU run script (uses file 2 - grpo_cot_neural_models.py)
cat > run_single_gpu.sh << 'EOF'
#!/bin/bash
source venv/bin/activate

echo "Running GRPO with CoT on single GPU (no DeepSpeed)..."
python grpo_cot_neural_models.py \
    --policy_model EleutherAI/gpt-neo-125m \
    --reward_model microsoft/deberta-v3-base \
    --dataset gsm8k \
    --num_examples 100 \
    --num_epochs 3 \
    --batch_size 2 \
    --lr 5e-6 \
    --device cuda
EOF

# Single GPU with DeepSpeed run script (uses file 3)
cat > run_deepspeed_single.sh << 'EOF'
#!/bin/bash
source venv/bin/activate

echo "Running GRPO with DeepSpeed on single GPU..."
deepspeed --num_gpus=1 grpo_cot_deepspeed.py \
    --policy_model EleutherAI/gpt-neo-125m \
    --reward_model microsoft/deberta-v3-base \
    --dataset gsm8k \
    --num_examples 100 \
    --num_epochs 3 \
    --batch_size 2 \
    --lr 5e-6 \
    --zero_stage 2
EOF

# Multi-GPU DeepSpeed run script (uses file 3)
cat > run_deepspeed_multi.sh << 'EOF'
#!/bin/bash
source venv/bin/activate

# Number of GPUs to use
NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "Detected $NUM_GPUS GPUs"

if [ $NUM_GPUS -eq 0 ]; then
    echo "No GPUs detected. Exiting..."
    exit 1
fi

echo "Running GRPO with DeepSpeed on $NUM_GPUS GPUs..."
deepspeed --num_gpus=$NUM_GPUS grpo_cot_deepspeed.py \
    --policy_model EleutherAI/gpt-neo-125m \
    --reward_model microsoft/deberta-v3-base \
    --dataset gsm8k \
    --num_examples 1000 \
    --num_epochs 3 \
    --batch_size 4 \
    --lr 5e-6 \
    --zero_stage 2
EOF

# Llama-3 with DeepSpeed run script (uses file 4)
cat > run_llama3_deepspeed.sh << 'EOF'
#!/bin/bash
source venv/bin/activate

# Check available GPU memory
if command -v nvidia-smi &> /dev/null; then
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1)
    echo "GPU Memory: ${GPU_MEM}MB"
    NUM_GPUS=$(nvidia-smi -L | wc -l)
else
    echo "No GPU detected. This model requires GPU."
    exit 1
fi

# Determine optimal settings based on GPU memory
if [ $GPU_MEM -lt 24000 ]; then
    echo "Warning: Less than 24GB GPU memory. Using LoRA and aggressive optimizations..."
    USE_LORA="--use_lora"
    BATCH_SIZE=1
    ZERO_STAGE=3
elif [ $GPU_MEM -lt 40000 ]; then
    echo "24-40GB GPU memory detected. Using LoRA..."
    USE_LORA="--use_lora"
    BATCH_SIZE=1
    ZERO_STAGE=3
else
    echo "40GB+ GPU memory detected. Running without LoRA..."
    USE_LORA=""
    BATCH_SIZE=2
    ZERO_STAGE=2
fi

echo "Running GRPO for Llama-3 with DeepSpeed..."
deepspeed --num_gpus=$NUM_GPUS grpo_llama3_deepspeed.py \
    --policy_model meta-llama/Meta-Llama-3-8B-Instruct \
    --dataset gsm8k \
    --num_examples 50 \
    --num_epochs 1 \
    --batch_size $BATCH_SIZE \
    --lr 2e-5 \
    --zero_stage $ZERO_STAGE \
    $USE_LORA \
    --K 2
EOF

# Multi-node DeepSpeed run script
cat > run_deepspeed_multinode.sh << 'EOF'
#!/bin/bash
source venv/bin/activate

# Multi-node configuration
# Edit these values for your cluster
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"29500"}
NODE_RANK=${NODE_RANK:-0}
NUM_NODES=${NUM_NODES:-1}

echo "Running GRPO with DeepSpeed multi-node..."
echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "Node rank: $NODE_RANK / $NUM_NODES"

deepspeed --num_nodes=$NUM_NODES \
    --num_gpus=8 \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    grpo_cot_deepspeed.py \
    --policy_model EleutherAI/gpt-neo-1.3B \
    --reward_model microsoft/deberta-v3-base \
    --dataset gsm8k \
    --num_examples 10000 \
    --num_epochs 3 \
    --batch_size 8 \
    --lr 5e-6 \
    --zero_stage 3
EOF

# Make run scripts executable
chmod +x run_single_gpu.sh run_deepspeed_single.sh run_deepspeed_multi.sh run_llama3_deepspeed.sh run_deepspeed_multinode.sh

# Create hostfile for multi-node training
cat > hostfile << 'EOF'
# Example hostfile for DeepSpeed multi-node training
# Format: hostname slots=num_gpus
# Uncomment and modify for your cluster:
# node1 slots=8
# node2 slots=8
# node3 slots=8
EOF

# Create a DeepSpeed launcher helper
cat > launch_deepspeed.py << 'EOF'
#!/usr/bin/env python3
"""Helper script to launch DeepSpeed with proper configuration"""

import argparse
import subprocess
import torch
import os

def get_gpu_memory():
    """Get GPU memory in GB"""
    if torch.cuda.is_available():
        return torch.cuda.get_device_properties(0).total_memory / 1024**3
    return 0

def recommend_settings(model_size, gpu_memory, num_gpus):
    """Recommend DeepSpeed settings based on model and hardware"""
    settings = {}
    
    # Model size estimates (in billions of parameters)
    model_sizes = {
        "gpt2": 0.124,
        "gpt-neo-125m": 0.125,
        "gpt-neo-1.3b": 1.3,
        "gpt-neo-2.7b": 2.7,
        "llama-7b": 7.0,
        "llama-13b": 13.0,
        "llama-3-8b": 8.0
    }
    
    # Determine ZeRO stage
    total_memory = gpu_memory * num_gpus
    model_memory = model_sizes.get(model_size, 1.0) * 4  # 4 bytes per parameter
    
    if model_memory > total_memory * 0.8:
        settings["zero_stage"] = 3
        settings["cpu_offload"] = True
    elif model_memory > total_memory * 0.4:
        settings["zero_stage"] = 2
        settings["cpu_offload"] = True
    else:
        settings["zero_stage"] = 2
        settings["cpu_offload"] = False
    
    # Batch size recommendations
    if gpu_memory < 16:
        settings["batch_size"] = 1
        settings["gradient_accumulation"] = 8
    elif gpu_memory < 24:
        settings["batch_size"] = 2
        settings["gradient_accumulation"] = 4
    elif gpu_memory < 40:
        settings["batch_size"] = 4
        settings["gradient_accumulation"] = 2
    else:
        settings["batch_size"] = 8
        settings["gradient_accumulation"] = 1
    
    return settings

def main():
    parser = argparse.ArgumentParser(description="DeepSpeed launcher helper")
    parser.add_argument("--model", type=str, help="Model name/size")
    parser.add_argument("--check", action="store_true", help="Just check settings")
    args = parser.parse_args()
    
    # Get hardware info
    num_gpus = torch.cuda.device_count()
    gpu_memory = get_gpu_memory()
    
    print(f"Hardware detected:")
    print(f"- GPUs: {num_gpus}")
    print(f"- GPU Memory: {gpu_memory:.1f} GB per GPU")
    
    if args.model:
        settings = recommend_settings(args.model.lower(), gpu_memory, num_gpus)
        print(f"\nRecommended settings for {args.model}:")
        for key, value in settings.items():
            print(f"- {key}: {value}")

if __name__ == "__main__":
    main()
EOF

chmod +x launch_deepspeed.py

# Test installation
print_status "Testing installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"
python -c "import deepspeed; print(f'DeepSpeed version: {deepspeed.__version__}')"

# Check CUDA and DeepSpeed compatibility
print_status "Checking DeepSpeed environment..."
ds_report=$(python -c "import deepspeed; deepspeed.env_report()" 2>&1)
echo "$ds_report" | grep -E "(torch|cuda|deepspeed)" || true

# Create requirements.txt for reproducibility
print_status "Creating requirements.txt..."
pip freeze > requirements.txt

print_status "Setup completed successfully!"
echo ""
echo "=================================================="
echo "Installation Summary:"
echo "- Virtual environment created in ./venv"
echo "- All dependencies installed"
echo "- DeepSpeed configured and ready"
echo "- Project directories created"
echo "- Run scripts created:"
echo "  - ./run_single_gpu.sh       - Single GPU without DeepSpeed (file 2)"
echo "  - ./run_deepspeed_single.sh - Single GPU with DeepSpeed (file 3)"
echo "  - ./run_deepspeed_multi.sh  - Multi-GPU with DeepSpeed (file 3)"
echo "  - ./run_llama3_deepspeed.sh - Llama-3 with DeepSpeed (file 4)"
echo "  - ./run_deepspeed_multinode.sh - Multi-node training"
echo ""
echo "DeepSpeed configs created in ./configs/:"
echo "  - ds_zero2_config.json - For medium models"
echo "  - ds_zero3_config.json - For large models"
echo ""
echo "To activate the environment:"
echo "  source venv/bin/activate"
echo ""
echo "To check DeepSpeed setup:"
echo "  python -m deepspeed.env_report"
echo ""
echo "To get hardware recommendations:"
echo "  ./launch_deepspeed.py --check --model gpt-neo-1.3b"
echo "=================================================="

# Create a comprehensive test script
cat > test_deepspeed_setup.py << 'EOF'
#!/usr/bin/env python3
"""Test script to verify DeepSpeed installation"""

import sys
import torch
import transformers
import deepspeed
import subprocess

print("Testing GRPO DeepSpeed setup...")
print("="*50)
print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"Transformers: {transformers.__version__}")
print(f"DeepSpeed: {deepspeed.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")

# Test DeepSpeed components
print("\nDeepSpeed components:")
try:
    import deepspeed.ops.adam as adam
    print("- FusedAdam: Available")
except:
    print("- FusedAdam: Not available")

try:
    import deepspeed.ops.transformer as transformer
    print("- Transformer kernels: Available")
except:
    print("- Transformer kernels: Not available")

# Check MPI
print("\nMPI support:")
result = subprocess.run(['which', 'mpirun'], capture_output=True, text=True)
if result.returncode == 0:
    print(f"- MPI available at: {result.stdout.strip()}")
else:
    print("- MPI not found (required for multi-node training)")

print("\nSetup test completed!")
print("Run 'ds_report' or 'python -m deepspeed.env_report' for detailed DeepSpeed diagnostics")
EOF

chmod +x test_deepspeed_setup.py

# Deactivate virtual environment
deactivate

print_status "Run 'source venv/bin/activate' to activate the environment"
print_status "Then run './test_deepspeed_setup.py' to verify DeepSpeed setup"
