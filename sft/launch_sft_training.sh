cd ~/sft_reasoning
source ~/sft_env/bin/activate

deepspeed --hostfile=hostfile \
    --master_addr=10.65.0.2 \
    --master_port=29500 \
    sft_cot_trainer.py \
    --num_train_samples 1000 \
    --num_epochs 2 \
    --batch_size 1


# Alternative ways to call SFT
# python sft_cot_trainer.py --model meta-llama/Llama-2-7b-hf --output_dir ./sft_math_model
# 
# Edit launch_grpo_training.sh:
# export MODEL_PATH=./sft_math_model  # Use SFT model, not base LLaMA
# lsunch grpo training using this model path for the policy model defined in the MODEL_PATH
#
# For Single GPU:
# python sft_cot_trainer.py --num_train_samples 100 --num_epochs 1
#
# For Single Node Multi-GPU:
# deepspeed --num_gpus=8 sft_cot_trainer.py --num_train_samples 1000 --num_epochs 3
#
# For Multi-Node:
# export IS_CLUSTER=true
# export MASTER_IP="10.65.0.2"
# export SLAVE_IP="10.65.0.3"
# ./setup_runpod.sh
#
#
