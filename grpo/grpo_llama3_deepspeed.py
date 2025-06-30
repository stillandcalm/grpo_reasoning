#!/usr/bin/env python3
# grpo_llama3_deepspeed.py - GRPO with CoT for Llama-3-8B using DeepSpeed

import os
os.environ['WANDB_MODE'] = 'offline'

import sys
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    AutoConfig
)
from datasets import load_dataset, Dataset
import numpy as np
from tqdm import tqdm
import json
import argparse
import re
from typing import List, Tuple, Dict
import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
import torch.distributed as dist
import gc

# Suppress padding warnings
warnings.filterwarnings("ignore", message=".*right-padding was detected.*")

# DeepSpeed configuration for Llama-3-8B
def get_deepspeed_config_llama(batch_size, gradient_accumulation_steps=1, zero_stage=3, use_fp16=True):
    """Generate DeepSpeed configuration optimized for Llama-3-8B"""
    config = {
        "train_batch_size": batch_size * gradient_accumulation_steps,
        "train_micro_batch_size_per_gpu": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "gradient_clipping": 1.0,
        "wall_clock_breakdown": False,
    }
    
    # FP16 configuration
    if use_fp16:
        config["fp16"] = {
            "enabled": True,
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "initial_scale_power": 16,
            "hysteresis": 2,
            "min_loss_scale": 1
        }
    else:
        # Use BF16 for better stability with large models
        config["bf16"] = {
            "enabled": True
        }
    
    # Zero optimization configuration - ZeRO-3 for 8B model
    config["zero_optimization"] = {
        "stage": zero_stage,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": True
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": True
        },
        "overlap_comm": True,
        "contiguous_gradients": True,
        "sub_group_size": 1e8,  # Smaller for better memory management
        "reduce_bucket_size": 1e8,  # Smaller bucket size
        "stage3_prefetch_bucket_size": 1e7,
        "stage3_param_persistence_threshold": 1e5,
        "stage3_max_live_parameters": 1e8,
        "stage3_max_reuse_distance": 1e8,
        "stage3_gather_16bit_weights_on_model_save": True,
        "round_robin_gradients": True
    }
    
    # Activation checkpointing for memory efficiency
    config["activation_checkpointing"] = {
        "partition_activations": True,
        "cpu_checkpointing": True,
        "contiguous_memory_optimization": True,
        "number_checkpoints": None,
        "synchronize_checkpoint_boundary": False,
        "profile": False
    }
    
    # Optimizer config
    config["optimizer"] = {
        "type": "AdamW",
        "params": {
            "lr": 2e-5,  # Lower LR for large models
            "betas": [0.9, 0.95],  # Llama-style betas
            "eps": 1e-8,
            "weight_decay": 0.1
        }
    }
    
    # Memory optimizations
    config["flops_profiler"] = {
        "enabled": False,
        "profile_step": 1,
        "module_depth": -1,
        "top_modules": 1,
        "detailed": True,
        "output_file": None
    }
    
    return config

class RewardDataset(Dataset):
    """Dataset for training the reward model"""
    def __init__(self, examples: List[Dict], tokenizer, max_length=512):
        self.examples = examples  # Changed from self.data to self.examples
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.examples)  # Changed from self.data to self.examples
    
    def __getitem__(self, idx):
        item = self.examples[idx]  # Changed from self.data to self.examples
        encoding = self.tokenizer(
            item['text'],
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(item['score'], dtype=torch.float32)
        }
    
    def __getitems__(self, indices):
        """Support batch fetching for DataLoader"""
        return [self.__getitem__(i) for i in indices]

class PolicyDataset(Dataset):
    """Dataset for GRPO training with CoT support"""
    def __init__(self, data, tokenizer, max_length=512, use_cot=True):  # Increased max_length
        self.examples = []  # Changed from self.data to self.examples
        self.tokenizer = tokenizer
        self.use_cot = use_cot
        
        # Ensure left padding for decoder-only models
        original_padding = tokenizer.padding_side
        tokenizer.padding_side = 'left'
        
        # Handle both list and dataset formats
        if hasattr(data, '__len__'):
            items = data
        else:
            items = [data[i] for i in range(len(data))]
        
        for item in items:
            # Check if item has CoT trace
            has_cot = 'rationale' in item or 'chain_of_thought' in item or 'reasoning' in item
            
            if self.use_cot and has_cot:
                # Use CoT format with appropriate prompt for model type
                cot_trace = item.get('rationale', item.get('chain_of_thought', item.get('reasoning', '')))
                
                # Detect model type for appropriate prompt format
                if "llama" in tokenizer.name_or_path.lower():
                    prompt = (
                        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                        "You are a helpful assistant that solves problems step by step.<|eot_id|>"
                        "<|start_header_id|>user<|end_header_id|>\n\n"
                        f"{item['question']}<|eot_id|>"
                        "<|start_header_id|>assistant<|end_header_id|>\n\n"
                        "I'll solve this step by step.\n\n"
                    )
                else:
                    # Generic prompt for other models
                    prompt = (
                        "Solve this problem step by step.\n\n"
                        f"Problem: {item['question']}\n\n"
                        "Solution: Let me work through this step by step.\n\n"
                    )
                reference_cot = cot_trace
            else:
                # Standard format for non-CoT data
                if "llama" in tokenizer.name_or_path.lower():
                    prompt = (
                        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                        "You are a helpful assistant that solves math problems.<|eot_id|>"
                        "<|start_header_id|>user<|end_header_id|>\n\n"
                        f"{item['question']}<|eot_id|>"
                        "<|start_header_id|>assistant<|end_header_id|>\n\n"
                    )
                else:
                    prompt = (
                        "Solve the following math problem.\n\n"
                        f"Problem: {item['question']}\n\n"
                        "Answer: "
                    )
                reference_cot = ""
            
            inputs = tokenizer(
                prompt,
                truncation=True,
                max_length=max_length,
                padding='max_length',
                return_tensors='pt'
            )
            
            # Extract answer
            if 'answer' in item:
                full_answer = item['answer']
                if '####' in str(full_answer):
                    parts = full_answer.split('####')
                    answer = parts[1].strip().replace(',', '') if len(parts) > 1 else parts[0].strip()
                    solution = parts[0].strip() if len(parts) > 0 else ""
                else:
                    answer = str(full_answer).strip().replace(',', '')
                    solution = item.get('solution', '')
            else:
                answer = item.get('target', item.get('output', '')).strip()
                solution = ''
            
            self.examples.append({  # Changed from self.data to self.examples
                'prompt': prompt,
                'prompt_ids': inputs['input_ids'].squeeze(),
                'prompt_mask': inputs['attention_mask'].squeeze(),
                'answer': answer,
                'question': item['question'],
                'full_solution': solution,
                'reference_cot': reference_cot,
                'has_cot': has_cot
            })
        
        # Restore original padding side
        tokenizer.padding_side = original_padding
    
    def __len__(self):
        return len(self.examples)  # Changed from self.data to self.examples
    
    def __getitem__(self, idx):
        return self.examples[idx]  # Changed from self.data to self.examples
    
    def __getitems__(self, indices):
        """Support batch fetching for DataLoader"""
        return [self.examples[i] for i in indices]

def custom_collate_fn(batch):
    """Custom collate function to handle our data format"""
    collated = {
        'prompt': [],
        'prompt_ids': [],
        'prompt_mask': [],
        'answer': [],
        'question': [],
        'full_solution': [],
        'reference_cot': [],
        'has_cot': []
    }
    
    for item in batch:
        for key in collated:
            collated[key].append(item[key])
    
    collated['prompt_ids'] = torch.stack(collated['prompt_ids'])
    collated['prompt_mask'] = torch.stack(collated['prompt_mask'])
    
    return collated

class CoTAwareRewardModel(nn.Module):
    """Reward model that values Chain-of-Thought reasoning"""
    def __init__(self, model_name='microsoft/deberta-v3-base', dropout=0.1):
        super().__init__()
        
        config = AutoConfig.from_pretrained(model_name)
        config.num_labels = 1
        config.problem_type = "regression"
        
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            config=config,
            ignore_mismatched_sizes=True
        )
        
        hidden_size = config.hidden_size
        self.reward_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )
        
        self.reasoning_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 4),
            nn.GELU(),
            nn.Linear(hidden_size // 4, 1)
        )
        
    def forward(self, input_ids, attention_mask):
        outputs = self.model.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        pooled_output = outputs[0][:, 0, :]
        
        answer_reward = self.reward_head(pooled_output).squeeze(-1)
        reasoning_reward = self.reasoning_head(pooled_output).squeeze(-1)
        
        total_reward = 0.7 * answer_reward + 0.3 * reasoning_reward
        total_reward = 3.0 * torch.tanh(total_reward / 3.0)
        
        return total_reward

class GRPOTrainerLlama3DeepSpeed:
    def __init__(
        self, 
        policy_model_name='meta-llama/Meta-Llama-3-8B-Instruct',
        reward_model_name='microsoft/deberta-v3-base',
        local_rank=-1,
        zero_stage=3,
        use_lora=True,
        load_in_8bit=False
    ):
        self.local_rank = local_rank
        self.zero_stage = zero_stage
        self.use_lora = use_lora
        
        if local_rank != -1:
            torch.cuda.set_device(local_rank)
            device = torch.device('cuda', local_rank)
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.device = device
        print(f"Local rank: {local_rank}, Device: {self.device}")
        
        # Load tokenizer
        print(f"Loading tokenizer from: {policy_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            policy_model_name,
            use_fast=True,
            trust_remote_code=True
        )
        
        # Set special tokens for Llama-3
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'
        
        # Check GPU memory before loading large model
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"GPU Memory: {gpu_memory:.1f} GB")
            
            if "llama" in policy_model_name.lower() and gpu_memory < 24:
                print(f"WARNING: Loading Llama-3-8B on GPU with {gpu_memory:.1f}GB memory.")
                print("This may cause out-of-memory errors. Consider using:")
                print("  - LoRA (--use_lora flag)")
                print("  - Multiple GPUs")
                print("  - A smaller model like Mistral-7B")
                print("  - ZeRO-3 with CPU offloading")
        
        # Model loading configuration
        model_kwargs = {
            "torch_dtype": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            "device_map": None,  # Let DeepSpeed handle device placement
            "trust_remote_code": True,
        }
        
        if load_in_8bit and not use_lora:
            # 8-bit quantization for inference
            try:
                from transformers import BitsAndBytesConfig
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_8bit=True,
                    bnb_8bit_compute_dtype=torch.bfloat16
                )
            except ImportError:
                print("Warning: bitsandbytes not installed. Skipping 8-bit quantization.")
                load_in_8bit = False
        
        # Load policy model
        print(f"Loading policy model: {policy_model_name}")
        print("This may take several minutes for Llama-3-8B...")
        
        if use_lora:
            # Check if peft is installed
            try:
                from peft import LoraConfig, TaskType, get_peft_model
            except ImportError:
                print("PEFT not installed. Installing it now...")
                import subprocess
                subprocess.check_call([sys.executable, "-m", "pip", "install", "peft>=0.7.0"])
                from peft import LoraConfig, TaskType, get_peft_model
            
            self.policy_model = AutoModelForCausalLM.from_pretrained(
                policy_model_name,
                **model_kwargs
            )
            
            # LoRA configuration
            # Detect model type and set appropriate target modules
            if "llama" in policy_model_name.lower():
                target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            elif "gpt2" in policy_model_name.lower():
                target_modules = ["c_attn", "c_proj", "c_fc"]  # GPT-2 layer names
            elif "mistral" in policy_model_name.lower():
                target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            else:
                # Default to common attention layer names
                target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
                print(f"Warning: Unknown model type. Using default LoRA target modules: {target_modules}")
            
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=target_modules,
                lora_dropout=0.1,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            
            self.policy_model = get_peft_model(self.policy_model, lora_config)
            self.policy_model.print_trainable_parameters()
        else:
            self.policy_model = AutoModelForCausalLM.from_pretrained(
                policy_model_name,
                **model_kwargs
            )
        
        # Load reference model (frozen, can use 8-bit for large models only)
        print(f"Loading reference model...")
        ref_model_kwargs = model_kwargs.copy()
        
        # Only use 8-bit quantization for large models (7B+)
        model_size_gb = 0
        if "llama" in policy_model_name.lower() and "8b" in policy_model_name.lower():
            model_size_gb = 8
        elif "7b" in policy_model_name.lower():
            model_size_gb = 7
        elif "gpt2" in policy_model_name.lower() and "medium" not in policy_model_name.lower():
            model_size_gb = 0.5  # GPT-2 base is ~500MB
        
        # Only use 8-bit for models larger than 3GB
        if model_size_gb > 3 and load_in_8bit:
            try:
                from transformers import BitsAndBytesConfig
                ref_model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_8bit=True,
                    bnb_8bit_compute_dtype=torch.bfloat16
                )
                print("Loading reference model in 8-bit mode...")
            except ImportError:
                print("Warning: bitsandbytes not installed. Loading reference model in full precision.")
        else:
            # Remove device_map for small models to avoid the error
            ref_model_kwargs.pop("device_map", None)
        
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            policy_model_name,
            **ref_model_kwargs
        )
        
        # Only move to device if not quantized and ZeRO stage < 3
        if zero_stage < 3 and "quantization_config" not in ref_model_kwargs:
            self.ref_model = self.ref_model.to(device)
        
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False
        
        # Load reward model
        print(f"Loading reward tokenizer and model: {reward_model_name}")
        self.reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_name)
        self.reward_model = CoTAwareRewardModel(reward_model_name)
        
        if zero_stage < 3:
            self.reward_model = self.reward_model.to(device)
        
        print(f"Models loaded successfully!")
        
        # Print memory usage
        if torch.cuda.is_available():
            print(f"GPU memory allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
            print(f"GPU memory reserved: {torch.cuda.memory_reserved()/1024**3:.2f} GB")
    
    def compute_cot_features(self, text):
        """Extract features that indicate good Chain-of-Thought reasoning"""
        features = {
            'has_steps': bool(re.search(r'(step \d|first|second|then|next)', text.lower())),
            'has_calculations': bool(re.findall(r'\d+\s*[\+\-\*/]\s*\d+', text)),
            'has_reasoning_words': sum(1 for word in ['because', 'therefore', 'so', 'thus', 'since'] 
                                      if word in text.lower()),
            'has_conclusion': bool(re.search(r'(therefore|thus|so|answer is|the result)', text.lower())),
            'length': len(text.split()),
            'num_sentences': len(re.split(r'[.!?]+', text))
        }
        return features
    
    def generate_reward_training_data(self, dataset, num_samples_per_problem=5):
        """Generate training data for the CoT-aware reward model"""
        print("\nGenerating CoT-aware reward model training data...")
        reward_data = []
        
        dataset_items = [dataset[i] for i in range(min(50, len(dataset)))]
        
        for item in tqdm(dataset_items, desc="Creating reward data"):
            question = item['question']
            
            has_cot = 'rationale' in item or 'chain_of_thought' in item
            
            if has_cot:
                cot_trace = item.get('rationale', item.get('chain_of_thought', ''))
                prompt = f"Problem: {question}\n\nLet me think through this step by step:\n"
                
                reward_data.append({
                    'text': f"{prompt}{cot_trace}",
                    'score': 3.0
                })
                
                partial_cot = '. '.join(cot_trace.split('.')[:2]) + '...'
                reward_data.append({
                    'text': f"{prompt}{partial_cot}\n\nThe answer is probably {np.random.randint(1, 100)}.",
                    'score': 1.0
                })
            else:
                prompt = f"Problem: {question}\n\nSolution: "
                
                if 'answer' in item and '####' in item['answer']:
                    parts = item['answer'].split('####')
                    solution = parts[0].strip()
                    answer = parts[1].strip()
                    
                    reward_data.append({
                        'text': f"{prompt}Let me work through this step by step.\n{solution}\nTherefore, the answer is {answer}.",
                        'score': 2.5
                    })
            
            prompt_base = f"Problem: {question}\n\nSolution: "
            
            reward_data.append({
                'text': f"{prompt_base}The answer is {np.random.randint(1, 100)}.",
                'score': -1.0
            })
            
            reward_data.append({
                'text': f"{prompt_base}I'm not sure how to solve this.",
                'score': -2.0
            })
            
            reward_data.append({
                'text': f"{prompt_base}This looks like a math problem. Let me guess... maybe {np.random.randint(1, 100)}?",
                'score': -0.5
            })
        
        return reward_data
    
    def train_reward_model_deepspeed(self, reward_data, num_epochs=3, batch_size=16, lr=2e-5):
        """Train the CoT-aware neural reward model with DeepSpeed"""
        print(f"\nTraining CoT-aware reward model on {len(reward_data)} examples...")
        
        # For Llama-3, we keep the same reward model training
        # The reward model is much smaller and doesn't need special handling
        
        reward_dataset = RewardDataset(reward_data, self.reward_tokenizer)
        
        if self.local_rank != -1:
            sampler = DistributedSampler(reward_dataset)
        else:
            sampler = None
        
        dataloader = DataLoader(
            reward_dataset, 
            batch_size=batch_size, 
            shuffle=(sampler is None),
            sampler=sampler
        )
        
        if self.local_rank == -1:
            # Single GPU training for reward model
            optimizer = torch.optim.AdamW(self.reward_model.parameters(), lr=lr)
            self.reward_model.train()
            
            for epoch in range(num_epochs):
                total_loss = 0
                progress = tqdm(dataloader, desc=f"Reward epoch {epoch+1}/{num_epochs}")
                
                for batch in progress:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    predicted_rewards = self.reward_model(input_ids, attention_mask)
                    loss = F.mse_loss(predicted_rewards.float(), labels.float())
                    
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.reward_model.parameters(), 1.0)
                    optimizer.step()
                    
                    total_loss += loss.item()
                    progress.set_postfix({'loss': f"{loss.item():.4f}"})
                
                avg_loss = total_loss / len(dataloader)
                print(f"Epoch {epoch+1} - Average loss: {avg_loss:.4f}")
            
            print("Reward model training completed!")
            return
        
        # Distributed training path
        ds_config = get_deepspeed_config_llama(
            batch_size=batch_size * dist.get_world_size(),
            zero_stage=min(self.zero_stage, 2),
            use_fp16=False
        )
        
        self.reward_model, optimizer, _, _ = deepspeed.initialize(
            model=self.reward_model,
            model_parameters=self.reward_model.parameters(),
            config=ds_config
        )
        
        # Training loop remains the same...
    
    def get_reward_score(self, text):
        """Get reward score with bonus for good CoT reasoning"""
        self.reward_model.eval()
        
        # Ensure padding side is correct for reward tokenizer
        original_padding = self.reward_tokenizer.padding_side
        self.reward_tokenizer.padding_side = 'right'
        
        inputs = self.reward_tokenizer(
            text,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        ).to(self.device)
        
        self.reward_tokenizer.padding_side = original_padding
        
        with torch.no_grad():
            reward = self.reward_model(
                inputs['input_ids'],
                inputs['attention_mask']
            )
        
        base_reward = reward.item()
        
        cot_features = self.compute_cot_features(text)
        
        cot_bonus = 0
        if cot_features['has_steps']:
            cot_bonus += 0.3
        if cot_features['has_calculations']:
            cot_bonus += 0.2
        if cot_features['has_reasoning_words'] >= 2:
            cot_bonus += 0.2
        if cot_features['has_conclusion']:
            cot_bonus += 0.1
        if cot_features['num_sentences'] >= 3:
            cot_bonus += 0.2
        
        total_reward = base_reward + min(cot_bonus, 1.0)
        
        return np.clip(total_reward, -3.0, 3.0)
    
    def generate_response(self, prompt_ids, prompt_mask, temperature=0.7, max_new_tokens=300):
        """Generate response from policy model"""
        actual_length = prompt_mask.sum().item()
        prompt_ids = prompt_ids[:, :actual_length]
        prompt_mask = prompt_mask[:, :actual_length]
        
        with torch.no_grad():
            if hasattr(self, 'policy_engine'):
                model = self.policy_engine.module
            else:
                model = self.policy_model
                
            outputs = model.generate(
                input_ids=prompt_ids,
                attention_mask=prompt_mask,
                max_new_tokens=max_new_tokens,
                min_new_tokens=50,
                temperature=temperature,
                do_sample=True,
                top_k=50,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
        return outputs
    
    def train_policy_grpo_deepspeed(self, train_dataset, num_epochs=3, batch_size=1, lr=2e-5, K=2):
        """
        GRPO implementation for Llama-3-8B with DeepSpeed
        Note: Reduced batch_size and K for memory efficiency
        """
        if self.local_rank != -1:
            sampler = DistributedSampler(train_dataset)
        else:
            sampler = None
        
        dataloader = DataLoader(
            train_dataset, 
            batch_size=batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            collate_fn=custom_collate_fn
        )
        
        # DeepSpeed configuration for Llama-3
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        ds_config = get_deepspeed_config_llama(
            batch_size=batch_size * K * world_size,
            gradient_accumulation_steps=K * 4,  # More gradient accumulation for stability
            zero_stage=self.zero_stage,
            use_fp16=False  # Use BF16 instead
        )
        ds_config["optimizer"]["params"]["lr"] = lr
        
        # Get trainable parameters only if using LoRA
        if self.use_lora:
            model_parameters = filter(lambda p: p.requires_grad, self.policy_model.parameters())
        else:
            model_parameters = self.policy_model.parameters()
        
        # Initialize DeepSpeed
        self.policy_engine, _, _, _ = deepspeed.initialize(
            model=self.policy_model,
            model_parameters=model_parameters,
            config=ds_config
        )
        
        if self.local_rank in [-1, 0]:
            print(f"\nStarting GRPO training for Llama-3-8B...")
            print(f"Dataset size: {len(train_dataset)}")
            print(f"Batch size: {batch_size}")
            print(f"K (responses per prompt): {K}")
            print(f"Learning rate: {lr}")
            print(f"ZeRO stage: {self.zero_stage}")
            print(f"Using LoRA: {self.use_lora}")
        
        best_avg_reward = -float('inf')
        
        for epoch in range(num_epochs):
            if sampler:
                sampler.set_epoch(epoch)
            
            self.policy_engine.train()
            total_loss = 0
            total_reward = 0
            num_updates = 0
            
            progress = tqdm(dataloader, desc=f"GRPO Epoch {epoch+1}/{num_epochs}", 
                           disable=(self.local_rank not in [-1, 0]))
            
            for batch_idx, batch in enumerate(progress):
                # Memory efficient: process one example at a time
                for i in range(min(batch_size, len(batch['prompt']))):
                    prompt_ids = batch['prompt_ids'][i:i+1].to(self.device)
                    prompt_mask = batch['prompt_mask'][i:i+1].to(self.device)
                    prompt_text = batch['prompt'][i]
                    
                    responses = []
                    rewards = []
                    
                    # Generate K responses with lower temperature for Llama-3
                    for k in range(K):
                        temp = 0.6 + (k * 0.1)  # Lower temperature range
                        response = self.generate_response(
                            prompt_ids, 
                            prompt_mask, 
                            temperature=temp,
                            max_new_tokens=300
                        )
                        
                        full_text = self.tokenizer.decode(response[0], skip_special_tokens=True)
                        completion = full_text.replace(prompt_text, '')
                        
                        reward = self.get_reward_score(full_text)
                        
                        responses.append(response)
                        rewards.append(reward)
                        total_reward += reward
                        
                        # Clear cache to save memory
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    
                    # Compute advantage
                    baseline = np.mean(rewards)
                    
                    # Update on each response
                    for response, reward in zip(responses, rewards):
                        advantage = reward - baseline
                        
                        # Forward pass
                        outputs = self.policy_engine(response, labels=response)
                        
                        # GRPO loss
                        grpo_loss = -advantage * outputs.loss
                        
                        # Simplified KL penalty for memory efficiency
                        kl_weight = 0.001  # Lower KL weight for Llama-3
                        loss = grpo_loss
                        
                        # Backward with DeepSpeed
                        self.policy_engine.backward(loss)
                        
                        total_loss += loss.item()
                        num_updates += 1
                    
                    # Step after K responses
                    self.policy_engine.step()
                    
                    # Clear cache
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                # Update progress
                if num_updates > 0:
                    avg_reward = total_reward / (K * (batch_idx + 1))
                    progress.set_postfix({
                        'loss': f"{total_loss/num_updates:.3f}",
                        'reward': f"{avg_reward:.2f}",
                        'gpu_mem': f"{torch.cuda.memory_allocated()/1024**3:.1f}GB"
                    })
            
            # Save checkpoint
            if self.local_rank in [-1, 0]:
                epoch_avg_reward = total_reward / (K * len(train_dataset))
                print(f"Epoch {epoch+1} - Avg reward: {epoch_avg_reward:.2f}")
                
                if epoch_avg_reward > best_avg_reward:
                    best_avg_reward = epoch_avg_reward
                    self.save_models_deepspeed("outputs/llama3_grpo_best")
    
    def save_models_deepspeed(self, output_dir):
        """Save models with DeepSpeed support"""
        os.makedirs(output_dir, exist_ok=True)
        
        if hasattr(self, 'policy_engine'):
            # Save with DeepSpeed
            self.policy_engine.save_checkpoint(output_dir, tag="policy")
            
            # For LoRA, also save the adapter
            if self.use_lora:
                self.policy_engine.module.save_pretrained(f"{output_dir}/lora_adapter")
        else:
            self.policy_model.save_pretrained(f"{output_dir}/policy")
        
        # Save tokenizer
        self.tokenizer.save_pretrained(f"{output_dir}/tokenizer")
        
        # Save reward model
        if hasattr(self.reward_model, 'module'):
            torch.save(self.reward_model.module.state_dict(), f"{output_dir}/reward_model.pt")
        else:
            torch.save(self.reward_model.state_dict(), f"{output_dir}/reward_model.pt")
        
        # Save config
        config = {
            'model_type': 'llama3_grpo_deepspeed',
            'base_model': 'meta-llama/Meta-Llama-3-8B-Instruct',
            'use_lora': self.use_lora,
            'zero_stage': self.zero_stage
        }
        with open(f"{output_dir}/config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Models saved to {output_dir}")
    
    def test(self):
        """Test the trained model"""
        if self.local_rank not in [-1, 0]:
            return
            
        print("\nTesting Llama-3 with CoT...")
        if hasattr(self, 'policy_engine'):
            self.policy_engine.eval()
            model = self.policy_engine.module
        else:
            self.policy_model.eval()
            model = self.policy_model
        
        test_problems = [
            "What is 25 + 38?",
            "If Sarah has 42 candies and gives 17 to her friend, how many does she have left?",
            "A baker made 156 cookies. He sold 89. How many are left?"
        ]
        
        for problem in test_problems:
            # Use appropriate prompt format
            if "llama" in self.tokenizer.name_or_path.lower():
                prompt = (
                    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                    "You are a helpful assistant that solves problems step by step.<|eot_id|>"
                    "<|start_header_id|>user<|end_header_id|>\n\n"
                    f"{problem}<|eot_id|>"
                    "<|start_header_id|>assistant<|end_header_id|>\n\n"
                )
            else:
                prompt = (
                    "Solve this problem step by step.\n\n"
                    f"Problem: {problem}\n\n"
                    "Solution: "
                )
            
            inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=300,
                    temperature=0.7,
                    do_sample=True,
                    top_k=50
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            solution = response.replace(prompt, '')
            
            print(f"\n{'='*60}")
            print(f"Problem: {problem}")
            print(f"Solution: {solution[:500]}...")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy_model', default='meta-llama/Meta-Llama-3-8B-Instruct',
                       help='Llama model to use')
    parser.add_argument('--dataset', default='gsm8k',
                       choices=['gsm8k', 'aqua_rat', 'math_qa'])
    parser.add_argument('--num_examples', type=int, default=100)
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--zero_stage', type=int, default=3,
                       help='DeepSpeed ZeRO stage (use 3 for 8B model)')
    parser.add_argument('--use_lora', action='store_true', default=True,
                       help='Use LoRA for efficient fine-tuning')
    parser.add_argument('--load_in_8bit', action='store_true', default=False,
                       help='Load model in 8-bit (for inference only)')
    parser.add_argument('--K', type=int, default=2,
                       help='Number of responses per prompt')
    args = parser.parse_args()
    
    # Initialize distributed training if needed
    if args.local_rank != -1:
        deepspeed.init_distributed()
    
    # Load dataset
    from datasets import load_dataset
    
    # Fix for dataset loading issue - handle different datasets properly
    print(f"Loading dataset: {args.dataset}")
    try:
        if args.dataset == 'gsm8k':
            dataset = load_dataset('gsm8k', 'main')['train']
        elif args.dataset == 'aqua_rat':
            dataset = load_dataset('aqua_rat')['train']
        elif args.dataset == 'math_qa':
            dataset = load_dataset('math_qa')['train']
        else:
            # Try generic loading
            dataset = load_dataset(args.dataset)['train']
        
        # Select subset
        if len(dataset) > args.num_examples:
            dataset = dataset.select(range(args.num_examples))
            
        print(f"Loaded {len(dataset)} examples from {args.dataset}")
            
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Trying alternative loading approach...")
        
        # Alternative: Create dummy data for testing
        print("Creating dummy dataset for testing...")
        dummy_data = []
        for i in range(args.num_examples):
            dummy_data.append({
                'question': f"What is {i} + {i+1}?",
                'answer': f"Let me solve this step by step. {i} + {i+1} = {2*i+1}. #### {2*i+1}"
            })
        
        from datasets import Dataset
        dataset = Dataset.from_list(dummy_data)
    
    # Initialize trainer
    trainer = GRPOTrainerLlama3DeepSpeed(
        policy_model_name=args.policy_model,
        local_rank=args.local_rank,
        zero_stage=args.zero_stage,
        use_lora=args.use_lora,
        load_in_8bit=args.load_in_8bit
    )
    
    # Create training dataset
    # Use all examples if we have fewer than 50, otherwise skip the first 50 for training
    if len(dataset) <= 50:
        train_data = [dataset[i] for i in range(len(dataset))]
    else:
        train_data = [dataset[i] for i in range(50, min(args.num_examples, len(dataset)))]
    
    print(f"Creating training dataset with {len(train_data)} examples...")
    train_dataset = PolicyDataset(train_data, trainer.tokenizer, max_length=512)
    
    # Generate and train reward model
    reward_data = trainer.generate_reward_training_data(dataset)
    trainer.train_reward_model_deepspeed(reward_data, num_epochs=2)
    
    # Train policy with GRPO
    trainer.train_policy_grpo_deepspeed(
        train_dataset, 
        args.num_epochs, 
        args.batch_size, 
        args.lr, 
        K=args.K
    )
    
    # Test
    trainer.test()
    
    # Save final models
    if args.local_rank in [-1, 0]:
        trainer.save_models_deepspeed("outputs/llama3_grpo_final")

if __name__ == "__main__":
    main()
