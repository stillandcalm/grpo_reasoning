#!/usr/bin/env python3
# grpo_cot_deepspeed.py - GRPO with Chain-of-Thought reasoning using DeepSpeed

import os
os.environ['WANDB_MODE'] = 'offline'

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
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import json
import argparse
import re
from typing import List, Tuple, Dict
import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
import torch.distributed as dist

# DeepSpeed configuration
def get_deepspeed_config(batch_size, gradient_accumulation_steps=1, zero_stage=2, use_fp16=True):
    """Generate DeepSpeed configuration"""
    config = {
        "train_batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "gradient_clipping": 1.0,
        "wall_clock_breakdown": False
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
    
    # Zero optimization configuration
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
        "sub_group_size": 1e9,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_16bit_weights_on_model_save": True
    }
    
    # Always add optimizer config
    config["optimizer"] = {
        "type": "AdamW",
        "params": {
            "lr": 5e-6,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01
        }
    }
    
    return config

class RewardDataset(Dataset):
    """Dataset for training the reward model"""
    def __init__(self, examples: List[Dict], tokenizer, max_length=512):
        self.data = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
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

class PolicyDataset(Dataset):
    """Dataset for GRPO training with CoT support"""
    def __init__(self, data, tokenizer, max_length=256, use_cot=True):
        self.data = []
        self.tokenizer = tokenizer
        self.use_cot = use_cot
        
        # Handle both list and dataset formats
        if hasattr(data, '__len__'):
            items = data
        else:
            items = [data[i] for i in range(len(data))]
        
        for item in items:
            # Check if item has CoT trace
            has_cot = 'rationale' in item or 'chain_of_thought' in item or 'reasoning' in item
            
            if self.use_cot and has_cot:
                # Use CoT format
                cot_trace = item.get('rationale', item.get('chain_of_thought', item.get('reasoning', '')))
                prompt = (
                    "Solve this problem step by step.\n\n"
                    f"Problem: {item['question']}\n\n"
                    "Let me think through this step by step:\n"
                )
                reference_cot = cot_trace
            else:
                # Standard format for non-CoT data
                prompt = (
                    "Solve the following math problem step by step.\n\n"
                    f"Problem: {item['question']}\n\n"
                    "Solution: I'll solve this step by step.\n\n"
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
            
            self.data.append({
                'prompt': prompt,
                'prompt_ids': inputs['input_ids'].squeeze(),
                'prompt_mask': inputs['attention_mask'].squeeze(),
                'answer': answer,
                'question': item['question'],
                'full_solution': solution,
                'reference_cot': reference_cot,
                'has_cot': has_cot
            })
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

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

class GRPOTrainerWithCoTDeepSpeed:
    def __init__(
        self, 
        policy_model_name='EleutherAI/gpt-neo-125m',
        reward_model_name='microsoft/deberta-v3-base',
        local_rank=-1,
        zero_stage=2
    ):
        self.local_rank = local_rank
        self.zero_stage = zero_stage
        
        if local_rank != -1:
            torch.cuda.set_device(local_rank)
            device = torch.device('cuda', local_rank)
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.device = device
        print(f"Local rank: {local_rank}, Device: {self.device}")
        
        # Load tokenizer
        print(f"Loading tokenizer from: {policy_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(policy_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'
        
        # Load policy model (will be wrapped by DeepSpeed)
        print(f"Loading policy model: {policy_model_name}")
        self.policy_model = AutoModelForCausalLM.from_pretrained(
            policy_model_name,
            torch_dtype=torch.float32  # Load in FP32, DeepSpeed will handle FP16 conversion
        )
        
        # Load reference model (keep in FP32 for stability)
        print(f"Loading reference model...")
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            policy_model_name,
            torch_dtype=torch.float32
        )
        
        # Only move ref_model to device if not using ZeRO-3
        if zero_stage < 3:
            self.ref_model = self.ref_model.to(device)
        
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False
        
        # Load reward model components
        print(f"Loading reward tokenizer and model: {reward_model_name}")
        self.reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_name)
        self.reward_model = CoTAwareRewardModel(reward_model_name)
        
        # Only move reward model to device if not using ZeRO-3
        if zero_stage < 3:
            self.reward_model = self.reward_model.to(device)
        
        print(f"Models loaded successfully!")
    
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
        print(f"\nTraining CoT-aware reward model on {len(reward_data)} examples with DeepSpeed...")
        
        # Create dataset
        reward_dataset = RewardDataset(reward_data, self.reward_tokenizer)
        
        # Create sampler for distributed training
        if self.local_rank != -1:
            sampler = DistributedSampler(reward_dataset)
        else:
            sampler = None
        
        # Create dataloader
        dataloader = DataLoader(
            reward_dataset, 
            batch_size=batch_size, 
            shuffle=(sampler is None),
            sampler=sampler
        )
        
        # For single GPU, use regular training without DeepSpeed
        if self.local_rank == -1:
            print("Using single GPU training for reward model...")
            optimizer = torch.optim.AdamW(self.reward_model.parameters(), lr=lr)
            self.reward_model.to(self.device)
            self.reward_model.train()
            
            for epoch in range(num_epochs):
                total_loss = 0
                progress = tqdm(dataloader, desc=f"Reward epoch {epoch+1}/{num_epochs}")
                
                for batch in progress:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    # Forward pass
                    predicted_rewards = self.reward_model(input_ids, attention_mask)
                    
                    # MSE loss - ensure both tensors are same dtype
                    loss = F.mse_loss(predicted_rewards.float(), labels.float())
                    
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.reward_model.parameters(), 1.0)
                    optimizer.step()
                    
                    total_loss += loss.item()
                    progress.set_postfix({'loss': f"{loss.item():.4f}"})
                
                avg_loss = total_loss / len(dataloader)
                print(f"Epoch {epoch+1} - Average loss: {avg_loss:.4f}")
                
            print("CoT-aware reward model training completed!")
            return
        
        # DeepSpeed config for reward model (for distributed training)
        ds_config = get_deepspeed_config(
            batch_size=batch_size * dist.get_world_size(),
            zero_stage=min(self.zero_stage, 2),  # Use at most ZeRO-2 for reward model
            use_fp16=False  # Disable FP16 for reward model to avoid dtype issues
        )
        if "optimizer" not in ds_config:
            ds_config["optimizer"] = {
                "type": "AdamW",
                "params": {
                    "lr": lr,
                    "betas": [0.9, 0.999],
                    "eps": 1e-8,
                    "weight_decay": 0.01
                }
            }
        else:
            ds_config["optimizer"]["params"]["lr"] = lr
        
        # Initialize DeepSpeed
        self.reward_model, optimizer, _, _ = deepspeed.initialize(
            model=self.reward_model,
            model_parameters=self.reward_model.parameters(),
            config=ds_config
        )
        
        self.reward_model.train()
        
        for epoch in range(num_epochs):
            if sampler:
                sampler.set_epoch(epoch)
            
            total_loss = 0
            progress = tqdm(dataloader, desc=f"Reward epoch {epoch+1}/{num_epochs}")
            
            for batch in progress:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                predicted_rewards = self.reward_model(input_ids, attention_mask)
                
                # MSE loss
                loss = F.mse_loss(predicted_rewards, labels)
                
                # Backward pass with DeepSpeed
                self.reward_model.backward(loss)
                self.reward_model.step()
                
                total_loss += loss.item()
                progress.set_postfix({'loss': f"{loss.item():.4f}"})
            
            avg_loss = total_loss / len(dataloader)
            if self.local_rank in [-1, 0]:
                print(f"Epoch {epoch+1} - Average loss: {avg_loss:.4f}")
        
        print("CoT-aware reward model training completed!")
    
    def get_reward_score(self, text):
        """Get reward score with bonus for good CoT reasoning"""
        self.reward_model.eval()
        
        inputs = self.reward_tokenizer(
            text,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        ).to(self.device)
        
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
    
    def generate_response(self, prompt_ids, prompt_mask, temperature=0.8):
        """Generate response from policy model"""
        actual_length = prompt_mask.sum().item()
        prompt_ids = prompt_ids[:, :actual_length]
        prompt_mask = prompt_mask[:, :actual_length]
        
        with torch.no_grad():
            # Check if model is wrapped by DeepSpeed
            if hasattr(self, 'policy_engine'):
                model = self.policy_engine.module
            else:
                model = self.policy_model
                
            outputs = model.generate(
                input_ids=prompt_ids,
                attention_mask=prompt_mask,
                max_new_tokens=200,
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
    
    def train_policy_grpo_deepspeed(self, train_dataset, num_epochs=3, batch_size=2, lr=5e-6, K=4):
        """
        True GRPO implementation with CoT awareness using DeepSpeed
        """
        # Create sampler for distributed training
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
        
        # DeepSpeed configuration
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        ds_config = get_deepspeed_config(
            batch_size=batch_size * K * world_size,
            gradient_accumulation_steps=K,
            zero_stage=self.zero_stage,
            use_fp16=True  # Use FP16 for policy model
        )
        ds_config["optimizer"]["params"]["lr"] = lr
        
        # Initialize DeepSpeed for policy model
        self.policy_engine, _, _, _ = deepspeed.initialize(
            model=self.policy_model,
            model_parameters=self.policy_model.parameters(),
            config=ds_config
        )
        
        if self.local_rank in [-1, 0]:
            print(f"\nStarting GRPO training with CoT-aware rewards using DeepSpeed...")
            print(f"Dataset size: {len(train_dataset)}")
            print(f"Batch size: {batch_size}")
            print(f"K (responses per prompt): {K}")
            print(f"Learning rate: {lr}")
            print(f"ZeRO stage: {self.zero_stage}")
        
        best_avg_reward = -float('inf')
        
        for epoch in range(num_epochs):
            if sampler:
                sampler.set_epoch(epoch)
            
            self.policy_engine.train()
            total_loss = 0
            total_reward = 0
            num_updates = 0
            positive_rewards = 0
            high_quality_cot = 0
            
            progress = tqdm(dataloader, desc=f"GRPO Epoch {epoch+1}/{num_epochs}", 
                           disable=(self.local_rank not in [-1, 0]))
            
            for batch_idx, batch in enumerate(progress):
                # Collect ALL responses and rewards
                all_responses = []
                all_rewards = []
                all_texts = []
                
                for i in range(min(batch_size, len(batch['prompt']))):
                    prompt_ids = batch['prompt_ids'][i:i+1].to(self.device)
                    prompt_mask = batch['prompt_mask'][i:i+1].to(self.device)
                    prompt_text = batch['prompt'][i]
                    has_cot_ref = batch['has_cot'][i]
                    
                    # Generate K responses
                    prompt_responses = []
                    prompt_rewards = []
                    prompt_texts = []
                    
                    for k in range(K):
                        temp = 0.6 + (k * 0.2)
                        response = self.generate_response(prompt_ids, prompt_mask, temperature=temp)
                        
                        full_text = self.tokenizer.decode(response[0], skip_special_tokens=True)
                        completion = full_text.replace(prompt_text, '')
                        
                        reward = self.get_reward_score(full_text)
                        
                        prompt_responses.append(response)
                        prompt_rewards.append(reward)
                        prompt_texts.append(completion)
                        
                        if reward > 1.0:
                            positive_rewards += 1
                        if reward > 2.0:
                            high_quality_cot += 1
                        
                        # Show examples from first batch
                        if self.local_rank in [-1, 0] and epoch == 0 and batch_idx == 0 and i == 0 and k < 2:
                            print(f"\n[Response {k+1}, reward={reward:.2f}]")
                            print(f"Question: {batch['question'][i]}")
                            print(f"Completion: {completion[:150]}...")
                            cot_features = self.compute_cot_features(completion)
                            print(f"CoT features: steps={cot_features['has_steps']}, "
                                  f"calc={cot_features['has_calculations']}, "
                                  f"reasoning={cot_features['has_reasoning_words']}")
                    
                    all_responses.extend(prompt_responses)
                    all_rewards.extend(prompt_rewards)
                    all_texts.extend(prompt_texts)
                    total_reward += sum(prompt_rewards)
                
                # Skip if all rewards are terrible
                if max(all_rewards) < -1.5:
                    continue
                
                # Compute baseline
                baseline = np.mean(all_rewards)
                
                # GRPO update on ALL responses
                total_grpo_loss = 0
                num_responses = 0
                
                for response, reward, text in zip(all_responses, all_rewards, all_texts):
                    # Advantage
                    advantage = reward - baseline
                    
                    # Forward pass
                    outputs = self.policy_engine(response, labels=response)
                    
                    # GRPO loss
                    grpo_loss = -advantage * outputs.loss
                    
                    # KL penalty
                    with torch.no_grad():
                        ref_outputs = self.ref_model(response)
                    
                    kl_div = F.kl_div(
                        F.log_softmax(outputs.logits, dim=-1),
                        F.softmax(ref_outputs.logits, dim=-1),
                        reduction='batchmean'
                    )
                    
                    # Combine losses
                    loss = grpo_loss + 0.01 * kl_div
                    
                    # Backward with DeepSpeed (accumulates gradients)
                    self.policy_engine.backward(loss)
                    
                    total_grpo_loss += loss.item()
                    num_responses += 1
                
                # Step optimizer after all K responses
                self.policy_engine.step()
                
                if num_responses > 0:
                    avg_loss = total_grpo_loss / num_responses
                    total_loss += avg_loss
                    num_updates += 1
                
                # Update progress
                if num_updates > 0:
                    avg_loss = total_loss / num_updates
                    avg_reward = total_reward / (K * (batch_idx + 1) * batch_size)
                    progress.set_postfix({
                        'loss': f"{avg_loss:.3f}",
                        'reward': f"{avg_reward:.2f}",
                        'baseline': f"{baseline:.2f}",
                        'high_cot': high_quality_cot
                    })
            
            # Epoch summary
            epoch_avg_reward = total_reward / (K * len(train_dataset))
            if self.local_rank in [-1, 0]:
                print(f"Epoch {epoch+1} - Loss: {total_loss/max(num_updates,1):.4f}, "
                      f"Avg reward: {epoch_avg_reward:.2f}, "
                      f"High-quality CoT: {high_quality_cot}/{K*len(train_dataset)}")
            
            # Save best model
            if epoch_avg_reward > best_avg_reward:
                best_avg_reward = epoch_avg_reward
                if self.local_rank in [-1, 0]:
                    self.save_models_deepspeed("outputs/grpo_cot_deepspeed_best")
                    print(f"Saved best model with avg reward: {best_avg_reward:.2f}")
    
    def save_models_deepspeed(self, output_dir):
        """Save both policy and reward models with DeepSpeed support"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save policy model with DeepSpeed
        if hasattr(self, 'policy_engine'):
            self.policy_engine.save_checkpoint(output_dir, tag="policy")
        else:
            # Save without DeepSpeed
            self.policy_model.save_pretrained(f"{output_dir}/policy")
        
        # Save tokenizer
        self.tokenizer.save_pretrained(f"{output_dir}/tokenizer")
        
        # Save reward model state dict
        if hasattr(self.reward_model, 'module'):
            torch.save(self.reward_model.module.state_dict(), f"{output_dir}/reward_model.pt")
        else:
            torch.save(self.reward_model.state_dict(), f"{output_dir}/reward_model.pt")
        
        # Save config
        config = {
            'model_type': 'grpo_cot_deepspeed',
            'policy_model': self.policy_model.config._name_or_path if hasattr(self.policy_model, 'config') else 'unknown',
            'reward_model': 'cot_aware_deberta',
            'zero_stage': self.zero_stage
        }
        with open(f"{output_dir}/config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Models saved to {output_dir}")
    
    def test(self):
        """Test the trained policy model"""
        if self.local_rank not in [-1, 0]:
            return
            
        print("\nTesting CoT-trained policy model...")
        if hasattr(self, 'policy_engine'):
            self.policy_engine.eval()
            model = self.policy_engine.module
        else:
            self.policy_model.eval()
            model = self.policy_model
        
        test_problems = [
            "What is 25 + 38?",
            "If Sarah has 42 candies and gives 17 to her friend, how many does she have left?",
            "What is 9 times 8?",
            "A baker made 156 cookies. He sold 89. How many are left?",
            "What is 144 divided by 12?"
        ]
        
        for problem in test_problems:
            prompt = (
                "Solve this problem step by step.\n\n"
                f"Problem: {problem}\n\n"
                "Let me think through this step by step:\n"
            )
            
            inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.7,
                    do_sample=True,
                    top_k=50
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            solution = response.replace(prompt, '')
            
            reward = self.get_reward_score(response)
            cot_features = self.compute_cot_features(solution)
            
            print(f"\n{'='*60}")
            print(f"Problem: {problem}")
            print(f"Solution: {solution[:300]}...")
            print(f"Reward score: {reward:.2f}")
            print(f"CoT quality: steps={cot_features['has_steps']}, "
                  f"calculations={cot_features['has_calculations']}, "
                  f"reasoning_words={cot_features['has_reasoning_words']}")

def load_cot_dataset(dataset_name, split="train", max_examples=1000):
    """Load various CoT datasets"""
    print(f"Loading CoT dataset: {dataset_name}")
    
    try:
        if dataset_name == "gsm8k":
            # Standard GSM8K (has some reasoning in answers)
            # Try different loading methods due to version compatibility
            try:
                dataset = load_dataset("gsm8k", "main", split=split)
            except:
                # Alternative loading method
                dataset = load_dataset("openai/gsm8k", "main", split=split)
            
            # Manually slice to max_examples
            if len(dataset) > max_examples:
                dataset = dataset.select(range(max_examples))
        
        elif dataset_name == "aqua_rat":
            # AQuA-RAT has rationales
            dataset = load_dataset("deepmind/aqua_rat", split=split)
            if len(dataset) > max_examples:
                dataset = dataset.select(range(max_examples))
        
        elif dataset_name == "math_qa":
            # MathQA includes reasoning chains
            dataset = load_dataset("math_qa", split=split)
            if len(dataset) > max_examples:
                dataset = dataset.select(range(max_examples))
            
        elif dataset_name == "cot_collection":
            # If available, use dedicated CoT collections
            try:
                dataset = load_dataset("kaist-ai/CoT-Collection", split=split)
                if len(dataset) > max_examples:
                    dataset = dataset.select(range(max_examples))
            except:
                print("CoT Collection not available, falling back to GSM8K")
                dataset = load_dataset("gsm8k", "main", split=split)
                if len(dataset) > max_examples:
                    dataset = dataset.select(range(max_examples))
        
        else:
            print(f"Unknown dataset {dataset_name}, using GSM8K")
            dataset = load_dataset("gsm8k", "main", split=split)
            if len(dataset) > max_examples:
                dataset = dataset.select(range(max_examples))
        
        return dataset
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Trying alternative loading approach...")
        
        # Fallback: load without config name
        try:
            if dataset_name == "gsm8k":
                dataset = load_dataset("gsm8k", split=split)
            else:
                dataset = load_dataset(dataset_name, split=split)
            
            if len(dataset) > max_examples:
                dataset = dataset.select(range(max_examples))
            
            return dataset
        except:
            raise ValueError(f"Could not load dataset {dataset_name}. Please check your internet connection and dataset availability.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy_model', default='EleutherAI/gpt-neo-125m',
                       choices=['gpt2', 'gpt2-medium', 'EleutherAI/gpt-neo-125m', 'EleutherAI/gpt-neo-1.3B'])
    parser.add_argument('--reward_model', default='microsoft/deberta-v3-base',
                       choices=['microsoft/deberta-v3-base', 'roberta-base', 'albert-base-v2'])
    parser.add_argument('--dataset', default='gsm8k',
                       choices=['gsm8k', 'aqua_rat', 'math_qa', 'cot_collection'])
    parser.add_argument('--num_examples', type=int, default=100)
    parser.add_argument('--num_epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=5e-6)
    parser.add_argument('--local_rank', type=int, default=-1,
                       help='Local rank for distributed training')
    parser.add_argument('--zero_stage', type=int, default=2,
                       help='DeepSpeed ZeRO optimization stage (0, 1, 2, or 3)')
    parser.add_argument('--use_cot', action='store_true', default=True,
                       help='Use CoT-aware training')
    args = parser.parse_args()
    
    # Initialize distributed training if needed
    if args.local_rank != -1:
        deepspeed.init_distributed()
    
    # Load dataset
    dataset = load_cot_dataset(args.dataset, max_examples=args.num_examples)
    
    # Initialize trainer with DeepSpeed
    trainer = GRPOTrainerWithCoTDeepSpeed(
        policy_model_name=args.policy_model,
        reward_model_name=args.reward_model,
        local_rank=args.local_rank,
        zero_stage=args.zero_stage
    )
    
    # Create datasets
    tokenizer = trainer.tokenizer
    if args.num_examples > 50:
        policy_data = [dataset[i] for i in range(50, args.num_examples)]
    else:
        policy_data = [dataset[i] for i in range(args.num_examples)]
    
    train_dataset = PolicyDataset(policy_data, tokenizer, use_cot=args.use_cot)
    
    # Generate reward training data and train reward model
    reward_data = trainer.generate_reward_training_data(dataset)
    trainer.train_reward_model_deepspeed(reward_data, num_epochs=3)
    
    # Train policy with GRPO using DeepSpeed
    trainer.train_policy_grpo_deepspeed(train_dataset, args.num_epochs, args.batch_size, args.lr, K=4)
    
    # Test the trained model
    trainer.test()
    
    # Save final models
    if args.local_rank in [-1, 0]:
        trainer.save_models_deepspeed("outputs/grpo_cot_deepspeed_final")

if __name__ == "__main__":
    main()
