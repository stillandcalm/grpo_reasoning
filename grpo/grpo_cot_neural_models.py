#!/usr/bin/env python3
# grpo_cot_neural_models.py - GRPO with Chain-of-Thought reasoning support

import os
os.environ['WANDB_MODE'] = 'offline'

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
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
            'labels': torch.tensor(item['score'], dtype=torch.float)
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
                # Store the CoT trace for training
                reference_cot = cot_trace
            else:
                # Standard format for non-CoT data
                prompt = (
                    "Solve the following math problem step by step.\n\n"
                    f"Problem: {item['question']}\n\n"
                    "Solution: I'll solve this step by step.\n\n"
                )
                reference_cot = ""  # Changed from None to empty string
            
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
                # Handle different answer formats
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
                'reference_cot': reference_cot,  # Now always a string
                'has_cot': has_cot
            })
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def custom_collate_fn(batch):
    """Custom collate function to handle our data format"""
    # Initialize lists for each field
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
    
    # Collect all items
    for item in batch:
        for key in collated:
            collated[key].append(item[key])
    
    # Stack tensors
    collated['prompt_ids'] = torch.stack(collated['prompt_ids'])
    collated['prompt_mask'] = torch.stack(collated['prompt_mask'])
    
    return collated

class CoTAwareRewardModel(nn.Module):
    """Reward model that values Chain-of-Thought reasoning"""
    def __init__(self, model_name='microsoft/deberta-v3-base', dropout=0.1):
        super().__init__()
        
        # Load pretrained model
        config = AutoConfig.from_pretrained(model_name)
        config.num_labels = 1
        config.problem_type = "regression"
        
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            config=config,
            ignore_mismatched_sizes=True
        )
        
        # Custom head for reward prediction
        hidden_size = config.hidden_size
        self.reward_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # Additional head for reasoning quality
        self.reasoning_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 4),
            nn.GELU(),
            nn.Linear(hidden_size // 4, 1)
        )
        
    def forward(self, input_ids, attention_mask):
        # Get model outputs
        outputs = self.model.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Pool the outputs
        pooled_output = outputs[0][:, 0, :]
        
        # Get reward components
        answer_reward = self.reward_head(pooled_output).squeeze(-1)
        reasoning_reward = self.reasoning_head(pooled_output).squeeze(-1)
        
        # Combine rewards (answer correctness + reasoning quality)
        total_reward = 0.7 * answer_reward + 0.3 * reasoning_reward
        
        # Bound between -3 and 3
        total_reward = 3.0 * torch.tanh(total_reward / 3.0)
        
        return total_reward

class GRPOTrainerWithCoT:
    def __init__(
        self, 
        policy_model_name='EleutherAI/gpt-neo-125m',
        reward_model_name='microsoft/deberta-v3-base',
        device='cuda'
    ):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load policy model
        print(f"Loading policy model: {policy_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(policy_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'
        
        self.policy_model = AutoModelForCausalLM.from_pretrained(
            policy_model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        ).to(self.device)
        
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            policy_model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        ).to(self.device).eval()
        
        for param in self.ref_model.parameters():
            param.requires_grad = False
        
        # Load CoT-aware reward model
        print(f"Loading CoT-aware reward model: {reward_model_name}")
        self.reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_name)
        self.reward_model = CoTAwareRewardModel(reward_model_name).to(self.device)
        
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
        
        # Convert dataset slice to list
        dataset_items = [dataset[i] for i in range(min(50, len(dataset)))]
        
        for item in tqdm(dataset_items, desc="Creating reward data"):
            question = item['question']
            
            # Check if we have CoT data
            has_cot = 'rationale' in item or 'chain_of_thought' in item
            
            if has_cot:
                # High-quality CoT example
                cot_trace = item.get('rationale', item.get('chain_of_thought', ''))
                prompt = f"Problem: {question}\n\nLet me think through this step by step:\n"
                
                reward_data.append({
                    'text': f"{prompt}{cot_trace}",
                    'score': 3.0  # Highest score for real CoT
                })
                
                # Medium quality - partial CoT
                partial_cot = '. '.join(cot_trace.split('.')[:2]) + '...'
                reward_data.append({
                    'text': f"{prompt}{partial_cot}\n\nThe answer is probably {np.random.randint(1, 100)}.",
                    'score': 1.0
                })
            else:
                # For non-CoT data, create synthetic examples
                prompt = f"Problem: {question}\n\nSolution: "
                
                # Parse answer if available
                if 'answer' in item and '####' in item['answer']:
                    parts = item['answer'].split('####')
                    solution = parts[0].strip()
                    answer = parts[1].strip()
                    
                    # Good solution with steps
                    reward_data.append({
                        'text': f"{prompt}Let me work through this step by step.\n{solution}\nTherefore, the answer is {answer}.",
                        'score': 2.5
                    })
            
            # Poor examples (same for all data types)
            prompt_base = f"Problem: {question}\n\nSolution: "
            
            # No reasoning
            reward_data.append({
                'text': f"{prompt_base}The answer is {np.random.randint(1, 100)}.",
                'score': -1.0
            })
            
            # Confused response
            reward_data.append({
                'text': f"{prompt_base}I'm not sure how to solve this.",
                'score': -2.0
            })
            
            # Some effort but poor reasoning
            reward_data.append({
                'text': f"{prompt_base}This looks like a math problem. Let me guess... maybe {np.random.randint(1, 100)}?",
                'score': -0.5
            })
        
        return reward_data
    
    def train_reward_model(self, reward_data, num_epochs=3, batch_size=16, lr=2e-5):
        """Train the CoT-aware neural reward model"""
        print(f"\nTraining CoT-aware reward model on {len(reward_data)} examples...")
        
        # Create dataset and dataloader
        reward_dataset = RewardDataset(reward_data, self.reward_tokenizer)
        dataloader = DataLoader(reward_dataset, batch_size=batch_size, shuffle=True)
        
        # Optimizer
        optimizer = torch.optim.AdamW(self.reward_model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=num_epochs * len(dataloader)
        )
        
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
                
                # MSE loss
                loss = F.mse_loss(predicted_rewards, labels)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.reward_model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                total_loss += loss.item()
                progress.set_postfix({'loss': f"{loss.item():.4f}"})
            
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1} - Average loss: {avg_loss:.4f}")
        
        print("CoT-aware reward model training completed!")
    
    def get_reward_score(self, text):
        """Get reward score with bonus for good CoT reasoning"""
        self.reward_model.eval()
        
        # Get neural model score
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
        
        # Extract CoT features for additional bonus
        cot_features = self.compute_cot_features(text)
        
        # Bonus for good reasoning patterns
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
        
        # Combine neural reward with CoT bonus
        total_reward = base_reward + min(cot_bonus, 1.0)
        
        return np.clip(total_reward, -3.0, 3.0)
    
    def generate_response(self, prompt_ids, prompt_mask, temperature=0.8):
        """Generate response from policy model"""
        actual_length = prompt_mask.sum().item()
        prompt_ids = prompt_ids[:, :actual_length]
        prompt_mask = prompt_mask[:, :actual_length]
        
        with torch.no_grad():
            outputs = self.policy_model.generate(
                input_ids=prompt_ids,
                attention_mask=prompt_mask,
                max_new_tokens=200,  # Longer for CoT
                min_new_tokens=50,    # Ensure some reasoning
                temperature=temperature,
                do_sample=True,
                top_k=50,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
        return outputs
    
    def train_policy_grpo(self, train_dataset, num_epochs=3, batch_size=2, lr=5e-6, K=4):
        """
        True GRPO implementation with CoT awareness
        """
        dataloader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            collate_fn=custom_collate_fn  # Use custom collate function
        )
        optimizer = torch.optim.AdamW(self.policy_model.parameters(), lr=lr)
        
        print(f"\nStarting GRPO training with CoT-aware rewards...")
        print(f"Dataset size: {len(train_dataset)}")
        print(f"Batch size: {batch_size}")
        print(f"K (responses per prompt): {K}")
        print(f"Learning rate: {lr}")
        
        self.policy_model.train()
        best_avg_reward = -float('inf')
        
        for epoch in range(num_epochs):
            total_loss = 0
            total_reward = 0
            num_updates = 0
            positive_rewards = 0
            high_quality_cot = 0
            
            progress = tqdm(dataloader, desc=f"GRPO Epoch {epoch+1}/{num_epochs}")
            
            for batch_idx, batch in enumerate(progress):
                optimizer.zero_grad()
                
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
                        # Vary temperature for diversity
                        temp = 0.6 + (k * 0.2)
                        response = self.generate_response(prompt_ids, prompt_mask, temperature=temp)
                        
                        # Decode and get reward
                        full_text = self.tokenizer.decode(response[0], skip_special_tokens=True)
                        completion = full_text.replace(prompt_text, '')
                        
                        # Get CoT-aware reward
                        reward = self.get_reward_score(full_text)
                        
                        prompt_responses.append(response)
                        prompt_rewards.append(reward)
                        prompt_texts.append(completion)
                        
                        if reward > 1.0:
                            positive_rewards += 1
                        if reward > 2.0:
                            high_quality_cot += 1
                        
                        # Show examples from first batch
                        if epoch == 0 and batch_idx == 0 and i == 0 and k < 2:
                            print(f"\n[Response {k+1}, reward={reward:.2f}]")
                            print(f"Question: {batch['question'][i]}")
                            print(f"Completion: {completion[:150]}...")
                            cot_features = self.compute_cot_features(completion)
                            print(f"CoT features: steps={cot_features['has_steps']}, "
                                  f"calc={cot_features['has_calculations']}, "
                                  f"reasoning={cot_features['has_reasoning_words']}")
                    
                    # Add ALL K responses for TRUE GRPO
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
                    outputs = self.policy_model(response, labels=response)
                    
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
                    total_grpo_loss += loss
                    num_responses += 1
                
                if num_responses > 0:
                    # Average and backward
                    avg_loss = total_grpo_loss / num_responses
                    avg_loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), 1.0)
                    
                    optimizer.step()
                    total_loss += avg_loss.item()
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
            print(f"Epoch {epoch+1} - Loss: {total_loss/max(num_updates,1):.4f}, "
                  f"Avg reward: {epoch_avg_reward:.2f}, "
                  f"High-quality CoT: {high_quality_cot}/{K*len(train_dataset)}")
            
            # Save best model
            if epoch_avg_reward > best_avg_reward:
                best_avg_reward = epoch_avg_reward
                self.save_models("outputs/grpo_cot_best")
                print(f"Saved best model with avg reward: {best_avg_reward:.2f}")
    
    def save_models(self, output_dir):
        """Save both policy and reward models"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save policy model
        self.policy_model.save_pretrained(f"{output_dir}/policy")
        self.tokenizer.save_pretrained(f"{output_dir}/policy")
        
        # Save reward model
        torch.save(self.reward_model.state_dict(), f"{output_dir}/reward_model.pt")
        
        # Save config
        config = {
            'model_type': 'grpo_cot',
            'policy_model': self.policy_model.config._name_or_path,
            'reward_model': 'cot_aware_deberta'
        }
        with open(f"{output_dir}/config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Models saved to {output_dir}")
    
    def test(self):
        """Test the trained policy model"""
        print("\nTesting CoT-trained policy model...")
        self.policy_model.eval()
        
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
                outputs = self.policy_model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.7,
                    do_sample=True,
                    top_k=50
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            solution = response.replace(prompt, '')
            
            # Get reward and CoT features
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
    
    if dataset_name == "gsm8k":
        # Standard GSM8K (has some reasoning in answers)
        dataset = load_dataset("gsm8k", "main", split=f"{split}[:{max_examples}]")
    
    elif dataset_name == "aqua_rat":
        # AQuA-RAT has rationales
        dataset = load_dataset("aqua_rat", split=f"{split}[:{max_examples}]")
    
    elif dataset_name == "math_qa":
        # MathQA includes reasoning chains
        dataset = load_dataset("math_qa", split=f"{split}[:{max_examples}]")
        
    elif dataset_name == "cot_collection":
        # If available, use dedicated CoT collections
        try:
            dataset = load_dataset("kaist-ai/CoT-Collection", split=f"{split}[:{max_examples}]")
        except:
            print("CoT Collection not available, falling back to GSM8K")
            dataset = load_dataset("gsm8k", "main", split=f"{split}[:{max_examples}]")
    
    else:
        print(f"Unknown dataset {dataset_name}, using GSM8K")
        dataset = load_dataset("gsm8k", "main", split=f"{split}[:{max_examples}]")
    
    return dataset

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
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--use_cot', action='store_true', default=True,
                       help='Use CoT-aware training')
    args = parser.parse_args()
    
    # Load appropriate dataset
    dataset = load_cot_dataset(args.dataset, max_examples=args.num_examples)
    
    # Initialize CoT-aware trainer
    trainer = GRPOTrainerWithCoT(
        policy_model_name=args.policy_model,
        reward_model_name=args.reward_model,
        device=args.device
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
    trainer.train_reward_model(reward_data, num_epochs=3)
    
    # Train policy with GRPO
    trainer.train_policy_grpo(train_dataset, args.num_epochs, args.batch_size, args.lr, K=4)
    
    # Test the trained model
    trainer.test()
    
    # Save final models
    trainer.save_models("outputs/grpo_cot_final")

if __name__ == "__main__":
    main()
