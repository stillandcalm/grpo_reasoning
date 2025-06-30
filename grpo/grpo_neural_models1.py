#!/usr/bin/env python3
# grpo_neural_models.py - GRPO with proper neural reward and policy models

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
    """Dataset for GRPO training"""
    def __init__(self, data, tokenizer, max_length=256):
        self.data = []
        self.tokenizer = tokenizer
        
        # Handle both list and dataset formats
        if hasattr(data, '__len__'):
            items = data
        else:
            items = [data[i] for i in range(len(data))]
        
        for item in items:
            # Create a clear prompt for math problems
            prompt = (
                "Solve the following math problem step by step.\n\n"
                f"Problem: {item['question']}\n\n"
                "Solution: I'll solve this step by step.\n\n"
            )
            
            inputs = tokenizer(
                prompt,
                truncation=True,
                max_length=max_length,
                padding='max_length',
                return_tensors='pt'
            )
            
            # Extract numerical answer
            full_answer = item['answer']
            parts = full_answer.split('####')
            answer = parts[1].strip().replace(',', '') if len(parts) > 1 else parts[0].strip().replace(',', '')
            
            self.data.append({
                'prompt': prompt,
                'prompt_ids': inputs['input_ids'].squeeze(),
                'prompt_mask': inputs['attention_mask'].squeeze(),
                'answer': answer,
                'question': item['question'],
                'full_solution': parts[0].strip() if len(parts) > 0 else ""
            })
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

class NeuralRewardModel(nn.Module):
    """Reward model based on a pretrained classifier"""
    def __init__(self, model_name='microsoft/deberta-v3-base', dropout=0.1):
        super().__init__()
        
        # Load pretrained model
        config = AutoConfig.from_pretrained(model_name)
        config.num_labels = 1  # Regression task
        config.problem_type = "regression"
        
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            config=config,
            ignore_mismatched_sizes=True
        )
        
        # Add custom head for better reward prediction
        hidden_size = config.hidden_size
        self.reward_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )
        
    def forward(self, input_ids, attention_mask):
        # Get model outputs
        outputs = self.model.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Pool the outputs (use [CLS] token)
        pooled_output = outputs[0][:, 0, :]
        
        # Get reward score
        reward = self.reward_head(pooled_output).squeeze(-1)
        
        # Bound rewards between -3 and 3 using tanh
        reward = 3.0 * torch.tanh(reward / 3.0)
        
        return reward

class GRPOTrainerWithNeuralModels:
    def __init__(
        self, 
        policy_model_name='EleutherAI/gpt-neo-125m',  # Better than GPT-2 for math
        reward_model_name='microsoft/deberta-v3-base',  # Good for understanding text
        device='cuda'
    ):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load policy model and tokenizer
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
        
        # Load reward model and tokenizer
        print(f"Loading reward model: {reward_model_name}")
        self.reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_name)
        self.reward_model = NeuralRewardModel(reward_model_name).to(self.device)
        
        print(f"Models loaded successfully!")
    
    def generate_reward_training_data(self, dataset, num_samples_per_problem=5):
        """Generate training data for the reward model"""
        print("\nGenerating reward model training data...")
        reward_data = []
        
        # Convert dataset slice to list
        dataset_items = [dataset[i] for i in range(min(50, len(dataset)))]
        
        for item in tqdm(dataset_items, desc="Creating reward data"):  # Use first 50 for reward training
            prompt = (
                "Solve the following math problem step by step.\n\n"
                f"Problem: {item['question']}\n\n"
                "Solution: I'll solve this step by step.\n\n"
            )
            
            # GSM8K format: answer contains both solution and final answer separated by ####
            full_answer = item['answer']
            parts = full_answer.split('####')
            solution_steps = parts[0].strip() if len(parts) > 0 else ""
            correct_answer = parts[1].strip().replace(',', '') if len(parts) > 1 else ""
            
            # High-quality correct solution
            reward_data.append({
                'text': f"{prompt}{solution_steps}\n\nTherefore, the answer is {correct_answer}.",
                'score': 3.0
            })
            
            # Good attempt with step-by-step but wrong answer
            if solution_steps:
                steps = solution_steps.split('.')[:2]  # Take first few steps
                wrong_answer = str(int(float(correct_answer)) + np.random.randint(-10, 10)) if correct_answer.isdigit() else "42"
                if wrong_answer != correct_answer:
                    reward_data.append({
                        'text': f"{prompt}{'. '.join(steps)}.\n\nSo the answer is {wrong_answer}.",
                        'score': 0.5
                    })
            
            # Poor attempt - just wrong answer
            random_answer = str(np.random.randint(0, 100))
            reward_data.append({
                'text': f"{prompt}The answer is {random_answer}.",
                'score': -1.0
            })
            
            # Very poor - no math
            reward_data.append({
                'text': f"{prompt}I don't know how to solve this.",
                'score': -2.0
            })
            
            # Medium quality - shows some work
            reward_data.append({
                'text': f"{prompt}Let me think... This involves some calculations... The answer might be around {np.random.randint(0, 100)}.",
                'score': -0.5
            })
        
        return reward_data
    
    def train_reward_model(self, reward_data, num_epochs=3, batch_size=16, lr=2e-5):
        """Train the neural reward model"""
        print(f"\nTraining reward model on {len(reward_data)} examples...")
        
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
        
        print("Reward model training completed!")
    
    def get_reward_score(self, text):
        """Get reward score from the neural reward model"""
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
        
        return reward.item()
    
    def generate_response(self, prompt_ids, prompt_mask, temperature=0.8):
        """Generate response from policy model"""
        actual_length = prompt_mask.sum().item()
        prompt_ids = prompt_ids[:, :actual_length]
        prompt_mask = prompt_mask[:, :actual_length]
        
        with torch.no_grad():
            outputs = self.policy_model.generate(
                input_ids=prompt_ids,
                attention_mask=prompt_mask,
                max_new_tokens=150,  # Allow longer for step-by-step
                min_new_tokens=30,
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
        True GRPO implementation that uses all K responses, not just the best.
        
        GRPO formula: ∇J = E[∑(r_i - b) * ∇log π(y_i|x)]
        where b is baseline (mean reward)
        """
        dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.AdamW(self.policy_model.parameters(), lr=lr)
        
        print(f"\nStarting TRUE GRPO training with neural reward model...")
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
            
            progress = tqdm(dataloader, desc=f"GRPO Epoch {epoch+1}/{num_epochs}")
            
            for batch_idx, batch in enumerate(progress):
                optimizer.zero_grad()
                
                # Collect ALL responses and rewards for TRUE GRPO
                all_responses = []
                all_rewards = []
                all_prompts = []
                
                for i in range(min(batch_size, len(batch['prompt']))):
                    prompt_ids = batch['prompt_ids'][i:i+1].to(self.device)
                    prompt_mask = batch['prompt_mask'][i:i+1].to(self.device)
                    prompt_text = batch['prompt'][i]
                    
                    # Generate K responses for this prompt
                    prompt_responses = []
                    prompt_rewards = []
                    
                    for k in range(K):
                        # Vary temperature for diversity
                        temp = 0.6 + (k * 0.2)  # 0.6, 0.8, 1.0, 1.2 for K=4
                        response = self.generate_response(prompt_ids, prompt_mask, temperature=temp)
                        
                        # Decode and get reward
                        full_text = self.tokenizer.decode(response[0], skip_special_tokens=True)
                        completion = full_text.replace(prompt_text, '')
                        
                        # Get reward from neural model
                        reward = self.get_reward_score(full_text)
                        
                        prompt_responses.append(response)
                        prompt_rewards.append(reward)
                        
                        if reward > 1.0:
                            positive_rewards += 1
                        
                        # Debug: show examples from first batch
                        if epoch == 0 and batch_idx == 0 and i == 0:
                            print(f"\n[Response {k+1}, reward={reward:.2f}]")
                            print(f"Completion preview: {completion[:100]}...")
                    
                    # Add ALL K responses (not just best!) - this is TRUE GRPO
                    all_responses.extend(prompt_responses)
                    all_rewards.extend(prompt_rewards)
                    all_prompts.extend([prompt_text] * K)
                    total_reward += sum(prompt_rewards)
                
                # Skip if all rewards are terrible
                if max(all_rewards) < -1.5:
                    continue
                
                # Compute baseline (mean reward) - key component of GRPO
                baseline = np.mean(all_rewards)
                
                # TRUE GRPO: Update using ALL responses weighted by advantage
                total_grpo_loss = 0
                num_responses = 0
                
                for response, reward in zip(all_responses, all_rewards):
                    # Compute advantage (reward - baseline)
                    advantage = reward - baseline
                    
                    # Forward pass
                    outputs = self.policy_model(response, labels=response)
                    
                    # GRPO loss: -advantage * log_prob
                    # Positive advantage -> increase log prob
                    # Negative advantage -> decrease log prob
                    grpo_loss = -advantage * outputs.loss
                    
                    # Optional: Add KL penalty for stability
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
                    # Average loss over all K*batch_size responses
                    avg_loss = total_grpo_loss / num_responses
                    avg_loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), 1.0)
                    
                    optimizer.step()
                    total_loss += avg_loss.item()
                    num_updates += 1
                
                # Update progress bar
                if num_updates > 0:
                    avg_loss = total_loss / num_updates
                    avg_reward = total_reward / (K * (batch_idx + 1) * batch_size)
                    progress.set_postfix({
                        'loss': f"{avg_loss:.3f}",
                        'avg_reward': f"{avg_reward:.2f}",
                        'baseline': f"{baseline:.2f}",
                        'positive': positive_rewards
                    })
            
            # Epoch summary
            epoch_avg_reward = total_reward / (K * len(train_dataset))
            print(f"Epoch {epoch+1} - Loss: {total_loss/max(num_updates,1):.4f}, "
                  f"Avg reward: {epoch_avg_reward:.2f}, Positive rewards: {positive_rewards}/{K*len(train_dataset)}")
            
            # Save best model
            if epoch_avg_reward > best_avg_reward:
                best_avg_reward = epoch_avg_reward
                self.save_models("outputs/grpo_neural_best")
                print(f"Saved best model with avg reward: {best_avg_reward:.2f}")
    
    def save_models(self, output_dir):
        """Save both policy and reward models"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save policy model
        self.policy_model.save_pretrained(f"{output_dir}/policy")
        self.tokenizer.save_pretrained(f"{output_dir}/policy")
        
        # Save reward model
        torch.save(self.reward_model.state_dict(), f"{output_dir}/reward_model.pt")
        
        print(f"Models saved to {output_dir}")
    
    def test(self):
        """Test the trained policy model"""
        print("\nTesting trained policy model...")
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
                "Solve the following math problem step by step.\n\n"
                f"Problem: {problem}\n\n"
                "Solution: I'll solve this step by step.\n\n"
            )
            
            inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
            
            with torch.no_grad():
                outputs = self.policy_model.generate(
                    **inputs,
                    max_new_tokens=150,
                    temperature=0.7,
                    do_sample=True,
                    top_k=50
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            solution = response.replace(prompt, '')
            
            # Get reward score
            reward = self.get_reward_score(response)
            
            print(f"\n{'='*60}")
            print(f"Problem: {problem}")
            print(f"Solution: {solution[:200]}...")
            print(f"Reward score: {reward:.2f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy_model', default='EleutherAI/gpt-neo-125m',
                       choices=['gpt2', 'gpt2-medium', 'EleutherAI/gpt-neo-125m', 'EleutherAI/gpt-neo-1.3B'])
    parser.add_argument('--reward_model', default='microsoft/deberta-v3-base',
                       choices=['microsoft/deberta-v3-base', 'roberta-base', 'albert-base-v2'])
    parser.add_argument('--num_examples', type=int, default=100)
    parser.add_argument('--num_epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=5e-6)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    
    # Load dataset
    print(f"Loading GSM8K dataset ({args.num_examples} examples)...")
    dataset = load_dataset("gsm8k", "main", split=f"train[:{args.num_examples}]")
    
    # Initialize trainer
    trainer = GRPOTrainerWithNeuralModels(
        policy_model_name=args.policy_model,
        reward_model_name=args.reward_model,
        device=args.device
    )
    
    # Create policy dataset from the latter part of the dataset
    tokenizer = trainer.tokenizer
    if args.num_examples > 50:
        # Use examples after the first 50 for policy training
        policy_data = [dataset[i] for i in range(50, args.num_examples)]
    else:
        # If less than 50 examples, use all for both
        policy_data = [dataset[i] for i in range(args.num_examples)]
    
    train_dataset = PolicyDataset(policy_data, tokenizer)
    
    # Generate reward training data and train reward model
    reward_data = trainer.generate_reward_training_data(dataset)
    trainer.train_reward_model(reward_data, num_epochs=3)
    
    # Train policy with GRPO
    trainer.train_policy_grpo(train_dataset, args.num_epochs, args.batch_size, args.lr, K=4)
    
    # Test the trained model
    trainer.test()
    
    # Save final models
    trainer.save_models("outputs/grpo_neural_final")

if __name__ == "__main__":
    main()
