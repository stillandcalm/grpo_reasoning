#!/usr/bin/env python3
"""
Supervised Fine-Tuning (SFT) for Chain-of-Thought Mathematical Reasoning
This should be run BEFORE GRPO training to teach the model how to reason step-by-step
UPDATED: All fixes from testing included
"""

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
import datasets
from typing import List, Dict, Optional
import re
from dataclasses import dataclass
import logging
from tqdm import tqdm
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SFTConfig:
    """Configuration for SFT training"""
    model_name: str = "meta-llama/Llama-3.1-8B"  # Updated to Llama 3.1
    output_dir: str = "./sft_math_model"
    
    # Training parameters
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    max_length: int = 1024
    
    # Data parameters
    dataset_name: str = "gsm8k"
    num_train_samples: Optional[int] = None
    add_reasoning_templates: bool = True
    
    # DeepSpeed
    use_deepspeed: bool = True
    deepspeed_config: str = "./ds_config_sft.json"


class ChainOfThoughtDataset(Dataset):
    """Dataset for chain-of-thought reasoning training"""
    
    def __init__(
        self,
        data: List[Dict],
        tokenizer,
        max_length: int = 1024,
        add_reasoning_templates: bool = True
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.add_reasoning_templates = add_reasoning_templates
        
        # Reasoning templates to add variety
        self.reasoning_templates = [
            "Let me solve this step by step.",
            "I'll work through this problem systematically.",
            "Let me break this down step by step.",
            "I'll approach this problem step by step.",
            "Let me think through this carefully.",
        ]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Create the full conversation with reasoning
        if self.add_reasoning_templates:
            template = self.reasoning_templates[idx % len(self.reasoning_templates)]
            text = self._create_reasoning_example(
                item["question"],
                item["solution"],
                item["answer"],
                template
            )
        else:
            text = self._create_basic_example(
                item["question"],
                item["solution"],
                item["answer"]
            )
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Set up labels for language modeling
        labels = encoding["input_ids"].clone()
        
        # Find where the answer starts to compute loss only on the response
        question_part = f"Problem: {item['question']}\n\nSolution:"
        question_tokens = self.tokenizer(question_part, add_special_tokens=False)["input_ids"]
        
        # Mask the question part from loss computation
        if len(question_tokens) < self.max_length:
            labels[0, :len(question_tokens)] = -100
        
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": labels.squeeze()
        }
    
    def _create_reasoning_example(
        self,
        question: str,
        solution: str,
        answer: float,
        template: str
    ) -> str:
        """Create a full reasoning example with chain-of-thought"""
        
        # Clean up the solution
        solution = solution.strip()
        
        # Ensure solution has clear steps
        if "\n" not in solution:
            # Try to add line breaks at sentence boundaries
            solution = solution.replace(". ", ".\n")
        
        # Format the complete example
        text = f"""Problem: {question}

Solution: {template}

{solution}

Therefore, the answer is {answer}."""
        
        return text
    
    def _create_basic_example(
        self,
        question: str,
        solution: str,
        answer: float
    ) -> str:
        """Create a basic example without template"""
        return f"""Problem: {question}

Solution: {solution}

Therefore, the answer is {answer}."""


class ChainOfThoughtDataProcessor:
    """Process datasets for chain-of-thought training"""
    
    def __init__(self):
        self.answer_pattern = r"####\s*([\-\d\.]+)"
    
    def process_gsm8k(self, split: str = "train", max_samples: Optional[int] = None) -> List[Dict]:
        """Process GSM8K dataset"""
        logger.info(f"Loading GSM8K {split} split...")
        
        dataset = datasets.load_dataset("gsm8k", "main", split=split)
        processed_data = []
        
        for idx, example in enumerate(tqdm(dataset, desc="Processing GSM8K")):
            if max_samples and idx >= max_samples:
                break
            
            # Extract components
            question = example["question"]
            full_answer = example["answer"]
            
            # Split solution and answer
            match = re.search(self.answer_pattern, full_answer)
            if match:
                answer = float(match.group(1))
                solution = full_answer.split("####")[0].strip()
                
                # Enhance the solution with clear step indicators
                solution = self._enhance_solution_steps(solution)
                
                processed_data.append({
                    "question": question,
                    "solution": solution,
                    "answer": answer
                })
        
        logger.info(f"Processed {len(processed_data)} examples")
        return processed_data
    
    def _enhance_solution_steps(self, solution: str) -> str:
        """Enhance solution with clear step indicators"""
        lines = solution.split('\n')
        enhanced_lines = []
        step_count = 0
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Add step indicators for lines with calculations
            if any(op in line for op in ['+', '-', '*', '/', '=']):
                step_count += 1
                if not line.startswith(("Step", "First", "Next", "Then", "Finally")):
                    enhanced_lines.append(f"Step {step_count}: {line}")
                else:
                    enhanced_lines.append(line)
            else:
                enhanced_lines.append(line)
        
        return '\n'.join(enhanced_lines)
    
    def create_diverse_reasoning_examples(self, data: List[Dict]) -> List[Dict]:
        """Create diverse reasoning examples with different styles"""
        diverse_data = []
        
        reasoning_styles = [
            {
                "prefix": "Let me solve this step by step.",
                "step_format": "Step {}: {}",
                "conclusion": "Therefore, the answer is"
            },
            {
                "prefix": "I'll break this problem down:",
                "step_format": "{}) {}",
                "conclusion": "So the final answer is"
            },
            {
                "prefix": "To solve this problem:",
                "step_format": "- {}",
                "conclusion": "Thus, we get"
            },
            {
                "prefix": "Working through this systematically:",
                "step_format": "• {}",
                "conclusion": "This gives us"
            }
        ]
        
        for idx, item in enumerate(data):
            style = reasoning_styles[idx % len(reasoning_styles)]
            
            # Reformat the solution with the chosen style
            solution_lines = item["solution"].split('\n')
            reformatted_lines = [style["prefix"]]
            
            step_num = 1
            for line in solution_lines:
                if line.strip() and any(op in line for op in ['+', '-', '*', '/', '=']):
                    if "{}" in style["step_format"]:
                        formatted_line = style["step_format"].format(step_num, line.strip())
                    else:
                        formatted_line = style["step_format"].format(line.strip())
                    reformatted_lines.append(formatted_line)
                    step_num += 1
                elif line.strip():
                    reformatted_lines.append(line.strip())
            
            # Create new solution
            new_solution = '\n'.join(reformatted_lines)
            
            diverse_data.append({
                "question": item["question"],
                "solution": new_solution,
                "answer": item["answer"],
                "style": idx % len(reasoning_styles)
            })
        
        return diverse_data


class SFTTrainer:
    """Supervised Fine-Tuning trainer for chain-of-thought reasoning"""
    
    def __init__(self, config: SFTConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize tokenizer and model
        self._setup_model()
        
        # Process data
        self._setup_data()
    
    def _setup_model(self):
        """Initialize model and tokenizer"""
        logger.info(f"Loading model: {self.config.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model WITHOUT device_map for DeepSpeed compatibility
        if self.config.use_deepspeed:
            # For DeepSpeed, load model without device_map
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                # Don't use device_map with DeepSpeed
            )
            # Move model to GPU for DeepSpeed
            if torch.cuda.is_available():
                self.model = self.model.to(self.device)
        else:
            # For non-DeepSpeed, can use device_map
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map="auto"
            )
        
        logger.info(f"Model loaded on {self.device}")
    
    def _setup_data(self):
        """Setup training data"""
        processor = ChainOfThoughtDataProcessor()
        
        # Load and process data
        train_data = processor.process_gsm8k("train", self.config.num_train_samples)
        eval_data = processor.process_gsm8k("test", max_samples=500)
        
        # Create diverse examples if requested
        if self.config.add_reasoning_templates:
            train_data = processor.create_diverse_reasoning_examples(train_data)
        
        # Create datasets
        self.train_dataset = ChainOfThoughtDataset(
            train_data,
            self.tokenizer,
            self.config.max_length,
            self.config.add_reasoning_templates
        )
        
        self.eval_dataset = ChainOfThoughtDataset(
            eval_data,
            self.tokenizer,
            self.config.max_length,
            False  # Don't add templates to eval
        )
        
        logger.info(f"Train dataset: {len(self.train_dataset)} examples")
        logger.info(f"Eval dataset: {len(self.eval_dataset)} examples")
    
    def save_eval_metrics(self, metrics: Dict[str, float], step: int):
        """Save evaluation metrics to JSON file for monitoring"""
        # Save to checkpoint directory
        checkpoint_dir = os.path.join(self.config.output_dir, f"checkpoint_step_{step}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        metrics_file = os.path.join(checkpoint_dir, "eval_metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump({
                "step": step,
                "timestamp": datetime.now().isoformat(),
                **metrics
            }, f, indent=2)
    
    def train(self):
        """Run supervised fine-tuning"""
        
        # Setup training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_ratio=self.config.warmup_ratio,
            logging_steps=50,
            save_steps=500,
            eval_steps=500,
            eval_strategy="steps",  # Fixed from evaluation_strategy
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            bf16=torch.cuda.is_available(),
            tf32=torch.cuda.is_available(),
            dataloader_num_workers=4,
            remove_unused_columns=False,
            deepspeed=self.config.deepspeed_config if self.config.use_deepspeed else None,
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            processing_class=self.tokenizer,  # Fixed from tokenizer
            data_collator=DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False,
                pad_to_multiple_of=8
            )
        )
        
        # Train
        logger.info("Starting supervised fine-tuning...")
        trainer.train()
        
        # Save final model
        logger.info(f"Saving final model to {self.config.output_dir}")
        trainer.save_model()
        self.tokenizer.save_pretrained(self.config.output_dir)
    
    def test_generation(self, num_examples: int = 3):
        """Test the model's generation capabilities"""
        logger.info("Testing model generation...")
        
        test_problems = [
            "Sarah has 24 cookies. She wants to share them equally among her 6 friends. How many cookies will each friend get?",
            "A store sells apples for $2 each. If John buys 7 apples and pays with a $20 bill, how much change will he receive?",
            "There are 45 students in a class. If 3/5 of them are girls, how many boys are in the class?"
        ]
        
        self.model.eval()
        for problem in test_problems[:num_examples]:
            prompt = f"Problem: {problem}\n\nSolution: Let me solve this step by step.\n\n"
            
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"\n{'='*50}")
            print(generated)


def create_sft_deepspeed_config():
    """Create DeepSpeed config for SFT"""
    config = {
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
                "pin_memory": True
            },
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 2e8,
            "contiguous_gradients": True
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
        
        "wall_clock_breakdown": False
    }
    
    with open("ds_config_sft.json", "w") as f:
        json.dump(config, f, indent=2)
    
    return "ds_config_sft.json"


def main():
    """Main training function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="SFT for Chain-of-Thought Reasoning")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--output_dir", type=str, default="./sft_math_model")
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--num_train_samples", type=int, default=None)
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")  # ADDED
    args = parser.parse_args()
    
    # Create config
    config = SFTConfig(
        model_name=args.model,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_train_samples=args.num_train_samples
    )
    
    # Create DeepSpeed config
    config.deepspeed_config = create_sft_deepspeed_config()
    
    # Train
    trainer = SFTTrainer(config)
    trainer.train()
    
    # Test generation
    trainer.test_generation()


if __name__ == "__main__":
    main()
