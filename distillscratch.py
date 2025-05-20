#!/usr/bin/env python
"""
This script implements training a causal language model from scratch:
1) Configures a model architecture similar to Llama-3 with customizable parameters
2) Loads a Python code dataset for training
3) Trains the model using Hugging Face's Trainer
4) Saves the final trained model

Usage:
    python trainscratch.py --hidden_size 768 --num_layers 12 --num_heads 12 ...
"""

import os
os.environ['HF_HOME'] = os.environ.get('SCRATCH') + '/iaifi_lab/Lab/ericjm/.cache/huggingface'
import argparse
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import (
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
)
from pathlib import Path

class DistillationTrainer(Trainer):
    """
    Custom Trainer for distillation training.
    Computes the Hinton-style distillation loss between teacher and student outputs,
    and logs both the teacher's and student's cross-entropy (CE) losses on the data,
    along with their exponentially smoothed values.
    """
    def __init__(self, teacher_model: nn.Module, temperature: float, alpha: float = 0.5, ce_loss_alpha: float = 0.9, *args, **kwargs):
        """
        Args:
            teacher_model: The pretrained teacher model.
            temperature: Temperature for distillation softening.
            alpha: Weight for hard-label loss (1-alpha is weight for distillation loss).
            ce_loss_alpha: Smoothing factor (EMA) for the CE losses.
        """
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.temperature = temperature
        self.alpha = alpha  # Weight for hard-label loss
        self.ce_loss_alpha = ce_loss_alpha
        self.teacher_loss_smoothed = None
        self.student_loss_smoothed = None
        self.teacher_model.eval()  # Ensure teacher remains in eval mode
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Compute teacher outputs and cross-entropy loss without tracking gradients.
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)
        teacher_loss = teacher_outputs.loss
        teacher_logits = teacher_outputs.logits

        # Compute student outputs and cross-entropy loss.
        student_outputs = model(**inputs)
        student_loss = student_outputs.loss  # Hard-label cross-entropy loss
        student_logits = student_outputs.logits

        # Compute the distillation loss using softened probabilities.
        T = self.temperature
        
        labels = inputs["labels"]
        valid_mask = (labels != -100)
        # teacher_logits, student_logits are [batch_size, seq_len, vocab_size]
        # valid_mask is [batch_size, seq_len]
        # We need to "flatten" them consistently before applying KL:
        teacher_probs = torch.softmax(teacher_logits / T, dim=-1)
        student_log_probs = torch.log_softmax(student_logits / T, dim=-1)
        # Reshape to [batch_size * seq_len, vocab_size] 
        # then index only the valid tokens
        teacher_probs = teacher_probs.view(-1, teacher_probs.size(-1))[valid_mask.view(-1)]
        student_log_probs = student_log_probs.view(-1, student_log_probs.size(-1))[valid_mask.view(-1)]
        loss_fn = nn.KLDivLoss(reduction="batchmean")
        distill_loss = loss_fn(student_log_probs, teacher_probs) * (T * T)
        combined_loss = self.alpha * student_loss + (1 - self.alpha) * distill_loss

        # Update exponentially smoothed teacher and student losses.
        if self.teacher_loss_smoothed is None:
            self.teacher_loss_smoothed = teacher_loss.item()
        else:
            self.teacher_loss_smoothed = (
                self.ce_loss_alpha * self.teacher_loss_smoothed
                + (1 - self.ce_loss_alpha) * teacher_loss.item()
            )
        if self.student_loss_smoothed is None:
            self.student_loss_smoothed = student_loss.item()
        else:
            self.student_loss_smoothed = (
                self.ce_loss_alpha * self.student_loss_smoothed
                + (1 - self.ce_loss_alpha) * student_loss.item()
            )

        # Log losses every logging_steps.
        if self.state.global_step % self.args.logging_steps == 0:
            self.log({
                "teacher_ce_loss": teacher_loss.item(),
                "teacher_ce_loss_smoothed": self.teacher_loss_smoothed,
                "student_ce_loss": student_loss.item(),
                "student_ce_loss_smoothed": self.student_loss_smoothed,
                "distill_loss": distill_loss.item(),
                "combined_loss": combined_loss.item(),
            })

        return (combined_loss, student_outputs) if return_outputs else combined_loss

def parse_args():
    parser = argparse.ArgumentParser(description="Train a language model from scratch")
    # Model architecture parameters
    parser.add_argument("--teacher_model_name", type=str, default="NousResearch/Meta-Llama-3.1-8B",
                        help="Pretrained teacher model name or path.")
    parser.add_argument("--hidden_size", type=int, default=768, 
                        help="Hidden size of the model")
    parser.add_argument("--num_layers", type=int, default=12, 
                        help="Number of transformer layers")
    parser.add_argument("--num_heads", type=int, default=12, 
                        help="Number of attention heads")
    parser.add_argument("--num_key_value_heads", type=int, default=None,
                        help="Number of key/value heads for GQA (defaults to num_heads if not set)")
    parser.add_argument("--intermediate_size", type=int, default=3072, 
                        help="Size of the intermediate layer in MLP")
    
    # Distillation parameters
    parser.add_argument("--temperature", type=float, default=2.0,
                        help="Temperature for distillation softening.")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Weight for hard-label loss (1-alpha is weight for distillation loss).")
    parser.add_argument("--smoothing_alpha", type=float, default=0.9995,
                        help="Exponential smoothing factor for loss tracking.")
    
    # Training parameters
    parser.add_argument("--output_dir", type=str, default="./model_from_scratch",
                        help="Directory to store model checkpoints and logs")
    parser.add_argument("--lr", type=float, default=5e-4,
                        help="Learning rate for the optimizer")
    parser.add_argument("--max_steps", type=int, default=100000,
                        help="Total number of training steps to run")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum sequence length for tokenization")
    parser.add_argument("--num_train_epochs", type=int, default=3,
                        help="Number of epochs to train (if not using max_steps)")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Per-device batch size")
    parser.add_argument("--accumulations", type=int, default=4,
                        help="Number of gradient accumulation steps")
    parser.add_argument("--eval_steps", type=int, default=1000,
                        help="Evaluate every N steps")
    parser.add_argument("--logging_steps", type=int, default=100,
                        help="Log every N steps")
    parser.add_argument("--save_steps", type=int, default=5000,
                        help="Save checkpoint every N steps")
    parser.add_argument("--limit_checkpoints", type=int, default=3,
                        help="Limit the number of checkpoints saved. Set to -1 for unlimited")
    parser.add_argument("--use_streaming", action="store_true",
                        help="Use a streaming dataset if set")
    
    return parser.parse_args()

def prepare_dataset(dataset, tokenizer, max_length):
    """
    Tokenize the dataset for causal language modeling
    """
    def tokenize_function(examples):
        return tokenizer(
            examples["code"],
            truncation=True,
            max_length=max_length,
        )

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
    )
    return tokenized_dataset

def main():
    args = parse_args()
    print("Arguments:", args)

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset (Python code)
    if args.use_streaming:
        dataset = load_dataset("codeparrot/github-code",
                              streaming=True,
                              languages=["Python"],
                              split="train",
                              trust_remote_code=True)
        train_dataset = dataset
        val_dataset = None
    else:
        dataset = load_dataset("codeparrot/github-code",
                              languages=["Python"],
                              split="train[:5%]",  # Using a small subset for demonstration
                              trust_remote_code=True)
        train_size = int(0.9 * len(dataset))
        train_dataset = dataset.select(range(train_size))
        val_dataset = dataset.select(range(train_size, len(dataset)))

    # Load tokenizer (using an existing one for simplicity)
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load teacher and student models with bf16 precision to match training args
    teacher_model = AutoModelForCausalLM.from_pretrained(
        args.teacher_model_name,
        torch_dtype=torch.bfloat16,  # Changed from float32 to bfloat16
        device_map="auto"
    )
    teacher_model.eval()
        
    # Tokenize datasets
    tokenized_train = prepare_dataset(train_dataset, tokenizer, args.max_length)
    tokenized_val = prepare_dataset(val_dataset, tokenizer, args.max_length) if val_dataset else None

    # Configure model from scratch
    config = AutoConfig.from_pretrained("NousResearch/Meta-Llama-3.1-8B")  # as a starting point
    config.vocab_size = len(tokenizer)
    config.hidden_size = args.hidden_size 
    config.num_hidden_layers = args.num_layers
    config.intermediate_size = args.intermediate_size
    config.num_attention_heads = args.num_heads
    config.num_key_value_heads = args.num_key_value_heads if args.num_key_value_heads else args.num_heads
    config.pad_token_id = tokenizer.pad_token_id
    config.bos_token_id = tokenizer.bos_token_id
    config.eos_token_id = tokenizer.eos_token_id

    student_model = AutoModelForCausalLM.from_config(config)

    # Data collator for causal language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # causal language modeling
    )

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.accumulations,
        per_device_eval_batch_size=args.batch_size,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=args.logging_steps,
        evaluation_strategy="steps" if val_dataset else "no",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.limit_checkpoints,
        save_only_model=True,
        optim="adamw_torch_fused",
        learning_rate=args.lr,
        warmup_steps=min(10000, args.max_steps // 10),
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        bf16=True,  # Use mixed precision if available
    )

    # Initialize the Trainer
    trainer = DistillationTrainer(
        model=student_model,
        teacher_model=teacher_model,
        temperature=args.temperature,
        alpha=args.alpha,
        ce_loss_alpha=args.smoothing_alpha,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # Train the model
    trainer.train()

    # Save the final model and tokenizer
    trainer.save_model(os.path.join(args.output_dir, "final_model"))
    tokenizer.save_pretrained(os.path.join(args.output_dir, "final_model"))

if __name__ == "__main__":
    main()
