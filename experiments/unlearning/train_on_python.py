#!/usr/bin/env python3
"""
Train pruned models on Python code.
Converts pruned models to VariableSizeLlamaForCausalLM to physically remove pruned neurons.
"""

import os
import sys
import argparse
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

# Setup HF cache location
os.environ['HF_HOME'] = os.environ.get('SCRATCH', '/tmp') + '/iaifi_lab/Lab/ericjm/.cache/huggingface'

# Add narrow module to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from narrow.modeling import convert_pruned_to_variable_size


def prepare_dataset(dataset, tokenizer, max_length):
    """Tokenize dataset for causal language modeling."""
    def tokenize_function(examples):
        return tokenizer(
            examples["code"],
            truncation=True,
            max_length=max_length,
        )
    
    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
    )
    return tokenized


def main():
    parser = argparse.ArgumentParser(description="Train pruned model on Python code")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to pruned model"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save trained models"
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=10000,
        help="Total training steps"
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=2500,
        help="Save checkpoint every N steps"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Per-device batch size"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=1024,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--convert_to_variable_size",
        action="store_true",
        help="Convert pruned model to VariableSizeLlamaForCausalLM"
    )
    
    args = parser.parse_args()
    
    print(f"{'='*80}")
    print(f"Training Pruned Model on Python Code")
    print(f"{'='*80}")
    print(f"Model: {args.model_path}")
    print(f"Output: {args.output_dir}")
    print(f"Max steps: {args.max_steps}")
    print(f"Save steps: {args.save_steps}")
    print(f"Convert to variable size: {args.convert_to_variable_size}")
    print(f"{'='*80}\n")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float32,
        device_map="auto"
    )
    
    # Convert to variable size if requested
    if args.convert_to_variable_size:
        print("Converting to VariableSizeLlamaForCausalLM...")
        original_params = sum(p.numel() for p in model.parameters())
        model = convert_pruned_to_variable_size(model)
        new_params = sum(p.numel() for p in model.parameters())
        print(f"  Original parameters: {original_params:,}")
        print(f"  New parameters: {new_params:,}")
        print(f"  Reduction: {(1 - new_params/original_params):.1%}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load Python code dataset (stream from local cache for speed!)
    print("Loading Python code dataset from local cache...")
    
    # Load from cached dataset and convert to iterable
    from datasets import load_from_disk
    cache_dir = os.path.join(
        os.environ['HF_HOME'],
        'datasets/codeparrot___github-code/all-all-4b2efe4a27feed92'
    )
    
    print(f"Loading from: {cache_dir}")
    dataset = load_from_disk(cache_dir)
    
    # Convert to iterable dataset for streaming from local files
    train_dataset = dataset.to_iterable_dataset()
    
    # Take enough samples for training
    total_samples = args.max_steps * args.batch_size * args.gradient_accumulation_steps
    print(f"Streaming {total_samples:,} samples from local cache")
    
    train_dataset = train_dataset.take(total_samples)
    
    # Tokenize
    print("Tokenizing dataset...")
    tokenized_train = prepare_dataset(train_dataset, tokenizer, args.max_length)
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        save_steps=args.save_steps,
        save_total_limit=None,  # Keep all checkpoints
        logging_steps=50,
        logging_first_step=True,
        bf16=False,
        fp16=False,
        dataloader_num_workers=4,  # Use multiple workers for faster data loading
        dataloader_prefetch_factor=2,
        remove_unused_columns=True,
        report_to="none",
        save_safetensors=True,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        data_collator=data_collator,
    )
    
    # Train
    print("\nStarting training...")
    trainer.train()
    
    # Save final model
    print("\nSaving final model...")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    
    print(f"\n{'='*80}")
    print(f"Training complete!")
    print(f"Saved to: {args.output_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()



