"""
This script implements prune-finetuning. We
1) Subclass the Trainer class from the transformers library to
    allow for a loss function which incentivizes structural
    sparsity in the model.
2) Implements a few such loss functions.
3) Performs training.
"""

import os
os.environ['HF_HOME'] = '/om/user/ericjm/.cache/huggingface'
import argparse
import torch
import torch.nn as nn
from transformers import Trainer, TrainingArguments
from typing import Dict, List, Optional, Union, Callable
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import DataCollatorForLanguageModeling

class SparsityTrainer(Trainer):
    """Custom trainer that adds sparsity-inducing regularization to the loss for CausalLM."""
    def __init__(self, *args, compute_sparsity_loss: Callable[[nn.Module], torch.Tensor], sparsity_lambda: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.compute_sparsity_loss = compute_sparsity_loss
        self.sparsity_lambda = sparsity_lambda

    def compute_loss(self, model, inputs, return_outputs=False):
        """Override compute_loss to add sparsity regularization."""
        data_loss, outputs = super().compute_loss(model, inputs, return_outputs=return_outputs)
        total_loss = data_loss + self.sparsity_lambda * self.compute_sparsity_loss(model)
        return (total_loss, outputs) if return_outputs else total_loss

def l1_sparsity_loss(model: nn.Module) -> torch.Tensor:
    """Compute the L1 norm of the model's weights."""
    return sum(p.abs().sum() for p in model.parameters() if p.requires_grad)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="google/gemma-2-2b")
    parser.add_argument("--output_dir", type=str, default="./pruning_output")
    parser.add_argument("--sparsity_lambda", type=float, default=0.01)
    parser.add_argument("--max_steps", type=int, default=10_000)
    parser.add_argument("--max_length", type=int, default=512)
    return parser.parse_args()

def prepare_dataset(dataset, tokenizer, max_length):
    """Prepare dataset for causal language modeling."""
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            return_overflowing_tokens=True,
            return_length=True,
        )

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
    )

    return tokenized_dataset

def main():
    args = parse_args()
    dataset = load_dataset("codeparrot/github-code", streaming=True, languages=["Python"], split="train")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float32,
        device_map="cuda:0"  # Automatically handle model parallelism
    )

    # Prepare dataset
    tokenized_dataset = prepare_dataset(dataset, tokenizer, args.max_length)

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # We're doing causal language modeling, not masked
    )

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=4,  # Adjust based on GPU memory
        per_device_eval_batch_size=4,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=100,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        gradient_accumulation_steps=4,  # Adjust based on needs
        fp16=False,  # Don't do mixed precision training
        gradient_checkpointing=True,  # Memory efficiency
        learning_rate=2e-5,
        warmup_steps=500,
    )

    # Initialize custom trainer
    trainer = SparsityTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_sparsity_loss=l1_sparsity_loss,
        sparsity_lambda=args.sparsity_lambda,
    )

    # Train
    trainer.train()

    # Save final model
    trainer.save_model(os.path.join(args.output_dir, "final_model"))

if __name__ == "__main__":
    main()
