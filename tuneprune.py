#!/usr/bin/env python
"""
This script implements prune-finetuning by:
1) Subclassing the Hugging Face Trainer to allow a loss function that
   incentivizes structural sparsity in the model (e.g., L1 regularization).
2) Training a causal language model with an additional L1 penalty on its weights.
3) Saving the final model checkpoints.

Usage:
    python prune_finetune.py --model_name <model> --num_train_epochs <epochs> ...
"""

import os
os.environ['HF_HOME'] = '/om/user/ericjm/.cache/huggingface'
import argparse
import torch
import torch.nn as nn

from typing import Callable
from datasets import load_dataset
from transformers import (
    Trainer,
    TrainingArguments,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)

class SparsityTrainer(Trainer):
    """
    Custom Trainer that adds a sparsity-inducing regularization term (e.g., L1)
    to the loss for causal language modeling.
    """
    def __init__(
        self,
        *args,
        compute_sparsity_loss: Callable[[nn.Module], torch.Tensor],
        sparsity_lambda: float = 0.0,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.compute_sparsity_loss = compute_sparsity_loss
        self.sparsity_lambda = sparsity_lambda

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        data_loss, outputs = super().compute_loss(
            model, inputs,
            return_outputs=True,
            **kwargs
        )
        reg_loss = self.sparsity_lambda * self.compute_sparsity_loss(model)
        total_loss = data_loss + reg_loss
        self.log({"reg_loss": reg_loss.item() / self.sparsity_lambda})
        self.log({"reg_loss_weighted": reg_loss.item()})
        self.log({"data_loss": data_loss.item()})
        return (total_loss, outputs) if return_outputs else total_loss


def l1_sparsity_loss(model: nn.Module) -> torch.Tensor:
    """
    Compute the L1 norm of the model's trainable parameters.
    Potentially huge for large models—be mindful of scaling.
    """
    return sum(p.abs().sum() for p in model.parameters() if p.requires_grad)

def l1_sparsity_loss_mlps(model: nn.Module) -> torch.Tensor:
    """
    Compute the L1 norm of the model's MLP parameters.
    """
    return sum(p.abs().sum() for name, p in model.named_parameters() if p.requires_grad and 'mlp' in name)

def l1_of_l2_of_mlps(model: nn.Module) -> torch.Tensor:
    """
    Computes the L1 norm of the L2 norm of the parameters specific to each MLP neuron.
    """
    L1 = 0.0
    for layeri in range(len(model.model.layers)):
        gate_proj = model.model.layers[layeri].mlp.gate_proj.weight # (4x, x)
        up_proj = model.model.layers[layeri].mlp.up_proj.weight     # (4x, x)
        down_proj = model.model.layers[layeri].mlp.down_proj.weight # (x, 4x)
        L2 = torch.sqrt(
            gate_proj.pow(2).sum(dim=1) + \
            up_proj.pow(2).sum(dim=1) + \
            down_proj.pow(2).sum(dim=0)
        )
        L1 += L2.abs().sum()
    return L1

def lhalf_of_l2_of_mlps(model: nn.Module) -> torch.Tensor:
    """
    Computes the L1/2 norm of the L2 norm of the parameters specific to each MLP neuron.
    """
    Lhalf = 0.0
    for layeri in range(len(model.model.layers)):
        gate_proj = model.model.layers[layeri].mlp.gate_proj.weight # (4x, x)
        up_proj = model.model.layers[layeri].mlp.up_proj.weight     # (4x, x)
        down_proj = model.model.layers[layeri].mlp.down_proj.weight # (x, 4x)
        L2 = torch.sqrt(
            gate_proj.pow(2).sum(dim=1) + \
            up_proj.pow(2).sum(dim=1) + \
            down_proj.pow(2).sum(dim=0)
        )
        Lhalf += L2.abs().pow(0.5).sum()
    return Lhalf.pow(2)

REGULARIZERS = {
    "l1_sparsity_loss": l1_sparsity_loss,
    "l1_sparsity_loss_mlps": l1_sparsity_loss_mlps,
    "l1_of_l2_of_mlps": l1_of_l2_of_mlps,
    "lhalf_of_l2_of_mlps": lhalf_of_l2_of_mlps,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Prune-finetuning script")
    parser.add_argument("--model_name", type=str, default="NousResearch/Llama-3.2-1B",
                        help="Model name or path to a pretrained model.")
    parser.add_argument("--output_dir", type=str, default="./pruning_output",
                        help="Directory to store model checkpoints and logs.")
    parser.add_argument("--sparsity_lambda", type=float, default=0.01,
                        help="Regularization strength for L1 penalty.")
    parser.add_argument("--regularizer", type=str, default="l1_sparsity_loss_mlps",
                        choices=REGULARIZERS.keys(),
                        help="Regularization function to use.")
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="Learning rate for Adam optimizer.")
    parser.add_argument("--max_steps", type=int, default=10000,
                        help="Total number of training steps to run.")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum sequence length for tokenization.")
    parser.add_argument("--num_train_epochs", type=int, default=1,
                        help="Number of total epochs to train (if not using max_steps).")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Per-device batch size.")
    parser.add_argument("--eval_steps", type=int, default=500,
                        help="Perform evaluation every N steps.")
    parser.add_argument("--logging_steps", type=int, default=5,
                        help="Log every N steps.")
    parser.add_argument("--save_steps", type=int, default=500,
                        help="Save checkpoint every N steps.")
    parser.add_argument("--use_streaming", action="store_true",
                        help="Use streaming dataset if set.")
    return parser.parse_args()


def prepare_dataset(dataset, tokenizer, max_length):
    """
    Tokenize the dataset for causal language modeling.
    """
    def tokenize_function(examples):
        return tokenizer(
            examples["code"],
            truncation=True,
            max_length=max_length,
        )

    # For non-streaming dataset, we can .map() directly.
    # For streaming, we may have to do things differently.
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
    )
    return tokenized_dataset


def main():
    args = parse_args()
    print("args: ", args)

    # Load or stream dataset
    if args.use_streaming:
        dataset = load_dataset("codeparrot/github-code",
                               streaming=True,
                               languages=["Python"],
                               split="train")
        # Streaming datasets typically don't have "train" / "validation" splits
        # or random access. One might need to do something like:
        #   train_dataset = dataset.take(100000)
        #   val_dataset   = dataset.skip(100000).take(20000)
        # Adjust accordingly for your use case.
        # For simplicity, let's pretend we just do a single dataset:
        train_dataset = dataset
        val_dataset = None
    else:
        dataset = load_dataset("codeparrot/github-code",
                               languages=["Python"],
                               split="train[:1%]")  # For demonstration
        # Create a small validation split just as an example
        train_dataset = dataset.select(range(0, int(0.8 * len(dataset))))
        val_dataset = dataset.select(range(int(0.8 * len(dataset)), len(dataset)))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token_id is None:
        # Ensure we have a pad token, especially important for GPT-2-like models
        tokenizer.pad_token = tokenizer.eos_token

    tokenized_train = prepare_dataset(train_dataset, tokenizer, args.max_length)
    tokenized_val = prepare_dataset(val_dataset, tokenizer, args.max_length) if val_dataset else None

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float32,  # or float16 if your GPU supports it
        device_map="auto"           # could also specify device like "cuda:0"
    )

    # Data collator for causal LM
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=args.logging_steps,
        evaluation_strategy="steps" if val_dataset else "no",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        gradient_accumulation_steps=1,
        fp16=False,  # Set to True if you want mixed precision (and your GPU supports it)
        gradient_checkpointing=True,  # Potential memory savings
        learning_rate=args.lr,
        warmup_steps=1000,
    )

    trainer = SparsityTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_sparsity_loss=REGULARIZERS[args.regularizer],
        sparsity_lambda=args.sparsity_lambda,
    )

    # Train
    trainer.train()

    # Save final model
    trainer.save_model(os.path.join(args.output_dir, "final_model"))

    # You might also want to save the tokenizer
    tokenizer.save_pretrained(os.path.join(args.output_dir, "final_model"))

if __name__ == "__main__":
    main()
