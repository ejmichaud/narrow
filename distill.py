#!/usr/bin/env python
"""
This script implements standard Hinton-style distillation:
1) Loads a teacher and a student causal language model.
2) Trains the student to mimic the teacher's softened outputs using a KL-divergence loss.
3) Computes and logs both the teacher's and the student's cross-entropy loss on the data, along with smoothed versions.
4) Saves the final distilled student model.

Usage:
    python distill_finetune.py --teacher_model_name <teacher_model> --student_model_name <student_model> --temperature <temp> ...
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
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)


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
    parser = argparse.ArgumentParser(description="Distillation finetuning script")
    parser.add_argument("--teacher_model_name", type=str, default="NousResearch/Meta-Llama-3.1-8B",
                        help="Pretrained teacher model name or path.")
    parser.add_argument("--student_model_name", type=str, default="NousResearch/Llama-3.2-1B",
                        help="Pretrained student model name or path.")
    parser.add_argument("--temperature", type=float, default=5.0,
                        help="Temperature for distillation softening.")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Weight for hard-label loss (1-alpha is weight for distillation loss).")
    parser.add_argument("--smoothing_alpha", type=float, default=0.995,
                        help="Exponential smoothing factor for loss tracking.")
    parser.add_argument("--output_dir", type=str, default="./distillation_output",
                        help="Directory to store the final student model and logs.")
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="Learning rate for the optimizer.")
    parser.add_argument("--max_steps", type=int, default=10000,
                        help="Total number of training steps to run.")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum sequence length for tokenization.")
    parser.add_argument("--num_train_epochs", type=int, default=1,
                        help="Number of epochs to train (if not using max_steps).")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Per-device batch size.")
    parser.add_argument("--accumulations", type=int, default=2,
                        help="Number of gradient accumulation steps.")
    parser.add_argument("--eval_steps", type=int, default=500,
                        help="Evaluate every N steps.")
    parser.add_argument("--logging_steps", type=int, default=10,
                        help="Log every N steps.")
    parser.add_argument("--save_steps", type=int, default=5000,
                        help="Save checkpoint every N steps.")
    parser.add_argument("--limit_checkpoints", type=int, default=1,
                        help="Limit the number of checkpoints saved. Set to -1 for unlimited.")
    parser.add_argument("--use_streaming", action="store_true",
                        help="Use a streaming dataset if set.")
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

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
    )
    return tokenized_dataset


def main():
    args = parse_args()
    print("args:", args)

    # Load (or stream) the dataset.
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
                               split="train[:1%]",  # For demonstration purposes
                               trust_remote_code=True)
        train_dataset = dataset.select(range(0, int(0.8 * len(dataset))))
        val_dataset = dataset.select(range(int(0.8 * len(dataset)), len(dataset)))

    # Assume teacher and student share the same vocabulary; use the teacher's tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenized_train = prepare_dataset(train_dataset, tokenizer, args.max_length)
    tokenized_val = prepare_dataset(val_dataset, tokenizer, args.max_length) if val_dataset else None

    # Load teacher and student models with bf16 precision to match training args
    teacher_model = AutoModelForCausalLM.from_pretrained(
        args.teacher_model_name,
        torch_dtype=torch.bfloat16,  # Changed from float32 to bfloat16
        device_map="auto"
    )
    teacher_model.eval()

    student_model = AutoModelForCausalLM.from_pretrained(
        args.student_model_name,
        torch_dtype=torch.bfloat16,  # Changed from float32 to bfloat16
        device_map="auto"
    )

    # Data collator for causal language modeling.
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False # causal language modeling
    )

    # Define training arguments.
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
        bf16=True,
        gradient_checkpointing=False,
        learning_rate=args.lr,
        warmup_steps=1000,
    )

    # Create the DistillationTrainer.
    trainer = DistillationTrainer(
        teacher_model=teacher_model,
        temperature=args.temperature,
        alpha=args.alpha,
        ce_loss_alpha=args.smoothing_alpha,
        model=student_model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # Train the student model using distillation loss.
    trainer.train()

    # Save the final distilled student model and tokenizer.
    trainer.save_model(os.path.join(args.output_dir, "final_model"))
    tokenizer.save_pretrained(os.path.join(args.output_dir, "final_model"))


if __name__ == "__main__":
    main()
