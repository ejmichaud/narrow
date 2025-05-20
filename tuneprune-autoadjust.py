#!/usr/bin/env python
"""
This script implements prune-finetuning by:
1) Subclassing the Hugging Face Trainer to allow a loss function that
   incentivizes structural sparsity in the model (e.g., L1 regularization).
2) Training a causal language model with an additional L1 penalty on its weights.
3) Saving the final model checkpoints.

It now also supports an *adaptive* regularization strength that adjusts
over training based on whether the (smoothed) training loss is going up or down.

Usage:
    python prune_finetune.py --model_name <model> --num_train_epochs <epochs> ...
"""

import os
# os.environ['HF_HOME'] = '/om/user/ericjm/.cache/huggingface'
os.environ['HF_HOME'] = os.environ.get('SCRATCH') + '/iaifi_lab/Lab/ericjm/.cache/huggingface'
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
    to the loss for causal language modeling. Also supports adaptive regularization.
    """
    def __init__(
        self,
        *args,
        compute_sparsity_loss: Callable[[nn.Module], torch.Tensor],
        sparsity_lambda: float = 0.0,
        use_adaptive_reg: bool = False,
        adjust_steps: int = 100,
        alpha: float = 0.9,
        inc_factor: float = 1.01,
        dec_factor: float = 0.99,
        reg_warmup_steps: int = 1000,
        logging_steps: int = 10,
        **kwargs
    ):
        """
        Args:
            compute_sparsity_loss: A callable that takes in a model and returns
                a scalar regularization (e.g., L1) penalty.
            sparsity_lambda: Initial regularization strength.
            use_adaptive_reg: If True, enable adaptive regularization behavior.
            adjust_steps: Perform regularization strength adjustments every N steps.
            alpha: Exponential smoothing factor for the training loss.
            inc_factor: Multiplicative factor to *increase* regularization if
                the smoothed loss is not rising.
            dec_factor: Multiplicative factor to *decrease* regularization if
                the smoothed loss is rising.
            reg_warmup_steps: Number of steps before starting adaptive regularization.
            logging_steps: Log every N steps.
        """
        super().__init__(*args, **kwargs)
        self.compute_sparsity_loss = compute_sparsity_loss
        self.sparsity_lambda = sparsity_lambda

        # Adaptive-reg parameters
        self.use_adaptive_reg = use_adaptive_reg
        self.adjust_steps = adjust_steps
        self.alpha = alpha
        self.inc_factor = inc_factor
        self.dec_factor = dec_factor
        self.reg_warmup_steps = reg_warmup_steps

        # For tracking smoothed training loss
        self.smoothed_loss = None
        self.last_smoothed_loss = None
        self._global_step = 0  # We'll increment this manually to track steps
        self._loss_buffer = []  # Add a buffer for initial loss values
        self.logging_steps = logging_steps

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Compute the total loss = data_loss + reg_loss with proper handling of gradient accumulation.
        """
        # Get the standard language modeling loss
        data_loss, outputs = super().compute_loss(
            model, inputs,
            return_outputs=True,
            **kwargs
        )
        
        # Store the unscaled data loss for logging and adaptive regularization
        # This ensures consistent values regardless of gradient_accumulation_steps
        unscaled_data_loss = data_loss.item() * self.args.gradient_accumulation_steps
        
        # Track smoothed training loss for adaptive regularization
        # Use the unscaled loss for consistent behavior
        self._update_smoothed_loss(unscaled_data_loss)
        
        # Compute the regularization penalty
        # The regularization should also be scaled down for gradient accumulation
        # to maintain the same effective regularization strength
        reg_loss = (self.sparsity_lambda * self.compute_sparsity_loss(model)) / self.args.gradient_accumulation_steps
        
        # The total loss is what will be used for backpropagation
        total_loss = data_loss + reg_loss
        
        # For logging, we want to show the unscaled values
        if self._global_step % self.logging_steps == 0:
            # For the raw regularization value (before lambda scaling)
            if self.sparsity_lambda != 0.0:
                raw_reg_value = reg_loss.item() * self.args.gradient_accumulation_steps / self.sparsity_lambda
                self.log({"reg_loss": raw_reg_value})
            
            # For the weighted regularization (actual contribution to the loss)
            weighted_reg = reg_loss.item() * self.args.gradient_accumulation_steps
            self.log({"reg_loss_weighted": weighted_reg})
            
            # Log the unscaled data loss
            self.log({"data_loss": unscaled_data_loss})
            
            # Other logging values
            self.log({"current_lambda": self.sparsity_lambda})
            self.log({"smoothed_loss": self.smoothed_loss})
            
            # Optionally log the total unscaled loss
            total_unscaled = unscaled_data_loss + weighted_reg
            self.log({"total_loss": total_unscaled})
        
        return (total_loss, outputs) if return_outputs else total_loss

    def _update_smoothed_loss(self, current_loss: float):
        """
        Update the exponential-moving-average training loss,
        and adjust `sparsity_lambda` if needed.
        """
        self._global_step += 1
        
        # Collect initial losses for better initialization
        if self.smoothed_loss is None and self._global_step <= 10:
            self._loss_buffer.append(current_loss)
            if self._global_step == 10:
                # Initialize with mean of first 10 batches
                self.smoothed_loss = sum(self._loss_buffer) / len(self._loss_buffer)
            return
        
        if self.smoothed_loss is None:
            # First step, just set smoothed_loss = current_loss
            self.smoothed_loss = current_loss
        else:
            # Exponential smoothing
            self.smoothed_loss = (
                self.alpha * self.smoothed_loss
                + (1 - self.alpha) * current_loss
            )

        if self.use_adaptive_reg and self._global_step > self.reg_warmup_steps:
            if self._global_step % self.adjust_steps == 0:
                # Compare smoothed_loss to the last time we adjusted
                if self.last_smoothed_loss is not None:
                    if self.smoothed_loss > self.last_smoothed_loss:
                        # If the (smoothed) training loss has increased,
                        # we want to *decrease* the regularization strength
                        self.sparsity_lambda *= self.dec_factor
                    else:
                        # Otherwise, we *increase* it slightly
                        self.sparsity_lambda *= self.inc_factor

                # Update last_smoothed_loss
                self.last_smoothed_loss = self.smoothed_loss


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
    return sum(p.abs().sum() for name, p in model.named_parameters()
               if p.requires_grad and 'mlp' in name)

def l1_of_l2_of_mlps(model: nn.Module) -> torch.Tensor:
    """
    Computes the L1 norm of the L2 norm of the parameters for each MLP neuron.
    """
    L1 = 0.0
    for layeri in range(len(model.model.layers)):
        gate_proj = model.model.layers[layeri].mlp.gate_proj.weight # (4x, x)
        up_proj = model.model.layers[layeri].mlp.up_proj.weight     # (4x, x)
        down_proj = model.model.layers[layeri].mlp.down_proj.weight # (x, 4x)
        L2 = torch.sqrt(
            gate_proj.pow(2).sum(dim=1) +
            up_proj.pow(2).sum(dim=1) +
            down_proj.pow(2).sum(dim=0)
        )
        L1 += L2.abs().sum()
    return L1

def lhalf_of_l2_of_mlps(model: nn.Module) -> torch.Tensor:
    """
    Computes the L1/2 norm of the L2 norm of the parameters for each MLP neuron.
    """
    Lhalf = 0.0
    for layeri in range(len(model.model.layers)):
        gate_proj = model.model.layers[layeri].mlp.gate_proj.weight # (4x, x)
        up_proj = model.model.layers[layeri].mlp.up_proj.weight     # (4x, x)
        down_proj = model.model.layers[layeri].mlp.down_proj.weight # (x, 4x)
        L2 = torch.sqrt(
            gate_proj.pow(2).sum(dim=1) +
            up_proj.pow(2).sum(dim=1) +
            down_proj.pow(2).sum(dim=0)
        )
        Lhalf += L2.abs().pow(0.5).sum()
    return Lhalf.pow(2)

def group_hoyer_neurons(model: nn.Module) -> torch.Tensor:
    """
    Computes the Hoyer norm of the L2s of the parameters for each MLP neuron.
    """
    n_layers = len(model.model.layers)
    intermediate_size = model.config.intermediate_size
    device = model.model.layers[0].mlp.gate_proj.weight.device
    dtype = model.model.layers[0].mlp.gate_proj.weight.dtype
    neuron_l2s = torch.zeros(n_layers * intermediate_size, device=device, dtype=dtype)
    for layeri in range(len(model.model.layers)):
        gate_proj = model.model.layers[layeri].mlp.gate_proj.weight # (4x, x)
        up_proj = model.model.layers[layeri].mlp.up_proj.weight     # (4x, x)
        down_proj = model.model.layers[layeri].mlp.down_proj.weight # (x, 4x)
        L2 = torch.sqrt(
            gate_proj.pow(2).sum(dim=1) +
            up_proj.pow(2).sum(dim=1) +
            down_proj.pow(2).sum(dim=0)
        )
        neuron_l2s[layeri * intermediate_size:(layeri + 1) * intermediate_size] = L2
    L1 = neuron_l2s.abs().sum()
    L2 = neuron_l2s.pow(2).sum().sqrt()
    return L1 / L2
    

def group_lasso_residual_stream(model: nn.Module) -> torch.Tensor:
    """
    Computes the L1 norm of the L2 norm (per residual dimension) of all parameters that read from or write to the residual stream,
    but in a vectorized and faster manner.
    """
    d_model = model.config.hidden_size
    device = model.model.embed_tokens.weight.device
    dtype = model.model.embed_tokens.weight.dtype
    sq_sums = torch.zeros(d_model, device=device, dtype=dtype)
    sq_sums += model.model.embed_tokens.weight.pow(2).sum(dim=0)
    for layer in model.model.layers:
        sq_sums += layer.input_layernorm.weight.pow(2)
        sq_sums += layer.post_attention_layernorm.weight.pow(2)
        sq_sums += layer.mlp.gate_proj.weight.pow(2).sum(dim=0)
        sq_sums += layer.mlp.up_proj.weight.pow(2).sum(dim=0)
        sq_sums += layer.mlp.down_proj.weight.pow(2).sum(dim=1)
        sq_sums += layer.self_attn.q_proj.weight.pow(2).sum(dim=0)
        sq_sums += layer.self_attn.k_proj.weight.pow(2).sum(dim=0)
        sq_sums += layer.self_attn.v_proj.weight.pow(2).sum(dim=0)
        sq_sums += layer.self_attn.o_proj.weight.pow(2).sum(dim=1)
    sq_sums += model.model.norm.weight.pow(2)
    return sq_sums.sqrt().sum()

def group_hoyer_residual_stream(model: nn.Module) -> torch.Tensor:
    """
    Computes the Hoyer norm of the L2s of the parameters that read from or write to the residual stream,
    but in a vectorized and faster manner.
    """
    d_model = model.config.hidden_size
    device = model.model.embed_tokens.weight.device
    dtype = model.model.embed_tokens.weight.dtype
    sq_sums = torch.zeros(d_model, device=device, dtype=dtype)
    sq_sums += model.model.embed_tokens.weight.pow(2).sum(dim=0)
    for layer in model.model.layers:
        sq_sums += layer.input_layernorm.weight.pow(2)
        sq_sums += layer.post_attention_layernorm.weight.pow(2)
        sq_sums += layer.mlp.gate_proj.weight.pow(2).sum(dim=0)
        sq_sums += layer.mlp.up_proj.weight.pow(2).sum(dim=0)
        sq_sums += layer.mlp.down_proj.weight.pow(2).sum(dim=1)
        sq_sums += layer.self_attn.q_proj.weight.pow(2).sum(dim=0)
        sq_sums += layer.self_attn.k_proj.weight.pow(2).sum(dim=0)
        sq_sums += layer.self_attn.v_proj.weight.pow(2).sum(dim=0)
        sq_sums += layer.self_attn.o_proj.weight.pow(2).sum(dim=1)
    sq_sums += model.model.norm.weight.pow(2)
    res_l2s = sq_sums.sqrt()
    L1 = res_l2s.abs().sum()
    L2 = res_l2s.pow(2).sum().sqrt()
    return L1 / L2

REGULARIZERS = {
    "l1_sparsity_loss": l1_sparsity_loss,
    "l1_sparsity_loss_mlps": l1_sparsity_loss_mlps,
    "l1_of_l2_of_mlps": l1_of_l2_of_mlps,
    "lhalf_of_l2_of_mlps": lhalf_of_l2_of_mlps,
    "group_lasso_residual_stream": group_lasso_residual_stream,
    "group_hoyer_neurons": group_hoyer_neurons,
    "group_hoyer_residual_stream": group_hoyer_residual_stream,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Prune-finetuning script")
    parser.add_argument("--model_name", type=str, default="NousResearch/Llama-3.2-1B",
                        help="Model name or path to a pretrained model.")
    parser.add_argument("--output_dir", type=str, default="./pruning_output",
                        help="Directory to store model checkpoints and logs.")
    parser.add_argument("--sparsity_lambda", type=float, default=0.01,
                        help="Initial regularization strength for L1 penalty.")
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
    parser.add_argument("--accumulations", type=int, default=2,
                        help="Number of gradient accumulation steps.")
    parser.add_argument("--eval_steps", type=int, default=500,
                        help="Perform evaluation every N steps.")
    parser.add_argument("--logging_steps", type=int, default=10,
                        help="Log every N steps.")
    parser.add_argument("--save_steps", type=int, default=5_000,
                        help="Save checkpoint every N steps.")
    parser.add_argument("--limit_checkpoints", type=int, default=1,
                        help="Limit the number of checkpoints saved. Set to -1 for unlimited.")
    parser.add_argument("--use_streaming", action="store_true",
                        help="Use streaming dataset if set.")

    # Adaptive reg arguments
    parser.add_argument("--use_adaptive_reg", action="store_true",
                        help="Enable or disable adaptive regularization.")
    parser.add_argument("--adjust_steps", type=int, default=200,
                        help="Adjust regularization strength every N steps.")
    parser.add_argument("--alpha", type=float, default=0.999,
                        help="Smoothing factor for training loss (EMA).")
    parser.add_argument("--inc_factor", type=float, default=1.1,
                        help="Factor to multiply `sparsity_lambda` if loss is decreasing.")
    parser.add_argument("--dec_factor", type=float, default=0.6,
                        help="Factor to multiply `sparsity_lambda` if loss is increasing.")

    # Add reg_warmup_steps to the argument parser
    parser.add_argument("--reg_warmup_steps", type=int, default=1000,
                       help="Number of steps before starting adaptive regularization.")

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
    print("args: ", args)

    # Load or stream dataset
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
                               split="train[:1%]",  # For demonstration
                               trust_remote_code=True)
        train_dataset = dataset.select(range(0, int(0.8 * len(dataset))))
        val_dataset = dataset.select(range(int(0.8 * len(dataset)), len(dataset)))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token_id is None:
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
        optim="adamw_torch_fused",  # FASTER OPTIMIZER
        bf16=True,  # Mixed precision BF16 if your GPU supports it
        gradient_checkpointing=False,  # Potential memory savings
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
        use_adaptive_reg=args.use_adaptive_reg,
        adjust_steps=args.adjust_steps,
        alpha=args.alpha,
        inc_factor=args.inc_factor,
        dec_factor=args.dec_factor,
        reg_warmup_steps=args.reg_warmup_steps,
        logging_steps=args.logging_steps,
    )

    # Train
    trainer.train()

    # Save final model
    trainer.save_model(os.path.join(args.output_dir, "final_model"))
    tokenizer.save_pretrained(os.path.join(args.output_dir, "final_model"))


if __name__ == "__main__":
    main()
