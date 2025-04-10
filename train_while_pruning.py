#!/usr/bin/env python
"""
This script implements training of a causal language model while simultaneously pruning the model.
1) Loads a pretrained model and tokenizer
2) Loads a Python code dataset for training
3) Trains the model using a custom Trainer which implements attribution-based pruning
4) Saves the final trained model

Usage:
    python train_while_pruning.py --model_name meta-llama/Meta-Llama-3.1-8B ...
"""

import os
os.environ['HF_HOME'] = os.environ.get('SCRATCH') + '/iaifi_lab/Lab/ericjm/.cache/huggingface'
import json
import argparse
import torch
from datasets import load_dataset
from transformers import (
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
)
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Train a language model from scratch")
    
    parser.add_argument("--model_name", type=str, default="NousResearch/Meta-Llama-3.1-8B",
                        help="Model name to use (can be from an existing model)")

    # Pruning parameters
    parser.add_argument("--neuron_sparsity", type=float, default=0.8,
                        help="Maximum neuron sparsity to target for pruning")
    parser.add_argument("--neuron_prune_steps", type=int, default=-1,
                        help="Number of steps between neuron pruning events")
    parser.add_argument("--neuron_prune_number", type=int, default=10,
                        help="Number of neurons to prune at each neuron pruning event")
    parser.add_argument("--residual_sparsity", type=float, default=0.75,
                        help="Maximum residual stream sparsity to target for pruning")
    parser.add_argument("--residual_prune_steps", type=int, default=-1,
                        help="Number of steps between residual stream pruning events")
    parser.add_argument("--residual_prune_number", type=int, default=2,
                        help="Number of residual stream neurons to prune at each residual stream pruning event")
    parser.add_argument("--mask_steps", type=int, default=1,
                        help="Number of steps between mask applications to weights (default is every step).")
    parser.add_argument("--ema_alpha", type=float, default=0.99,
                        help="Exponential moving average alpha for gradient averaging.")
    parser.add_argument("--pruning_warmup_steps", type=int, default=100,
                        help="Number of steps to wait before starting any pruning")
    
    # Training parameters
    parser.add_argument("--output_dir", type=str, default="./model_from_scratch",
                        help="Directory to store model checkpoints and logs")
    parser.add_argument("--lr", type=float, default=5e-4,
                        help="Learning rate for the optimizer")
    parser.add_argument("--max_steps", type=int, default=100_000,
                        help="Total number of training steps to run")
    parser.add_argument("--max_length", type=int, default=1024,
                        help="Maximum sequence length for tokenization")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Per-device batch size")
    parser.add_argument("--accumulations", type=int, default=4,
                        help="Number of gradient accumulation steps")
    parser.add_argument("--logging_steps", type=int, default=5,
                        help="Log every N steps")
    parser.add_argument("--save_steps", type=int, default=5000,
                        help="Save checkpoint every N steps")
    parser.add_argument("--limit_checkpoints", type=int, default=-1,
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

class PruningTrainer(Trainer):
    """
    Custom Trainer that implements attribution-based pruning during training.
    Features:
    1) Maintains exponential moving average (EMA) of attribution scores
    2) Prunes neurons and residual connections at specified intervals
    3) Applies mask to parameters during training to maintain pruning
    """
    def __init__(
        self, 
        neuron_sparsity=0.0,
        neuron_prune_steps=100,
        neuron_prune_number=100,
        residual_sparsity=0.0,
        residual_prune_steps=100,
        residual_prune_number=2,
        mask_steps=1,
        ema_decay=0.9,
        pruning_warmup_steps=0,
        **kwargs
    ):
        """
        Initialize the PruningTrainer.
        
        Parameters
        ----------
        neuron_sparsity : float
            Maximum fraction of neurons to prune
        neuron_prune_steps : int
            Number of steps between neuron pruning events
        neuron_prune_number : int
            Number of neurons to prune at each neuron pruning event
        residual_sparsity : float
            Maximum fraction of residual connections to prune
        residual_prune_steps : int
            Number of steps between residual stream pruning events
        residual_prune_number : int
            Number of residual stream neurons to prune at each residual stream pruning event
        mask_steps : int
            Number of steps between mask applications to weights
        ema_decay : float
            Decay factor for exponential moving average of attribution scores
        pruning_warmup_steps : int
            Number of steps to wait before starting any pruning
        **kwargs
            Arguments to pass to parent Trainer
        """
        super().__init__(**kwargs)
        
        # Pruning parameters
        self.neuron_sparsity = neuron_sparsity
        self.neuron_prune_steps = neuron_prune_steps
        self.neuron_prune_number = neuron_prune_number
        self.residual_sparsity = residual_sparsity
        self.residual_prune_steps = residual_prune_steps
        self.residual_prune_number = residual_prune_number
        self.mask_steps = mask_steps
        self.ema_decay = ema_decay
        self.pruning_warmup_steps = pruning_warmup_steps

        # Count neurons and residual stream dimensions
        self.n_neurons = sum(layer.mlp.gate_proj.out_features for layer in self.model.model.layers)
        self.n_residuals = self.model.config.hidden_size

        # Initialize masks and gradient EMA
        self.mask = {name: torch.ones_like(param) for name, param in self.model.named_parameters()} 
        self.gradient_ema = {name: torch.zeros_like(param) for name, param in self.model.named_parameters()}

        # use sets for faster lookup
        self.pruned_neurons = set()
        self.pruned_residuals = set()
        
        # Track the order in which components are pruned
        self.pruned_neurons_order = list()
        self.pruned_residuals_order = list()
    
    def get_neuron_sparsity(self) -> float:
        """
        Fraction of neurons pruned out of the total available.
        Example: 0.8 means 80% of the neurons have been pruned.
        """
        # Number of pruned so far vs the total number of neurons
        return len(self.pruned_neurons) / float(self.n_neurons) if self.n_neurons > 0 else 0.0

    def get_residual_sparsity(self) -> float:
        """
        Fraction of residual stream dimensions pruned out of the total available.
        Example: 0.8 means 80% of the residual stream dimensions have been pruned.
        """
        return len(self.pruned_residuals) / float(self.n_residuals) if self.n_residuals > 0 else 0.0
    
    def _neuron_abs_attribution_scores(self):
        """
        Computes absolute attribution scores, a linear estimate of the effect on the loss of ablating a neuron, using ema of gradients.
        
        Returns
        -------
        dict
            Dictionary mapping neuron identifiers to their attribution scores.
        """
        scores = dict()
        for layeri, layer in enumerate(self.model.model.layers):
            gp = layer.mlp.gate_proj.weight
            up = layer.mlp.up_proj.weight
            dp = layer.mlp.down_proj.weight
            gp_grad = self.gradient_ema[f"model.layers.{layeri}.mlp.gate_proj.weight"]
            up_grad = self.gradient_ema[f"model.layers.{layeri}.mlp.up_proj.weight"]
            dp_grad = self.gradient_ema[f"model.layers.{layeri}.mlp.down_proj.weight"]
            scores[layeri] = torch.sum( \
                (gp_grad * -gp) + \
                (up_grad * -up) + \
                (dp_grad.T * -dp.T), dim=1).abs().tolist()
        return scores
    
    def _residual_abs_attribution_scores(self):
        """
        Computes absolute attribution scores, a linear estimate of the effect on the loss of ablating a residual connection, using ema of gradients.

        Returns
        -------
        list
            List of attribution scores for each residual stream dimension.
        """
        d_model = self.model.config.hidden_size
        device = self.model.model.embed_tokens.weight.device
        dtype = self.model.model.embed_tokens.weight.dtype
        scores = torch.zeros(d_model, device=device, dtype=dtype)
        scores += (self.gradient_ema[f"model.embed_tokens.weight"] * -self.model.model.embed_tokens.weight).sum(dim=0)
        for layeri, layer in enumerate(self.model.model.layers):
            scores += self.gradient_ema[f"model.layers.{layeri}.input_layernorm.weight"] * -layer.input_layernorm.weight
            scores += self.gradient_ema[f"model.layers.{layeri}.post_attention_layernorm.weight"] * -layer.post_attention_layernorm.weight
            scores += (self.gradient_ema[f"model.layers.{layeri}.mlp.gate_proj.weight"] * -layer.mlp.gate_proj.weight).sum(dim=0)
            scores += (self.gradient_ema[f"model.layers.{layeri}.mlp.up_proj.weight"] * -layer.mlp.up_proj.weight).sum(dim=0)
            scores += (self.gradient_ema[f"model.layers.{layeri}.mlp.down_proj.weight"] * -layer.mlp.down_proj.weight).sum(dim=1)
            scores += (self.gradient_ema[f"model.layers.{layeri}.self_attn.q_proj.weight"] * -layer.self_attn.q_proj.weight).sum(dim=0)
            scores += (self.gradient_ema[f"model.layers.{layeri}.self_attn.k_proj.weight"] * -layer.self_attn.k_proj.weight).sum(dim=0)
            scores += (self.gradient_ema[f"model.layers.{layeri}.self_attn.v_proj.weight"] * -layer.self_attn.v_proj.weight).sum(dim=0)
            scores += (self.gradient_ema[f"model.layers.{layeri}.self_attn.o_proj.weight"] * -layer.self_attn.o_proj.weight).sum(dim=1)
        scores += self.gradient_ema[f"model.norm.weight"] * -self.model.model.norm.weight
        return scores.abs().tolist()

    @torch.no_grad()
    def _prune_neurons(self):
        """
        Prunes neurons from the model based on the absolute attribution scores.
        """
        if len(self.pruned_neurons) >= self.n_neurons * self.neuron_sparsity:
            return
        neuron_scores = self._neuron_abs_attribution_scores()
        neuron_score_tuples = [
            (layeri, neuroni, neuron_scores[layeri][neuroni]) 
            for layeri in neuron_scores for neuroni in range(len(neuron_scores[layeri]))
        ]
        neuron_score_tuples.sort(key=lambda x: x[2])
        neurons_to_prune = []
        while len(neurons_to_prune) < self.neuron_prune_number:
            layeri, neuroni, _ = neuron_score_tuples.pop(0)
            if (layeri, neuroni) not in self.pruned_neurons:
                neurons_to_prune.append((layeri, neuroni))
                self.pruned_neurons.add((layeri, neuroni))
                self.pruned_neurons_order.append((layeri, neuroni))
        for layeri, neuroni in neurons_to_prune:
            self.mask[f"model.layers.{layeri}.mlp.gate_proj.weight"][neuroni, :] = 0
            self.mask[f"model.layers.{layeri}.mlp.up_proj.weight"][neuroni, :] = 0
            self.mask[f"model.layers.{layeri}.mlp.down_proj.weight"][:, neuroni] = 0
        for name, param in self.model.named_parameters():
            param.data *= self.mask[name]

    @torch.no_grad()
    def _prune_residuals(self):
        """
        Prunes residual connections from the model based on the absolute attribution scores.
        """
        if len(self.pruned_residuals) >= self.n_residuals * self.residual_sparsity:
            return
        residual_scores = self._residual_abs_attribution_scores()
        residual_score_tuples = [
            (i, residual_scores[i]) for i in range(self.n_residuals)
        ]
        residual_score_tuples.sort(key=lambda x: x[1])
        residuals_to_prune = []
        while len(residuals_to_prune) < self.residual_prune_number:
            i, _ = residual_score_tuples.pop(0)
            if i not in self.pruned_residuals:
                residuals_to_prune.append(i)
                self.pruned_residuals.add(i)
                self.pruned_residuals_order.append(i)
        self.mask[f"model.embed_tokens.weight"][:, residuals_to_prune] = 0
        for layeri, layer in enumerate(self.model.model.layers):
            self.mask[f"model.layers.{layeri}.input_layernorm.weight"][residuals_to_prune] = 0
            self.mask[f"model.layers.{layeri}.post_attention_layernorm.weight"][residuals_to_prune] = 0
            self.mask[f"model.layers.{layeri}.mlp.gate_proj.weight"][:, residuals_to_prune] = 0
            self.mask[f"model.layers.{layeri}.mlp.up_proj.weight"][:, residuals_to_prune] = 0
            self.mask[f"model.layers.{layeri}.mlp.down_proj.weight"][residuals_to_prune, :] = 0
            self.mask[f"model.layers.{layeri}.self_attn.q_proj.weight"][:, residuals_to_prune] = 0
            self.mask[f"model.layers.{layeri}.self_attn.k_proj.weight"][:, residuals_to_prune] = 0
            self.mask[f"model.layers.{layeri}.self_attn.v_proj.weight"][:, residuals_to_prune] = 0
            self.mask[f"model.layers.{layeri}.self_attn.o_proj.weight"][residuals_to_prune, :] = 0
        self.mask[f"model.norm.weight"][residuals_to_prune] = 0
        for name, param in self.model.named_parameters():
            param.data *= self.mask[name]

    def log(self, logs, start_time=None):
        """
        Override the Trainer's log method to add sparsity metrics to every log event.
        """
        logs["neuron_sparsity"] = self.get_neuron_sparsity()
        logs["residual_sparsity"] = self.get_residual_sparsity()
        super().log(logs, start_time)

class PruningCallback(TrainerCallback):
    """
    A callback that:
      - Maintains and updates the gradient EMA every step (after gradient accumulation).
      - Prunes neurons/residuals periodically.
      - Re-applies the pruning mask periodically.
      - Logs neuron/residual sparsities every `logging_steps`.
    """

    def __init__(self, trainer):
        super().__init__()
        self.trainer = trainer

    def on_pre_optimizer_step(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        """Update the gradient EMA after gradient accumulation and before optimizer.step()."""
        if state.global_step < 1:
            return control

        # Update gradient EMA
        for name, param in self.trainer.model.named_parameters():
            if param.grad is not None:
                self.trainer.gradient_ema[name].mul_(self.trainer.ema_decay).add_(
                    (1.0 - self.trainer.ema_decay) * param.grad
                )
        return control

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        """Prune and re-apply the mask at the end of each training step."""
        global_step = state.global_step
        
        if global_step < 1:
            return control

        # Skip pruning during warmup period
        if global_step <= self.trainer.pruning_warmup_steps:
            return control

        # Possibly prune neurons
        if (
            self.trainer.neuron_prune_steps > 0
            and (global_step % self.trainer.neuron_prune_steps == 0)
        ):
            self.trainer._prune_neurons()

        # Possibly prune residuals
        if (
            self.trainer.residual_prune_steps > 0
            and (global_step % self.trainer.residual_prune_steps == 0)
        ):
            self.trainer._prune_residuals()

        # Re-apply mask
        if (
            self.trainer.mask_steps > 0
            and (global_step % self.trainer.mask_steps == 0)
        ):
            for name, param in self.trainer.model.named_parameters():
                param.data.mul_(self.trainer.mask[name])
        
        return control

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs=None,
        **kwargs
    ):
        """This is invoked every time the Trainer does `self.log(...)`."""
        # We've moved the sparsity logging to on_step_end
        return control


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

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Tokenize datasets
    tokenized_train = prepare_dataset(train_dataset, tokenizer, args.max_length)
    tokenized_val = prepare_dataset(val_dataset, tokenizer, args.max_length) if val_dataset else None

    # Load pretrained model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )

    # save model stats about parameter count
    model_stats = {
        "n_params": sum(p.numel() for p in model.parameters()),
        "n_trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
    with open(os.path.join(args.output_dir, "model_stats.json"), "w") as f:
        json.dump(model_stats, f)

    # Data collator for causal language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # causal language modeling
    )

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.accumulations,
        per_device_eval_batch_size=args.batch_size,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=args.logging_steps,
        evaluation_strategy="steps" if val_dataset else "no",
        eval_steps=args.logging_steps * 10 if val_dataset else None,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.limit_checkpoints,
        save_only_model=True,
        optim="adamw_torch_fused",
        learning_rate=args.lr,
        warmup_steps=min(10000, args.max_steps // 10),
        lr_scheduler_type="cosine",
        weight_decay=0.0,
        bf16=torch.cuda.is_available(),  # Use mixed precision if available
    )

    # Initialize the PruningTrainer
    trainer = PruningTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
        tokenizer=tokenizer,
        # Pruning parameters
        neuron_sparsity=args.neuron_sparsity,
        neuron_prune_steps=args.neuron_prune_steps,
        neuron_prune_number=args.neuron_prune_number,
        residual_sparsity=args.residual_sparsity,
        residual_prune_steps=args.residual_prune_steps,
        residual_prune_number=args.residual_prune_number,
        mask_steps=args.mask_steps,
        pruning_warmup_steps=args.pruning_warmup_steps,
    )

    pruning_callback = PruningCallback(trainer)
    trainer.add_callback(pruning_callback)

    # Train the model
    trainer.train()

    # Save the final model and tokenizer
    final_model_path = os.path.join(args.output_dir, "final_model")
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)

    # save learned mask
    torch.save(trainer.mask, os.path.join(args.output_dir, "mask.pt"))

    # save the mask, pruned neurons, and pruned residuals
    with open(os.path.join(args.output_dir, "pruned_neurons_order.json"), "w") as f:
        json.dump(trainer.pruned_neurons_order, f)
    with open(os.path.join(args.output_dir, "pruned_residuals_order.json"), "w") as f:
        json.dump(trainer.pruned_residuals_order, f)

    # Save the training arguments
    args_dict = vars(args)
    with open(os.path.join(args.output_dir, "training_args.json"), "w") as f:
        json.dump(args_dict, f, indent=4)

    # Save final sparsity metrics
    final_metrics = {
        "final_neuron_sparsity": trainer.get_neuron_sparsity(),
        "final_residual_sparsity": trainer.get_residual_sparsity(),
        "total_steps": trainer.state.global_step,
    }
    with open(os.path.join(args.output_dir, "final_metrics.json"), "w") as f:
        json.dump(final_metrics, f, indent=4)

if __name__ == "__main__":
    main()
