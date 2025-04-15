"""
This script first prunes a model based on attribution scores, then performs
training to hopefully recover performance lost during pruning.
"""

import argparse
import json
import os
# os.environ['HF_HOME'] = '/om/user/ericjm/.cache/huggingface'
os.environ['HF_HOME'] = os.environ.get('SCRATCH') + '/iaifi_lab/Lab/ericjm/.cache/huggingface'
from collections.abc import Mapping
from collections import defaultdict
from typing import Any, Dict, Tuple, Union

from tqdm.auto import tqdm
import numpy as np
import einops
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
)


def move_to_device(data: Any, device: torch.device) -> Any:
    """
    Recursively move tensors (or containers of tensors) to the specified device.
    """
    if isinstance(data, torch.Tensor):
        return data.to(device)
    if isinstance(data, Mapping):
        return type(data)({k: move_to_device(v, device) for k, v in data.items()})
    if isinstance(data, (list, tuple)):
        return type(data)(move_to_device(v, device) for v in data)
    return data


def prepare_data(
    dataset_name: str,
    model_name: str,
    max_length: int,
    batch_size: int,
    num_samples: int,
    split: str = "train",
    streaming: bool = True,
    skip_samples: int = 0,
) -> DataLoader:
    """
    Load and tokenize a dataset for pruning or evaluation.

    If the dataset is streamed, you can optionally skip a number of documents,
    allowing you to use one part of the stream for attribution and a different part for evaluation.
    
    Args:
        dataset_name: Name of the dataset to load.
        model_name: Name of the model used to load its tokenizer.
        max_length: Maximum token length.
        batch_size: Batch size to use in the DataLoader.
        num_samples: Number of samples to use from the dataset.
        split: Which split of the dataset to use (e.g. "train", "test").
        streaming: Whether to load the dataset in streaming mode.
        skip_samples: Number of samples to skip from the beginning of the stream.
    
    Returns:
        A DataLoader yielding batches suitable for language modeling.
    """
    if streaming:
        dataset = load_dataset(dataset_name, split=split, languages=['Python'], streaming=True)
        if skip_samples > 0:
            dataset = dataset.skip(skip_samples)
        dataset = dataset.take(num_samples)
        # Convert streaming dataset to a list to allow tokenization.
        dataset = list(dataset)
    else:
        dataset = load_dataset(dataset_name, split=split, languages=['Python'])
        # For non-streaming, we assume random access is available.
        dataset = dataset.select(range(skip_samples, skip_samples + num_samples))

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        # Assumes the field "code" holds the text; adjust as necessary.
        return tokenizer(
            examples["code"],
            truncation=True,
            max_length=max_length,
        )

    tokenized_dataset = (
        dataset
        if isinstance(dataset, list)
        else dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
    )
    # If tokenized_dataset is a list (from streaming), tokenize each sample.
    if isinstance(tokenized_dataset, list):
        tokenized_dataset = [tokenize_function(sample) for sample in tokenized_dataset]

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    dataloader = DataLoader(tokenized_dataset, batch_size=batch_size, collate_fn=data_collator)
    return dataloader


@torch.no_grad()
def evaluate_model(model: nn.Module, dataloader: DataLoader, device: torch.device) -> Dict[str, float]:
    """
    Evaluate the model on the provided data and compute average loss.
    
    Args:
        model: The language model to evaluate.
        dataloader: DataLoader providing evaluation batches.
        device: The device on which the model and data reside.
    
    Returns:
        A dictionary of evaluation statistics.
    """
    model.eval()
    losses = []
    for batch in tqdm(dataloader, desc="evaluating..."):
        batch = move_to_device(batch, device)
        outputs = model(**batch)
        losses.append(outputs.loss.item())
    return {
        "mean_loss": np.mean(losses).item(),
        "std_of_mean": (np.std(losses) / np.sqrt(len(losses))).item(),
        "losses": losses,
    }


def mask_by_gradient_attribution(
    model: nn.Module,
    dataloader: DataLoader,
    neuron_sparsity: float,
    residual_sparsity: float,
    num_attribution_batches: int,
    output_dir: str, 
):
    """
    Prune neurons and residual stream dimensions based on their attribution scores.
    
    Args:
        model: The language model to prune.
        dataloader: DataLoader providing training batches for attribution.
        neuron_sparsity: Fraction of neurons to prune.
        residual_sparsity: Fraction of residual stream dimensions to prune.
        num_attribution_batches: Number of batches to use for computing attribution scores.
        output_dir: Directory to save pruning information.
    """
    # model.train()  # Set to train mode to enable gradients

    # param_grads = {name: torch.zeros_like(param) for name, param in model.named_parameters()}
    # num_samples = 0
    # for i, batch in enumerate(tqdm(dataloader, desc="computing mean gradients...")):
    #     if i >= num_attribution_batches:
    #         break
    #     model.zero_grad()
    #     batch = move_to_device(batch, model.device)
    #     outputs = model(**batch)
    #     loss = outputs.loss
    #     loss.backward()
    #     for name, param in model.named_parameters():
    #         if param.grad is not None:
    #             param_grads[name] += param.grad.abs().detach()
    #     num_samples += batch['input_ids'].size(0)
    #     model.zero_grad()
    # for name in param_grads:
    #     if num_samples > 0:
    #         param_grads[name] /= num_samples

    # neuron_scores = {}
    # for layeri, layer in enumerate(model.model.layers):
    #     gp_grad = param_grads[f"model.layers.{layeri}.mlp.gate_proj.weight"]
    #     up_grad = param_grads[f"model.layers.{layeri}.mlp.up_proj.weight"]
    #     dp_grad = param_grads[f"model.layers.{layeri}.mlp.down_proj.weight"]
    #     gp = layer.mlp.gate_proj.weight
    #     up = layer.mlp.up_proj.weight
    #     dp = layer.mlp.down_proj.weight
    #     neuron_scores[layeri] = torch.sum(
    #         (gp_grad * -gp) + 
    #         (up_grad * -up) + 
    #         (dp_grad.T * -dp.T), 
    #         dim=1
    #     ).abs().tolist()
    
    # d_model = model.config.hidden_size
    # device = model.model.embed_tokens.weight.device
    # dtype = model.model.embed_tokens.weight.dtype
    # residual_scores = torch.zeros(d_model, device=device, dtype=dtype)
    # residual_scores += (param_grads[f"model.embed_tokens.weight"] * -model.model.embed_tokens.weight).sum(dim=0)
    # for layeri, layer in enumerate(model.model.layers):
    #     residual_scores += param_grads[f"model.layers.{layeri}.input_layernorm.weight"] * -layer.input_layernorm.weight
    #     residual_scores += param_grads[f"model.layers.{layeri}.post_attention_layernorm.weight"] * -layer.post_attention_layernorm.weight
    #     residual_scores += (param_grads[f"model.layers.{layeri}.mlp.gate_proj.weight"] * -layer.mlp.gate_proj.weight).sum(dim=0)
    #     residual_scores += (param_grads[f"model.layers.{layeri}.mlp.up_proj.weight"] * -layer.mlp.up_proj.weight).sum(dim=0)
    #     residual_scores += (param_grads[f"model.layers.{layeri}.mlp.down_proj.weight"] * -layer.mlp.down_proj.weight).sum(dim=1)
    #     residual_scores += (param_grads[f"model.layers.{layeri}.self_attn.q_proj.weight"] * -layer.self_attn.q_proj.weight).sum(dim=0)
    #     residual_scores += (param_grads[f"model.layers.{layeri}.self_attn.k_proj.weight"] * -layer.self_attn.k_proj.weight).sum(dim=0)
    #     residual_scores += (param_grads[f"model.layers.{layeri}.self_attn.v_proj.weight"] * -layer.self_attn.v_proj.weight).sum(dim=0)
    #     residual_scores += (param_grads[f"model.layers.{layeri}.self_attn.o_proj.weight"] * -layer.self_attn.o_proj.weight).sum(dim=1)
    # residual_scores += param_grads[f"model.norm.weight"] * -model.model.norm.weight
    # residual_scores = residual_scores.abs().tolist()

    mask = {name: torch.ones_like(param) for name, param in model.named_parameters()}

    # neuron_score_tuples = [
    #     (layeri, neuroni, neuron_scores[layeri][neuroni]) 
    #     for layeri in neuron_scores for neuroni in range(len(neuron_scores[layeri]))
    # ]
    # neuron_score_tuples.sort(key=lambda x: x[2])  # Sort by score (ascending)
    n_neurons = sum(layer.mlp.gate_proj.out_features for layer in model.model.layers)
    neurons_to_prune_count = int(n_neurons * neuron_sparsity)
    pruned_neurons = []
    for i in range(neurons_to_prune_count):
        # select a random layer and neuron
        layeri = np.random.randint(0, len(model.model.layers))
        neuroni = np.random.randint(0, model.config.intermediate_size)
        # layeri, neuroni, _ = neuron_score_tuples[i]
        pruned_neurons.append((layeri, neuroni))
    for layeri, neuroni in pruned_neurons:
        mask[f"model.layers.{layeri}.mlp.gate_proj.weight"][neuroni, :] = 0
        mask[f"model.layers.{layeri}.mlp.up_proj.weight"][neuroni, :] = 0
        mask[f"model.layers.{layeri}.mlp.down_proj.weight"][:, neuroni] = 0
    
    # residual_score_tuples = [(i, residual_scores[i]) for i in range(len(residual_scores))]
    # residual_score_tuples.sort(key=lambda x: x[1])  # Sort by score (ascending)
    n_residuals = model.config.hidden_size
    residuals_to_prune_count = int(n_residuals * residual_sparsity)
    # sample random subset of range(0, n_residuals)
    pruned_residuals = np.random.choice(range(n_residuals), residuals_to_prune_count, replace=False).tolist()
    mask[f"model.embed_tokens.weight"][:, pruned_residuals] = 0
    for layeri, layer in enumerate(model.model.layers):
        mask[f"model.layers.{layeri}.input_layernorm.weight"][pruned_residuals] = 0
        mask[f"model.layers.{layeri}.post_attention_layernorm.weight"][pruned_residuals] = 0
        mask[f"model.layers.{layeri}.mlp.gate_proj.weight"][:, pruned_residuals] = 0
        mask[f"model.layers.{layeri}.mlp.up_proj.weight"][:, pruned_residuals] = 0
        mask[f"model.layers.{layeri}.mlp.down_proj.weight"][pruned_residuals, :] = 0
        mask[f"model.layers.{layeri}.self_attn.q_proj.weight"][:, pruned_residuals] = 0
        mask[f"model.layers.{layeri}.self_attn.k_proj.weight"][:, pruned_residuals] = 0
        mask[f"model.layers.{layeri}.self_attn.v_proj.weight"][:, pruned_residuals] = 0
        mask[f"model.layers.{layeri}.self_attn.o_proj.weight"][pruned_residuals, :] = 0
    mask[f"model.norm.weight"][pruned_residuals] = 0
    
    # Apply mask to model parameters
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in mask:
                param.data *= mask[name]
    
    stats = {
        "pruned_neurons": pruned_neurons,
        "pruned_residuals": pruned_residuals,
        "total_neurons": n_neurons,
        "total_residuals": n_residuals,
        "total_neurons_pruned": len(pruned_neurons),
        "total_residuals_pruned": len(pruned_residuals),
    }

    # Print statistics for debugging
    print("\n=== Pruning Statistics ===")
    print(f"Total neurons: {n_neurons}")
    print(f"Total residuals: {n_residuals}")
    print(f"Neurons pruned: {len(pruned_neurons)} / {n_neurons} ({len(pruned_neurons)/n_neurons:.2%})")
    print(f"Residuals pruned: {len(pruned_residuals)} / {n_residuals} ({len(pruned_residuals)/n_residuals:.2%})")
    
    # Print some of the pruned indices for verification
    # if pruned_neurons:
    #     print(f"\nSample of pruned neurons: {pruned_neurons[:5]}{'...' if len(pruned_neurons) > 5 else ''}")
    # if pruned_residuals:
    #     print(f"Sample of pruned residuals: {pruned_residuals[:5]}{'...' if len(pruned_residuals) > 5 else ''}")
    
    # Print some attribution score statistics
    # if neuron_scores:
    #     # Convert dictionary of lists to a flat numpy array
    #     neuron_scores_array = np.concatenate([np.array(scores) for scores in neuron_scores.values()])
    #     print(f"\nNeuron attribution scores - min: {neuron_scores_array.min():.6f}, max: {neuron_scores_array.max():.6f}, mean: {neuron_scores_array.mean():.6f}")
    
    # if residual_scores:
    #     residual_scores_array = np.array(residual_scores)
    #     print(f"Residual attribution scores - min: {residual_scores_array.min():.6f}, max: {residual_scores_array.max():.6f}, mean: {residual_scores_array.mean():.6f}")
    # print("===========================\n")
    
    return mask, stats


class MaskedTrainer(Trainer):
    """
    Custom Trainer that applies a mask to model parameters every mask_steps steps.
    This ensures pruned neurons remain pruned during training.
    """
    def __init__(self, mask=None, mask_steps=1, **kwargs):
        """
        Initialize the MaskedTrainer.
        
        Args:
            mask: Dictionary mapping parameter names to binary masks
            mask_steps: Apply mask every this many steps
            **kwargs: Arguments to pass to the parent Trainer
        """
        super().__init__(**kwargs)
        self.mask = mask
        self.mask_steps = mask_steps
        
    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], num_items_in_batch=None) -> torch.Tensor:
        """Override training_step to apply masks periodically"""
        # Call the parent's training_step to handle the training logic
        loss = super().training_step(model, inputs, num_items_in_batch)
        
        # Apply mask every mask_steps
        if self.state.global_step % self.mask_steps == 0 and self.mask is not None:
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name in self.mask:
                        param.data *= self.mask[name]
        
        return loss


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for both pruning and training phases.
    """
    parser = argparse.ArgumentParser(
        description="Prune a model based on attribution scores, then train to recover performance."
    )
    # Model and dataset parameters
    parser.add_argument("--model_name", type=str, default="NousResearch/Llama-3.2-1B",
                        help="Pretrained model name or path.")
    parser.add_argument("--dataset_name", type=str, default="codeparrot/github-code",
                        help="Dataset name for pruning and training.")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for data loading.")
    parser.add_argument("--accumulations", type=int, default=4,
                        help="Number of gradient accumulation steps.")
    parser.add_argument("--streaming", action="store_true",
                        help="Load the dataset in streaming mode.")
    parser.add_argument("--output_dir", type=str, default="./pruned_trained_models",
                        help="Directory to save the pruned and trained model.")

    # Pruning parameters
    parser.add_argument("--neuron_sparsity", type=float, default=0.8,
                        help="Fraction of neurons to prune.")
    parser.add_argument("--residual_sparsity", type=float, default=0.5,
                        help="Fraction of residual stream dimensions to prune.")
    parser.add_argument("--prune_samples", type=int, default=1000,
                        help="Number of samples to use for pruning data.")
    parser.add_argument("--prune_skip", type=int, default=0,
                        help="Number of samples to skip for pruning (if streaming).")

    # Training parameters
    # parser.add_argument("--train_samples", type=int, default=,
    #                     help="Number of samples to use for training data.")
    parser.add_argument("--train_skip", type=int, default=0,
                        help="Number of samples to skip for training (if streaming).")
    parser.add_argument("--max_steps", type=int, default=-1,
                        help="Total number of training steps to run. -1 means use num_train_epochs.")
    parser.add_argument("--lr", type=float, default=5e-5,
                        help="Learning rate for training.")
    parser.add_argument("--mask_steps", type=int, default=1,
                        help="Apply mask every this many steps during training.")
    parser.add_argument("--eval_steps", type=int, default=500,
                        help="Evaluate every N steps.")
    parser.add_argument("--save_steps", type=int, default=1000,
                        help="Save checkpoint every N steps.")
    parser.add_argument("--limit_checkpoints", type=int, default=3,
                        help="Limit the number of checkpoints saved. Set to -1 for unlimited")
    parser.add_argument("--logging_steps", type=int, default=100,
                        help="Log every N steps.")
    parser.add_argument("--warmup_steps", type=int, default=500,
                        help="Number of warmup steps for learning rate scheduler.")

    # Evaluation parameters
    parser.add_argument("--eval", action="store_true",
                        help="Whether to perform evaluation after pruning and training.")
    parser.add_argument("--eval_samples", type=int, default=200,
                        help="Number of samples to use for evaluation.")
    parser.add_argument("--eval_skip", type=int, default=0,
                        help="Number of samples to skip for evaluation (if streaming).")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float32,
        device_map=str(device)
    )

    # ===== STEP 1: PRUNING PHASE =====
    
    # Load pruning data
    print("Preparing pruning data...")
    pruning_dataloader = prepare_data(
        dataset_name=args.dataset_name,
        model_name=args.model_name,
        max_length=args.max_length,
        batch_size=args.batch_size,
        num_samples=args.prune_samples,
        split="train",
        streaming=args.streaming,
        skip_samples=args.prune_skip
    )
    
    # Create mask based on attribution scores
    print("Creating pruning mask based on attribution scores...")
    num_attribution_batches = args.prune_samples // args.batch_size
    mask, pruning_stats = mask_by_gradient_attribution(
        model=model,
        dataloader=pruning_dataloader,
        neuron_sparsity=args.neuron_sparsity,
        residual_sparsity=args.residual_sparsity,
        num_attribution_batches=num_attribution_batches,
        output_dir=args.output_dir
    )
    
    # Save initial pruning statistics
    pruning_stats_file = os.path.join(args.output_dir, "pruning_stats.json")
    with open(pruning_stats_file, "w") as f:
        json.dump(pruning_stats, f, indent=4)
    print(f"Pruning statistics saved to {pruning_stats_file}")
    # save mask as a torch file
    mask_file = os.path.join(args.output_dir, "pruning_mask.pt")
    torch.save(mask, mask_file)
    
    # ===== STEP 2: TRAINING PHASE =====
    
    # Load tokenizer for training data preparation
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load training data
    print("Preparing training data...")
    if args.streaming:
        train_dataset = load_dataset(
            args.dataset_name, 
            split="train", 
            languages=['Python'], 
            streaming=True
        )
        if args.train_skip > 0:
            train_dataset = train_dataset.skip(args.train_skip)
        train_dataset = train_dataset.take(args.batch_size * args.max_steps * args.accumulations)
    else:
        train_dataset = load_dataset(
            args.dataset_name, 
            split="train", 
            languages=['Python']
        )
        train_dataset = train_dataset.select(range(args.train_skip, args.train_skip + args.batch_size * args.max_steps * args.accumulations))
    
    # Tokenize the training dataset
    def tokenize_function(examples):
        return tokenizer(
            examples["code"],
            truncation=True,
            max_length=args.max_length,
        )
    
    tokenized_train = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=train_dataset.column_names,
    )
    
    # Load evaluation data if needed
    eval_dataset = None
    if args.eval:
        print("Preparing evaluation data...")
        if args.streaming:
            eval_dataset = load_dataset(
                args.dataset_name, 
                split="train", 
                languages=['Python'], 
                streaming=True
            )
            if args.eval_skip > 0:
                eval_dataset = eval_dataset.skip(args.eval_skip)
            eval_dataset = eval_dataset.take(args.eval_samples)
        else:
            eval_dataset = load_dataset(
                args.dataset_name, 
                split="train", 
                languages=['Python']
            )
            eval_dataset = eval_dataset.select(range(args.eval_skip, args.eval_skip + args.eval_samples))
        
        tokenized_eval = eval_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=eval_dataset.column_names,
        )
    else:
        tokenized_eval = None
    
    # Data collator for training
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.accumulations,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=args.logging_steps,
        evaluation_strategy="steps" if tokenized_eval else "no",
        eval_steps=args.eval_steps if tokenized_eval else None,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.limit_checkpoints,
        load_best_model_at_end=tokenized_eval is not None,
        bf16=True if torch.cuda.is_available() else False,
        optim="adamw_torch_fused",
        lr_scheduler_type="cosine",
        weight_decay=0.00,
    )
    
    # Initialize the MaskedTrainer with the mask
    trainer = MaskedTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        data_collator=data_collator,
        tokenizer=tokenizer,
        mask=mask,
        mask_steps=args.mask_steps,
    )
    
    # Train the model while maintaining the pruned structure
    print("Starting training phase...")
    trainer.train()
    
    # Save the final model
    trainer.save_model(os.path.join(args.output_dir, "final_model"))
    tokenizer.save_pretrained(os.path.join(args.output_dir, "final_model"))
    print(f"Final model saved to {os.path.join(args.output_dir, 'final_model')}")
    
    # Evaluate the final model if requested
    if args.eval:
        print("Evaluating final model...")
        eval_dataloader = DataLoader(
            tokenized_eval, 
            batch_size=args.batch_size, 
            collate_fn=data_collator
        )
        eval_stats = evaluate_model(model, eval_dataloader, device)
        
        # Save evaluation results
        eval_file = os.path.join(args.output_dir, "final_evaluation_results.json")
        with open(eval_file, "w") as f:
            json.dump(eval_stats, f, indent=4)
        print(f"Final evaluation results saved to {eval_file}")


if __name__ == "__main__":
    main()







