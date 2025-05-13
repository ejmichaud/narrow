#!/usr/bin/env python
"""
Implements gradient routing for finetuning LLMs.

Routes data loss gradients to high-attribution MLP neurons
and sparsity loss gradients to low-attribution MLP neurons.
Attribution is periodically recalculated based on L2 norm.


python slowtransfer.py \
    --output_dir=./grad_route_output \
    --model_name="NousResearch/Llama-3.2-1B" \
    --target_sparsity=0.5 \
    --attribution_update_steps=500 \
    --sparsity_regularizer=lhalf_of_l2_of_mlps \
    --lr=4e-6 \
    --max_steps=100000 \
    --batch_size=32 \
    --use_streaming \
    --save_steps=50000 \
    --sparsity_loss_weight=2.0
"""
import os
import argparse
import torch
import torch.nn as nn
from typing import Dict, Union, Any
from datasets import load_dataset, IterableDataset
from transformers import (
    Trainer,
    TrainingArguments,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    trainer_utils,
)
import sys
import wandb
from torch.optim import AdamW
from tqdm import tqdm

# --- Configuration ---
# Set environment variables early
os.environ["WANDB_PROJECT"] = os.environ.get("WANDB_PROJECT", "gradient_routing")
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
# cache_dir = # Set cache directory if needed


# --- Sparsity Regularizers ---
def l1_of_l2_of_mlps(model: nn.Module) -> torch.Tensor:
    """L1 norm of the L2 norm of MLP neuron parameters."""
    # Use the device of the first parameter as the reference
    target_device = next(model.parameters()).device
    total_l1_of_l2 = torch.tensor(0.0, device=target_device)
    layer_l2_norms = []
    for layer in getattr(getattr(model, "model", model), "layers", []):
        mlp = getattr(layer, "mlp", None)
        if not mlp:
            continue
        gate_w = getattr(getattr(mlp, "gate_proj", None), "weight", None)
        up_w = getattr(getattr(mlp, "up_proj", None), "weight", None)
        down_w = getattr(getattr(mlp, "down_proj", None), "weight", None)
        if gate_w is None or up_w is None or down_w is None:
            continue

        # Calculate L2 norm on the layer's device - REMOVE .detach() to allow gradient flow
        l2_sq = (
            gate_w.pow(2).sum(dim=1) + up_w.pow(2).sum(dim=1) + down_w.pow(2).sum(dim=0)
        )
        # Move the result to the target device before storing
        layer_l2_norms.append(torch.sqrt(l2_sq).to(target_device))

    # Concatenate and sum norms on the target device
    if layer_l2_norms:
        all_l2_norms = torch.cat(layer_l2_norms)
        total_l1_of_l2 = all_l2_norms.abs().sum()

    return total_l1_of_l2


def generalized_norm(
    model: nn.Module, outer_norm: float, inner_norm: float, eps: float = 1e-8
) -> torch.Tensor:
    """Outer norm of the inner norm of MLP neuron parameters."""
    # Use the device of the first parameter as the reference
    target_device = next(model.parameters()).device
    all_inner_norms_list = []
    for layer in getattr(getattr(model, "model", model), "layers", []):
        mlp = getattr(layer, "mlp", None)
        if not mlp:
            continue
        gate_w = getattr(getattr(mlp, "gate_proj", None), "weight", None)
        up_w = getattr(getattr(mlp, "up_proj", None), "weight", None)
        down_w = getattr(getattr(mlp, "down_proj", None), "weight", None)
        if gate_w is None or up_w is None or down_w is None:
            continue

        # Calculate inner norm on the layer's device - REMOVE .detach() to allow gradient flow
        inner = (
            gate_w.pow(inner_norm).sum(dim=1)
            + up_w.pow(inner_norm).sum(dim=1)
            + down_w.pow(inner_norm).sum(dim=0)
            + eps
        ) ** (1.0 / inner_norm)
        # Move the result to the target device before storing
        all_inner_norms_list.append(inner.to(target_device))

    if not all_inner_norms_list:
        return torch.tensor(0.0, device=target_device)

    # Concatenate norms on the target device
    all_norms_cat = torch.cat(all_inner_norms_list)
    # Calculate outer norm on the target device
    return (all_norms_cat.abs().pow(outer_norm).sum() + eps) ** (1.0 / outer_norm)


REGULARIZERS = {
    "l1_of_l2_of_mlps": l1_of_l2_of_mlps,
    "lhalf_of_l2_of_mlps": lambda model: generalized_norm(model, 0.5, 2),
    "lfifth_of_l2_of_mlps": lambda model: generalized_norm(model, 0.2, 5),
    "lhalf_of_l1_of_mlps": lambda model: generalized_norm(model, 0.5, 1),
}


# --- Attribution Helper ---
def move_to_device(data: Any, device: torch.device) -> Any:
    """
    Recursively move tensors (or containers of tensors) to the specified device.
    (Helper function copied from prune_abs.py for convenience)
    """
    if isinstance(data, torch.Tensor):
        return data.to(device)
    if isinstance(data, dict):  # Changed Mapping to dict for simplicity
        return type(data)({k: move_to_device(v, device) for k, v in data.items()})
    if isinstance(data, (list, tuple)):
        return type(data)(move_to_device(v, device) for v in data)
    return data


def get_mlp_neuron_gradient_attribution(
    model: nn.Module,
    attribution_dataloader: torch.utils.data.DataLoader,
    num_attribution_batches: int,
    device: torch.device,  # Pass device explicitly
) -> Dict[int, torch.Tensor]:
    """
    Compute attribution scores based on gradient * activation.
    Similar to the implementation in prune_abs.py.
    """
    print(f"Computing gradient attributions using {num_attribution_batches} batches...")
    model.train()  # Ensure gradients are computed

    num_layers = len(getattr(getattr(model, "model", model), "layers", []))
    # Initialize scores on the target device to avoid issues later
    scores = {layeri: torch.tensor(0.0, device=device) for layeri in range(num_layers)}
    neuron_counts = {
        layeri: torch.tensor(0.0, device=device) for layeri in range(num_layers)
    }  # Keep track of how many activations summed

    data_iter = iter(attribution_dataloader)
    batches_processed = 0

    # Use tqdm for progress indication
    pbar = tqdm(total=num_attribution_batches, desc="Attribution batches")

    while batches_processed < num_attribution_batches:
        try:
            batch = next(data_iter)
        except StopIteration:
            print(
                "Warning: Attribution dataloader exhausted before reaching num_attribution_batches."
            )
            break  # Stop if dataloader runs out

        cache = {}
        forward_hooks = {}
        backward_handles = (
            {}
        )  # To store backward hook handles associated with forward hooks

        def get_attribution_hook(cache, layer_idx, backward_handles):
            def attribution_hook(module, input, output):
                # The backward hook is registered on the *output* tensor of act_fn
                def backward_hook(grad):
                    # Compute attribution: -output * grad (element-wise)
                    # Using .detach() as we only need the values, not graph history
                    modified_grad = (
                        -output.detach() * grad.detach()
                    ).abs()  # Use absolute value
                    # Sum across batch and sequence length dimensions
                    summed_attr = modified_grad.sum(
                        dim=tuple(range(modified_grad.ndim - 1))
                    )
                    cache[layer_idx] = (
                        cache.get(layer_idx, torch.zeros_like(summed_attr))
                        + summed_attr
                    )
                    # Store neuron activation counts for averaging later
                    neuron_counts[layer_idx] = neuron_counts[layer_idx] + (
                        output.shape[0] * output.shape[1]
                    )
                    # Return original grad
                    return grad

                # Register backward hook on the output tensor and store the handle
                backward_handles[layer_idx] = output.register_hook(backward_hook)
                return None  # Forward hook doesn't modify output

            return attribution_hook

        # Register hooks for each layer's activation function
        model_base = getattr(model, "model", model)
        for layeri in range(num_layers):
            mlp_act_fn = getattr(
                getattr(model_base.layers[layeri], "mlp", {}), "act_fn", None
            )
            if mlp_act_fn:
                forward_hooks[layeri] = mlp_act_fn.register_forward_hook(
                    get_attribution_hook(cache, layeri, backward_handles)
                )

        # Perform forward and backward pass
        batch = move_to_device(batch, device)
        model.zero_grad()  # Ensure grads are clean
        outputs = model(**batch)
        loss = outputs.loss
        if isinstance(loss, torch.Tensor):  # Handle potential multi-GPU loss
            loss = loss.mean()
        loss.backward()

        # Accumulate scores from cache (values computed in backward hooks)
        for layeri in range(num_layers):
            if layeri in cache:
                # Ensure scores are on the correct device before accumulation
                scores[layeri] = scores[layeri].to(device) + cache[layeri].to(device)

        # Remove hooks
        for handle in forward_hooks.values():
            handle.remove()
        for handle in backward_handles.values():
            handle.remove()

        # Cleanup references
        del cache, forward_hooks, backward_handles, batch, outputs, loss
        batches_processed += 1
        pbar.update(1)

    pbar.close()

    # Average scores
    final_scores = {}
    for layeri in range(num_layers):
        if neuron_counts[layeri] > 0:
            final_scores[layeri] = (scores[layeri] / neuron_counts[layeri]).detach()
        else:
            # Handle case where a layer might not have had activations/scores
            mlp_layer = getattr(
                getattr(model, "model", model).layers[layeri], "mlp", None
            )
            if mlp_layer:
                # Try to get the expected size, e.g., from gate_proj output dim
                output_dim = getattr(
                    getattr(mlp_layer, "gate_proj", None), "weight", torch.zeros(0)
                ).shape[0]
                final_scores[layeri] = torch.zeros(output_dim, device=device)
            else:
                final_scores[layeri] = torch.tensor(
                    [], device=device
                )  # Empty tensor if no MLP

    model.zero_grad(set_to_none=True)  # Clean up grads
    print("Gradient attribution calculation complete.")
    return final_scores


# --- Gradient Routing Trainer ---
class GradientRoutingTrainer(Trainer):
    def __init__(
        self,
        *args,
        target_sparsity: float,
        attribution_update_steps: int,
        sparsity_regularizer: str,
        attribution_dataloader,
        sparsity_loss_weight: float = 1.0,
        attribution_batch_size: int,  # Keep this, now used for # batches
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if target_sparsity < 0 or target_sparsity > 1:
            raise ValueError("target_sparsity must be between 0.0 and 1.0")
        self.target_sparsity = target_sparsity
        self.attribution_update_steps = attribution_update_steps
        self.sparsity_regularizer_fn = REGULARIZERS[sparsity_regularizer]
        self.attribution_dataloader = attribution_dataloader
        self.sparsity_loss_weight = sparsity_loss_weight
        self.num_attribution_batches = attribution_batch_size  # Rename for clarity

        self.neuron_masks: Dict[int, torch.Tensor] = {}
        self.mlp_params_map: Dict[
            torch.Tensor, tuple[int, str, torch.Tensor | None]
        ] = {}
        self._prepare_mlp_param_map()
        self._last_attribution_update_step = -1

        print(
            f"GradientRoutingTrainer: Target Sparsity={self.target_sparsity:.1%}, "
            f"Update Steps={self.attribution_update_steps}, Regularizer={sparsity_regularizer}, "
            f"Sparsity Loss Weight={self.sparsity_loss_weight}, "
            f"Attribution Batches={self.num_attribution_batches}"  # Log the number of batches
        )

    def _prepare_mlp_param_map(self):
        """Creates a map from trainable MLP weight tensors to their layer index and type."""
        self.mlp_params_map = {}
        model_base = getattr(
            self.model, "model", self.model
        )  # Handle potential wrapping
        for i, layer in enumerate(getattr(model_base, "layers", [])):
            mlp = getattr(layer, "mlp", None)
            if isinstance(mlp, nn.Module):
                for name, param in mlp.named_parameters():
                    if "weight" in name and param.requires_grad:
                        param_type = name.split(".")[0]
                        self.mlp_params_map[param] = (
                            i,
                            param_type,
                            None,
                        )  # Mask added later
        print(f"Mapped {len(self.mlp_params_map)} trainable MLP weight tensors.")

    def _update_attribution_masks(self):
        """Calculates attribution scores (gradient-based) and updates masks."""
        if not self.mlp_params_map:
            print("Warning: No MLP params mapped, skipping attribution update.")
            self._last_attribution_update_step = self.state.global_step
            return
        print(
            f"Step {self.state.global_step}: Updating attribution masks using gradient attribution..."
        )
        self.model.eval()  # Set to eval before calling attribution func, func will set train

        # Use the new attribution function
        layer_neuron_scores = get_mlp_neuron_gradient_attribution(
            self.model,
            self.attribution_dataloader,
            self.num_attribution_batches,
            self.model.device,
        )
        # Ensure model is back in training mode after attribution calculation
        self.model.train()

        if not layer_neuron_scores:
            print("Warning: No gradient attribution scores calculated.")
            self._last_attribution_update_step = self.state.global_step
            return

        # Move all score tensors to CPU for concatenation and quantile calculation
        # This avoids potential cross-device issues if scores were somehow on different devices
        all_scores_list = [
            scores.cpu()
            for scores in layer_neuron_scores.values()
            if scores.numel() > 0
        ]
        if not all_scores_list:
            print("Warning: No valid attribution scores found across layers.")
            self._last_attribution_update_step = self.state.global_step
            return

        all_scores = torch.cat(all_scores_list)

        if all_scores.numel() == 0:
            print("Warning: Concatenated gradient attribution tensor is empty.")
            self._last_attribution_update_step = self.state.global_step
            return

        # We want to keep high-attribution neurons, so prune based on low scores
        # The threshold identifies the score at the target_sparsity quantile (lower scores are pruned)
        threshold = torch.quantile(all_scores.float(), self.target_sparsity)
        print(
            f"Attribution threshold (Gradient q={self.target_sparsity:.2f}): {threshold.item():.6f}"
        )

        self.neuron_masks.clear()
        total_low_attrib = 0
        total_neurons = 0
        model_base = getattr(self.model, "model", self.model)

        for i, scores in layer_neuron_scores.items():
            # Ensure scores are on the same device as the threshold for comparison
            mask = (
                scores.to(threshold.device) >= threshold
            ).detach()  # True for high attribution
            self.neuron_masks[i] = mask  # Store mask (can be on GPU)
            total_neurons += mask.numel()
            total_low_attrib += (~mask).sum().item()

            layer = model_base.layers[i]
            mlp = getattr(layer, "mlp", None)
            if not mlp:
                continue
            # Update map with computed mask reference
            for proj_name in ["gate_proj", "up_proj", "down_proj"]:
                proj_layer = getattr(mlp, proj_name, None)
                if (
                    proj_layer
                    and hasattr(proj_layer, "weight")
                    and proj_layer.weight in self.mlp_params_map
                ):
                    # Mask itself is stored in self.neuron_masks[i]
                    self.mlp_params_map[proj_layer.weight] = (
                        i,
                        proj_name,
                        self.neuron_masks[i],
                    )

        actual_sparsity = total_low_attrib / total_neurons if total_neurons > 0 else 0
        print(
            f"Actual low-attribution fraction: {total_low_attrib}/{total_neurons} ({actual_sparsity:.2%})"
        )
        self.log(
            {
                "attribution_threshold_grad": threshold.item(),  # Log grad attribution
                "actual_sparsity": actual_sparsity,
            }
        )
        self._last_attribution_update_step = self.state.global_step

    def training_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        batch_idx: int,
    ) -> torch.Tensor:
        """Performs a training step with gradient routing."""
        # Update masks periodically or on the first step
        if self._last_attribution_update_step < 0 or (
            self.state.global_step > self._last_attribution_update_step
            and (self.state.global_step + 1) % self.attribution_update_steps == 0
        ):
            self._update_attribution_masks()

        model.train()
        inputs = self._prepare_inputs(inputs)

        # Data loss and gradients
        with self.compute_loss_context_manager():
            loss_data = self.compute_loss(model, inputs, return_outputs=False)
        if self.args.n_gpu > 1:
            loss_data = loss_data.mean()
        scaled_loss_data = loss_data / self.args.gradient_accumulation_steps
        self.accelerator.backward(
            scaled_loss_data
        )  # Populates .grad for data loss path

        # Store data gradients
        data_grads = {
            param: param.grad.clone()
            for param in model.parameters()
            if param.grad is not None
        }

        # Zero gradients before sparsity backward pass
        self.optimizer.zero_grad(set_to_none=True)

        # Sparsity loss and gradients
        loss_sparsity = self.sparsity_regularizer_fn(model)
        if self.args.n_gpu > 1:
            loss_sparsity = loss_sparsity.mean()
        # Apply the sparsity loss weight
        weighted_loss_sparsity = self.sparsity_loss_weight * loss_sparsity
        scaled_loss_sparsity = (
            weighted_loss_sparsity / self.args.gradient_accumulation_steps
        )
        # This backward pass calculates gradients *only* for the sparsity loss path, overwriting .grad
        self.accelerator.backward(scaled_loss_sparsity)

        # Route gradients
        with torch.no_grad():
            for param in model.parameters():
                if not param.requires_grad:
                    continue

                # Get stored data gradient (might be None)
                data_grad = data_grads.get(param)

                if param in self.mlp_params_map:
                    layer_idx, param_type, neuron_mask = self.mlp_params_map[param]

                    # Get sparsity grad (current param.grad, might be None)
                    sparsity_grad = param.grad

                    if neuron_mask is None:  # Mask not ready yet
                        param.grad = data_grad  # Assign data_grad (could be None)
                        continue

                    # Ensure mask is on the correct device
                    neuron_mask = neuron_mask.to(param.device)

                    # Prepare grads, ensuring they are tensors on the correct device
                    sparsity_grad_mlp = (
                        sparsity_grad
                        if sparsity_grad is not None
                        else torch.zeros_like(param)
                    )
                    data_grad_mlp = (
                        data_grad if data_grad is not None else torch.zeros_like(param)
                    )

                    # Create float masks for multiplication
                    high_attrib_mask = neuron_mask.float()
                    low_attrib_mask = (~neuron_mask).float()

                    # Adjust mask shape for broadcasting
                    if param_type == "down_proj":
                        # Unsqueeze dim 0 (input channels)
                        high_attrib_mask = high_attrib_mask.unsqueeze(0)
                        low_attrib_mask = low_attrib_mask.unsqueeze(0)
                    else:  # gate_proj, up_proj
                        # Unsqueeze dim 1 (output channels)
                        high_attrib_mask = high_attrib_mask.unsqueeze(1)
                        low_attrib_mask = low_attrib_mask.unsqueeze(1)

                    # Combine gradients using broadcasted masks
                    param.grad = (
                        data_grad_mlp * high_attrib_mask
                        + sparsity_grad_mlp * low_attrib_mask
                    )

                else:
                    # Non-MLP parameters: Keep only data gradient
                    param.grad = data_grad  # Assign data_grad (could be None)

        log_dict = {
            "loss_data": loss_data.item(),
            "loss_sparsity": loss_sparsity.item(),
            "weighted_loss_sparsity": weighted_loss_sparsity.item(),
        }
        self.log(log_dict)

        # Return scaled loss consistent with Trainer expectation
        return loss_data.detach() / self.args.gradient_accumulation_steps


# --- Argument Parsing ---
def parse_args():
    parser = argparse.ArgumentParser(description="Gradient Routing Finetuning Script")
    # Core args
    parser.add_argument("--model_name", type=str, default="NousResearch/Llama-3.2-1B")
    parser.add_argument("--output_dir", type=str, default="./grad_route_output")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Per-device batch size"
    )
    parser.add_argument(
        "--accumulations", type=int, default=2, help="Gradient accumulation steps"
    )
    # Dataset args
    parser.add_argument("--use_streaming", action="store_true")
    parser.add_argument("--max_length", type=int, default=512)
    # Routing args
    parser.add_argument("--target_sparsity", type=float, default=0.5)
    parser.add_argument("--attribution_update_steps", type=int, default=500)
    parser.add_argument(
        "--sparsity_regularizer",
        type=str,
        default="l1_of_l2_of_mlps",
        choices=REGULARIZERS.keys(),
    )
    parser.add_argument(
        "--sparsity_loss_weight",
        type=float,
        default=1.0,
        help="Weight for sparsity loss relative to data loss",
    )
    parser.add_argument(
        "--attribution_batch_size",
        type=int,
        default=32,
        help="Number of batches to use for gradient attribution calculation.",
    )
    # Logging/Saving args
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--save_steps", type=int, default=2000)
    parser.add_argument("--eval_steps", type=int, default=1000)
    parser.add_argument(
        "--num_train_epochs", type=int, default=1
    )  # Used if max_steps <= 0

    args = parser.parse_args()

    # Basic check for output directory
    try:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Output directory: {args.output_dir}")
    except OSError as e:
        print(
            f"Error creating output directory {args.output_dir}: {e}", file=sys.stderr
        )
        sys.exit(1)
    return args


# --- Data Preparation ---
def prepare_dataset(dataset, tokenizer, max_length, is_streaming):
    """Tokenizes dataset"""

    def tokenize_function(examples):
        # Find text field heuristically
        text_field = next(
            (k for k in ["text", "code", "content"] if k in examples), None
        )
        if not text_field:
            potential_fields = [
                k
                for k, v in examples.items()
                if isinstance(v, list) and v and isinstance(v[0], str)
            ]
            if not potential_fields:
                raise ValueError(
                    f"Cannot find text field in examples: {list(examples.keys())}"
                )
            text_field = potential_fields[0]
        texts = [str(t) if t is not None else "" for t in examples[text_field]]
        return tokenizer(texts, truncation=True, max_length=max_length)

    column_names = (
        list(next(iter(dataset)).keys()) if is_streaming else dataset.column_names
    )
    tokenized_dataset = dataset.map(
        tokenize_function, batched=True, remove_columns=column_names
    )
    return tokenized_dataset


# --- Main Execution ---
def main():
    args = parse_args()
    print("Args:", args)

    wandb.init(config=vars(args))

    # --- Load Data ---
    dataset_name = "codeparrot/github-code"
    dataset_kwargs = {
        "cache_dir": "../.cache/huggingface/datasets"
    }  # Centralize cache dir
    if args.use_streaming:
        train_dataset = load_dataset(
            dataset_name,
            streaming=True,
            languages=["Python"],
            split="train",
            **dataset_kwargs,
        )
        val_dataset = None
        attr_stream_take_count = (
            args.attribution_batch_size * args.batch_size * 8
        )  # Heuristic: aim for enough data
        attribution_dataset = load_dataset(
            dataset_name,
            streaming=True,
            languages=["Python"],
            split="train",
            **dataset_kwargs,
        ).take(attr_stream_take_count)
        print(
            f"Using streaming dataset. Attribution stream approx {attr_stream_take_count} samples."
        )
    else:
        dataset_subset = "Python-small"
        dataset = load_dataset(
            dataset_name, dataset_subset, split="train", **dataset_kwargs
        ).shuffle(seed=42)
        split_idx = int(0.98 * len(dataset))
        train_dataset, val_dataset = dataset.select(range(split_idx)), dataset.select(
            range(split_idx, len(dataset))
        )
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        attr_sample_count = args.attribution_batch_size * args.batch_size * num_gpus
        attr_sample_count = min(
            len(train_dataset), attr_sample_count
        )  # Don't take more than available

        attribution_dataset = train_dataset.select(range(attr_sample_count))
        print(
            f"Using non-streaming '{dataset_subset}'. Train: {len(train_dataset)}, Val: {len(val_dataset)}, Attr: {attr_sample_count} samples for {args.attribution_batch_size} batches."
        )

    # --- Tokenizer and Model ---
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    if (
        tokenizer.pad_token_id is not None
        and getattr(model.config, "pad_token_id", None) is None
    ):
        model.config.pad_token_id = tokenizer.pad_token_id
    print(f"Model loaded: {args.model_name} on device: {model.device}")

    # --- Tokenize Data ---
    tokenized_train = prepare_dataset(
        train_dataset, tokenizer, args.max_length, args.use_streaming
    )
    tokenized_val = (
        prepare_dataset(val_dataset, tokenizer, args.max_length, False)
        if val_dataset
        else None
    )
    tokenized_attribution = prepare_dataset(
        attribution_dataset, tokenizer, args.max_length, args.use_streaming
    )  # Attr dataset can be streaming or not

    # --- Collator, Optimizer, Dataloader ---
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    attribution_dataloader = torch.utils.data.DataLoader(
        tokenized_attribution,
        batch_size=args.batch_size,
        collate_fn=data_collator,
        shuffle=False,
        num_workers=0,
    )

    # --- Training Arguments ---
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        num_train_epochs=args.num_train_epochs if args.max_steps <= 0 else 100,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.accumulations,
        per_device_eval_batch_size=args.batch_size * 2,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        evaluation_strategy="steps" if tokenized_val else "no",
        eval_steps=args.eval_steps if tokenized_val else None,
        save_strategy="steps",
        save_steps=args.save_steps,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported() and torch.cuda.is_available(),
        gradient_checkpointing=False,
        save_total_limit=3,
        load_best_model_at_end=False,
        save_only_model=True,
        learning_rate=args.lr,  # Passed to Trainer but optimizer has priority
        warmup_steps=0,
        lr_scheduler_type="constant",
        report_to=["wandb"],
        seed=42,
        save_safetensors=True,
        logging_dir=os.path.join(args.output_dir, "logs"),
    )

    # --- Initialize Trainer ---
    trainer = GradientRoutingTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
        tokenizer=tokenizer,
        optimizers=(optimizer, None),
        # Routing specific args
        target_sparsity=args.target_sparsity,
        attribution_update_steps=args.attribution_update_steps,
        sparsity_regularizer=args.sparsity_regularizer,
        sparsity_loss_weight=args.sparsity_loss_weight,
        attribution_dataloader=attribution_dataloader,
        attribution_batch_size=args.attribution_batch_size,
    )

    # --- Train ---
    def save_final_model(suffix="final_model"):
        try:
            output_path = os.path.join(args.output_dir, suffix)
            print(f"Saving final model to {output_path}...")
            trainer.save_model(output_path)
            tokenizer.save_pretrained(output_path)
            print("Save complete.")
        except Exception as e:
            print(f"Error saving final model: {e}", file=sys.stderr)

    print("\nStarting training...")
    train_result = None
    try:
        last_checkpoint = trainer_utils.get_last_checkpoint(args.output_dir)
        resume = last_checkpoint if last_checkpoint else False
        print(
            f"Resuming from checkpoint: {resume}"
            if resume
            else "Starting from scratch."
        )
        train_result = trainer.train(resume_from_checkpoint=resume)
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
    except Exception as e:
        print(f"\nTraining interrupted: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        save_final_model("error_checkpoint")  # Attempt save on error
    finally:
        save_final_model()  # Save final model state
        wandb.finish()
        print("Training finished.")


if __name__ == "__main__":
    main()
