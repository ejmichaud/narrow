import torch
import torch.nn.utils.prune as prune
from typing import Dict, Union, Any, Mapping
import einops
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling
import os


def prune_by_attribution(
    model: nn.Module,
    train_dataloader,
    importance_threshold: float = 1e-7,
    attribution_batch_size: int = 4,
    num_attribution_batches: int = 4,
    save_pruned_model: bool = False,
    output_dir: str = "./pruned_model",
):
    device = next(model.parameters()).device
    neuron_to_avg_effect = {}

    def compute_neuron_importances():
        print(
            f"Processing {num_attribution_batches} of size {attribution_batch_size} batches for attribution..."
        )
        for batch_idx, inputs in enumerate(train_dataloader):
            print(f"Processing batch {batch_idx + 1}/{num_attribution_batches}")
            if batch_idx >= num_attribution_batches:
                break
            inputs = _prepare_inputs(inputs)

            # Compute the effect of each neuron on the loss
            _, cache, grad_cache = get_cache_fwd_and_bwd(model, inputs)

            for layer_idx, layer in enumerate(model.model.layers):
                for mat_name in ["gate_proj", "up_proj", "down_proj"]:
                    matrix = getattr(layer.mlp, mat_name)
                    cache_key = id(matrix)

                    neuron_acts = cache.cache_dict[cache_key]
                    neuron_grads = grad_cache.cache_dict[cache_key]

                    for neuron_idx in range(matrix.weight.shape[0]):
                        # takes element-wise product between neuron gradients and activations, sums over sequence dimension
                        neuron_effect = einops.einsum(
                            neuron_grads[:, :, neuron_idx],
                            neuron_acts[:, :, neuron_idx],
                            "batch seq, batch seq -> batch",
                        )
                        avg_neuron_effect = neuron_effect.mean().item()

                        cache_name = f"layer_{layer_idx}_{mat_name}"
                        if cache_name not in neuron_to_avg_effect:
                            neuron_to_avg_effect[cache_name] = {}

                        if neuron_idx not in neuron_to_avg_effect[cache_name]:
                            neuron_to_avg_effect[cache_name][neuron_idx] = 0.0

                        neuron_to_avg_effect[cache_name][
                            neuron_idx
                        ] += avg_neuron_effect

        # Average the neuron effects over the batches
        for cache_name in neuron_to_avg_effect:
            for neuron_idx in neuron_to_avg_effect[cache_name]:
                neuron_to_avg_effect[cache_name][neuron_idx] /= num_attribution_batches

    def prune_neurons():
        total_neurons_pruned = 0
        total_neurons = 0

        for layer_idx in range(len(model.model.layers)):
            layer = model.model.layers[layer_idx]
            for mat_name in ["gate_proj", "up_proj", "down_proj"]:
                matrix = getattr(layer.mlp, mat_name)
                total_neurons += matrix.weight.shape[0]
                cache_name = f"layer_{layer_idx}_{mat_name}"

                neuron_effects = neuron_to_avg_effect[cache_name]

                # Prune neurons with absolute effect below threshold
                neurons_to_prune = [
                    neuron
                    for neuron, effect in neuron_effects.items()
                    if abs(effect) < importance_threshold
                ]

                # Create mask
                output_mask = torch.ones(
                    matrix.weight.shape[0], device=matrix.weight.device
                )
                output_mask[neurons_to_prune] = 0

                num_pruned = (output_mask == 0).sum().item()
                total_neurons_pruned += num_pruned

                full_mask = output_mask.unsqueeze(1).expand_as(matrix.weight)
                prune.custom_from_mask(matrix, name="weight", mask=full_mask)

            print(f"Layer {layer_idx}:")
            print(
                f"- Pruning {num_pruned} out of {layer.mlp.gate_proj.weight.shape[0]} neurons"
            )
        print(
            f"Total neurons pruned across all layers: {total_neurons_pruned} / {total_neurons} ({total_neurons_pruned/total_neurons:.2%})"
        )

    def _prepare_inputs(
        inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> Dict[str, Union[torch.Tensor, Any]]:
        if isinstance(inputs, Mapping):
            return type(inputs)({k: _prepare_input(v) for k, v in inputs.items()})

    def _prepare_input(data: Union[torch.Tensor, Any]) -> Union[torch.Tensor, Any]:
        if isinstance(data, Mapping):
            return type(data)({k: _prepare_input(v) for k, v in data.items()})
        elif isinstance(data, (tuple, list)):
            return type(data)(_prepare_input(v) for v in data)
        elif isinstance(data, torch.Tensor):
            kwargs = {"device": device}
            return data.to(**kwargs)
        return data

    def get_cache_fwd_and_bwd(model, tokens):
        cache = {}
        grad_cache = {}

        def forward_hook(module, input, output):
            cache[id(module)] = output.detach()
            return output

        def backward_hook(module, grad_input, grad_output):
            grad_cache[id(module)] = grad_output[0].detach()
            return grad_input

        # Register hooks for MLP layers
        for name, module in model.named_modules():
            if "mlp" in name:
                module.register_forward_hook(forward_hook)
                module.register_full_backward_hook(backward_hook)

        # Forward and backward pass
        outputs = model(**tokens)
        loss = outputs.loss
        loss.backward()

        class SimpleCache:
            def __init__(self, cache_dict):
                self.cache_dict = cache_dict

            def __getitem__(self, key):
                return self.cache_dict[key]

        return (
            loss.item(),
            SimpleCache(cache),
            SimpleCache(grad_cache),
        )

    # Start the pruning process
    compute_neuron_importances()
    prune_neurons()

    if save_pruned_model:
        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(output_dir)
        print(f"Pruned model saved to {output_dir}")


def prune_by_weight_norm(
    model: nn.Module,
    pruning_threshold: float,
    save_pruned_model: bool = False,
    output_dir: str = "./pruned_model",
):
    total_neurons_pruned = 0
    total_neurons = 0

    for i, layer in enumerate(model.model.layers):
        gate_proj = layer.mlp.gate_proj
        up_proj = layer.mlp.up_proj
        down_proj = layer.mlp.down_proj

        total_neurons += gate_proj.weight.shape[0]
        L2 = torch.sqrt(
            gate_proj.weight.pow(2).sum(dim=1)
            + up_proj.weight.pow(2).sum(dim=1)
            + down_proj.weight.pow(2).sum(dim=0)
        )
        # L2 Shape: torch.Size([8192])

        # Mask neurons based on the L2 norm of the contributing parameters
        output_mask = L2 >= pruning_threshold
        num_pruned = (~output_mask).sum().item()
        total_neurons_pruned += num_pruned

        mask = output_mask.unsqueeze(1).expand_as(gate_proj.weight)
        prune.custom_from_mask(gate_proj, name="weight", mask=mask)
        prune.custom_from_mask(up_proj, name="weight", mask=mask)
        prune.custom_from_mask(down_proj, name="weight", mask=mask.T)

        print(f"Layer {i}:")
        print(f"- Pruned {num_pruned} out of {len(output_mask)} neurons")

    print(
        f"Total neurons pruned across all layers: {total_neurons_pruned} / {total_neurons} ({total_neurons_pruned/total_neurons:.2%})"
    )

    if save_pruned_model:
        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(output_dir)
        print(f"Pruned model saved to {output_dir}")


def prepare_data_for_pruning(
    dataset_name: str,
    model_name: str,
    max_length: int,
    batch_size: int,
    num_samples: int = 1000,
):
    dataset = load_dataset(
        dataset_name, languages=["Python"], split="train", streaming=True
    )
    attribution_dataset = dataset.take(num_samples)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    def prepare_dataset(dataset, tokenizer, max_length):
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

    attribution_tokenized = prepare_dataset(
        attribution_dataset, tokenizer, max_length=max_length
    )
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    attribution_dataloader = DataLoader(
        attribution_tokenized, batch_size=batch_size, collate_fn=data_collator
    )
    return attribution_dataloader


def main():
    # Choose a model
    model_name = "NousResearch/Llama-3.2-1B"
    # Choose a pruning strategy — "attribution" or "weight_norm"
    pruning_strategy = "attribution"
    # Choose whether to save the pruned model to a specified output directory
    save_pruned_model = True
    output_dir = "./pruned_models/llama_1B_checkpoint_1"

    model = AutoModelForCausalLM.from_pretrained(model_name)

    if pruning_strategy == "attribution":
        attribution_dataloader = prepare_data_for_pruning(
            dataset_name="codeparrot/github-code",
            model_name=model_name,
            max_length=512,
            batch_size=8,
        )
        prune_by_attribution(
            model=model,
            train_dataloader=attribution_dataloader,
            importance_threshold=1e-7,
            attribution_batch_size=2,
            num_attribution_batches=1,
            save_pruned_model=save_pruned_model,
            output_dir=output_dir,
        )
    elif pruning_strategy == "weight_norm":
        pruning_threshold = 1.5
        prune_by_weight_norm(
            model=model,
            pruning_threshold=pruning_threshold,
            save_pruned_model=save_pruned_model,
            output_dir=output_dir,
        )
    else:
        print("Invalid pruning strategy selected.")


if __name__ == "__main__":
    main()
