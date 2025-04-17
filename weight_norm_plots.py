import torch
import matplotlib.pyplot as plt
import argparse
from transformers import AutoModelForCausalLM


def load_model(checkpoint_path):
    """
    Loads a finetuned Llama model from a checkpoint path.
    This uses the Hugging Face AutoModelForCausalLM loader.
    """
    # Note: AutoModelForCausalLM.from_pretrained() can load either a local directory or a hub ID.
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path, torch_dtype=torch.float16
    )
    model.eval()  # Set model to evaluation mode
    # Optionally, move to CPU if desired: model.to("cpu")
    return model


def get_neuron_weight_norms(model):
    """
    Computes neuron weight norms for each MLP layer in the model.

    For each layer, the neuron weight norm is computed as:

        norm = gate_proj.weight.abs().sum(dim=1)
             + up_proj.weight.abs().sum(dim=1)
             + down_proj.weight.abs().sum(dim=0)

    Instead of concatenating neurons from all layers (global norms),
    this returns a dictionary mapping each layer index to the corresponding
    neuron norms (each neuron from that layer).
    """
    neuron_norms_by_layer = {}

    # Loop over layers. For Llama models the transformer layers live in model.model.layers.
    for i, layer in enumerate(model.model.layers):
        # Access the three projection layers in the MLP component
        gate_proj = layer.mlp.gate_proj
        up_proj = layer.mlp.up_proj
        down_proj = layer.mlp.down_proj

        neuron_norms_layer = (
            gate_proj.weight.abs().sum(dim=1)
            + up_proj.weight.abs().sum(dim=1)
            + down_proj.weight.abs().sum(dim=0)
        )
        # Save neuron norms for this layer as a numpy array.
        neuron_norms_by_layer[i] = neuron_norms_layer.detach().cpu().numpy()

    return neuron_norms_by_layer


def plot_norm_distribution(norms_by_layer):
    """
    Plots the histogram of neuron weight norms and prints percentiles,
    computed per layer (i.e. percentiles across neurons within each layer).

    norms_by_layer: dict mapping layer index to numpy array of neuron norms.
    """
    percentiles = [5, 10, 25, 50, 75, 90, 95]

    for layer, norms in norms_by_layer.items():
        plt.figure(figsize=(12, 8))
        plt.hist(norms, bins=100, density=True, alpha=0.75)
        plt.xlabel("Neuron Weight Norm")
        plt.ylabel("Density")
        plt.title(f"Distribution of Neuron Weight Norms for Layer {layer}")
        plt.grid(True)
        plt.show()

        norms_tensor = torch.tensor(norms, dtype=torch.float32)
        print(f"Layer {layer} neuron weight norm percentiles:")
        for p in percentiles:
            q_value = torch.quantile(norms_tensor, p / 100.0).item()
            print(f"  {p}th percentile: {q_value:.4f}")


# ----------------------------------------------------------------------
# Main execution
# ----------------------------------------------------------------------
checkpoint = "/afs/csail.mit.edu/u/a/asher/narrow/experiments/weightpruning1/logs/checkpoint-2000"
model = load_model(checkpoint)

print("Computing neuron weight norms per layer...")
norms_by_layer = get_neuron_weight_norms(model)

for layer, norms in norms_by_layer.items():
    print(f"Layer {layer}: Computed neuron weight norms for {len(norms)} neurons.")

plot_norm_distribution(norms_by_layer)
