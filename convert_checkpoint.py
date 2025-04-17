import os
import json
import shutil
import torch
from safetensors.torch import load_file, save_file


def convert_pruned_checkpoint(checkpoint_dir, output_path, write_index=True):
    """
    Convert a pruned checkpoint (safetensors) by reconstructing pruned weights
    (stored as <param>_mask and <param>_orig) and saving a new safetensors file.
    """
    # Load and merge all model shards
    state_dict = {}
    shard_files = sorted(
        [
            f
            for f in os.listdir(checkpoint_dir)
            if f.startswith("model-") and f.endswith(".safetensors")
        ]
    )

    print("Loading from shards:", shard_files)
    for shard in shard_files:
        shard_path = os.path.join(checkpoint_dir, shard)
        shard_dict = load_file(shard_path)
        state_dict.update(shard_dict)

    print(f"\nLoaded {len(state_dict)} keys from the original checkpoint.")

    # Reconstruct pruned weights (i.e. keys ending in _mask and _orig)
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.endswith("_mask"):
            base_key = key[:-5]  # remove '_mask'
            orig_key = base_key + "_orig"
            if orig_key in state_dict:
                new_state_dict[base_key] = state_dict[orig_key] * value
            else:
                print(f"Warning: Found mask {key} without corresponding {orig_key}.")

    # Add non-pruned parameters
    for key, value in state_dict.items():
        if key.endswith("_mask") or key.endswith("_orig"):
            continue
        if key not in new_state_dict:
            new_state_dict[key] = value

    print(f"\nNew state dict has {len(new_state_dict)} keys. Example keys:")
    for i, key in enumerate(new_state_dict.keys()):
        if i < 5:
            print(f"  {key}")

    # Ensure that all values are torch.Tensor instances.
    for key, value in new_state_dict.items():
        if not isinstance(value, torch.Tensor):
            new_state_dict[key] = torch.tensor(value)

    # Prepare output directory and file path
    os.makedirs(output_path, exist_ok=True)
    model_path = os.path.join(output_path, "model.safetensors")
    if os.path.exists(model_path):
        os.remove(model_path)

    # Save the new state dict with explicit metadata.
    metadata = {"format": "pt"}
    save_file(new_state_dict, model_path, metadata=metadata)
    print(f"\nSaved new checkpoint to {model_path}")

    # Immediately verify the saved file by reloading it.
    try:
        reloaded = load_file(model_path)
        print(
            "Reloaded saved file successfully. Keys (first 5):",
            list(reloaded.keys())[:5],
        )
    except Exception as e:
        print("Error reloading saved file:", e)

    # Optionally write an index file. If you suspect the index file might be causing issues,
    # you can set write_index=False or delete the file before loading the model in Transformers.
    if write_index:
        weights_map = {key: "model.safetensors" for key in new_state_dict.keys()}
        index_data = {"metadata": metadata, "weight_map": weights_map}
        index_path = os.path.join(output_path, "model.safetensors.index.json")
        with open(index_path, "w") as f:
            json.dump(index_data, f, indent=2)
        print(f"Saved index file to {index_path}")
    else:
        print("Index file not written (per user setting).")

    # Copy over additional configuration files if present.
    files_to_copy = [
        "config.json",
        "generation_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ]
    for filename in files_to_copy:
        src = os.path.join(checkpoint_dir, filename)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(output_path, filename))

    print("\nSuccessfully converted checkpoint!")


if __name__ == "__main__":
    checkpoint_dir = "/afs/csail.mit.edu/u/a/asher/narrow/experiments/weightpruning1/logs/checkpoint-2000"
    output_path = "/afs/csail.mit.edu/u/a/asher/narrow/experiments/weightpruning1/logs/checkpoint-2000-converted"
    # You can try setting write_index=False if the index file appears to be problematic.
    convert_pruned_checkpoint(checkpoint_dir, output_path, write_index=True)
