import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from sfttrainer import SFTPruner
from huggingface_hub import login
from dotenv import load_dotenv
import sys
from transformers import AutoModelForCausalLM

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pruning import pruning_loss

import torch.nn.functional as F
import torch

load_dotenv()
hf_token = os.getenv("HF_TOKEN")
login(token=hf_token)


dataset = load_dataset("code_search_net", "python", split="train[:100]")
dataset = dataset.rename_column("whole_func_string", "text")


model_name = "NousResearch/Llama-3.2-1B"

# Load the Llama model
model = AutoModelForCausalLM.from_pretrained(model_name)

# Print the model architecture
print(model)

training_args = SFTConfig(
    max_seq_length=512,
    output_dir="/tmp",
)


def custom_loss(outputs, labels, num_items_in_batch):
    print(f"num_items_in_batch: {num_items_in_batch}")
    print(f"outputs shape: {outputs.logits.shape}")
    print(f"labels shape: {labels.shape}")
    breakpoint()
    logits = outputs.logits.permute(0, 2, 1)
    ce_loss = F.cross_entropy(logits, labels)
    penalty = pruning_loss(model, penalty_type="tied_l2_with_lhalf")
    total_loss = ce_loss + 0.5 * penalty / penalty.detach()
    return total_loss


trainer = SFTPruner(
    model_name,
    train_dataset=dataset,
    args=training_args,
    # compute_loss_func=custom_loss,
)

trainer.train()
