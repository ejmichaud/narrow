import os
import sys
import json
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.optim import AdamW
from tqdm import tqdm
import datasets

# Setup
os.environ['HF_HOME'] = os.environ.get('SCRATCH', '/tmp') + '/iaifi_lab/Lab/ericjm/.cache/huggingface'
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="NousResearch/Llama-3.2-1B")
parser.add_argument("--output_dir", type=str, default=".")
args = parser.parse_args()

# Load dataset
print("Loading dataset...")
ds = datasets.load_dataset("cais/wmdp", "wmdp-bio")
# Create 80/20 train/test split from test set
full_data = ds['test']
split_idx = int(0.8 * len(full_data))
train_data = full_data.select(range(split_idx))
test_data = full_data.select(range(split_idx, len(full_data)))
print(f"Train: {len(train_data)}, Test: {len(test_data)}")

# Format function
def format_wmdp_prompt(item):
    prompt = f"Question: {item['question']}\n"
    choices = item['choices']
    labels = ['A', 'B', 'C', 'D']
    for label, choice_text in zip(labels[:len(choices)], choices):
        prompt += f"{label}. {choice_text}\n"
    prompt += "Answer:"
    return prompt

# Evaluation function
@torch.no_grad()
def eval_accuracy(model, tokenizer, dataset):
    correct = 0
    labels = ['A', 'B', 'C', 'D']
    for item in dataset:
        prompt = format_wmdp_prompt(item)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        logits = model(**inputs).logits[0, -1]
        
        num_choices = len(item['choices'])
        letter_probs = {}
        for i, label in enumerate(labels[:num_choices]):
            label_token = tokenizer.encode(" " + label, add_special_tokens=False)[0]
            letter_probs[label] = logits[label_token].item()
        
        pred = max(letter_probs, key=letter_probs.get)
        correct_label = labels[item['answer']]
        if pred == correct_label:
            correct += 1
    return correct / len(dataset)

# Load model
print(f"Loading model from {args.model_path}...")
model = AutoModelForCausalLM.from_pretrained(
    args.model_path,
    torch_dtype=torch.float32,
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(args.model_path)
tokenizer.pad_token = tokenizer.eos_token

# Baseline evaluation
print("Baseline evaluation...")
model.eval()
baseline_acc = eval_accuracy(model, tokenizer, test_data)
print(f"Baseline test accuracy: {baseline_acc:.3f}")

# Clear cache
torch.cuda.empty_cache()

# Fine-tuning
print("Fine-tuning...")
optimizer = AdamW(model.parameters(), lr=5e-6)
batch_size = 1
accumulation_steps = 4
epochs = 2

loss_history = []
model.train()
for epoch in range(epochs):
    total_loss = 0
    ema_loss = None
    optimizer.zero_grad()
    
    pbar = tqdm(range(len(train_data)), desc=f"Epoch {epoch+1}")
    for i in pbar:
        item = train_data[i]
        prompt = format_wmdp_prompt(item)
        full_text = prompt + " " + ['A', 'B', 'C', 'D'][item['answer']]
        
        prompt_inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        full_inputs = tokenizer(full_text, return_tensors="pt").to(model.device)
        
        labels = full_inputs.input_ids.clone()
        prompt_len = prompt_inputs.input_ids.shape[1]
        labels[0, :prompt_len] = -100
        
        outputs = model(**full_inputs, labels=labels)
        loss = outputs.loss / accumulation_steps
        loss.backward()
        
        current_loss = loss.item() * accumulation_steps
        ema_loss = current_loss if ema_loss is None else 0.9 * ema_loss + 0.1 * current_loss
        pbar.set_postfix({'loss': f'{ema_loss:.4f}'})
        
        loss_history.append({'epoch': epoch + 1, 'step': i, 'loss': current_loss, 'ema_loss': ema_loss})
        
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        total_loss += current_loss
    
    optimizer.step()
    optimizer.zero_grad()
    print(f"Epoch {epoch+1} avg loss: {total_loss / len(train_data):.4f}")

# Final evaluation
print("Final evaluation...")
model.eval()
final_acc = eval_accuracy(model, tokenizer, test_data)
print(f"Final test accuracy: {final_acc:.3f}")

# Save results
results = {
    "dataset": "wmdp-bio",
    "model_path": args.model_path,
    "baseline_accuracy": baseline_acc,
    "final_accuracy": final_acc,
    "epochs": epochs,
    "learning_rate": 5e-6,
    "batch_size": batch_size * accumulation_steps,
    "loss_history": loss_history,
    "train_size": len(train_data),
    "test_size": len(test_data)
}

os.makedirs(args.output_dir, exist_ok=True)
output_path = os.path.join(args.output_dir, "results.json")
with open(output_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to {output_path}")
print(f"Accuracy change: {baseline_acc:.3f} -> {final_acc:.3f} ({final_acc - baseline_acc:+.3f}")



