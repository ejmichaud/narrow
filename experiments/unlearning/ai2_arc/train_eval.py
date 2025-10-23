import os
import sys
import json
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.optim import AdamW
from tqdm import tqdm
import datasets

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="NousResearch/Llama-3.2-1B", 
                    help="Path to model (HF name or local path)")
parser.add_argument("--output_dir", type=str, default=".", 
                    help="Directory to save results")
parser.add_argument("--learning_rate", type=float, default=5e-6,
                    help="Learning rate for fine-tuning")
args = parser.parse_args()

# Setup
os.environ['HF_HOME'] = os.environ.get('SCRATCH', '/tmp') + '/iaifi_lab/Lab/ericjm/.cache/huggingface'

# Load model and tokenizer
print(f"Loading model from {args.model_path}...")
model = AutoModelForCausalLM.from_pretrained(
    args.model_path,
    torch_dtype=torch.float32,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(args.model_path)
tokenizer.pad_token = tokenizer.eos_token

# Load dataset
print("Loading dataset...")
ds = datasets.load_dataset("allenai/ai2_arc", "ARC-Easy")

# Format prompt function
def format_prompt(item):
    prompt = f"Question: {item['question']}\n"
    for label, text in zip(item['choices']['label'], item['choices']['text']):
        prompt += f"{label}. {text}\n"
    prompt += "Answer:"
    return prompt

# Evaluation function
@torch.no_grad()
def eval_accuracy(model, tokenizer, dataset):
    correct = 0
    for item in dataset:
        prompt = format_prompt(item)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        logits = model(**inputs).logits[0, -1]
        
        letter_probs = {}
        for label in item['choices']['label']:
            label_token = tokenizer.encode(" " + label, add_special_tokens=False)[0]
            letter_probs[label] = logits[label_token].item()
        
        pred = max(letter_probs, key=letter_probs.get)
        if pred == item['answerKey']:
            correct += 1
    
    return correct / len(dataset)

# Baseline evaluation
print("Baseline evaluation...")
model.eval()
baseline_acc = eval_accuracy(model, tokenizer, ds['test'])
print(f"Baseline test accuracy: {baseline_acc:.3f}")

# Clear cache
torch.cuda.empty_cache()

# Fine-tuning
print(f"Fine-tuning with lr={args.learning_rate}...")
optimizer = AdamW(model.parameters(), lr=args.learning_rate)
batch_size = 1
accumulation_steps = 4
epochs = 2  # Optimal from notebook testing

train_data = ds['train']
loss_history = []
epoch_accuracies = []

model.train()
for epoch in range(epochs):
    total_loss = 0
    ema_loss = None
    optimizer.zero_grad()
    
    pbar = tqdm(range(len(train_data)), desc=f"Epoch {epoch+1}")
    for i in pbar:
        item = train_data[i]
        
        prompt = format_prompt(item)
        full_text = prompt + " " + item['answerKey']
        
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
        
        # Save loss for curve
        loss_history.append({
            'epoch': epoch + 1,
            'step': i,
            'loss': current_loss,
            'ema_loss': ema_loss
        })
        
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        total_loss += current_loss
    
    optimizer.step()
    optimizer.zero_grad()
    epoch_loss = total_loss / len(train_data)
    print(f"Epoch {epoch+1} avg loss: {epoch_loss:.4f}")
    
    # Evaluate after each epoch
    model.eval()
    test_acc = eval_accuracy(model, tokenizer, ds['test'])
    print(f"  Test accuracy: {test_acc:.3f}")
    epoch_accuracies.append({'epoch': epoch + 1, 'test_accuracy': test_acc})
    model.train()

# Final evaluation
print("\nFinal evaluation...")
model.eval()
final_acc = eval_accuracy(model, tokenizer, ds['test'])
print(f"Final test accuracy: {final_acc:.3f}")

# Save results
results = {
    "dataset": "ai2_arc",
    "split": "ARC-Easy",
    "model_path": args.model_path,
    "baseline_accuracy": baseline_acc,
    "final_accuracy": final_acc,
    "epochs": epochs,
    "learning_rate": args.learning_rate,
    "batch_size": batch_size * accumulation_steps,
    "loss_history": loss_history,
    "epoch_accuracies": epoch_accuracies
}

os.makedirs(args.output_dir, exist_ok=True)
output_path = os.path.join(args.output_dir, "results.json")
with open(output_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to {output_path}")
print(f"Accuracy change: {baseline_acc:.3f} -> {final_acc:.3f} ({final_acc - baseline_acc:+.3f})")

