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
parser.add_argument("--learning_rate", type=float, default=1e-5,
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
ds = datasets.load_dataset("azhx/counterfact")
train_prompts = [item['prompt'].format(item['subject']) for item in ds['train']['requested_rewrite']]
train_answers = [" " + item['target_true']['str'] for item in ds['train']['requested_rewrite']]
test_prompts = [item['prompt'].format(item['subject']) for item in ds['test']['requested_rewrite']]
test_answers = [" " + item['target_true']['str'] for item in ds['test']['requested_rewrite']]

# Evaluation function
@torch.no_grad()
def eval_next_token_accuracy(model, tokenizer, prompts, answers):
    correct = 0
    for prompt, answer in zip(prompts, answers):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        logits = model(**inputs).logits[0, -1]
        pred_token = tokenizer.decode(logits.argmax())
        answer_first_token = tokenizer.decode(tokenizer(answer, add_special_tokens=False).input_ids[0])
        if pred_token == answer_first_token:
            correct += 1
    return correct / len(prompts)

# Baseline evaluation
print("Baseline evaluation...")
model.eval()
baseline_acc = eval_next_token_accuracy(model, tokenizer, test_prompts, test_answers)
print(f"Baseline test accuracy: {baseline_acc:.3f}")

# Clear cache
torch.cuda.empty_cache()

# Fine-tuning
print(f"Fine-tuning with lr={args.learning_rate}...")
optimizer = AdamW(model.parameters(), lr=args.learning_rate)
batch_size = 1
accumulation_steps = 4
epochs = 3

loss_history = []

model.train()
for epoch in range(epochs):
    total_loss = 0
    ema_loss = None
    optimizer.zero_grad()
    
    pbar = tqdm(range(0, len(train_prompts), batch_size), desc=f"Epoch {epoch+1}")
    for i in pbar:
        batch_prompts = train_prompts[i:i+batch_size]
        batch_answers = train_answers[i:i+batch_size]
        
        prompt_inputs = tokenizer(batch_prompts, padding=True, truncation=True, return_tensors="pt").to(model.device)
        full_texts = [p + a for p, a in zip(batch_prompts, batch_answers)]
        full_inputs = tokenizer(full_texts, padding=True, truncation=True, return_tensors="pt").to(model.device)
        
        labels = full_inputs.input_ids.clone()
        for j in range(len(batch_prompts)):
            prompt_len = (prompt_inputs.input_ids[j] != tokenizer.pad_token_id).sum()
            labels[j, :prompt_len] = -100
        
        outputs = model(**full_inputs, labels=labels)
        loss = outputs.loss / accumulation_steps
        loss.backward()
        
        current_loss = loss.item() * accumulation_steps
        ema_loss = current_loss if ema_loss is None else 0.9 * ema_loss + 0.1 * current_loss
        pbar.set_postfix({'loss': f'{ema_loss:.4f}'})
        
        # Save loss for curve
        loss_history.append({
            'epoch': epoch + 1,
            'step': i // batch_size,
            'loss': current_loss,
            'ema_loss': ema_loss
        })
        
        if (i // batch_size + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        total_loss += current_loss
    
    optimizer.step()
    optimizer.zero_grad()
    print(f"Epoch {epoch+1} avg loss: {total_loss / len(train_prompts):.4f}")

# Final evaluation
print("Final evaluation...")
model.eval()
final_acc = eval_next_token_accuracy(model, tokenizer, test_prompts, test_answers)
print(f"Final test accuracy: {final_acc:.3f}")

# Save results
results = {
    "dataset": "counterfact",
    "model_path": args.model_path,
    "baseline_accuracy": baseline_acc,
    "final_accuracy": final_acc,
    "epochs": epochs,
    "learning_rate": args.learning_rate,
    "batch_size": batch_size * accumulation_steps,
    "loss_history": loss_history
}

os.makedirs(args.output_dir, exist_ok=True)
output_path = os.path.join(args.output_dir, "results.json")
with open(output_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to {output_path}")
print(f"Accuracy change: {baseline_acc:.3f} -> {final_acc:.3f} ({final_acc - baseline_acc:+.3f})")

