#!/usr/bin/env python3
"""
Evaluate model on downstream tasks with learning rate sweep.
"""

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

# Import narrow module for VariableSize models
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))


def load_counterfact_data():
    """Load CounterFact dataset."""
    ds = datasets.load_dataset("azhx/counterfact")
    train_prompts = [item['prompt'].format(item['subject']) for item in ds['train']['requested_rewrite']]
    train_answers = [" " + item['target_true']['str'] for item in ds['train']['requested_rewrite']]
    test_prompts = [item['prompt'].format(item['subject']) for item in ds['test']['requested_rewrite']]
    test_answers = [" " + item['target_true']['str'] for item in ds['test']['requested_rewrite']]
    return train_prompts, train_answers, test_prompts, test_answers


def load_ai2_arc_data():
    """Load AI2-ARC dataset."""
    ds = datasets.load_dataset("allenai/ai2_arc", "ARC-Easy")
    return ds


def load_wmdp_data(subset):
    """Load WMDP dataset (bio/cyber/chem)."""
    ds = datasets.load_dataset("cais/wmdp", f"wmdp-{subset}")
    # Create 80/20 train/test split
    full_data = ds['test']
    split_idx = int(0.8 * len(full_data))
    train_data = full_data.select(range(split_idx))
    test_data = full_data.select(range(split_idx, len(full_data)))
    return train_data, test_data


def format_arc_prompt(item):
    """Format AI2-ARC prompt."""
    prompt = f"Question: {item['question']}\n"
    for label, text in zip(item['choices']['label'], item['choices']['text']):
        prompt += f"{label}. {text}\n"
    prompt += "Answer:"
    return prompt


def format_wmdp_prompt(item):
    """Format WMDP prompt."""
    prompt = f"Question: {item['question']}\n"
    choices = item['choices']
    labels = ['A', 'B', 'C', 'D']
    for label, choice_text in zip(labels[:len(choices)], choices):
        prompt += f"{label}. {choice_text}\n"
    prompt += "Answer:"
    return prompt


@torch.no_grad()
def eval_counterfact_accuracy(model, tokenizer, prompts, answers):
    """Evaluate CounterFact next-token accuracy."""
    correct = 0
    for prompt, answer in zip(prompts, answers):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        logits = model(**inputs).logits[0, -1]
        pred_token = tokenizer.decode(logits.argmax())
        answer_first_token = tokenizer.decode(tokenizer(answer, add_special_tokens=False).input_ids[0])
        if pred_token == answer_first_token:
            correct += 1
    return correct / len(prompts)


@torch.no_grad()
def eval_arc_accuracy(model, tokenizer, dataset):
    """Evaluate AI2-ARC accuracy."""
    correct = 0
    for item in dataset:
        prompt = format_arc_prompt(item)
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


@torch.no_grad()
def eval_wmdp_accuracy(model, tokenizer, dataset):
    """Evaluate WMDP accuracy."""
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


def finetune_counterfact(model, tokenizer, train_prompts, train_answers, lr, epochs=3):
    """Fine-tune on CounterFact with given learning rate."""
    optimizer = AdamW(model.parameters(), lr=lr)
    batch_size = 1
    accumulation_steps = 4
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        for i in range(0, len(train_prompts), batch_size):
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
            
            if (i // batch_size + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
        
        optimizer.step()
        optimizer.zero_grad()


def finetune_arc(model, tokenizer, train_data, lr, epochs=2):
    """Fine-tune on AI2-ARC with given learning rate."""
    optimizer = AdamW(model.parameters(), lr=lr)
    batch_size = 1
    accumulation_steps = 4
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        for i in range(len(train_data)):
            item = train_data[i]
            
            prompt = format_arc_prompt(item)
            full_text = prompt + " " + item['answerKey']
            
            prompt_inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            full_inputs = tokenizer(full_text, return_tensors="pt").to(model.device)
            
            labels = full_inputs.input_ids.clone()
            prompt_len = prompt_inputs.input_ids.shape[1]
            labels[0, :prompt_len] = -100
            
            outputs = model(**full_inputs, labels=labels)
            loss = outputs.loss / accumulation_steps
            loss.backward()
            
            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
        
        optimizer.step()
        optimizer.zero_grad()


def finetune_wmdp(model, tokenizer, train_data, lr, epochs=2):
    """Fine-tune on WMDP with given learning rate."""
    optimizer = AdamW(model.parameters(), lr=lr)
    batch_size = 1
    accumulation_steps = 4
    labels = ['A', 'B', 'C', 'D']
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        for i in range(len(train_data)):
            item = train_data[i]
            
            prompt = format_wmdp_prompt(item)
            full_text = prompt + " " + labels[item['answer']]
            
            prompt_inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            full_inputs = tokenizer(full_text, return_tensors="pt").to(model.device)
            
            labels_tensor = full_inputs.input_ids.clone()
            prompt_len = prompt_inputs.input_ids.shape[1]
            labels_tensor[0, :prompt_len] = -100
            
            outputs = model(**full_inputs, labels=labels_tensor)
            loss = outputs.loss / accumulation_steps
            loss.backward()
            
            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
        
        optimizer.step()
        optimizer.zero_grad()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True, 
                        choices=["counterfact", "ai2_arc", "wmdp_bio", "wmdp_cyber", "wmdp_chem"])
    parser.add_argument("--lr_sweep", type=float, nargs="+", required=True)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"{'='*80}")
    print(f"Downstream Evaluation with LR Sweep")
    print(f"{'='*80}")
    print(f"Model: {args.model_path}")
    print(f"Dataset: {args.dataset}")
    print(f"LR sweep: {args.lr_sweep}")
    print(f"{'='*80}\n")
    
    # Load model and tokenizer
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float32,
        device_map="auto",
        trust_remote_code=True  # For VariableSizeLlama
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataset
    if args.dataset == "counterfact":
        train_prompts, train_answers, test_prompts, test_answers = load_counterfact_data()
        
        # Baseline
        print("Baseline evaluation...")
        model.eval()
        baseline_acc = eval_counterfact_accuracy(model, tokenizer, test_prompts, test_answers)
        print(f"Baseline: {baseline_acc:.3f}\n")
        
        # LR sweep
        lr_results = {}
        for lr in args.lr_sweep:
            print(f"Fine-tuning with lr={lr}...")
            
            # Reload model for each LR
            model = AutoModelForCausalLM.from_pretrained(
                args.model_path,
                torch_dtype=torch.float32,
                device_map="auto",
                trust_remote_code=True
            )
            
            finetune_counterfact(model, tokenizer, train_prompts, train_answers, lr)
            
            model.eval()
            final_acc = eval_counterfact_accuracy(model, tokenizer, test_prompts, test_answers)
            print(f"  Final: {final_acc:.3f}")
            
            lr_results[lr] = final_acc
            
            del model
            torch.cuda.empty_cache()
        
        # Find best LR
        best_lr = max(lr_results, key=lr_results.get)
        best_acc = lr_results[best_lr]
        
        results = {
            "dataset": "counterfact",
            "model_path": args.model_path,
            "baseline_accuracy": baseline_acc,
            "lr_sweep_results": {str(lr): acc for lr, acc in lr_results.items()},
            "best_lr": best_lr,
            "best_accuracy": best_acc,
        }
        
    elif args.dataset == "ai2_arc":
        ds = load_ai2_arc_data()
        
        # Baseline
        print("Baseline evaluation...")
        model.eval()
        baseline_acc = eval_arc_accuracy(model, tokenizer, ds['test'])
        print(f"Baseline: {baseline_acc:.3f}\n")
        
        # LR sweep
        lr_results = {}
        for lr in args.lr_sweep:
            print(f"Fine-tuning with lr={lr}...")
            
            # Reload model
            model = AutoModelForCausalLM.from_pretrained(
                args.model_path,
                torch_dtype=torch.float32,
                device_map="auto",
                trust_remote_code=True
            )
            
            finetune_arc(model, tokenizer, ds['train'], lr)
            
            model.eval()
            final_acc = eval_arc_accuracy(model, tokenizer, ds['test'])
            print(f"  Final: {final_acc:.3f}")
            
            lr_results[lr] = final_acc
            
            del model
            torch.cuda.empty_cache()
        
        # Find best LR
        best_lr = max(lr_results, key=lr_results.get)
        best_acc = lr_results[best_lr]
        
        results = {
            "dataset": "ai2_arc",
            "model_path": args.model_path,
            "baseline_accuracy": baseline_acc,
            "lr_sweep_results": {str(lr): acc for lr, acc in lr_results.items()},
            "best_lr": best_lr,
            "best_accuracy": best_acc,
        }
    
    else:  # wmdp_bio, wmdp_cyber, or wmdp_chem
        subset = args.dataset.split('_')[1]  # Extract bio/cyber/chem
        train_data, test_data = load_wmdp_data(subset)
        
        # Baseline
        print("Baseline evaluation...")
        model.eval()
        baseline_acc = eval_wmdp_accuracy(model, tokenizer, test_data)
        print(f"Baseline: {baseline_acc:.3f}\n")
        
        # LR sweep
        lr_results = {}
        for lr in args.lr_sweep:
            print(f"Fine-tuning with lr={lr}...")
            
            # Reload model
            model = AutoModelForCausalLM.from_pretrained(
                args.model_path,
                torch_dtype=torch.float32,
                device_map="auto",
                trust_remote_code=True
            )
            
            finetune_wmdp(model, tokenizer, train_data, lr)
            
            model.eval()
            final_acc = eval_wmdp_accuracy(model, tokenizer, test_data)
            print(f"  Final: {final_acc:.3f}")
            
            lr_results[lr] = final_acc
            
            del model
            torch.cuda.empty_cache()
        
        # Find best LR
        best_lr = max(lr_results, key=lr_results.get)
        best_acc = lr_results[best_lr]
        
        results = {
            "dataset": args.dataset,
            "model_path": args.model_path,
            "baseline_accuracy": baseline_acc,
            "lr_sweep_results": {str(lr): acc for lr, acc in lr_results.items()},
            "best_lr": best_lr,
            "best_accuracy": best_acc,
        }
    
    # Save results
    output_file = os.path.join(args.output_dir, "results.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"Results saved to: {output_file}")
    print(f"Baseline: {baseline_acc:.3f}")
    print(f"Best LR: {best_lr}")
    print(f"Best Accuracy: {best_acc:.3f}")
    print(f"Improvement: {best_acc - baseline_acc:+.3f}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()





