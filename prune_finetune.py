from transformers import TrainingArguments
import os

def main():
    args = parse_args()
    print("args: ", args)
    # ... (dataset/tokenizer/model loading code) ...

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.accumulations,
        per_device_eval_batch_size=args.batch_size,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=args.logging_steps,
        evaluation_strategy="steps" if val_dataset else "no",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        optim="adamw_torch_fused",  # FASTER OPTIMIZER
        bf16=True,  # BF16
        gradient_checkpointing=False,  # Potential memory savings
        save_total_limit=2,  # EDITS: Only keep the latest two checkpoints
        learning_rate=args.lr,
        warmup_steps=2000,
        lr_scheduler_type="linear",  # <-- This line sets a linear decay scheduler
        load_best_model_at_end=False,  # EDITS: This prevents optimizer state saving
        save_only_model=True,  # EDITS: If using newer transformers versions
    )
    # EDITS: Disable saving optimizer state to save space.
    training_args.save_optimizer_state = False

    # ... (remainder of your main() code) ... 