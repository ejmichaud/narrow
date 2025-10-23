# Unlearning Experiments

Fine-tuning experiments to evaluate model performance on downstream tasks.

## Structure

```
unlearning/
├── counterfact/
│   └── train_eval.py                    # CounterFact dataset fine-tuning
├── ai2_arc/
│   └── train_eval.py                    # AI2 ARC-Easy dataset fine-tuning
├── slurm/
│   ├── base_counterfact.slurm           # Template for base model
│   ├── base_ai2_arc.slurm               # Template for base model
│   ├── tuneprune_counterfact.slurm      # Template for tuneprune
│   ├── tuneprune_ai2_arc.slurm          # Template for tuneprune
│   ├── generate_and_submit.sh           # Submit all experiments
│   ├── submit_random_pruned.sh          # Submit only random pruned experiments
│   └── generate_and_submit_lr_grid.sh   # Learning rate grid search
├── create_random_pruned_models.py       # Create randomly pruned models
├── results/                             # Results saved here
├── view_results.py                      # View/compare results
└── analyze_lr_grid.py                   # Analyze learning rate grid search
```

## Usage

### Create random pruned models (optional):
```bash
python create_random_pruned_models.py
```

This creates randomly pruned models at 30%, 63%, and 80% neuron sparsity levels.
Models are saved to `$SCRATCH/iaifi_lab/Lab/ericjm/narrow/random_pruned/`.

### Submit all experiments to SLURM (recommended):
```bash
cd slurm
./generate_and_submit.sh
```

This will:
1. Find all tuneprune models (all lambda values) at checkpoint-70000
2. Find all random pruned models (if they exist)
3. Generate SLURM job scripts for each model x dataset combination
4. Submit all jobs in parallel

Typically submits:
- ~8 jobs (base + 3 lambda values x 2 datasets each)
- +6 jobs if random pruned models exist (3 sparsity levels x 2 datasets)

### Submit only random pruned model experiments:
```bash
cd slurm
./submit_random_pruned.sh
```

This submits jobs only for random pruned models (6 jobs: 3 sparsity levels x 2 datasets).
Useful if you've already run base/lambda experiments and only want to add random pruning results.

### Learning Rate Grid Search:
```bash
cd slurm
./generate_and_submit_lr_grid.sh
```

This will:
1. Run experiments with multiple learning rates for each model
2. CounterFact LRs: [1e-6, 5e-6, 1e-5, 5e-5, 1e-4]
3. AI2-ARC LRs: [1e-6, 5e-6, 1e-5, 2e-5, 5e-5]
4. Results saved to `results/{model}_lr_{lr}/{dataset}/`

Analyze grid search results:
```bash
python analyze_lr_grid.py results/
```

### Run individual experiments:
```bash
# With default base model and default learning rate
python counterfact/train_eval.py

# With custom model and learning rate
python counterfact/train_eval.py --model_path /path/to/model --output_dir results/custom --learning_rate 1e-5

# AI2-ARC with custom learning rate
python ai2_arc/train_eval.py --learning_rate 5e-6
```

### View results:
```bash
python view_results.py results/
```

## Datasets

- **counterfact**: Factual knowledge editing dataset (single-token answers)
  - 3 epochs, default lr=1e-5, batch_size=4
  - Grid search LRs: [1e-6, 5e-6, 1e-5, 5e-5, 1e-4]
- **ai2_arc**: ARC-Easy multiple choice questions (letter prediction)
  - 2 epochs, default lr=5e-6, batch_size=4
  - Grid search LRs: [1e-6, 5e-6, 1e-5, 2e-5, 5e-5]

## Model Types

- **base_model**: Unmodified NousResearch/Llama-3.2-1B
- **lambda_X_bs_Y_acc_Z**: Models from tuneprune15-redo with L1 regularization (lambda=X)
- **sparsity_X**: Randomly pruned models with X% of neurons zeroed out
  - sparsity_0.3: 30% random neuron pruning
  - sparsity_0.63: 63% random neuron pruning  
  - sparsity_0.8: 80% random neuron pruning
  - Created with `create_random_pruned_models.py`

## Results

Each experiment saves `results.json` with:
- Baseline accuracy (before training)
- Final accuracy (after training)
- Training hyperparameters
- Loss history (for debugging)
- Epoch accuracies (AI2-ARC only)

## SLURM Configuration

Each job uses:
- Partition: `gpu`
- Memory: `32GB`
- GPU: `nvidia_a100-sxm4-80gb:1`
- Time: `120` minutes

