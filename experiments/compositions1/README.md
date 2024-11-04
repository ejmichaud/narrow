Now that we've established there's a strong curriculum effect on the compositional multitask sparse parity dataset, let's study fine-tuning for sparsity. We'll first pretrain a model with multiple compositional subtasks. Then some experiments:
- L1 regularization finetuning on the weights (this works, see `parity-compositions (l1).ipynb`) but with different regularization strengths. Let the fine-tuning dataset be half (out of six) of the compositional subtasks. As we increase regularization strength, will the model eventually sacrifice performance on some of these subtasks in favor of sparsity? In the Pareto frontier, are there sharp steps as the model drops subtasks. There might be 3 steps, but possibly much more.
- L1 regularization finetuning with a finite amount of fine-tuning data -- multi-epoch training. With too low L1 regularization, the model might memorize, but then I assume that with high enough L1 regularization, the model will be forced to generalize like in grokking. 


