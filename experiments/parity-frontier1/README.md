We want to run an experiment comparing tune-pruning with training from scratch. We'll need to do the following experiments:
1) Train a general model across all tasks
2) Perform prune-tuning on the general model to distill it on the compositional subtask
3) Train models from scratch on the compositional subtask, of varying size.

For step (2), the default approach would be to do what Asher did and take multiple checkpoints during a single tune-pruning run. So periodically during training, we prune the network as aggressively as we can while ensuring that ~100% test accuracy is preserved.

For step (3), we'll just need a script that trains models from scratch on only the compositional subtask.

For these experiments, I think I'll use atomic subtasks with k=3, and then the compositional subtasks will be composed from 3 atomic subtasks, for an effective k=9. I hope this will still be learnable by the models from scratch in reasonable time.


NOTE: As described above, we previously used a subtask structure with k=3 per subtask and the compositional subtask had 3 atomic subtasks, for a total of k=9. However, the networks trained from scratch could not learn this k=9, even after 10M training steps. So now, I'll repeat everything but for k=3 per atomic subtask but only 2 atomic subtasks in the combined one.


