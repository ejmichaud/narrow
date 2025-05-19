# Explorations in making general models narrow.

This respository accompanies the paper "On the creation of narrow AI: hierarchy and nonlocality of neural network skills" by Eric Michaud, Asher Parker-Sartori, and Max Tegmark. We provide instructions for replicating each of our figures below:

**Figure 2** (compositional-parity.pdf): This figure is created in `notebooks/parity-compositions5 (figure).ipynb`. This notebook loads data created by the runs in `experiments/compositions4`. Each run was of the `experiments/compositions4/train.py` script, called by each of four the slurm job array `.sh` scripts in `experiments/compositions4/`.

**Figure 7**: (compositional-parity-depth-comparisons.pdf): This figure is also created in `notebooks/parity-compositions5 (figure).ipynb`, using data from `experiments/compositions4`.

**Figure 3**: (cmspnetworkstructureandpruningresultsmain-labeled.png): This figure is created in `notebooks/parity-nonlocality-pruning.ipynb`. This notebook loads data created by the runs in `experiments/compositions3`. The script for these runs is `experiments/compositions3/trainprunesave.py` and the grid search over widths and seeds is defined in `experiments/compositions3/run.sh`.

**Figure 10**: (pruningresultspretrainedacrosswidthandseed.pdf): This figure is also created in `notebooks/parity-nonlocality-pruning.ipynb`, using data from `experiments/compositions3`.

**Figure 8** (ablationscoresk3m3composite2bothlayers.pdf) and **Figure 9** (ablationscoresk3m3composite2bothlayersimgshow.pdf) are both created in `notebooks/parity-nonlocality-compositional.ipynb`.




