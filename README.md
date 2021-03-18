# BERT EMD
The repository containing the experiment files for the explainable
document ranking model using multilingual BERT and Regularized Earth
Movers Distance.

## Requirements
Before starting the project make sure these requirements are available:
- [conda][conda]. For setting up your research environment and python dependencies.
- [git][git]. For versioning your code.
- [dvc][dvc]. For versioning your data (part of conda environment).

## Setup

**NOTE:** For each new project it is advisable to change the environment name
and to make sure that the required modules are in the `environment.yml` file.

### Install the conda environment

First create the new conda environment:

```bash
conda env create -f environment.yml
```

### Activate the environment

To activate the newly set environment run:

```bash
conda activate bert-emd
```

### Install the CUDA version of PyTorch

Use conda to install the appropriate version of PyTorch. **NOTE:** Be
sure you install the right CUDA version.

```bash
conda install pytorch cudatoolkit=11.1 -c pytorch -c conda-forge
```

### Deactivate the environment

When the environment is not in use anymore deactivate it by running:

```bash
conda deactivate
```

## Data Preparation

First download the data. Go into the `/data` folder and run

```bash
sh ./download.sh
```

This will download the data used in the experiments.

## Experiment Setup

To run the experiments one can manually change the `params.yaml` file with
different parameters. Then, simply run the following commands:

```bash
# trains the model with the provided parameters
python src/train.py
# evaluates the model
python src/evaluate.py
```

### Using DVC

We use DVC to automatically run experiments with different parameters. The `dvc`
is installed with `conda`. To run multiple experiments we execute the following
command:

```bash
# prepare the queue of experiments
dvc exp run --queue -S model.ranking=cls -S model.reg=None -S model.nit=None
dvc exp run --queue -S model.ranking=max -S model.reg=None -S model.nit=None
dvc exp run --queue -S model.ranking=mean -S model.reg=None -S model.nit=None
dvc exp run --queue -S model.ranking=emd -S model.reg=0.1
dvc exp run --queue -S model.ranking=emd -S model.reg=1
dvc exp run --queue -S model.ranking=emd -S model.reg=10

# execute all queued experiments (run 3 jobs in parallel)
dvc exp run --run-all --jobs 3
```

Afterwards, we can compare the performance of the models by running:

```bash
dvc exp show
```









[git]: https://git-scm.com/
[dvc]: https://dvc.org/
[conda]: https://docs.conda.io/en/latest/
