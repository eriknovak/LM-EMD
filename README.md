# LM-EMD
The repository containing the experiment files for the explainable
cross-lingual document ranking model using multilingual language model 
and Regularized Earth Movers Distance.


## Requirements
Before starting the project make sure these requirements are available:
- [conda][conda]. For setting up your research environment and python dependencies.
- [git][git]. For versioning your code.
- [dvc][dvc]. For versioning your data (part of conda environment).

## Setup

### Install the conda environment

First create the new conda environment:

```bash
conda env create -f environment.yml
```

### Activate the environment

To activate the newly set environment run:

```bash
conda activate lm-emd
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

# Acknowledgments
This work is developed by [Department of Artificial Intelligence][ailab] at [Jozef Stefan Institute][ijs].

The work is supported by the following EU Horizon 2020 projects:
- [Envirolens][elens] (grant no. 761758). The project demonstrates and promotes the use of Earth observation as direct evidence for environmental law enforcement,
including in a court of law and in related contractual negotiations.
- [X5GON][x5gon] (grant no. 821918). The projects goal is to connect different Open Educational Resources (OER) providers around the globe and to provide meaningful
 AI tools to support the learning process.









[git]: https://git-scm.com/
[dvc]: https://dvc.org/
[conda]: https://docs.conda.io/en/latest/

[ailab]: http://ailab.ijs.si/
[ijs]: https://www.ijs.si/
[elens]: https://envirolens.eu/
[x5gon]: https://www.x5gon.org/
