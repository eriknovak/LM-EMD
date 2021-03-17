# Deneir Script
The template for continuous and integrated research

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
conda activate deneir-script
```

### Deactivate the environment
When the environment is not in use anymore deactivate it by running:

```bash
conda deactivate
```


[git]: https://git-scm.com/
[dvc]: https://dvc.org/
[conda]: https://docs.conda.io/en/latest/
