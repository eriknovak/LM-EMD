# LM-EMD
The repository containing the experiment files for the interpretable
cross-lingual document ranking model using multilingual language model
and Regularized Earth Mover's Distance.


## ✔️ Requirements
Before starting the project make sure these requirements are available:
- [conda][conda]. For setting up your research environment and python dependencies.
- [git][git]. For versioning your code.
- [dvc][dvc]. For versioning your data (part of conda environment).

## 💻 Setup

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
sure you install the CUDA version that is supported by your machine.

```bash
conda install pytorch cudatoolkit=11.1 -c pytorch -c conda-forge
```

### Deactivate the environment

When the environment is not in use anymore deactivate it by running:

```bash
conda deactivate
```

## 💾 Data Preparation

First download the data. Go into the `/data` folder and run

```bash
sh ./download.sh
```

This will download the data used in the experiments.

## 🥼 Experiment Setup

**NOTE:** Training a single model requires approximate 10 GB of GPU space.

To run the experiments one can manually change the `params.yaml` file with
different parameters. Then, simply run the following commands:

```bash
# trains the model with the provided parameters
python src/train.py data/sasaki18 data/model.pth data/losses
# evaluates the model
python src/evaluate.py data/sasaki18 data/model.pth data/scores
```

### Using DVC

We use DVC to automatically run experiments with different parameters. The `dvc`
is installed with `conda`. To run multiple experiments we execute the following
command:

```bash
# prepare the queue of experiments using pairwise_ranking
dvc exp run --queue -S model.ranking=cls -S model.reg=None -S model.nit=None
dvc exp run --queue -S model.ranking=max -S model.reg=None -S model.nit=None
dvc exp run --queue -S model.ranking=mean -S model.reg=None -S model.nit=None
dvc exp run --queue -S model.ranking=emd -S model.reg=0.1
dvc exp run --queue -S model.ranking=emd -S model.reg=1
dvc exp run --queue -S model.ranking=emd -S model.reg=10

# execute all queued experiments (run 3 jobs in parallel)
dvc exp run --run-all --jobs 3
```

To train the models using cross entropy:
```bash
# to train the models using cross-entropy
dvc exp run --queue -S train.loss=cross_entropy -S model.ranking=cls -S model.reg=None -S model.nit=None
dvc exp run --queue -S train.loss=cross_entropy -S model.ranking=max -S model.reg=None -S model.nit=None
dvc exp run --queue -S train.loss=cross_entropy -S model.ranking=mean -S model.reg=None -S model.nit=None
dvc exp run --queue -S train.loss=cross_entropy -S model.ranking=emd -S model.reg=0.1
dvc exp run --queue -S train.loss=cross_entropy -S model.ranking=emd -S model.reg=1
dvc exp run --queue -S train.loss=cross_entropy -S model.ranking=emd -S model.reg=10

# execute all queued experiments (run 3 jobs in parallel)
dvc exp run --run-all --jobs 3
```

Afterwards, we can compare the performance of the models by running:

```bash
dvc exp show
```

To save the best performance parameters run:

```bash
# [exp-id] is the ID of the experiment that yielded the best performance
dvc exp apply [exp-id]
```


## 📋 Experiment Results


| Model    	| EN → DE     | EN → FR     | EN → TL     | EN → JA     | EN → SW     |
|-----------|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|
| BERT-CLS  | .978 / .987 | .978 / .987 | .851 / .912 | .955 / .973 | .913 / .947 |
| BERT-MAX  | .941 / .964 | .948 / .969 | .798 / .874 | .912 / .946 | .824 / .886 |
| BERT-MEAN | .967 / .980 | .958 / .976 | .786 / .874 | .941 / .965 | .835 / .897 |
| LM-EMD   	| .977 / .986 | .974 / .985 | .801 / .874 | .955 / .973 | .890 / .932 |

*Table 1. CLIR performance of the models. The scores are formatted as P@1/MAP scores.*



| Params  | Loss   | EN → DE     | EN → FR     | EN → TL     | EN → JA     | EN → SW     |
|---------|--------|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|
| γ = 0.1 | CE     | .876 / .927 | .843 / .909 | .674 / .793 | .846 / .908 | .631 / .754 |
|         | PR     | .977 / .986 | .974 / .985 | .801 / .874 | .955 / .973 | .890 / .932 |
| γ = 1   | CE     | .876 / .927 | .846 / .910 | .669 / .790 | .846 / .907 | .617 / .747 |
|         | PR     | .970 / .982 | .968 / .981 | .809 / .883 | .910 / .946 | .859 / .913 |
| γ = 10  | CE     | .878 / .928 | .846 / .911 | .671 / .792 | .848 / .909 | .628 / .753 |
|         | PR     | .965 / .979 | .961 / .978 | .805 / .881 | .899 / .941 | .835 / .899 |

*Table 2. The performance comparison of the LM-EMD model trained with different regularization
factor (γ) values, and using the cross-entropy (CE) and pairwise ranking (PR) loss functions
during training. The scores are formatted as P@1/MAP scores.*

## 🔎 Interpretability

The LM-EMD has one advantage over the rest of the evaluated models: interpetability.
Not only does Earth Mover's Distance return the final relevance score of the document,
it also returns the so called **transportation matrix** which shows which terms in the
document match the terms in the query; giving an idea from where the document
scores come from.

The transportation matrix is generated using the Sinkhorn algorihm by using the pre-generated
cost matrix containing the cosine distances between the query and document terms.
The transportation matrix tells us which terms in the query and document are closest, e.g.
have the biggest influence on the document score.

Here are the cost and transportation matrices of the interpretability example found in the
paper.
![president-usa](./data/interpretability/president-usa.png)


### Generating other Examples

Once the model is trained one can modify the `batch` values in [./src/interpret.py][interpret].
To visualize the interpertation graphs simply run the following command:

```bash
python src/interpret.py data/model.pth data/interpretability/{new-image-name}.png
```

where `{new-image-name}` is the name of the image. This will generate an image in the
[./data/interpretability][interdata] folder.

# 🏬 Acknowledgments
This work is developed by [Department of Artificial Intelligence][ailab] at [Jozef Stefan Institute][ijs].

The work is supported by the Slovenian Research Agency and the EU Horizon 2020 project [Humane AI NET][humaneai] (H2020-ICT-952026).









[git]: https://git-scm.com/
[dvc]: https://dvc.org/
[conda]: https://docs.conda.io/en/latest/

[interpret]: ./src/interpret.py
[interdata]: ./data/interpretability

[ailab]: http://ailab.ijs.si/
[ijs]: https://www.ijs.si/

[humaneai]: https://www.humane-ai.eu/
