# =====================================
# The Large-CLIR Dataset
# =====================================


# 1. download the datasets files
wget https://www.cs.jhu.edu/~kevinduh/a/wikiclir2018/sasaki18.tgz

# 2. extract the datasets
tar -xvzf ./sasaki18.tgz


# =====================================
# The FastText Files
# =====================================


# 1. create the folder
mkdir fasttext

# 2. go into the said folder
cd ./fasttext

# 3. download the models
wget https://dl.fbaipublicfiles.com/fasttext/vectors-aligned/wiki.de.align.vec
wget https://dl.fbaipublicfiles.com/fasttext/vectors-aligned/wiki.fr.align.vec
wget https://dl.fbaipublicfiles.com/fasttext/vectors-aligned/wiki.tl.align.vec