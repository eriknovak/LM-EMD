import os
import sys
import json
import yaml
import random
from tqdm import tqdm


# =====================================
# Import Inputs and Training Parameters
# =====================================


data_folder = sys.argv[1]
model_file = sys.argv[2]
SCORES_DIR = sys.argv[3]

params = yaml.safe_load(open("params.yaml"))


# =====================================
# Import Training Data
# =====================================


from library.data_loader import get_test_datasets

test_data = {}

data_folders = os.listdir(data_folder)
for folder in tqdm(data_folders, desc="Preparing Data"):
    test_data[folder] = get_test_datasets(folder, max_examples=100)


# =====================================
# Define the Model
# =====================================


import torch

# import transformers
from transformers import BertTokenizer

# import the NEW method
from library.bert_model import BERT

# set the device on which we will train the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# get the bert tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

# load the NEW method used for CLIR
ranking_type = params["model"]["ranking"]
reg = params["model"]["reg"]
nit = params["model"]["nit"]
model = BERT(type=ranking_type, reg=reg, nit=nit)
model.load_state_dict(torch.load(model_file))
model = model.eval().to(device)


# =====================================
# Prepare Evaluation Functions
# =====================================


def get_document_scores(batch):
    """Get the document scores

    Args:
        batch.query: The query texts.
        batch.documents: The documents text.

    Returns:
        outputs: The outputs of the model.
            outputs[0]: The distances scores.
            outputs[1]: The cost matrix.
            outputs[2]: The transportation matrix.

    """
    q_inputs = tokenizer(
        batch["query"], truncation=True, padding=True, return_tensors="pt"
    )
    d_inputs = tokenizer(
        batch["documents"], truncation=True, padding=True, return_tensors="pt"
    )
    # get the input batches
    examples = {
        "q_input_ids": q_inputs["input_ids"],
        "q_attention_mask": q_inputs["attention_mask"],
        "d_input_ids": d_inputs["input_ids"],
        "d_attention_mask": d_inputs["attention_mask"],
    }
    # move the batch tensors to the same device as the model
    examples = {k: v.to(device) for k, v in examples.items()}
    outputs = model(**examples)
    return outputs


def get_average_precision_at_k(batch):
    # get parameters for calculation
    labels = torch.Tensor(batch["relevance"])
    # get the loss values
    with torch.no_grad():
        outputs = get_document_scores(batch)
        # sort the instances
        distances = outputs[0].detach().cpu()
        # delete the outputs
        del outputs
    # Make sure deallocation has taken place
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # get the sort indices
    sort_indices = distances.argsort()
    # sort the labels values based on the similarity order
    RelAtK = labels[sort_indices]
    # get the cummulative sum over the whole labels list
    cum_labels = RelAtK.cumsum(dim=0)
    # calculate the precision at k value over the whole list
    PatK = cum_labels * torch.Tensor([1 / (i + 1) for i in range(labels.shape[0])])
    # Group Truth Positives
    GTP = RelAtK.sum()
    # Average Precision for given query
    AveP = 1 / GTP * (RelAtK * PatK).sum()
    return {"P@1": PatK[0].item(), "AveP": AveP.item()}


def evaluate_dataset(dataset_key: str):
    values = []
    dataset = test_data[dataset_key]
    # set a placeholder for evaluation results
    for example in tqdm(dataset, desc=dataset_key):
        query = example["query"]
        documents = example["documents"]
        relevance = example["relevance"]
        # sort the documents and relevance labels
        index_shuffle = list(range(len(documents)))
        random.shuffle(index_shuffle)
        documents = [documents[index_shuffle[i]] for i in range(len(documents))]
        relevance = [relevance[index_shuffle[i]] for i in range(len(relevance))]
        # calculate the performance of the dataset
        performance = get_average_precision_at_k(
            {"query": query, "documents": documents, "relevance": relevance,}
        )
        values.append(performance)

    # return the evaluation results
    return {
        "P@1": sum([v["P@1"] for v in values]) / len(values),
        "MAP": sum([v["AveP"] for v in values]) / len(values),
    }


# =====================================
# Execute the Evaluation Process
# =====================================

scores = {}
for key in test_data.keys():
    scores[key] = {
        "model": ranking_type,
        "scores": None,
    }
    scores[key]["scores"] = evaluate_dataset(key)

# create the scores folder
if not os.path.exists(SCORES_DIR):
    os.makedirs(SCORES_DIR)

# save all scores
with open(f"{SCORES_DIR}/scores.json", "w") as f:
    json.dump(scores, f, indent=4)

# save the scores in individual files
for key, score in scores.items():
    with open(f"{SCORES_DIR}/{key}.json", "w") as f:
        json.dump(score["scores"], f, indent=4)
