import sys
import yaml


# =====================================
# Import Inputs and Training Parameters
# =====================================


model_file = sys.argv[1]
image_file = sys.argv[2]

params = yaml.safe_load(open("params.yaml"))


# =====================================
# Define the Model
# =====================================


# import pytorch
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
# Prepare the scoring function
# =====================================


def get_document_scores(batch):
    """Get the document scores

    Args:
        batch.query: The query texts.
        batch.documents: The documents text.

    Returns:
        outputs: The outputs of the model.
            outputs[0]: The document scores.
            outputs[1]: The cost matrix.
            outputs[2]: The transportation matrix.

        examples: The query and documents text tokenizer.
            example.q_input_ids: The tensor containing the query token IDs.
            example.q_attention_mask: The query attention mask.
            example.d_input_ids: The tensor containing the document token IDs.
            example.d_attention_mask: The document attention mask.

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
    return outputs, examples


# =====================================
# Prepare the visualization function
# =====================================


import numpy as np
import matplotlib.pyplot as plt


def generate_visualization(outputs, examples, image_file):
    """Generates and saves the visualization

    Args:
        outputs: The outputs of the LM-EMD model.
            outputs[0]: The document scores.
            outputs[1]: The cost matrix.
            outputs[2]: The transportation matrix.

        examples: The query and documents text tokenizer.
            example.q_input_ids: The tensor containing the query token IDs.
            example.d_input_ids: The tensor containing the document token IDs.

        image_file: The path where the image should be stored.
    """
    # get the output values
    scores = outputs[0].cpu()
    cm = outputs[1].cpu()
    tm = outputs[2].cpu()

    # get the query and document input ids
    q_input_ids = examples["q_input_ids"].cpu()
    d_input_ids = examples["d_input_ids"].cpu()

    # get the max sizes (batch, query, document)
    bsize, qsize, dsize = cm.shape

    # prepare the figure size based on the input
    xsize = 1.2 * dsize
    ysize = 0.65 * bsize * qsize

    # initialize the figure
    fig, ax = plt.subplots(nrows=bsize, ncols=2, figsize=(xsize, ysize))
    if bsize == 1:
        ax = [ax]
    # iterate through the examples
    for i in range(bsize):

        # the cosine distance matrix
        ax[i][0].set_title("distance matrix", fontsize="xx-large")
        cmim = ax[i][0].imshow(cm[i].numpy(), cmap="PuBu", vmin=0)
        cbar = fig.colorbar(cmim, ax=ax[i][0], shrink=0.9)
        cbar.ax.set_ylabel("cosine distance", rotation=-90, va="bottom")

        # the EMD transport matrix
        ax[i][1].set_title("transportation matrix", fontsize="xx-large")
        tmim = ax[i][1].imshow(tm[i] / tm[i].max(), cmap="Greens", vmin=0, vmax=1)
        cbar = fig.colorbar(tmim, ax=ax[i][1], shrink=0.9)
        cbar.ax.set_ylabel("mass transport (match)", rotation=-90, va="bottom")

        # query and document tokens
        q_tokens = tokenizer.convert_ids_to_tokens(q_input_ids[i])
        d_tokens = tokenizer.convert_ids_to_tokens(d_input_ids[i])

        for j in range(2):
            # set the x and y ticks
            ax[i][j].set_xticks(np.arange(len(d_tokens)))
            ax[i][j].set_yticks(np.arange(len(q_tokens)))
            # add the x and y labels
            ax[i][j].set_xticklabels(d_tokens, fontsize=14)
            ax[i][j].set_yticklabels(q_tokens, fontsize=14)
            # rotate the x labels a bit
            plt.setp(
                ax[i][j].get_xticklabels(),
                rotation=45,
                ha="right",
                rotation_mode="anchor",
            )

        # assign the document score (lower scores -> greater rank)
        d_score = round(scores[i].item(), 3)
        ax[i][0].set_ylabel(
            f"Document score: {d_score}",
            rotation=-90,
            va="bottom",
            labelpad=30,
            fontsize=14,
        )

    # make the layout more tight
    plt.tight_layout()

    # save the plot in a file
    plt.savefig(image_file, dpi=500, transparent=True, bbox_inches="tight")


# =====================================
# Execute example interpretability
# =====================================


# initialize the batch examples
batch = {
    "query": [
        # TODO: modify query texts
        "Who was the first president of the United States?",
        "Who was the first president of the United States?",
        "Who was the first president of the United States?",
        "Who was the first president of the United States?",
        "Who was the first president of the United States?",
    ],
    "documents": [
        # TODO: modify document texts
        "George Washington war von 1789 bis 1797 der erste Präsident der Vereinigten Staaten von Amerika.",
        "Abraham Lincoln amtierte von 1861 bis 1865 als 16. Präsident der Vereinigten Staaten von Amerika.",
        "Marie Skłodowska Curie war eine Physikerin und Chemikerin polnischer Herkunft, die in Frankreich lebte und wirkte.",
        "Augusta Ada King-Noel, Countess of Lovelace, allgemein als Ada Lovelace bekannt war eine britische Mathematikerin.",
        "Christoph Kolumbus wurde der erste Vizekönig der las Indias genannten Gebiete.",
    ],
}

# calculate the similarities, cost and transportation matrices
with torch.no_grad():
    outputs, examples = get_document_scores(batch)

# generate the visualization and save it
generate_visualization(outputs, examples, image_file)

