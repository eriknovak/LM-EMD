import sys
import math
import yaml
import json
from tqdm import tqdm


# =====================================
# Import Inputs and Training Parameters
# =====================================


data_folder = sys.argv[1]
model_file = sys.argv[2]
LOSS_DIR = sys.argv[3]

params = yaml.safe_load(open("params.yaml"))

# =====================================
# Import Training Data
# =====================================


from library.data_loader import get_train_datasets, get_folders_in_dir, create_folder

train_data = {}
data_folders = get_folders_in_dir(data_folder)
for folder in tqdm(data_folders, desc="Preparing Data"):
    train_data[folder] = get_train_datasets(folder)


# =====================================
# Define the Model
# =====================================


import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim

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
model = BERT(type=ranking_type, reg=reg, nit=nit).train().to(device)


# =====================================
# Define the Loss Function
# =====================================


training_loss = params["train"]["loss"]

loss_fn = None
if training_loss == "cross_entropy":
    # use cross entropy as a loss function
    loss_fn = nn.CrossEntropyLoss()
elif training_loss == "margin_ranking":
    # use the margin ranking loss function
    margin_ranking_loss = nn.MarginRankingLoss()

    def modified_margin_ranking_loss(distances, labels):
        distances = distances.squeeze()
        relevant = distances[0].repeat(distances.size()[0] - 1)
        irrelevant = distances[1:]
        labels = torch.ones_like(relevant)
        return margin_ranking_loss(relevant, irrelevant, labels)

    # assign the modified margin ranking loss
    loss_fn = modified_margin_ranking_loss
else:
    # unsupported loss function
    raise Exception(f"Invalid loss function: '{training_loss}'")


# =====================================
# Define the Optimizer
# =====================================


training_optim = params["train"]["optimizer"]
training_lr = params["train"]["learning_rate"]
training_ep = params["train"]["epsilon"]
training_wd = params["train"]["weight_decay"]


optimizer = None
if training_optim == "adamw":
    optimizer = optim.AdamW(
        params=model.parameters(),
        lr=training_lr,
        eps=training_ep,
        weight_decay=training_wd,
    )
else:
    # unsupported optimizer type
    raise Exception(f"Invalid optimizer type: '{training_optim}'")


# =====================================
# Define the Training Function
# =====================================


def train(batch, loss_fn):
    query = batch["query"]
    documents = batch["documents"]
    relevance = batch["relevance"]
    q_inputs = tokenizer(query, truncation=True, padding=True, return_tensors="pt")
    d_inputs = tokenizer(documents, truncation=True, padding=True, return_tensors="pt")
    labels = torch.LongTensor([relevance.tolist().index(1)]).to(device)
    # get the input batches
    inputs = {
        "q_input_ids": q_inputs["input_ids"],
        "q_attention_mask": q_inputs["attention_mask"],
        "d_input_ids": d_inputs["input_ids"],
        "d_attention_mask": d_inputs["attention_mask"],
    }
    # move the batch tensors to the same device as the model
    inputs = {k: v.to(device) for k, v in inputs.items()}
    # get the model outpu ts
    outputs = model(**inputs)
    distances = (1 - outputs[0]).unsqueeze(0)
    loss = loss_fn(distances, labels)
    return loss


# =====================================
# Execute the Training Process
# =====================================


n_epochs = params["train"]["epochs"]
grad_update_step = params["train"]["grad_update_step"]

losses = {}
model_corrupted = False
# iterate through the data 'n_epochs' times
for epoch in tqdm(range(n_epochs), desc="Epochs"):
    # go through all datasets
    for key in train_data.keys():
        current_loss = 0
        losses[key] = []
        optimizer.zero_grad()
        # iterate through each batch of the train data
        for i, batch in enumerate(tqdm(train_data[key], desc=key)):
            loss = train(batch, loss_fn)
            loss.backward()
            current_loss += loss.item()
            # ACCUMULATIVE GRADIENT
            if i % grad_update_step == 0 and i > 0:
                # update the model using the optimizer
                optimizer.step()
                # once we update the model we set the gradients to zero
                optimizer.zero_grad()
                # store the loss value for visualization
                if math.isnan(current_loss):
                    model_corrupted = True
                    break
                tloss = current_loss / grad_update_step
                losses[key].append({"step": i, "loss": tloss})
                current_loss = 0
        if model_corrupted:
            break
        # update the model one last time for this epoch
        optimizer.step()
        optimizer.zero_grad()
        if math.isnan(current_loss):
            model_corrupted = True
            break
        tloss = current_loss / grad_update_step
        losses[key].append({"step": i, "loss": tloss})
    if model_corrupted:
        print("The model is corrupted")
        break


# =====================================
# Save the Loss Values and Model
# =====================================

# create the losses folder
create_folder(LOSS_DIR)

# save all losses
with open(f"{LOSS_DIR}/losses.json", "w") as f:
    json.dump(losses, f, indent=4)

# save the losses in individual files
for key, loss in losses.items():
    with open(f"{LOSS_DIR}/{key}.json", "w") as f:
        json.dump({"loss": loss}, f, indent=4)

# save the model
torch.save(model.state_dict(), model_file)
