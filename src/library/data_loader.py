import sys
import os
import re
import torch
from datasets import Dataset

# get the ROOT directory
__ROOTDIR__ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def readfile(filepath):
    """Opens and reads the line of the file

    Args:
        filepath (string): The file path.

    Returns:
        string[]: The lines of the file.

    """
    with open(filepath, "r", encoding="utf-8") as f:
        return f.readlines()


def format_row(row):
    """Formats the row

    Args:
        row (string): The row of the file.

    Returns:
        dict: The dictionary containing the following attributes:
            - query (string): The query.
            - document (string): The document.
            - relevance (integer): The relevance label.

    """
    splitted_values = re.split(r"\t+", row)

    if len(splitted_values) == 3:
        rel, query, document = splitted_values
        return {
            "query": query.strip(),
            "document": document.strip(),
            "relevance": 1 if int(rel.strip()) > 0 else 0,
        }
    else:
        return None


def prepare_dataset(filepath, max_examples):
    """Prepares the dataset

    Args:
        filepath (string): The path of the dataset file.

    Returns:
        dict: The dictionary of dataset attribute values:
            - query (string[]): The queries.
            - documents (string[]): The documents.
            - relevance (integer[]): The document relevance labels.

    """
    filerows = readfile(filepath)
    # the dataset placeholder
    dataset = {"query": [], "documents": [], "relevance": []}

    max_iter = max_examples if max_examples < len(filerows) else len(filerows)
    for i in range(max_iter):
        attrs = format_row(filerows[i])
        if attrs:
            dataset["query"].append(attrs["query"])
            dataset["documents"].append(attrs["document"])
            dataset["relevance"].append(attrs["relevance"])
    return dataset


def get_train_datasets(datatype, batch_size=5, max_examples=sys.maxsize):
    """Gets and prepares the training datasets

    Args:
        datatype (string): The training data type.
        batch_size (integer): The batch size (Default: 5).

    Returns:
        DataLoader: The dataset batches.

    """
    # prepare the dataset paths
    train_path = f"{__ROOTDIR__}/data/sasaki18/{datatype}/train.txt"
    # load the datasets
    data = prepare_dataset(train_path, max_examples=max_examples)
    data = Dataset.from_dict(data)
    data = torch.utils.data.DataLoader(data, batch_size=batch_size)
    return data


def get_test_datasets(datatype, batch_size=40, max_examples=sys.maxsize):
    """Gets and prepares the test datasets

    Args:
        datatype (string): The test data type.
        batch_size (integer): The batch size (Default: 40).

    Returns:
        DataLoader: The dataset batches.

    """
    # prepare the dataset paths
    test_path = f"{__ROOTDIR__}/data/sasaki18/{datatype}/test1.txt"
    # load the datasets
    data = prepare_dataset(test_path, max_examples=max_examples)
    data = Dataset.from_dict(data)
    data = torch.utils.data.DataLoader(data, batch_size=batch_size)
    return data

