import os
import re
import sys
import torch
from datasets import Dataset
from os.path import dirname, abspath, join, exists

# get the ROOT directory
ROOTDIR = dirname(dirname(dirname(abspath(__file__))))

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
    # QUICK HACK for parallel processing
    index = ROOTDIR.find("/.dvc")
    datapath = ROOTDIR[:index] if index >= 0 else ROOTDIR
    # prepare the dataset paths
    train_path = f"{datapath}/data/sasaki18/{datatype}/train.txt"
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
    # QUICK HACK for parallel processing
    index = ROOTDIR.find("/.dvc")
    datapath = ROOTDIR[:index] if index >= 0 else ROOTDIR
    # prepare the dataset paths
    test_path = f"{datapath}/data/sasaki18/{datatype}/test1.txt"
    # load the datasets
    data = prepare_dataset(test_path, max_examples=max_examples)
    data = Dataset.from_dict(data)
    data = torch.utils.data.DataLoader(data, batch_size=batch_size)
    return data


def get_folders_in_dir(path):
    """Gets the list of folders in directory

    Args:
        path (str): The relative path to the project root.

    Returns:
        List[str]: The list of folders.

    """
    # QUICK HACK for parallel processing
    index = ROOTDIR.find("/.dvc")
    datapath = ROOTDIR[:index] if index >= 0 else ROOTDIR
    # get the list of folders
    return os.listdir(join(datapath, path))


def create_folder(path):
    """Creates a folder if it does not exist

    Args:
        path (str): The relative path to the project root.

    """
    if not exists(join(ROOTDIR, path)):
        os.makedirs(join(ROOTDIR, path))
