import pytest
import numpy as np
from savanna.utils.dataset import *

DATA_PATH = "./data"
DATASET_NAME = "FashionMNIST"


def test_get_dataset():
    """
    Tests if four outputs of get_dataset() have correct number of dimensions
    """
    # trainset, testset = get_dataset("../savanna/data", "FashionMNIST", is_numpy=True)
    # assert isinstance(trainset, tuple)
    numpy_data = dict()
    (numpy_data["train_images"], numpy_data["train_labels"]), (
        numpy_data["test_images"],
        numpy_data["test_labels"],
    ) = get_dataset(DATA_PATH, DATASET_NAME, is_numpy=True)

    assert (
        numpy_data["train_images"].ndim == 4
        and numpy_data["train_labels"].ndim == 1
        and numpy_data["test_images"].ndim == 4
        and numpy_data["test_labels"].ndim == 1
    )


def test_get_subset_data():
    """
    Tests if four outputs of get_subset_data() have correct number of dimensions
    """
    # trainset, testset = get_dataset("../savanna/data", "FashionMNIST", is_numpy=True)
    numpy_data = dict()
    (numpy_data["train_images"], numpy_data["train_labels"]), (
        numpy_data["test_images"],
        numpy_data["test_labels"],
    ) = get_dataset(DATA_PATH, DATASET_NAME, is_numpy=True)

    (train_images, train_labels), (test_images, test_labels) = get_subset_data(
        dataset_name="FashionMNIST",
        data=numpy_data,
        choosen_classes=[0, 3],
        sub_train_indices=[
            25007,
            40074,
            53077,
            44949,
            44867,
            8986,
            19220,
            1749,
            55797,
            25755,
        ],
    )

    assert (
        train_images.ndim == 4
        and train_labels.ndim == 1
        and test_images.ndim == 4
        and test_labels.ndim == 1
    )
