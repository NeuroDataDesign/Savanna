import pytest
from savanna.utils.utils import run_experiment
from savanna.random_forest.naive_rf import run_naive_rf
from savanna.random_forest.naive_rerf import run_naive_rerf
from savanna.utils.dataset import get_dataset
import numpy as np

def test_run_on_naive_rf_thorough():
    DATASET_NAME = "FashionMNIST"
    DATA_PATH = "../savanna/data"
    numpy_data = dict()
    (numpy_data["train_images"], numpy_data["train_labels"]), (numpy_data["test_images"], numpy_data["test_labels"]) = get_dataset(DATA_PATH, DATASET_NAME, is_numpy=True)

    CHOSEN_CLASSES = [0,3]
    rf_type = "shared"
    accuracy, _ = run_naive_rf(DATASET_NAME, numpy_data, CHOSEN_CLASSES, np.arange(1000), rf_type)
    assert isinstance(accuracy, float)
    print("accuracy: ", accuracy)
    assert accuracy <= 1
    assert accuracy >= 0

def test_run_on_naive_rerf_thorough():
    DATASET_NAME = "FashionMNIST"
    DATA_PATH = "../savanna/data"
    numpy_data = dict()
    (numpy_data["train_images"], numpy_data["train_labels"]), (numpy_data["test_images"], numpy_data["test_labels"]) = get_dataset(DATA_PATH, DATASET_NAME, is_numpy=True)

    CHOSEN_CLASSES = [0,3]
    rf_type = "shared"
    accuracy, _ = run_naive_rerf(DATASET_NAME, numpy_data, CHOSEN_CLASSES, np.arange(1000), rf_type)
    assert isinstance(accuracy, float)
    print("accuracy: ", accuracy)
    assert accuracy <= 1
    assert accuracy >= 0
