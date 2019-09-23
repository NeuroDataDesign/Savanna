import pytest
from savanna.utils.dataset import *

def test_get_dataset():
    trainset, testset = get_dataset("../savanna/data", "FashionMNIST", is_numpy=True)
    assert isinstance(trainset, np.ndarray)

