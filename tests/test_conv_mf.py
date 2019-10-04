import pytest
from savanna.utils.dataset import *
from savanna.inference.conv_mf import ConvMF

def test_empty():
    """
    Placeholder
    """
    temp = ConvMF()
    assert True

def test_convmf():
    trainset, testset = get_dataset("../savanna/data", "FashionMNIST", is_numpy=True)

    test = ConvMF()
    test.fit(trainset[0], trainset[1])

    assert True
