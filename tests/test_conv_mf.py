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
    trainset, testset = get_dataset("savanna/datasets", "FashionMNIST", is_numpy=True)

    #trainset, testset = get_subset_data( choosen_classes, sub_train_indices, is_numpy=True, batch_size=None):


    test = ConvMF()
    test.fit(trainset[0], trainset[1])

    assert True
