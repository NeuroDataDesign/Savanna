import pytest
from savanna.utils.dataset import *
from savanna.inference.conv_mf import ConvMF

def test_convmf_native():
    numpy_data = dict()
    (numpy_data["train_images"], numpy_data["train_labels"]), (
        numpy_data["test_images"],
        numpy_data["test_labels"],
    ) = get_dataset("../savanna/data", "FashionMNIST", is_numpy=True)
    #trainset, testset = get_dataset("../savanna/data", "FashionMNIST", is_numpy=True)

    trainset, testset = get_subset_data(
                            dataset_name = "FashionMNIST",
                            data=numpy_data,
                            choosen_classes= np.arange(10),
                            sub_train_indices = np.arange(100)
                            )

    layer = ConvMF()
    layerOutput = layer.fit(trainset[0], trainset[1])

    nsamples, nclasses = layerOutput.shape
    assert nsamples == 100
    assert nclasses == 10

    layerOutput = layer.predict(testset[0])
    nsamples, nclasses = layerOutput.shape
    assert nsamples == 10000
    assert nclasses == 10

    results = layer.final_predict(testset[0])
    count = 0
    for i in range(len(results)):
        if results[i] == testset[1][i]:
            count += 1
    score = count/nsamples

    assert score > 0

    assert hasattr(layer, "type")
    assert hasattr(layer, "num_trees")
    assert hasattr(layer, "tree_type")
    assert hasattr(layer, "patch_height_min")
    assert hasattr(layer, "patch_height_max")
    assert hasattr(layer, "patch_width_max")
    assert hasattr(layer, "patch_width_min")

    assert layer.type == 'native'
    assert layer.num_trees == 1000
    assert layer.tree_type == 'S-RerF'
    assert layer.patch_height_min == 1
    assert layer.patch_height_max == 5
    assert layer.patch_width_min == 1
    assert layer.patch_width_max == 5



def test_convmf_kernel():
    numpy_data = dict()
    (numpy_data["train_images"], numpy_data["train_labels"]), (
        numpy_data["test_images"],
        numpy_data["test_labels"],
    ) = get_dataset("../savanna/data", "FashionMNIST", is_numpy=True)
    #trainset, testset = get_dataset("../savanna/data", "FashionMNIST", is_numpy=True)

    trainset, testset = get_subset_data(
                            dataset_name = "FashionMNIST",
                            data=numpy_data,
                            choosen_classes= np.arange(10),
                            sub_train_indices = np.arange(100)
                            )

    layer = ConvMF(type = 'kernel_patches', num_trees = 10)
    layerOutput = layer.fit(trainset[0], trainset[1])

    nsamples, outdim, _, nclasses = layerOutput.shape
    assert nsamples == 100
    assert outdim == 12
    assert nclasses == 10

    layerOutput = layer.predict(testset[0])
    nsamples, outdim, _, nclasses = layerOutput.shape
    assert nsamples == 10000
    assert outdim == 12
    assert nclasses == 10

    results = layer.final_predict(testset[0])
    count = 0
    for i in range(len(results)):
        if results[i] == testset[1][i]:
            count += 1
    score = count/nsamples

    assert score > 0

    assert hasattr(layer, "type")
    assert hasattr(layer, "num_trees")
    assert hasattr(layer, "tree_type")
    assert hasattr(layer, "patch_height_min")
    assert hasattr(layer, "patch_height_max")
    assert hasattr(layer, "patch_width_max")
    assert hasattr(layer, "patch_width_min")

    assert layer.type == 'kernel_patches'
    assert layer.num_trees == 10
    assert layer.tree_type == 'S-RerF'
    assert layer.patch_height_min == 1
    assert layer.patch_height_max == 5
    assert layer.patch_width_min == 1
    assert layer.patch_width_max == 5
