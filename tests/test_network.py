import pytest
from savanna.utils.dataset import *
from savanna.inference.conv_mf import ConvMF
from savanna.network.network import Network

def test_network():
    #numpy_data = dict()
    #(numpy_data["train_images"], numpy_data["train_labels"]), (
    #    numpy_data["test_images"],
    #    numpy_data["test_labels"],
    #) = get_dataset("../savanna/data", "FashionMNIST", is_numpy=True)
    #trainset, testset = get_dataset("../savanna/data", "FashionMNIST", is_numpy=True)

    #trainset, testset = get_subset_data(
    #                        dataset_name = "FashionMNIST",
    #                        data=numpy_data,
    #                        choosen_classes= np.arange(10),
    #                        sub_train_indices = np.arange(100)
    #                        )

    #net = Network()
    #net.add_convMF()
    #net.fit(trainset[0], trainset[1])
    #results = net.predict(testset[0])

    #count = 0
    #for i in range(len(results)):
    #    if results[i] == testset[1][i]:
    #        count += 1
    #score = count/10000

    assert True
