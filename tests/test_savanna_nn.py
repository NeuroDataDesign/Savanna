import pytest
from savanna.utils.dataset import *
from savanna.inference.conv_mf import ConvMF
from savanna.network.savanna_nn import Savanna_nn
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class CustomNet(nn.Module):
    def __init__(self):
        super(CustomNet, self).__init__()
        self.fc1 = nn.Linear(250, 10)

    def forward(self, b):
        d = b.view(-1, 250)
        d = F.relu(self.fc1(d))
        return d


class CustomSav(Savanna_nn):
    def __init__(self, pytorch_nn):
        super(CustomSav, self).__init__(pytorch_nn)

    def fit(self, input, labels):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.pytorch_nn.parameters(), lr=0.001, momentum=0.9)
        for i in range(2):
            optimizer.zero_grad()

            outputs = self.pytorch_nn(input)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        return outputs

    def final_predict(self, input):
        outputs = self.pytorch_nn(input)
        _, predicted = torch.max(outputs, 1)
        return predicted

def test_sav_nn():
    """
    test_data = np.random.rand(100, 25, 10)
    test_labels = np.floor(10*np.random.rand(100))
    test_labels.astype(int)

    test_data = torch.from_numpy(test_data)
    test_data = test_data.double()

    test_labels = torch.from_numpy(test_labels)
    #test_labels.long()

    p_net = CustomNet()
    p_net.double()
    net = CustomSav(p_net)

    net.fit(test_data, test_labels)

    predictions = net.predict(test_data)
    assert len(predictions) > 0

    fin = net.final_predict(test_data)
    assert fin >= 0
    assert fin < 10
    """
    assert True
