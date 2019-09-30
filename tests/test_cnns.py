import pytest
import copy
#from argparse import ArgumentParser
import torch
from savanna.utils.utils import run_experiment
from savanna.cnn.trainer import run_cnn
from savanna.cnn.models.simple import SimpleCNN1layer, SimpleCNN2Layers
from savanna.utils.deep_conv_rf_runners import run_one_layer_deep_conv_rf




def test_1layer_cnn():

    BATCH_SIZE = 8 #args.batch_size
    EPOCH = 10 #args.epochs
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    CHOOSEN_CLASSES = [0,3]
    NUM_CLASSES = len(CHOOSEN_CLASSES)
    IMG_SHAPE = (25,25,3)

    ### Make sure all dependencies are above this line
    CNN_CONFIG = {"batch_size": BATCH_SIZE, "epoch": EPOCH, "device": DEVICE}

    cnn_acc_vs_n_config = copy.deepcopy(CNN_CONFIG)
    cnn_acc_vs_n_config.update({'model': 0, 'lr': 0.001, 'weight_decay': 1e-05})
    #results = run_experiment(run_cnn, "cnn_acc_vs_n", "CNN (1-layer, 1-filter)",
    #               cnn_model=SimpleCNN1layer(1, NUM_CLASSES, IMG_SHAPE), cnn_config=cnn_acc_vs_n_config)
    results = run_experiment(run_one_layer_deep_conv_rf,
                   "deep_conv_rf_old_acc_vs_n", rf_type="unshared")

    assert isinstance(results, list)
