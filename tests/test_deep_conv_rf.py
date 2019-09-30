import pytest
from savanna.utils.deep_conv_rf_runners import run_one_layer_deep_conv_rf, run_two_layer_deep_conv_rf

def test_one_layer_deep_conv_rf():


    results = run_experiment(run_one_layer_deep_conv_rf,
                   "deep_conv_rf_old_acc_vs_n", rf_type="unshared")

    assert isinstance(results, list)

def test_two_layer_deep_conv_rf():

    results = run_experiment(run_two_layer_deep_conv_rf,
                       "deep_conv_rf_old_acc_vs_n", rf_type="unshared")

    assert isinstance(results, list)
