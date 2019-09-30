import pytest
from savanna.utils.utils import run_experiment
from savanna.random_forest.naive_rf import run_naive_rf
from savanna.random_forest.naive_rerf import run_naive_rerf

def test_run_on_naive_rf():
    results = run_experiment(run_naive_rf, "Naive RF")
    assert isinstance(results, list)

def test_naive_rerf():
    results = run_experiment(run_naive_rerf, "Naive Rerf")
    assert isinstance(results, list)