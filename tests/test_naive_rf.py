import pytest

import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from savanna.utils.dataset import get_subset_data
from savanna.utils.utils import run_experiment
from savanna.random_forest.naive_rf import run_naive_rf

def test_run_on_naive_rf():
    results = run_experiment(run_naive_rf, "Naive RF")
    assert isinstance(results, list)
