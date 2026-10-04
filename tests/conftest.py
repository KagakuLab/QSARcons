import pytest
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from qsarcons.modelling.lazy import LazyML
from qsarcons.consensus import RandomSearch, SystematicSearch, GeneticSearch


class MockEstimator:
    """A fast, deterministic stand-in for a real sklearn-style estimator - usable directly as a
    REGRESSORS/CLASSIFIERS entry (a class, constructible with arbitrary hopt-grid kwargs)."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.mean_y = 0.0

    def fit(self, x, y):
        self.mean_y = float(np.mean(list(y)))
        return self

    def predict(self, x):
        return np.full(len(x), self.mean_y)


# -----------------------------
# Dataset loader
# -----------------------------
def load_data():
    url = (
        "https://huggingface.co/datasets/KagakuData/Notebooks/"
        "resolve/main/chembl_200/CHEMBL1821.csv"
    )
    df = pd.read_csv(url, header=None)[[0, 2]].iloc[:140]
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
    df_train, df_val = train_test_split(df_train, test_size=0.2, random_state=42)
    return df_train, df_val, df_test

# -----------------------------
# Regression + Classification Data
# -----------------------------
@pytest.fixture
def regression_data():
    return load_data()

@pytest.fixture
def classification_data():
    df_train, df_val, df_test = load_data()
    for df in (df_train, df_val, df_test):
        df.iloc[:, 1] = (df.iloc[:, 1] > 6.5).astype(int)
    return df_train, df_val, df_test

# -----------------------------
# LazyML Training
# -----------------------------
@pytest.fixture
def regression_folder(regression_data):
    df_train, df_val, df_test = regression_data
    out = "regression_models"
    lazy = LazyML(task="continuous", hopt=True, output_folder=out, verbose=False)
    lazy.run(
        df_train.iloc[:, 0], df_train.iloc[:, 1],
        df_val.iloc[:, 0], df_val.iloc[:, 1],
        df_test.iloc[:, 0],
    )
    return out

@pytest.fixture
def classification_folder(classification_data):
    df_train, df_val, df_test = classification_data
    out = "classification_models"
    lazy = LazyML(task="binary", hopt=True, output_folder=out, verbose=False)
    lazy.run(
        df_train.iloc[:, 0], df_train.iloc[:, 1],
        df_val.iloc[:, 0], df_val.iloc[:, 1],
        df_test.iloc[:, 0],
    )
    return out

@pytest.fixture
def consensus_searchers():
    cons_size = "auto"
    return [
        ("Best", SystematicSearch(cons_size=1)),
        ("Random", RandomSearch(cons_size=cons_size, n_iter=50)),
        ("Systematic", SystematicSearch(cons_size=cons_size)),
        ("Genetic", GeneticSearch(cons_size=cons_size, n_iter=20)),
    ]
