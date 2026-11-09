import numpy as np
import pandas as pd

from typing import List, Union, Optional
from pandas import DataFrame, Series, Index
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error, f1_score
from .consensus import ConsensusSearch
from amltk.ensembling.weighted_ensemble_caruana import weighted_ensemble_caruana

class CaruanaSearch(ConsensusSearch):
    """Caruana greedy ensemble selection for consensus model search."""

    def __init__(self, n_iter=50, seed=None, **kwargs):
        super().__init__(**kwargs)
        self.n_iter = n_iter
        self.seed = seed
        self.weights_ = None
        self._weights_by_size = {}

    def _metric(self, y_true, y_pred):
        if self._task_type == "continuous":
            return r2_score(y_true, y_pred)
        elif self._task_type == "binary":
            return f1_score(y_true, np.round(y_pred).astype(int))

    def _run_with_cons_size(self, x: DataFrame, y: Series, cons_size: int):
        """Run Caruana ensemble selection — cons_size maps to n_iter here."""

        model_predictions = {col: x[col].values.astype(float) for col in x.columns}

        weights, trajectory, _ = weighted_ensemble_caruana(
            model_predictions=model_predictions,
            targets=np.array(y, dtype=float),
            size=cons_size,
            metric=self._metric,
            select=max,
            seed=self.seed,
        )

        # Store weights for weighted prediction
        self.weights_ = {k: v for k, v in weights.items() if v > 0}
        self._weights_by_size[cons_size] = self.weights_
        return list(self.weights_.keys())

    def run(self, x: DataFrame, y: Series):
        """Run the search, then restore the weights of the winning size (the loop leaves those of the last one)."""
        self._weights_by_size = {}
        best_cons = super().run(x, y)
        self.weights_ = self._weights_by_size[self.best_size_]
        return best_cons

    def predict(self, x_subset: DataFrame) -> List:
        """Weighted average prediction using Caruana weights."""

        if self.weights_ is None or not all(col in self.weights_ for col in x_subset.columns):
            return super().predict(x_subset)

        preds = np.sum(
            [self.weights_[col] * x_subset[col].values
             for col in x_subset.columns
             if col in self.weights_],
            axis=0
        )

        if self._task_type == "binary":
            return list(np.round(preds).astype(int))
        return list(preds)