from typing import Optional, List
import pandas as pd
from qsarcons.utils import detect_task
from qsarcons.hopt import StepwiseHopt, DEFAULT_PARAM_GRID_REGRESSORS, DEFAULT_PARAM_GRID_CLASSIFIERS
from qsarcons.logging import OutputSuppressor

class StackingAggregation:
    """Stacking regressor: uses **all filtered models** to train a meta-model"""

    def __init__(self, meta_model = None, hopt: bool = True):
        super().__init__()
        self.meta_model = meta_model
        self.hopt = hopt

        if self.meta_model is None:
            raise ValueError("meta_model must be provided")

    def run(self, x: pd.DataFrame, y: pd.Series) -> List:

        # 1. Hyperparameter optimization
        if self.hopt:

            est_name = self.meta_model.__class__.__name__

            task_type = detect_task(y)
            if task_type == "binary":
                param_grid = DEFAULT_PARAM_GRID_CLASSIFIERS.get(est_name)
            elif task_type == "continuous":
                param_grid = DEFAULT_PARAM_GRID_REGRESSORS.get(est_name)
                DEFAULT_PARAM_GRID_REGRESSORS["PLSRegression"]["n_components"] = [2, 4, 8, 16] # TODO remove
            else:
                raise ValueError("Task type not supported.")

            stepwise_hopt = StepwiseHopt(self.meta_model, param_grid, verbose=False)
            with OutputSuppressor():
                stepwise_hopt.fit(x, y)
            self.meta_model = stepwise_hopt.estimator

        # 3. Train final meta_model
        with OutputSuppressor():
            self.meta_model.fit(x, y)

        # 4. Return filtered models used
        return list(x.columns)

    def predict(self, x_subset: pd.DataFrame) -> List:
        y_pred = list(self.meta_model.predict(x_subset))
        return y_pred

    def predict_proba(self, x_subset: pd.DataFrame) -> List:
        if not hasattr(self.meta_model, "predict_proba"):
            raise ValueError(
                f"{self.meta_model.__class__.__name__} does not support predict_proba."
            )
        return self.meta_model.predict_proba(x_subset)[:, 1].tolist()