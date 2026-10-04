import random
from typing import List, Union, Optional

import numpy as np
import pandas as pd
from pandas import DataFrame, Series, Index
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from abc import ABC, abstractmethod
from qsarcons.genopt import Individual, GeneticAlgorithm
from sklearn.utils.multiclass import type_of_target
from amltk.ensembling.weighted_ensemble_caruana import weighted_ensemble_caruana

METRIC_MODES = {
    "r2": "maximize",
    "f1_score": "maximize",
}

def calc_accuracy(y_true, y_pred):
    """Compute performance metrics for regression or classification tasks."""

    y_true, y_pred = list(y_true), list(y_pred)

    task_type = type_of_target(y_true)
    if task_type == "continuous":
        return  r2_score(y_true, y_pred)
    elif task_type == "binary":
        return f1_score(y_true, y_pred)
    else:
        raise ValueError("Task type not supported.")

class ConsensusSearch(ABC):
    """Base class for consensus model selection."""

    def __init__(self, cons_size=9, cons_size_candidates=None):
        self.cons_size = cons_size
        self.cons_size_candidates = cons_size_candidates or range(2, 16, 2)
        self._task_type = None

    def _filter_models(self, x: DataFrame, y: List) -> DataFrame:
        """Filter out underperformed models based on baseline metric performance."""

        baseline_score = 0 if self._task_type == "continuous" else 0.5
        filtered_cols = [col for col in x.columns if calc_accuracy(y, x[col]) > baseline_score]

        filtered = x[filtered_cols]
        if filtered.shape[1] == 0:
            print("No models left after filtering. All models selected.")
            return x
        return filtered

    @abstractmethod
    def _run_with_cons_size(self, x, y, cons_size):
        return self._run_with_cons_size(x, y, cons_size)

    def run(self, x: DataFrame, y: List):
        """Execute consensus model search."""

        self._task_type = type_of_target(y)

        x_filtered = self._filter_models(x, y)
        if len(x_filtered.columns) < max(self.cons_size_candidates):
            print("WARNING: The number of filtered models is lower than the consensus size candidates. All models are used for consensus search.")
            x_filtered = x

        if isinstance(self.cons_size, int):
            return self._run_with_cons_size(x_filtered, y, self.cons_size)

        elif self.cons_size == 'auto':
            best_cons, best_score = None, None
            for size in self.cons_size_candidates:
                candidate = self._run_with_cons_size(x_filtered, y, size)
                y_pred = self.predict(x_filtered[candidate])
                score = calc_accuracy(y, y_pred)
                if best_score is None or score > best_score:
                    best_score = score
                    best_cons = candidate
            return list(best_cons)

    def predict(self, x_subset: DataFrame) -> List:

        if self._task_type == "continuous":
            return list(x_subset.mean(axis=1))

        elif self._task_type == "binary":
            votes = x_subset.sum(axis=1)
            majority = (votes >= (x_subset.shape[1] / 2)).astype(int)
            return list(majority)


class RandomSearch(ConsensusSearch):
    """Randomized search for optimal regression consensus."""

    def __init__(self, n_iter=1000, **kwargs):
        super().__init__(**kwargs)
        self.n_iter = n_iter

    def _run_with_cons_size(self, x: DataFrame, y: Series, cons_size: int) -> Index:
        """Run random search for a fixed consensus size."""
        results = []
        for _ in range(self.n_iter):
            cols = random.sample(list(x.columns), cons_size)
            y_pred = self.predict(x[cols])
            score = calc_accuracy(y, y_pred)
            results.append((cols, score))
        results.sort(key=lambda tup: tup[1], reverse=True)
        return pd.Index(results[0][0])

class SystematicSearch(ConsensusSearch):
    """Systematic selection of top-performing regression models."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _run_with_cons_size(self, x: DataFrame, y: Series, cons_size: int):
        """Run a systematic search for regression models."""

        scores = [(col, calc_accuracy(y, x[col])) for col in x.columns]
        scores.sort(key=lambda tup: tup[1], reverse=True)
        top_cols = [col for col, _ in scores[:cons_size]]
        return list(top_cols)

class GeneticSearch(ConsensusSearch):
    """Genetic algorithm-based search for optimal regression consensus. """

    def __init__(self, n_iter=50, verbose=False, **kwargs):
        super().__init__(**kwargs)
        self.n_iter = n_iter
        self.verbose = verbose

    def _run_with_cons_size(self, x, y, cons_size) -> Index:

        def objective(ind: Individual) -> float:
            y_pred = self.predict(x.iloc[:, list(ind.genes)])
            return calc_accuracy(y, y_pred)

        auto_mode = self.cons_size == "auto"

        if self.verbose:
            if auto_mode:
                print(f"Genetic optimization for consensus size {cons_size}:")
            else:
                print("Genetic consensus optimization:")

        ga = GeneticAlgorithm(
            fitness_func=objective,
            n_genes=len(x.columns),
            ind_size=cons_size,
            pop_size=100,
            crossover_prob=0.90,
            mutation_prob=0.2,
            verbose=self.verbose and not auto_mode,
        )
        ga.run(n_iter=self.n_iter)

        if self.verbose and auto_mode:
            print(f"The first generation best score: {ga.best_individuals[0].score:.3f}")
            print(f"The last generation best score: {ga.best_individuals[-1].score:.3f}")
            print()

        return x.columns[list(ga.get_solution().genes)]


class CaruanaSearch(ConsensusSearch):
    """Caruana greedy ensemble selection for consensus model search."""

    def __init__(self, n_iter=50, seed=None, **kwargs):
        super().__init__(**kwargs)
        self.n_iter = n_iter
        self.seed = seed
        self.weights_ = None

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
        return list(self.weights_.keys())

    def predict(self, x_subset: DataFrame) -> List:
        """Weighted average prediction using Caruana weights."""

        # Fall back to equal-weight averaging if weights not yet fitted
        # (e.g. called from auto cons_size search on a candidate not yet stored)
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
