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
        else:
            raise ValueError("Task type not set. Run `.run()` first.")


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
            y_pred = self.predict(x.iloc[:, list(ind)])
            return calc_accuracy(y, y_pred)

        space = range(len(x.columns))
        ga = GeneticAlgorithm(task="maximize", pop_size=100, crossover_prob=0.90, mutation_prob=0.2, elitism=True, verbose=False)
        ga.set_fitness(objective)
        ga.initialize(space, ind_size=cons_size)
        ga.run(n_iter=self.n_iter)

        return x.columns[list(ga.get_global_best())]
