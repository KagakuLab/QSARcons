import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import get_scorer
from sklearn.utils.multiclass import type_of_target
from sklearn.base import is_classifier


DEFAULT_PARAM_GRID_REGRESSORS = {
    "Ridge": {
        "random_state":42,
        "alpha": [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0],
        "solver": ["auto", "saga", "lsqr"],
    },
    "PLSRegression": {
        "n_components": [2, 4, 8, 16, 32],
    },
    "RandomForestRegressor": {
        "random_state":42,
        "n_estimators": [50, 100, 200, 400],
        "max_depth": [5, 10, 20, None],
        "max_features": ["sqrt", "log2", None],
    },
    "XGBRegressor": {
        "random_state":42,
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 6, 9],
        "learning_rate": [0.01, 0.05, 0.1, 0.3],
        "subsample": [0.6, 0.8, 1.0],
    },
    "MLPRegressor": {
        "random_state":42,
        "activation": ["relu", "tanh"],
        "learning_rate_init": [1e-4, 1e-3],
        "hidden_layer_sizes": [(128,), (512, 256, 128), (2048, 1024, 512, 256, 128, 64)],
        "max_iter": [300, 1000],
    },
    "SVR": {
        "random_state":42,
        "C": [0.1, 1, 10, 100],
        "kernel": ["linear", "rbf", "poly"],
        "gamma": ["scale", "auto"],
    },
    "LinearSVR": {
        "random_state":42,
        "C": [0.01, 0.1, 1.0, 10.0, 100.0],
        "epsilon": [0.001, 0.01, 0.1, 0.5],
        "loss": ["epsilon_insensitive", "squared_epsilon_insensitive"],
        "max_iter": [1000, 5000, 10000],
    },
}

DEFAULT_PARAM_GRID_CLASSIFIERS = {
    "RidgeClassifier": {
        "random_state":42,
        "alpha": [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0],
        "solver": ["auto", "saga", "lsqr"],
        "random_state":42
    },
    "LogisticRegression": {
        "random_state":42,
        "C": [0.01, 0.1, 1.0, 10.0, 100.0],
        "solver": ["liblinear", "lbfgs", "saga"],
        "max_iter": [500, 2000],
    },
    "RandomForestClassifier": {
        "random_state":42,
        "n_estimators": [50, 100, 200, 400],
        "max_depth": [5, 10, 20, None],
        "max_features": ["sqrt", "log2", None],
    },
    "XGBClassifier": {
        "random_state":42,
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 6, 9],
        "learning_rate": [0.01, 0.05, 0.1, 0.3],
        "subsample": [0.6, 0.8, 1.0],
    },
    "MLPClassifier": {
        "random_state":42,
        "activation": ["relu", "tanh"],
        "learning_rate_init": [1e-4, 1e-3],
        "hidden_layer_sizes": [(128,), (512, 256, 128), (2048, 1024, 512, 256, 128, 64)],
        "max_iter": [300, 1000],
         },
    "SVC": {
        "random_state":42,
        "C": [0.1, 1, 10, 100],
        "kernel": ["linear", "rbf", "poly"],
        "gamma": ["scale", "auto"],
    },
    "LinearSVC": {
        "random_state":42,
        "C": [0.01, 0.1, 1.0, 10.0, 100.0],
        "loss": ["hinge", "squared_hinge"],
        "penalty": ["l2"],
        "max_iter": [1000, 5000, 10000],
    },
}

def get_predictions(estimator, X):
    return estimator.predict(X).tolist()

def single_split_score(est, x, y, scoring, test_size=0.2, random_state=42):
    """
    Performs a single train/val split inside the function,
    fits the estimator, and returns the validation score.
    """

    x_train, x_val, y_train, y_val = train_test_split(
        x, y,
        test_size=test_size,
        random_state=random_state,
    )

    est.fit(x_train, y_train)
    y_pred = get_predictions(est, x_val)
    scorer = get_scorer(scoring)
    score = scorer._score_func(y_val, y_pred)

    return score

class StepwiseHopt:
    """
    Stepwise hyperparameter optimization for scikit-learn estimators.

    This optimizer iteratively tunes each hyperparameter one at a time
    while keeping the other parameters fixed at their current best values.
    """

    def __init__(self, estimator, param_grid, verbose=True):
        self.estimator = estimator
        self.param_grid = param_grid
        self.verbose = verbose
        self.best_params_ = {}

    def _evaluate_model(self, param, val, x, y, best_params, scoring):

        params = {**best_params, param: val}
        est = self.estimator.__class__(**params)
        score = single_split_score(est, x, y, scoring=scoring)
        return val, score

    def fit(self, x, y):

        if type_of_target(y) == "continuous":
            scoring = "r2"
        elif type_of_target(y) == "binary":
            scoring = "f1"
        else:
            raise ValueError("Unknown target type")

        if self.verbose:
            total_steps = sum(len(v) for v in self.param_grid.values())
            print(f"Stepwise optimization started with {total_steps} options")

        current_step = 0
        start_time = time.time()

        best_params = {}
        for param, options in self.param_grid.items():
            if not isinstance(options, (list, tuple)):
                best_params[param] = options
                continue

            if self.verbose:
                print(f"\nOptimizing '{param}' ({len(options)} options)")

            args = [(param, val, x, y, best_params, scoring) for val in options]
            results = [self._evaluate_model(*a) for a in args]

            # Select best value
            best_val, best_score = max(results, key=lambda x: x[1])  # higher is better

            best_params[param] = best_val
            current_step += len(options)
            if self.verbose:
                print(f"→ Best {param}: {best_val}, score={best_score:.4f}")

        self.best_params_ = best_params
        self.estimator = self.estimator.__class__(**best_params)
        total_time_min = (time.time() - start_time) / 60
        if self.verbose:
            print(f"\nStepwise optimization completed in {total_time_min:.1f} min")

        return self
