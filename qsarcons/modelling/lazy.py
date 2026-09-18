# ==========================================================
# Imports
# ==========================================================
import os
import gc
import shutil
import warnings

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression, Ridge, RidgeClassifier
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.svm import LinearSVR, LinearSVC
from sklearn.preprocessing import MinMaxScaler
from molfeat.trans import MoleculeTransformer
from molfeat.calc.pharmacophore import Pharmacophore2D

from qsarcons.hopt import StepwiseHopt, DEFAULT_PARAM_GRID_REGRESSORS, DEFAULT_PARAM_GRID_CLASSIFIERS
from qsarcons.logging import FailedMolecule

from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings("ignore")

# ==========================================================
# Configuration
# ==========================================================
DESCRIPTORS = {

    # fingerprints
    "avalon": MoleculeTransformer(featurizer='avalon', dtype=float),
    "rdkit": MoleculeTransformer(featurizer='rdkit', dtype=float),
    "maccs": MoleculeTransformer(featurizer='maccs', dtype=float),
    "atompair-count": MoleculeTransformer(featurizer='atompair-count', dtype=float),
    "fcfp": MoleculeTransformer(featurizer='fcfp', dtype=float),
    "fcfp-count": MoleculeTransformer(featurizer='fcfp-count', dtype=float),
    "ecfp": MoleculeTransformer(featurizer='ecfp', dtype=float),
    "ecfp-count": MoleculeTransformer(featurizer='ecfp-count', dtype=float),
    "topological": MoleculeTransformer(featurizer='topological', dtype=float),
    "topological-count": MoleculeTransformer(featurizer='topological-count', dtype=float),
    "secfp": MoleculeTransformer(featurizer='secfp', dtype=float),

    # scaffold
    "scaffoldkeys": MoleculeTransformer(featurizer='scaffoldkeys', dtype=float),

    # phys-chem
    "desc2D": MoleculeTransformer(featurizer='desc2D', dtype=float),

    # electrotopological
    "estate": MoleculeTransformer(featurizer='estate', dtype=float),

    # pharmacophore
    "erg": MoleculeTransformer(featurizer='erg', dtype=float),
    "cats2d": MoleculeTransformer(featurizer='cats2d', dtype=float),
    "pharm2D-cats": MoleculeTransformer(featurizer=Pharmacophore2D(factory='cats'), dtype=float),
    "pharm2D-gobbi": MoleculeTransformer(featurizer=Pharmacophore2D(factory='gobbi'), dtype=float),
    "pharm2D-pmapper": MoleculeTransformer(featurizer=Pharmacophore2D(factory='pmapper'), dtype=float),
}

REGRESSORS = {
    "RidgeRegression": Ridge,
    "PLSRegression": PLSRegression,
    "LinearSVR": LinearSVR,
    "MLPRegressor": MLPRegressor,
    "RandomForestRegressor": RandomForestRegressor,
    "XGBRegressor": XGBRegressor,
}

CLASSIFIERS = {
    "LogisticRegression": LogisticRegression,
    "RandomForestClassifier": RandomForestClassifier,
    "XGBClassifier": XGBClassifier,
    "MLPClassifier": MLPClassifier,
    "RidgeClassifier": RidgeClassifier,
    "LinearSVC": LinearSVC,
}

# Reduced hyperparameter grids for the two slowest estimators (deep MLPs, large forests), used instead of
# hopt.py's full grids to keep lazy model building fast. Every other estimator keeps its full grid.
REDUCED_PARAM_GRID_REGRESSORS = {
    "MLPRegressor": {
        "random_state": 42,
        "activation": ["relu", "tanh"],
        "learning_rate_init": [1e-3],
        "hidden_layer_sizes": [(128,), (512, 256, 128)],
        "max_iter": [500],
    },
    "RandomForestRegressor": {
        "random_state": 42,
        "n_estimators": [100, 200],
        "max_depth": [5, 10, 20],
        "max_features": ["sqrt", "log2"],
    },
}

REDUCED_PARAM_GRID_CLASSIFIERS = {
    "MLPClassifier": {
        "random_state": 42,
        "activation": ["relu", "tanh"],
        "learning_rate_init": [1e-3],
        "hidden_layer_sizes": [(128,), (512, 256, 128)],
        "max_iter": [500],
    },
    "RandomForestClassifier": {
        "random_state": 42,
        "n_estimators": [100, 200],
        "max_depth": [5, 10, 20],
        "max_features": ["sqrt", "log2"],
    },
}

# ==========================================================
# Utility Functions
# ==========================================================
def clean_descriptors(x):
    x = np.array(x, dtype=float)
    col_means = np.nanmean(x, axis=0)
    all_nan_cols = np.isnan(col_means)
    if np.any(all_nan_cols):
        x = x[:, ~all_nan_cols]
        col_means = col_means[~all_nan_cols]
    idx = np.where(np.isnan(x))
    x[idx] = np.take(col_means, idx[1])
    return x

def calc_descriptors(smi_list, calculator):
    x = calculator(smi_list)
    x = clean_descriptors(x)
    return x

def scale_descriptors(x_train, x_test):
    scaler = MinMaxScaler()
    scaler.fit(x_train)
    x_train_scaled = scaler.transform(x_train)
    if len(x_test) == 0:
        x_test_scaled = np.empty((0, x_train_scaled.shape[1]))
    else:
        x_test_scaled = scaler.transform(x_test)
    return x_train_scaled, x_test_scaled

def get_predictions(estimator, X):
    return estimator.predict(X).tolist()

def validate_smiles(smi_list, verbose=False):
    """Parse each SMILES with RDKit; unparsable ones become FailedMolecule sentinels."""
    mol_list = []
    for smi in smi_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None or mol.GetNumAtoms() == 0:
            mol_list.append(FailedMolecule(smi, message="SMILES parsing failed"))
        else:
            mol_list.append(smi)

    if verbose:
        n_valid = sum(isinstance(m, str) for m in mol_list)
        print(f"Validated {n_valid} of {len(mol_list)} molecules")

    return mol_list

def baseline_prediction(y, task):
    """Fallback prediction for molecules that can't be processed: the training mean or most frequent class."""
    y_arr = np.asarray(list(y))
    if task == "continuous":
        return float(np.mean(y_arr))
    values, counts = np.unique(y_arr, return_counts=True)
    return values[np.argmax(counts)]

def build_model(x_train, x_val, x_test, y_train, y_val, estimator_class, hopt, task):

    # 1. Scale train/val descriptors
    x_train_scaled, x_val_scaled = scale_descriptors(x_train, x_val)

    # 2. Optimize hyperparameters
    if hopt:
        est_name = estimator_class.__name__
        is_classification = task == "binary"

        if is_classification:
            param_grid = REDUCED_PARAM_GRID_CLASSIFIERS.get(est_name) or DEFAULT_PARAM_GRID_CLASSIFIERS.get(est_name)
        else:
            param_grid = REDUCED_PARAM_GRID_REGRESSORS.get(est_name) or DEFAULT_PARAM_GRID_REGRESSORS.get(est_name)

        estimator_instance = estimator_class()
        stepwise_hopt = StepwiseHopt(estimator_instance, param_grid, verbose=False)
        stepwise_hopt.fit(x_train_scaled, y_train)
        estimator_instance = stepwise_hopt.estimator
    else:
        estimator_instance = estimator_class()

    # 3. Train on train split only (not final training yet)
    estimator_instance.fit(x_train_scaled, y_train)
    pred_train = get_predictions(estimator_instance, x_train_scaled)
    pred_val = get_predictions(estimator_instance, x_val_scaled)

    # 4. Retrain model on full (train + val) and predict on test with that same fit
    x_full, y_full = np.vstack([x_train, x_val]), np.hstack([y_train, y_val])
    x_full_scaled, x_test_scaled = scale_descriptors(x_full, x_test)

    estimator_instance.fit(x_full_scaled, y_full)
    pred_test = get_predictions(estimator_instance, x_test_scaled)

    # 5. Release memory
    del estimator_instance
    gc.collect()

    return pred_train, pred_val, pred_test

class LazyML:
    """Train every built-in descriptor/estimator combination and predict on train/val/test in one pass."""

    def __init__(self, task, hopt=True, output_folder=None, verbose=True):
        self.task = task
        self.ESTIMATORS = REGRESSORS if task == "continuous" else CLASSIFIERS
        self.hopt = hopt
        self.output_folder = output_folder
        self.verbose = verbose

        if self.output_folder:
            if os.path.exists(self.output_folder):
                shutil.rmtree(self.output_folder)
            os.makedirs(self.output_folder)
        else:
            raise ValueError("output_folder must be specified.")

    def run(self, smiles_train, y_train, smiles_val, y_val, smiles_test):
        """Train every descriptor/estimator combination on SMILES + y, and predict on train/val/test."""

        smi_train_all, y_train_all = list(smiles_train), list(y_train)
        smi_val_all, y_val_all = list(smiles_val), list(y_val)
        smi_test_all = list(smiles_test)

        # 1. Validate SMILES; anything RDKit can't parse becomes a FailedMolecule sentinel
        if self.verbose:
            print("Step-1. SMILES validation")

        smi_all = smi_train_all + smi_val_all + smi_test_all
        mol_all = validate_smiles(smi_all, verbose=self.verbose)

        n_train, n_val = len(smi_train_all), len(smi_val_all)
        mol_train_all = mol_all[:n_train]
        mol_val_all = mol_all[n_train:n_train + n_val]
        mol_test_all = mol_all[n_train + n_val:]

        valid_idx_train = [i for i, m in enumerate(mol_train_all) if isinstance(m, str)]
        valid_idx_val = [i for i, m in enumerate(mol_val_all) if isinstance(m, str)]
        valid_idx_test = [i for i, m in enumerate(mol_test_all) if isinstance(m, str)]

        smi_train = [smi_train_all[i] for i in valid_idx_train]
        y_train = [y_train_all[i] for i in valid_idx_train]

        smi_val = [smi_val_all[i] for i in valid_idx_val]
        y_val = [y_val_all[i] for i in valid_idx_val]

        smi_test = [smi_test_all[i] for i in valid_idx_test]

        train_baseline = baseline_prediction(y_train, self.task)

        n_failed_train = len(mol_train_all) - len(valid_idx_train)
        n_failed_val = len(mol_val_all) - len(valid_idx_val)
        n_failed_test = len(mol_test_all) - len(valid_idx_test)

        if self.verbose:
            if n_failed_train:
                print(
                    f"{n_failed_train} training molecule(s) could not be processed and will be predicted "
                    "using the training set baseline value"
                )
            if n_failed_val:
                print(
                    f"{n_failed_val} validation molecule(s) could not be processed and will be predicted "
                    "using the training set baseline value"
                )
            if n_failed_test:
                print(
                    f"{n_failed_test} test molecule(s) could not be processed and will be predicted "
                    "using the training set baseline value"
                )

        result_df_train = pd.DataFrame({"SMILES": smi_train_all, "Y_TRUE": y_train_all})
        result_df_val = pd.DataFrame({"SMILES": smi_val_all, "Y_TRUE": y_val_all})
        result_df_test = pd.DataFrame({"SMILES": smi_test_all})

        # 2. Calculate descriptors for every descriptor set, once, up front
        if self.verbose:
            print("Step-2. Descriptor calculation")

        ready_descriptors = {}
        for desc_name, desc_calc in DESCRIPTORS.items():
            x_train = calc_descriptors(smi_train, desc_calc)
            x_val = calc_descriptors(smi_val, desc_calc)
            x_test = calc_descriptors(smi_test, desc_calc) if smi_test else np.empty((0, x_train.shape[1]))
            ready_descriptors[desc_name] = (x_train, x_val, x_test)

            if self.verbose:
                print(f"{desc_name}: done")

        # 3. Train every descriptor/estimator combination
        if self.verbose:
            print("Step-3. Individual model training")

        total_models = len(DESCRIPTORS) * len(self.ESTIMATORS)
        current_model = 0

        for desc_name, (x_train, x_val, x_test) in ready_descriptors.items():
            for est_name, estimator_class in self.ESTIMATORS.items():
                model_name = f"{desc_name}|{est_name}"
                current_model += 1

                pred_train, pred_val, pred_test = build_model(
                    x_train, x_val, x_test, y_train, y_val, estimator_class, self.hopt, self.task,
                )

                preds_train_by_smi = dict(zip(smi_train, pred_train))
                preds_val_by_smi = dict(zip(smi_val, pred_val))
                preds_test_by_smi = dict(zip(smi_test, pred_test))

                result_df_train[model_name] = [preds_train_by_smi.get(smi, train_baseline) for smi in smi_train_all]
                result_df_val[model_name] = [preds_val_by_smi.get(smi, train_baseline) for smi in smi_val_all]
                result_df_test[model_name] = [preds_test_by_smi.get(smi, train_baseline) for smi in smi_test_all]

                result_df_train.to_csv(os.path.join(self.output_folder, "train.csv"), index=False)
                result_df_val.to_csv(os.path.join(self.output_folder, "val.csv"), index=False)
                result_df_test.to_csv(os.path.join(self.output_folder, "test.csv"), index=False)

                if self.verbose:
                    print(f"[{current_model}/{total_models}] {model_name}")

        return result_df_train, result_df_val, result_df_test
