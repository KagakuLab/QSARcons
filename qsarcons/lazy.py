# ==========================================================
# Imports
# ==========================================================
import os
import gc
import psutil
import time
import shutil
import warnings

import numpy as np
import pandas as pd

from concurrent.futures import ProcessPoolExecutor

from sklearn.base import is_classifier
from sklearn.linear_model import LogisticRegression, Ridge, RidgeClassifier
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from xgboost import XGBRegressor, XGBClassifier
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.svm import SVR, SVC, LinearSVR, LinearSVC
from sklearn.utils.multiclass import type_of_target
from sklearn.preprocessing import MinMaxScaler
from molfeat.trans import MoleculeTransformer
from molfeat.calc.pharmacophore import Pharmacophore2D

from qsarcons.hopt import StepwiseHopt, DEFAULT_PARAM_GRID_REGRESSORS, DEFAULT_PARAM_GRID_CLASSIFIERS
from qsarcons.logging import OutputSuppressor
from sklearn.utils.multiclass import type_of_target

from rdkit import RDLogger
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
    return scaler.transform(x_train), scaler.transform(x_test)

def get_predictions(estimator, X):
    if is_classifier(estimator) and hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)[:, 1].tolist()
    else:
        return estimator.predict(X).tolist()

def build_model(x_train, x_val, x_test, y_train, y_val, y_test, estimator_class, hopt=True):

    # 1. Scale train/val descriptors
    x_train_scaled, x_val_scaled = scale_descriptors(x_train, x_val)

    # 2. Optimize hyperparameters
    if hopt:
        est_name = estimator_class.__name__
        task_type = type_of_target(y_train)
        is_classification = task_type in ["binary", "multiclass"]

        param_grid = (
            DEFAULT_PARAM_GRID_CLASSIFIERS.get(est_name)
            if is_classification
            else DEFAULT_PARAM_GRID_REGRESSORS.get(est_name)
        )

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

    # 4. Retrain model on full (train + val)
    x_full, y_full = np.vstack([x_train, x_val]), np.hstack([y_train, y_val])
    x_full_scaled, x_test_scaled = scale_descriptors(x_full, x_test)

    estimator_instance.fit(x_full_scaled, y_full)
    pred_test = get_predictions(estimator_instance, x_test_scaled)

    # 5. Release memory
    del estimator_instance
    gc.collect()

    return pred_train, pred_val, pred_test

class LazyML:
    def __init__(self, hopt=True, output_folder=None, verbose=True):
        self.hopt = hopt
        self.output_folder = output_folder
        self.verbose = verbose

        if self.output_folder:
            if os.path.exists(self.output_folder):
                shutil.rmtree(self.output_folder)
            os.makedirs(self.output_folder)
        else:
            raise ValueError("output_folder must be specified.")

    def run(self, df_train, df_val, df_test):

        # 1. Get data (smiles and prop)
        result_df_train = pd.DataFrame()
        smi_train, y_train = list(df_train.iloc[:, 0]), list(df_train.iloc[:, 1])
        result_df_train["SMILES"], result_df_train["Y_TRUE"] = smi_train, y_train

        result_df_val = pd.DataFrame()
        smi_val, y_val = list(df_val.iloc[:, 0]), list(df_val.iloc[:, 1])
        result_df_val["SMILES"], result_df_val["Y_TRUE"] = smi_val, y_val

        result_df_test = pd.DataFrame()
        smi_test, y_test = list(df_test.iloc[:, 0]), list(df_test.iloc[:, 1])
        result_df_test["SMILES"], result_df_test["Y_TRUE"] = smi_test, y_test

        # Task type
        task_type = type_of_target(y_train)
        if task_type == "continuous":
            estimators_dict = REGRESSORS
        elif task_type == "binary":
            estimators_dict = CLASSIFIERS
        else:
            raise ValueError("Task type not supported.")

        total_models = len(DESCRIPTORS) * len(estimators_dict)
        current_model = 0
        # 2. Calculate descriptors
        for desc_name, desc_calc in DESCRIPTORS.items():

            x_train = calc_descriptors(smi_train, desc_calc)
            x_val = calc_descriptors(smi_val, desc_calc)
            x_test = calc_descriptors(smi_test, desc_calc)

            # 3. Train models
            for est_name, estimator in estimators_dict.items():

                model_name = f"{desc_name}|{est_name}"
                current_model += 1
                if self.verbose:
                    print(f"[{current_model}/{total_models}] Running model: {model_name}", flush=True)

                start = time.time()
                pred_train, pred_val, pred_test = build_model(
                    x_train,
                    x_val,
                    x_test,
                    y_train,
                    y_val,
                    y_test,
                    estimator,
                    self.hopt
                )
                elapsed_min = (time.time() - start) / 60

                # 4. Write predictions
                result_df_train[model_name] = pred_train
                result_df_train.to_csv(os.path.join(self.output_folder, "train.csv"), index=False)

                result_df_val[model_name] = pred_val
                result_df_val.to_csv(os.path.join(self.output_folder, "val.csv"), index=False)

                result_df_test[model_name] = pred_test
                result_df_test.to_csv(os.path.join(self.output_folder, "test.csv"), index=False)

                if self.verbose:
                    process = psutil.Process()
                    mem_gb = process.memory_info().rss / (1024 ** 3)
                    print(f"  > Finished in {elapsed_min:.2f} min | Memory usage: {mem_gb:.3f} GB")

        return None
