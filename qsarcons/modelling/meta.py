import datetime
import os

from sklearn.model_selection import train_test_split

from qsarcons.consensus import GeneticSearch, RandomSearch, SystematicSearch
from qsarcons.modelling.lazy import LazyML

# Each entry knows how to build its own ConsensusSearch, since RandomSearch/SystematicSearch/GeneticSearch
# don't share one uniform constructor signature (only GeneticSearch takes n_iter/verbose).
CONSENSUS_BUILDERS = {
    "genetic": lambda cons_size, verbose: GeneticSearch(cons_size=cons_size, n_iter=50, verbose=verbose),
    "random": lambda cons_size, verbose: RandomSearch(cons_size=cons_size, n_iter=1000),
    "systematic": lambda cons_size, verbose: SystematicSearch(cons_size=cons_size),
}


class ConsensusEstimator:
    """Shared LazyML + consensus-search pipeline behind ConsensusRegressor/ConsensusClassifier."""

    _task = None
    _val_size = 0.2

    def __init__(self, consensus="genetic", hopt=True, output_folder=None, verbose=True, random_seed=42):
        """Build the underlying LazyML, defaulting output_folder to a timestamped folder name if not given."""
        if consensus not in CONSENSUS_BUILDERS:
            raise ValueError(f"Unknown consensus method {consensus!r}; choose from {list(CONSENSUS_BUILDERS)}.")

        self.consensus = consensus
        self.verbose = verbose
        self.random_seed = random_seed

        output_folder = output_folder or datetime.datetime.now().strftime("qsarcons_%d_%m_%Y_%H_%M_%S")
        self._lazy_model = LazyML(task=self._task, hopt=hopt, output_folder=output_folder, verbose=verbose)

        self.best_consensus = []
        self._consensus_search = None

    @property
    def output_folder(self):
        """Directory holding this model's files (train.csv/val.csv/test.csv)."""
        return self._lazy_model.output_folder

    def train_predict(self, smiles_train, y_train, smiles_test):
        """Train, select a model consensus, and predict on new SMILES - all in one call."""

        smi_train_all, y_train_all = list(smiles_train), list(y_train)
        idx_train, idx_val = train_test_split(
            range(len(smi_train_all)), test_size=self._val_size, random_state=self.random_seed,
        )
        smi_train = [smi_train_all[i] for i in idx_train]
        y_train_split = [y_train_all[i] for i in idx_train]
        smi_val = [smi_train_all[i] for i in idx_val]
        y_val = [y_train_all[i] for i in idx_val]

        _, result_df_val, result_df_test = self._lazy_model.run(
            smi_train, y_train_split, smi_val, y_val, list(smiles_test)
        )

        x_val, true_val = result_df_val.iloc[:, 2:], result_df_val.iloc[:, 1]

        if self.verbose:
            print("Step-4. Consensus search")

        cons_search = CONSENSUS_BUILDERS[self.consensus](cons_size="auto", verbose=self.verbose)
        best_cons = cons_search.run(x_val, true_val)

        self.best_consensus = list(best_cons)
        self._consensus_search = cons_search

        if self.verbose:
            print("Best consensus:")
            for name in self.best_consensus:
                print(f"  -{name}")

        x_test = result_df_test.iloc[:, 1:]
        pred_test = list(self._consensus_search.predict(x_test[self.best_consensus]))
        result_df_test.to_csv(os.path.join(self.output_folder, "test.csv"), index=False)

        return pred_test


class ConsensusRegressor(ConsensusEstimator):
    """ConsensusEstimator pipeline for continuous (regression) targets."""

    _task = "continuous"


class ConsensusClassifier(ConsensusEstimator):
    """ConsensusEstimator pipeline for binary classification targets."""

    _task = "binary"
