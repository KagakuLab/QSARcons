import pandas as pd
from sklearn.model_selection import train_test_split
from qsarcons.lazy import LazyML
from qsarcons.consensus import SystematicSearch, GeneticSearch

class ConsensusModel:

    def __init__(self, output_folder=None, hopt=True, verbose=False):
        self.output_folder = output_folder
        self.verbose = verbose
        self.lazy_ml = LazyML(hopt=hopt, output_folder=output_folder, verbose=verbose)
        self.cons_search = GeneticSearch(cons_size="auto", n_iter=50, verbose=verbose)

        self.best_cons = None

    def run(self, df_train, df_test):
        # 1. Fill fake test prop if needed
        if len(df_test.columns) == 1:
            df_test = df_test.copy()
            df_test[1] = None

        # 2. Train/val split
        df_train, df_val = train_test_split(df_train, test_size=0.2, random_state=42)

        # 3. Build multiple models
        self.lazy_ml.run(df_train, df_val, df_test)

        # 4. Load model predictions
        res_val = pd.read_csv(f"{self.output_folder}/val.csv")
        x_val, y_val = res_val.iloc[:, 2:], res_val.iloc[:, 1]

        # 5. Run genetic search on a validation set
        self.best_cons = self.cons_search.run(x_val, y_val)
        if self.verbose:
            print(f"Genetic consensus: {self.best_cons}")

        return self

    def predict(self, df_test):
        if self.best_cons is None:
            raise RuntimeError("Call run() before predict().")

        res_test = pd.read_csv(f"{self.output_folder}/test.csv")
        x_test = res_test.iloc[:, 2:]

        y_cons = self.cons_search.predict(x_test[self.best_cons])
        df_test["CONS_PRED"] = y_cons

        return df_test

    def run_predict(self, df_train, df_test):
        self.run(df_train, df_test)
        return self.predict(df_test)