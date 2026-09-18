
QSARcons - smart search for consensus of QSAR models
--------------------------------------------------------------------

The motivation behind this project is that there are many available chemical descriptors and machine learning
methods, and usually, it is not obvious which combination to prefer for modelling the target property of the
molecule. Therefore, the idea is just to build multiple (>100) simple individual QSAR models with diverse
descriptors and algorithms, and then search for the subset of models whose consensus performs best on a
validation dataset.

**Ready to use:**

- 2D descriptor calculation (RDKit and MolFeat fingerprints, physicochemical descriptors, and pharmacophores)
- Traditional machine learning algorithms (Ridge, PLS, SVR, Random Forest, XGBoost, MLP), with optional
  in-house stepwise hyperparameter optimization
- Consensus model search: random, systematic, and genetic search strategies
- Regression and binary classification tasks are currently supported

**Under development:**

- **QSARcons Pro:** additional workflows for building individual models (e.g. ``chemprop`` and ``QSARmil``) to
  combine traditional and advanced modelling approaches into stronger consensuses

Installation
--------------------------------------------------------------------

.. code-block:: bash

    pip install qsarcons

Beginner usage
--------------------------------------------------------------------

For a predefined pipeline that takes zero QSAR knowledge and zero setup: hand it your training data and get
predictions for new molecules back.

.. code-block:: python

    from qsarcons.modelling.meta import ConsensusRegressor

    smiles_train = [
        "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
        "COc1ccc2cc(ccc2c1)C(C)C(=O)O",
        "OC(=O)Cc1ccccc1Nc1c(Cl)cccc1Cl",
        "COc1ccc2c(c1)c(CC(=O)O)c(C)n2C(=O)c1ccc(Cl)cc1",
        "OC(=O)C(C)c1cccc(c1)C(=O)c1ccccc1",
        "CC(=O)Oc1ccccc1C(=O)O",
        "Cc1ccc(cc1)S(=O)(=O)N",
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        "OC(=O)c1ccccc1O",
        "Clc1ccc(cc1)C(c1ccc(Cl)cc1)C(Cl)(Cl)Cl",
        "CC(C)NCC(O)COc1cccc2ccccc12",
        "CCOC(=O)c1ccc(N)cc1",
    ]
    y_train = [5.2, 5.9, 6.1, 7.0, 5.6, 4.9, 3.8, 4.1, 4.6, 6.8, 5.0, 4.4]

    smiles_test = ["CC(C(=O)O)Oc1cccc(c1)-c1ccccc1", "CC(C(=O)O)c1ccc(cc1)-c1ccc(F)cc1"]

    # train and predict in one call - there's no separate save/load step
    model = ConsensusRegressor(consensus="genetic", hopt=False, verbose=True)
    y_pred = model.train_predict(smiles_train, y_train, smiles_test)

Use ``ConsensusRegressor`` for continuous properties and ``ConsensusClassifier`` for binary classification.
The ``consensus`` argument selects the search strategy: ``"genetic"`` (default), ``"random"``, or ``"systematic"``.
See the full walkthrough in
`Notebook_1_QSARcons_pipeline.ipynb <colab/Notebook_1_QSARcons_pipeline.ipynb>`_.

Professional usage
--------------------------------------------------------------------

Modify or build your own modelling pipeline by combining ``QSARcons``'s individual modules (descriptor
calculators, the individual model builder in ``qsarcons.modelling.lazy``, and the consensus search strategies
in ``qsarcons.consensus``) directly. See
`Notebook_1_QSARcons_pipeline.ipynb <colab/Notebook_1_QSARcons_pipeline.ipynb>`__ for a full example that loads an
external benchmark dataset, builds the individual model library, and compares all three consensus strategies.

Tutorials
--------------------------------------------------------------------

- `Notebook_1_QSARcons_pipeline.ipynb <colab/Notebook_1_QSARcons_pipeline.ipynb>`__ - the full pipeline, from raw
  SMILES to individual models to a searched consensus, also runnable directly in
  `Colab <https://colab.research.google.com/github/KagakuAI/QSARcons/blob/main/colab/Notebook_1_QSARcons_pipeline.ipynb>`_.

QSARcons Basic vs. QSARcons Pro
--------------------------------------------------------------------
The QSARcons idea is that diverse and strong individual models can be combined to even stronger consensus.
Currently, two versions are under development:

- **QSARcons Basic:** includes ``RDKit`` descriptors + ``scikit-learn`` ML methods
- **QSARcons Pro:** will include **QSARcons Basic** + other workflows for building individual models (e.g. ``chemprop`` and ``QSARmil``) to combine traditional and advanced modelling approaches into stronger consensuses
