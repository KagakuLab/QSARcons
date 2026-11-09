import numpy as np


def detect_task(y):
    """"binary" if every label is 0/1, otherwise "continuous" (sklearn calls whole-number targets "multiclass")."""
    return "binary" if set(np.unique(y)) <= {0, 1} else "continuous"
