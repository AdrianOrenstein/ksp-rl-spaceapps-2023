"""Helper functions required in the project."""

import os
import numpy as np


def validate_output_folder(path):
    """Checks if folder exists. If not, creates it and returns its name"""
    path = os.path.join(path, "")      # to ensure path ends with '/'
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return path


def get_weights_from_npy(filename, seed_idx=-1):
    """
    Returns the weights from an npy file storing [runs, weights].

    Args:
        filename: full path of npy file which stores the weights
        seed_idx: which seed's weights to return; -1 returns average over seeds
    Returns:
        weights: average of the weights stored in the file
    """
    data = np.load(filename, allow_pickle=True).item()
    assert isinstance(data, dict)

    if seed_idx == -1:
        weights = np.mean(data['weights_final'], axis=0)
    else:
        weights = data['weights_final'][seed_idx]
    return weights
