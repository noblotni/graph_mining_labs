import numba
import numpy as np


@numba.njit()
def set_diff1d(arr1: np.ndarray, arr2: np.ndarray):
    arr1 = np.unique(arr1)
    arr2 = np.unique(arr2)
    inds_to_remove = np.array([-1])
    for ind, value in enumerate(arr1):
        if value in arr2:
            inds_to_remove = np.append(inds_to_remove, ind)
    return np.delete(arr1, inds_to_remove)
