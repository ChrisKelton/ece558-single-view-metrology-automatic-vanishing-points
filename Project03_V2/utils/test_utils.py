import numpy as np


def compare_numpy_arrays(
    arr0: np.ndarray, arr1: np.ndarray, sub_th: float = 1e-10, pass_ratio: float = 0.99, verbose: bool = False
) -> bool:
    assert arr0.shape == arr1.shape

    temp = np.subtract(arr0.copy(), arr1.copy()) <= sub_th
    vals, cnts = np.unique(temp, return_counts=True)
    false_cnt: int = 0
    true_cnt: int = 0
    for val, cnt in zip(vals, cnts):
        if val:
            true_cnt = cnt
        else:
            false_cnt = cnt

    total_cnt = true_cnt + false_cnt
    true_ratio = true_cnt / total_cnt
    false_ratio = false_cnt / total_cnt
    print(f"True % = '{100 * true_ratio}%\nFalse % = '{100 * false_ratio}%'")
    if true_ratio >= pass_ratio:
        # if verbose:
        #     print(f"Array comparison PASSED.")
        return True
    else:
        # if verbose:
        #     print(f"Array comparison FAILED.")
        return False
