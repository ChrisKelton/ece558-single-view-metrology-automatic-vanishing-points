__all__ = ["load_image_as_grayscale", "image_scaling", "change_pixel_range_of_img"]
from pathlib import Path
from typing import Union

import cv2
import numpy as np


def load_image_as_grayscale(img: Union[np.ndarray, Path, str]) -> np.ndarray:
    if isinstance(img, Path) or isinstance(img, str):
        img = cv2.imread(str(img))

    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif img.ndim == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

    return img


def image_scaling(img: np.ndarray, scaling_factor: float = 0.8) -> np.ndarray:
    # scale by 80% to get rid of edge effects, such as staircase effect.

    big_sig = 0.6
    sigma = big_sig / scaling_factor

    rows = int(img.shape[0] * scaling_factor)
    cols = int(img.shape[1] * scaling_factor)
    dim = (cols, rows)

    blurred_img = cv2.GaussianBlur(img, (5, 5), sigma, borderType=cv2.BORDER_REFLECT)
    resized = cv2.resize(blurred_img, dim, interpolation=cv2.INTER_AREA)

    return resized


def change_pixel_range_of_img(img: np.ndarray, ub: int, lb: int, *, inplace: bool = False) -> np.ndarray:
    img_max = np.max(img)
    img_min = np.min(img)
    if inplace:
        return (ub - lb) * ((img - img_min) / (img_max - img_min)) + lb

    return (ub - lb) * ((img.copy() - img_min) / (img_max - img_min)) + lb
