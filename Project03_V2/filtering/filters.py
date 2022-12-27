__all__ = [
    "box_filter",
    "FilterDirection",
    "GradientDirection",
    "XOrYFilter",
    "first_order_derivative_filter",
    "roberts_filter",
    "prewitt_filter",
    "sobel_filter",
    "FilterType",
]
import enum
from dataclasses import dataclass
from typing import Optional

import numpy as np


def box_filter(filter_size: int = 3) -> np.ndarray:
    # want to keep the summation of the filter = 1
    return np.ones((filter_size, filter_size)) * (1 / (filter_size ** 2))


@dataclass
class FilterDirection(enum.Enum):
    HORIZONTAL: str = "horizontal"
    VERTICAL: str = "vertical"


@dataclass
class GradientDirection(enum.Enum):
    LEFT: str = "left"  # gradient will decrease from left (top) to right (bottom) (negative slope)
    RIGHT: str = "right"  # gradient will increase from left (top) to right (bottom) (positive slope)


def first_order_derivative_filter(
    filter_direction: FilterDirection = FilterDirection.HORIZONTAL,
    gradient_direction: GradientDirection = GradientDirection.LEFT,
) -> np.ndarray:
    if filter_direction is FilterDirection.HORIZONTAL:
        filter_ = np.ones((1, 2))
        if gradient_direction is GradientDirection.LEFT:
            filter_[0, -1] = -1
        elif gradient_direction is GradientDirection.RIGHT:
            filter_[0, 0] = -1
        else:
            raise ValueError(f"Got unsupported gradient_direction. Got '{gradient_direction}'.")
    elif filter_direction is FilterDirection.VERTICAL:
        filter_ = np.ones((2, 1))
        if gradient_direction is GradientDirection.LEFT:
            filter_[-1, 0] = -1
        elif gradient_direction is GradientDirection.RIGHT:
            filter_[0, 0] = -1
        else:
            raise ValueError(f"Got unsupported gradient_direction. Got '{gradient_direction}'.")
    else:
        raise ValueError(f"Got unsupported filter_direction. Got '{filter_direction}'.")

    return filter_


@dataclass
class XOrYFilter(enum.Enum):
    X = "x"
    Y = "y"


def roberts_filter(
    x_or_y: Optional[XOrYFilter] = None,  # specify "x" or "y" for Mx & My, respectively.
    filter_direction: FilterDirection = FilterDirection.HORIZONTAL,
    gradient_direction: GradientDirection = GradientDirection.LEFT,
) -> np.ndarray:
    filter_ = np.zeros((2, 2))
    if x_or_y is not None:
        if x_or_y is XOrYFilter.X:
            filter_direction = FilterDirection.VERTICAL
            gradient_direction = GradientDirection.RIGHT
        elif x_or_y is XOrYFilter.Y:
            filter_direction = FilterDirection.HORIZONTAL
            gradient_direction = GradientDirection.LEFT

    if filter_direction is FilterDirection.HORIZONTAL:
        if gradient_direction is GradientDirection.LEFT:
            filter_[0, 0] = 1
            filter_[1, 1] = -1
        elif gradient_direction is GradientDirection.RIGHT:
            filter_[0, 0] = -1
            filter_[1, 1] = 1
        else:
            raise ValueError(f"Got unsupported gradient_direction. Got '{gradient_direction}'.")
    elif filter_direction is FilterDirection.VERTICAL:
        if gradient_direction is GradientDirection.LEFT:
            filter_[1, 0] = 1
            filter_[0, 1] = -1
        elif gradient_direction is GradientDirection.RIGHT:
            filter_[1, 0] = -1
            filter_[0, 1] = 1
        else:
            raise ValueError(f"Got unsupported gradient_direction. Got '{gradient_direction}'.")
    else:
        raise ValueError(f"Got unsupported filter_direction. Got '{filter_direction}'.")

    return filter_


def prewitt_filter(x_or_y: XOrYFilter = XOrYFilter.X) -> np.ndarray:
    filter_ = np.zeros((3, 3))
    if x_or_y is XOrYFilter.X:
        filter_[:, 0] = -1
        filter_[:, -1] = 1
    elif x_or_y is XOrYFilter.Y:
        filter_[0, :] = 1
        filter_[-1, :] = -1
    else:
        raise ValueError(f"Got unsupported XOrYFilter type. Got '{x_or_y}'.")

    return filter_


def sobel_filter(x_or_y: XOrYFilter = XOrYFilter.X) -> np.ndarray:
    filter_ = np.zeros((3, 3))
    if x_or_y is XOrYFilter.X:
        filter_[:, 0] = -1
        filter_[1, 0] = -2
        filter_[:, -1] = 1
        filter_[1, -1] = 2
    elif x_or_y is XOrYFilter.Y:
        filter_[0, :] = 1
        filter_[0, 1] = 2
        filter_[-1, :] = -1
        filter_[-1, 1] = -2
    else:
        raise ValueError(f"Got unsupported XOrYFilter Type. Got '{x_or_y}'.")

    return filter_


@dataclass
class FilterType(enum.Enum):
    SOBEL = "sobel"
    PREWITT = "prewitt"
    ROBERTS = "roberts"
