__all__ = ["compute_gradients"]

from typing import Callable, Tuple, Optional

import cv2
import numpy as np

from Projects.Project03_V2.filtering.filters import FilterType, XOrYFilter, sobel_filter, roberts_filter, prewitt_filter
from Projects.Project03_V2.utils.frequency_utils import dft_2d, idft_2d, ifftshift
from Projects.Project03_V2.utils.padding_utils import pad_image, PadType


def compute_gradients(
    img: np.ndarray,
    x_filt: Optional[np.ndarray] = None,
    y_filt: Optional[np.ndarray] = None,
    filter_type: FilterType = FilterType.SOBEL,
    pad_type: PadType = PadType.ZERO,
    use_cv2: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    if img.ndim != 2:
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif img.ndim == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        else:
            raise ValueError(f"Got unexpected number of dimensions for image. Got '{img.ndim}' dimensions")

    if not use_cv2:
        pad_row = 1
        pad_col = 1
        if x_filt is None and y_filt is None:
            if filter_type is FilterType.SOBEL:
                filter_fun: Callable = sobel_filter
            elif filter_type is FilterType.ROBERTS:
                filter_fun: Callable = roberts_filter
            elif filter_type is FilterType.PREWITT:
                filter_fun: Callable = prewitt_filter
            else:
                raise ValueError(f"Got unrecognized filter_type '{filter_type}'.")

            x_filt = filter_fun(XOrYFilter.X)
            y_filt = filter_fun(XOrYFilter.Y)
        else:
            if x_filt is None:
                if y_filt.shape[0] != y_filt.shape[1]:
                    raise ValueError("Non-square kernels not supported")
                x_filt = y_filt.T
            elif y_filt is None:
                if x_filt.shape[0] != x_filt.shape[1]:
                    raise ValueError("Non-square kernels not supported")
                y_filt = x_filt.T
            pad_row = (x_filt.shape[0] - 1) // 2
            pad_col = (y_filt.shape[0] - 1) // 2

        img_pad = pad_image(img=img, pad_row=pad_row, pad_col=pad_col, pad_type=pad_type)
        x_filt_pad = pad_image(img=x_filt, target_shape=img_pad.shape)
        y_filt_pad = pad_image(img=y_filt, target_shape=img_pad.shape)
        img_pad_dft = dft_2d(img_pad, center_dc=True)
        x_filt_pad_dft = dft_2d(x_filt_pad, center_dc=True)
        y_filt_pad_dft = dft_2d(y_filt_pad, center_dc=True)

        dx_img_freq = np.multiply(img_pad_dft.copy(), x_filt_pad_dft)
        dx_img = idft_2d(dx_img_freq.copy())
        dx_img = ifftshift(dx_img.copy())

        dy_img_freq = np.multiply(img_pad_dft.copy(), y_filt_pad_dft)
        dy_img = idft_2d(dy_img_freq.copy())
        dy_img = ifftshift(dy_img)

        mag_img_64f = np.sqrt(abs(dx_img.copy()) + abs(dy_img.copy()))
        mag_img = cv2.convertScaleAbs(mag_img_64f.copy(), alpha=255 / mag_img_64f.max())

        # level-line angle is computed as the tangent to the the level line going through their base point
        # that's why it's x/-y
        phase_img = np.arctan2(dx_img.real, -dy_img.real)  # [pi/2, pi]

        if pad_row == 0 and pad_col == 0:
            return mag_img, phase_img
        elif pad_row == 0:
            return mag_img[:, pad_col:-pad_col], phase_img[:, pad_col:-pad_col]
        elif pad_col == 0:
            return mag_img[pad_row:-pad_row, :], phase_img[pad_row:-pad_row, :]

        return mag_img[pad_row:-pad_row, pad_col:-pad_col], phase_img[pad_row:-pad_row, pad_col:-pad_col]
    else:
        dx = cv2.Sobel(
            img.copy(), ddepth=cv2.CV_16S, dx=1, dy=0, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT
        )
        dy = cv2.Sobel(
            img.copy(), ddepth=cv2.CV_16S, dx=0, dy=1, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT
        )
        abs_grad_x = cv2.convertScaleAbs(dx)
        abs_grad_y = cv2.convertScaleAbs(dy)
        mag_img = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
        phase_img = np.arctan2(dx, -dy)

        return mag_img, phase_img
