__all__ = [
    "fftshift",
    "ifftshift",
    "dft_2d",
    "idft_2d",
    "magnitude_and_phase_from_complex_image",
    "log_of_frequency_image",
    "fourier_transform_test",
]
from collections import namedtuple
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np

from Projects.Project02.image_utils import *
from Projects.Project02.test_utils import compare_numpy_arrays

MagAndPhase = namedtuple("MagAndPhase", "magnitude phase")


def fftshift_(arr: np.ndarray, src_origin: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    this is simply a circular shift of the pixels. E.g.,

        [1, 2, 3, 4, 5]      [4, 5, 6, 2, 3]
        [2, 3, 4, 5, 6] ---> [5, 6, 7, 3, 4]
        [3, 4, 5, 6, 7]      [3, 4, 5, 1, 2]

    :param arr: input array (image) to be shifted. already in the frequency domain.
    :param src_origin: location of origin of original image (will be arr.shape[0] // 2, arr.shape[1] // 2)
    :return: fftshifted arr
    """
    arr_shape = arr.shape
    if src_origin is None:
        src_origin = (arr_shape[0] // 2, arr_shape[1] // 2)

    num_of_cols_right_side = arr_shape[1] - src_origin[1]
    num_of_cols_left_side = arr_shape[1] - num_of_cols_right_side
    num_of_rows_bottom_side = arr_shape[0] - src_origin[0]

    bottom_left_arr = arr[src_origin[0] :, :num_of_cols_left_side].copy()
    top_left_arr = arr[: src_origin[0], :num_of_cols_left_side].copy()
    left_arr = np.zeros((arr_shape[0], num_of_cols_left_side), dtype=arr.dtype)
    left_arr[:num_of_rows_bottom_side, :] = bottom_left_arr
    left_arr[num_of_rows_bottom_side:, :] = top_left_arr

    bottom_right_arr = arr[src_origin[0] :, num_of_cols_left_side:].copy()
    top_right_arr = arr[: src_origin[0], num_of_cols_left_side:].copy()
    right_arr = np.zeros((arr_shape[0], num_of_cols_right_side), dtype=arr.dtype)
    right_arr[:num_of_rows_bottom_side, :] = bottom_right_arr
    right_arr[num_of_rows_bottom_side:, :] = top_right_arr

    new_arr = np.zeros(arr_shape, dtype=arr.dtype)
    new_arr[:, :num_of_cols_right_side] = right_arr
    new_arr[:, num_of_cols_right_side:] = left_arr

    return new_arr


def fftshift(
    arr: np.ndarray, src_origin: Optional[Tuple[int, int]] = None, dst_origin: Optional[Tuple[int, int]] = None
) -> np.ndarray:

    channel_extension: bool = False
    if arr.ndim == 2:
        arr = np.expand_dims(arr, -1)
        channel_extension = True
    elif arr.ndim != 3:
        raise ValueError(f"Got array of unsupported shape. '{arr.shape}'.")

    new_arr = np.zeros(arr.shape, dtype=arr.dtype)
    if src_origin is None:
        src_origin = (arr.shape[0] // 2, arr.shape[1] // 2)
    if dst_origin is None:
        dst_origin = (0, 0)

    for i in range(arr.shape[-1]):
        new_arr[:, :, i] = fftshift_(arr[:, :, i].copy(), src_origin)

    if channel_extension:
        return np.squeeze(new_arr)

    return new_arr


def ifftshift_(arr: np.ndarray) -> np.ndarray:
    rows = arr.shape[0]
    cols = arr.shape[1]
    if rows % 2 == 0:
        lim_rows = 0
    else:
        lim_rows = 1
    if cols % 2 == 0:
        lim_cols = 0
    else:
        lim_cols = 1

    # since the fftshift is simply a circular shift of the image pixels, we can simply split the image up into
    # four quadrants and rearrange them appropriately.
    iarr = np.zeros((rows, cols), dtype=arr.dtype)
    iarr[: (rows // 2) + lim_rows, :] = arr[(rows // 2) :, :].copy()
    iarr[(rows // 2) + lim_rows :, :] = arr[: (rows // 2), :].copy()
    temp = iarr.copy()
    temp[:, : (cols // 2) + lim_cols] = iarr[:, (cols // 2) :].copy()
    temp[:, (cols // 2) + lim_cols :] = iarr[:, : (cols // 2)].copy()

    return temp


def ifftshift(arr: np.ndarray,) -> np.ndarray:

    channel_extension: bool = False
    if arr.ndim == 2:
        arr = np.expand_dims(arr, -1)
        channel_extension = True
    elif arr.ndim != 3:
        raise ValueError(f"Got array of unsupported shape. '{arr.shape}'")

    new_arr = np.zeros(arr.shape, dtype=arr.dtype)

    for i in range(arr.shape[-1]):
        new_arr[:, :, i] = ifftshift_(arr[:, :, i].copy())

    if channel_extension:
        return np.squeeze(new_arr)

    return new_arr


def ft_1d(dim_size: int, inverse: bool = False) -> np.ndarray:
    n = np.asmatrix(np.arange(dim_size))
    inv_exp_const = 1
    inv_norm_const = 1
    if inverse:
        # if inverse dft, then multiply the exponent by -1 & scale the total signal appropriately by the number of
        # samples in the 1D signal.
        inv_exp_const = -1
        inv_norm_const = dim_size

    # return the M-point transformation, W_M.
    # W_M is made up of elements w_M = e^(-i * 2pi / M); where M = the number of samples in our signal (dim_size).
    # each row of W_M is a polynomial of linear combination of w_M elements raised to a power related to the number of
    # samples in the signal.
    # E.g.
    #       [1  1           1           ...    1               ]
    #       [1  w_M         w_M^2       ...    w_M^(M-1)       ]
    #       [.  ...         ...         ...     ...            ]
    #       [1  w_M^(M-1)   w_M^2(M-1)  ...    w_M^((M-1)(M-1))]
    ret_val = np.exp((-2j * inv_exp_const * np.pi / dim_size) * n.T * n)
    return ret_val / inv_norm_const


def magnitude_and_phase_from_complex_image(complex_img: np.ndarray) -> MagAndPhase:
    real_img = complex_img.real
    imag_img = complex_img.imag
    magnitude = np.sqrt(real_img.copy() ** 2 + imag_img.copy() ** 2)
    # have to use arctan2 b/c this allows for full range of the 4 quadrants of the unit circle.
    # arctan only ranges from [-pi/2, pi/2]
    phase = np.arctan2(imag_img, real_img)

    return MagAndPhase(magnitude, phase)


def log_of_frequency_image(
    img: np.ndarray, return_magnitude_and_phase: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, MagAndPhase]]:

    mag_and_phase = magnitude_and_phase_from_complex_image(img.copy())
    # better visualization available using log transformation. The DC portion of the image (lower frequencies) take
    # up a larger portion of the frequency spectrum, so the values at the center of the image will be much higher than
    # values further away from the center.
    log_img = np.log(1 + mag_and_phase.magnitude)

    if return_magnitude_and_phase:
        return log_img, mag_and_phase

    return log_img


def dft_2d(
    img: np.ndarray,
    center_dc: bool = True,
    origin: Optional[Tuple[int, int]] = None,
    force_change_of_pixel_range: bool = False,
) -> np.ndarray:

    channel_extension: bool = False
    if img.ndim == 2:
        img = np.expand_dims(img, -1)
        channel_extension = True
    elif img.ndim != 3:
        raise ValueError(f"Got unsupported image shape. '{img.shape}'.")

    if force_change_of_pixel_range:
        if img.max() != 1 or img.min() != 0:
            img = change_pixel_range_of_img(img, 1, 0)

    # get 1D M-transformation matrices in the x and y directions for the image, given the image shape.
    rows_f = ft_1d(img.shape[0])
    cols_f = ft_1d(img.shape[1])
    img_f = np.zeros(img.shape, dtype=complex)
    for i in range(img.shape[-1]):
        # multiply the M-transformation matrices against each other and the image signal to get our frequency
        # transformation.
        temp = rows_f * img[:, :, i].copy() * cols_f
        if center_dc:
            temp = fftshift(temp, origin)
        img_f[:, :, i] = temp

    if channel_extension:
        return np.squeeze(img_f)

    return img_f


def idft_2d(img: np.ndarray, uncenter_dc: bool = True) -> np.ndarray:
    channel_extension: bool = False
    if img.ndim == 2:
        img = np.expand_dims(img, -1)
        channel_extension = True
    elif img.ndim != 3:
        raise ValueError(f"Got unsupported image shape. '{img.shape}'.")

    # get 1D inverse M-transformation matrices in the x and y directions of the image, given the image shape.
    rows_if = ft_1d(img.shape[0], inverse=True)
    cols_if = ft_1d(img.shape[1], inverse=True)
    img_if = np.zeros(img.shape, dtype=complex)
    for i in range(img.shape[-1]):
        if uncenter_dc:
            img[:, :, i] = ifftshift(img[:, :, i])
        img_if[:, :, i] = rows_if * img[:, :, i].copy() * cols_if

    if channel_extension:
        return np.squeeze(img_if)

    return img_if


def fourier_transform_test(img: Union[Path, str, np.ndarray], comparison_th: float = 1e-10):
    if isinstance(img, Path) or isinstance(img, str):
        img = load_image_as_grayscale(img)

    print(
        f"Comparing Own Implementation of Fourier Transform Methods to Numpy's.\n"
        f"Comparisons will be made with a threshold of '{comparison_th}'."
    )
    img_dft = dft_2d(img.copy(), center_dc=False, force_change_of_pixel_range=True)
    np_fft = np.fft.fft2(change_pixel_range_of_img(img.copy(), 1, 0))
    print(f"Comparing 2D-DFT with Numpy FFT")
    compare_numpy_arrays(img_dft.copy(), np_fft.copy(), sub_th=comparison_th)

    img_fshift = fftshift(img_dft.copy())
    np_fshift = np.fft.fftshift(np_fft.copy())
    print(f"Comparing FFTShifted Images")
    compare_numpy_arrays(img_fshift.copy(), np_fshift.copy(), sub_th=comparison_th)

    log_img = log_of_frequency_image(img_fshift.copy())
    mag_and_phase_img = magnitude_and_phase_from_complex_image(img_dft.copy())
    np_log_img = log_of_frequency_image(np_fshift.copy())
    np_mag_and_phase_img = magnitude_and_phase_from_complex_image(np_fft.copy())
    print("Comparing Magnitude Images")
    compare_numpy_arrays(
        mag_and_phase_img.magnitude.copy(), np_mag_and_phase_img.magnitude.copy(), sub_th=comparison_th
    )
    print("Comparing Phase Images")
    compare_numpy_arrays(mag_and_phase_img.phase.copy(), np_mag_and_phase_img.phase.copy(), sub_th=comparison_th)
    print("Comparing Log Images")
    compare_numpy_arrays(log_img.copy(), np_log_img.copy(), sub_th=comparison_th)

    show_images_side_by_side(
        [[log_img, np_log_img], [mag_and_phase_img.phase, np_mag_and_phase_img.phase]],
        [["My Log Spectrum", "Numpy's Log Spectrum"], ["My Phase", "Numpy's Phase"]],
        wait_key=True,
    )

    img_dft_ifshift = ifftshift(img_fshift.copy())
    np_fft_ifshift = np.fft.ifftshift(np_fshift.copy())
    print("Comparing IFFTShifted Images.")
    compare_numpy_arrays(img_dft_ifshift.copy(), np_fft_ifshift.copy(), sub_th=comparison_th)

    img_idft = abs(idft_2d(img_dft_ifshift.copy(), uncenter_dc=False))
    np_idft = abs(np.fft.ifft2(np_fft_ifshift.copy()))
    print("Comparing IDFT & IFFT Images.")
    compare_numpy_arrays(img_idft.copy(), np_idft.copy(), sub_th=comparison_th)

    show_images_side_by_side([img_idft, np_idft, img], ["My Image", "Numpy Image", "Original Image"], wait_key=True)

    original_img_idft = change_pixel_range_of_img(img_idft.copy(), img.max(), img.min())
    sub_original_img_and_idft_img = abs(np.subtract(original_img_idft.copy(), img.copy())).astype("uint8")
    print("Comparing Original and IDFT Image.")
    compare_numpy_arrays(img.copy(), original_img_idft.copy(), sub_th=comparison_th)
    cv2.imshow("Subtracted Image", sub_original_img_and_idft_img)
    cv2.waitKey(0)


def main():
    img_path = Path("./lena.png")
    fourier_transform_test(img_path)


if __name__ == "__main__":
    main()
