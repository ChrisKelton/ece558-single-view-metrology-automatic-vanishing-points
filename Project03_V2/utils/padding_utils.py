__all__ = ["PadType", "OutputSize", "pad_image", "unpad_2d_image", "padding_test_max_out"]
import enum
from dataclasses import dataclass
from pathlib import Path
from typing import Union, List, Optional, Tuple

import cv2
import numpy as np

from Projects.Project02.image_utils import show_images_side_by_side


@dataclass
class PadType(enum.Enum):
    ZERO = "zero"
    WRAP = "wrap"
    COPY = "copy"
    REFLECT = "reflect"


@dataclass
class OutputSize(enum.Enum):
    FULL = "full"
    SAME = "same"
    VALID = "valid"


def determine_pad_row_and_pad_col_from_target_shape(img: np.ndarray, target_shape: Tuple[int, int]) -> Tuple[int, int]:
    pad_row = (target_shape[0] // 2) - (img.shape[0] // 2)
    pad_col = (target_shape[1] // 2) - (img.shape[1] // 2)

    return pad_row, pad_col


# TODO: recursively call pad_image if pad_row and/or pad_col are greater than img.shape
def pad_image(
    img: np.ndarray,
    target_shape: Optional[Tuple[int, int]] = None,
    pad_row: Optional[int] = None,
    pad_col: Optional[int] = None,
    *,
    pad_type: PadType = PadType.ZERO,
) -> np.ndarray:
    if target_shape is None and (pad_row is None or pad_col is None):
        raise ValueError("Need target_shape specified or pad_row and pad_col specified.")

    if target_shape is not None:
        pad_row, pad_col = determine_pad_row_and_pad_col_from_target_shape(img, target_shape)

    if img.ndim == 3:
        temp = []
        channels_dim: int = 0
        if img.shape[-1] < 8:
            channels = img.shape[-1]
            channels_dim = -1
            for i in range(channels):
                temp.append(pad_image(img[:, :, i], pad_row, pad_col, pad_type=pad_type))
        elif img.shape[0] < 8:
            channels = img.shape[0]
            channels_dim = 0
            for i in range(channels):
                temp.append(pad_image(img[i, :, :], pad_row, pad_col, pad_type=pad_type))
        else:
            raise ValueError(f"Could not determine number of channels, given image with 3 dimensions.")

        pad_img_shape = temp[0].shape
        if channels_dim == 0:
            pad_img = np.zeros((channels, pad_img_shape[0], pad_img_shape[1]), dtype=temp[0].dtype)
            for cnt, temp_ in enumerate(temp):
                pad_img[cnt, :, :] = temp_.copy()
        else:
            pad_img = np.zeros((pad_img_shape[0], pad_img_shape[1], channels), dtype=temp[0].dtype)
            for cnt, temp_ in enumerate(temp):
                pad_img[:, :, cnt] = temp_.copy()

        return pad_img

    img_shape = img.shape
    img_pad_shape = img.shape + np.asarray([2 * pad_row, 2 * pad_col])
    target_shape_offset = [0, 0]
    if target_shape is not None:
        correct_shape_mask = img_pad_shape == target_shape
        for idx, mask in enumerate(correct_shape_mask):
            if not mask:
                offset = target_shape[idx] - img_pad_shape[idx]
                img_pad_shape[idx] += offset
                target_shape_offset[idx] = offset
    img_pad = np.zeros(img_pad_shape, dtype=img.dtype)
    img_pad[pad_row : img_shape[0] + pad_row, pad_col : img_shape[1] + pad_col] = img.copy()
    if pad_type is PadType.ZERO:
        # perform zero padding on image
        return img_pad
    elif pad_type is PadType.WRAP:
        # perform wrapping padding on image
        # Example with pad_row = 2 & pad_col = 2, img_shape = (20, 20)
        # img_pad.shape = (24, 24)
        # top columns padding
        # (0:2, 2:22) = img(-2:, :)
        if pad_row == 0:
            img_pad[:pad_row, pad_col : img_shape[1] + pad_col] = img[img_shape[0] :, :]
        else:
            img_pad[:pad_row, pad_col : img_shape[1] + pad_col] = img[-pad_row:, :]
        # bottom columns padding
        # (22:24, 2:22) = img(0:2, :)
        img_pad[img_shape[0] + pad_row :, pad_col : img_shape[1] + pad_col] = img[:pad_row, :]
        # left rows padding
        # (2:22, 0:2) = img(:, -2:)
        if pad_col == 0:
            img_pad[pad_row : img_shape[0] + pad_row, :pad_col] = img[:, img_shape[1] :]
        else:
            img_pad[pad_row : img_shape[0] + pad_row, :pad_col] = img[:, -pad_col:]
        # right rows padding
        # (2:22, 22:24) = img(:, 0:2)
        img_pad[pad_row : img_shape[0] + pad_row, img_shape[1] + pad_col :] = img[:, :pad_col]
        # top left quadrant
        # (0:2, 0:2) = img(-2:, -2:)
        if pad_row == 0 and pad_col == 0:
            img_pad[:pad_row, :pad_col] = img[img_shape[0] :, img_shape[1] :]
        elif pad_row == 0:
            img_pad[:pad_row, :pad_col] = img[img_shape[0] :, -pad_col:]
        elif pad_col == 0:
            img_pad[:pad_row, :pad_col] = img[-pad_row:, img_shape[1] :]
        else:
            img_pad[:pad_row, :pad_col] = img[-pad_row:, -pad_col:]
        # top right quadrant
        # (0:2, 22:24) = img(-2:, 0:2)
        if pad_row == 0:
            img_pad[:pad_row, img_shape[1] + pad_col :] = img[img_shape[0] :, :pad_col]
        else:
            img_pad[:pad_row, img_shape[1] + pad_col :] = img[-pad_row:, :pad_col]
        # bottom left quadrant
        # (22:24, 0:2) = img(:2, -2:)
        if pad_col == 0:
            img_pad[img_shape[0] + pad_row :, :pad_col] = img[:pad_row, img_shape[1] :]
        else:
            img_pad[img_shape[0] + pad_row :, :pad_col] = img[:pad_row, -pad_col:]
        # bottom right quadrant
        # (22:24, 22:24) = img(:2, :2)
        img_pad[img_shape[0] + pad_row :, img_shape[1] + pad_col :] = img[:pad_row, :pad_col]
    elif pad_type is PadType.COPY:
        # perform copy padding on image
        # Example with pad_row = 2 & pad_col = 2, img_shape = (20, 20)
        # img_pad.shape = (24, 24)
        # top columns padding
        # (:2, 2:22) = replicated(img(0, :), 2 [pad_rows])
        img_pad[:pad_row, pad_col : img_shape[1] + pad_col] = np.tile(
            np.expand_dims(img[0, :], 1), (pad_row, 1)
        ).reshape(pad_row, img.shape[1])
        # bottom columns padding
        # (22:24, 2:22) = replicated(img(-1, :), 2 [pad_rows])
        img_pad[img_shape[0] + pad_row :, pad_col : img_shape[1] + pad_col] = np.tile(
            np.expand_dims(img[-1, :], 1), (pad_row, 1)
        ).reshape(pad_row, img.shape[1])
        # left rows padding
        # (2:22, 0:2) = replicated(img(:, 0), 2 [pad_cols])
        img_pad[pad_row : img_shape[0] + pad_row, :pad_col] = np.tile(
            np.expand_dims(img[:, 0], 1), (1, pad_col)
        ).reshape(img.shape[0], pad_col)
        # right rows padding
        # (2:22, 22:24) = replicated(img(:, -1), 2 [pad_cols])
        img_pad[pad_row : img_shape[0] + pad_row, img_shape[1] + pad_col :] = np.tile(
            np.expand_dims(img[:, -1], 1), (1, pad_col)
        ).reshape(img.shape[0], pad_col)
        # top left quadrant
        # (0:2, 0:2) = np.ones(2, 2) * img(0, 0)
        img_pad[:pad_row, :pad_col] = np.ones((pad_row, pad_col)) * img[0, 0]
        # top right quadrant
        # (0:2, 22:24) = np.ones(2, 2) * img(0, -1)
        img_pad[:pad_row, img_shape[1] + pad_col :] = np.ones((pad_row, pad_col)) * img[0, -1]
        # bottom left quadrant
        # (22:24, 0:2) = np.ones(2, 2) * img(-1, 0)
        img_pad[img_shape[0] + pad_row :, :pad_col] = np.ones((pad_row, pad_col)) * img[-1, 0]
        # bottom right quadrant
        # (22:24, 22:24) = np.ones(2, 2) * img(-1, -1)
        img_pad[img_shape[0] + pad_row :, img_shape[1] + pad_col :] = np.ones((pad_row, pad_col)) * img[-1, -1]
    elif pad_type is PadType.REFLECT:
        # reflect padding across edges
        # top columns padding
        # (0:2, 2:22) = img(2:0:-1, :)
        if pad_row == 0:
            img_pad[:pad_row, pad_col : img_shape[1] + pad_col] = img[pad_row, :]
        else:
            img_pad[:pad_row, pad_col : img_shape[1] + pad_col] = img[pad_row - 1 :: -1, :]
        # bottom columns padding
        # (22:24, 2:22) = img(img_shape[0]-1:img_shape[0]-pad_row-1:-1, :)
        if pad_row == img_shape[0]:
            img_pad[img_shape[0] + pad_row :, pad_col : img_shape[1] + pad_col] = img[img_shape[0] - 1 :: -1, :]
        else:
            img_pad[img_shape[0] + pad_row :, pad_col : img_shape[1] + pad_col] = img[
                img_shape[0] - 1 : img_shape[0] - pad_row - 1 : -1, :
            ]
        # left row padding
        # (2:22, 0:2) = img(:, 2:0:-1)
        if pad_col == 0:
            img_pad[pad_row : img_shape[0] + pad_row, :pad_col] = np.expand_dims(img[:, pad_col], 1)
        else:
            img_pad[pad_row : img_shape[0] + pad_row, :pad_col] = img[:, pad_col - 1 :: -1]
        # right row padding
        # (2:22, 22:24) = img(:, img_shape[1]-1:img_shape[1]-pad_col-1:-1, :)
        if pad_col == img_shape[1]:
            img_pad[pad_row : img_shape[0] + pad_row, img_shape[1] + pad_col :] = img[:, img_shape[1] - 1 :: -1]
        else:
            img_pad[pad_row : img_shape[0] + pad_row, img_shape[1] + pad_col :] = img[
                :, img_shape[1] - 1 : img_shape[1] - pad_col - 1 : -1
            ]
        # top left quadrant
        # (0:2, 0:2) = np.flip(img(0:2, 0:2))
        img_pad[:pad_row, :pad_col] = np.flip(img[:pad_row, :pad_col])
        # top right quadrant
        # (0:2, 22:24) = np.flip(img(0:2, -2:))
        if pad_col == 0:
            img_pad[:pad_row, img_shape[1] + pad_col :] = np.flip(img[:pad_row, img_shape[1] :])
        else:
            img_pad[:pad_row, img_shape[1] + pad_col :] = np.flip(img[:pad_row, -pad_col:])
        # bottom left quadrant
        # (22:24, 0:2) = np.flip(img(-2:, 0:2)
        if pad_row == 0:
            img_pad[img_shape[0] + pad_row :, :pad_col] = np.flip(img[img_shape[0] :, :pad_col])
        else:
            img_pad[img_shape[0] + pad_row :, :pad_col] = np.flip(img[-pad_row:, :pad_col])
        # bottom right quadrant
        # (22:24, 22:24) = np.flip(img(-2:, -2:))
        if pad_row == 0 and pad_col == 0:
            img_pad[img_shape[0] + pad_row :, img_shape[1] + pad_col :] = np.flip(img[img_shape[0] :, img_shape[1] :])
        elif pad_col == 0:
            img_pad[img_shape[0] + pad_row :, img_shape[1] + pad_col :] = np.flip(img[-pad_row:, img_shape[1] :])
        elif pad_row == 0:
            img_pad[img_shape[0] + pad_row :, img_shape[1] + pad_col :] = np.flip(img[img_shape[0] :, -pad_col:])
        else:
            img_pad[img_shape[0] + pad_row :, img_shape[1] + pad_col :] = np.flip(img[-pad_row:, -pad_col:])
    else:
        raise ValueError(f"Got unsupported padding type. Got '{pad_type.name}'.")

    del img
    return img_pad


def unpad_2d_image(img: np.ndarray, pad_row: int, pad_col: int) -> np.ndarray:
    if img.ndim != 2:
        raise ValueError(f"Got image not with 2 dimensions. Got '{img.shape}'.")
    return img[pad_row : img.shape[0] - pad_row, pad_col : img.shape[1] - pad_col]


def padding_test_max_out(img: Union[str, Path, np.ndarray], output_path: Optional[Union[str, Path]] = None):
    if isinstance(img, str) or isinstance(img, Path):
        img = cv2.imread(str(img))
    if not Path(output_path).exists():
        Path(output_path).mkdir(exist_ok=True, parents=True)

    pad_types: List[PadType] = [PadType.ZERO, PadType.COPY, PadType.WRAP, PadType.REFLECT]
    pad_imgs = {}
    for pad_type in pad_types:
        pad_img = pad_image(img.copy(), img.shape[0], img.shape[1], pad_type=pad_type)
        pad_imgs.update({pad_type.value: pad_img.copy()})
        if output_path is not None:
            temp_output_path = str(output_path) + "/" + pad_type.value + ".png"
            cv2.imwrite(temp_output_path, pad_img.astype("uint8"))

    show_images_side_by_side(
        [
            [pad_imgs.get(pad_types[0].value), pad_imgs.get(pad_types[1].value)],
            [pad_imgs.get(pad_types[2].value), pad_imgs.get(pad_types[3].value)],
        ],
        [[pad_types[0].value, pad_types[1].value], [pad_types[2].value, pad_types[3].value]],
        close_plot=False,
    )


def main():
    img_path = "./lena.png"
    output_path = "./padding-tests"
    Path(output_path).mkdir(exist_ok=True, parents=True)
    padding_test_max_out(img_path, output_path)


if __name__ == "__main__":
    main()
