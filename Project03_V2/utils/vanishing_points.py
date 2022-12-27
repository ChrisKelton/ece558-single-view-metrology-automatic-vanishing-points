import enum
from typing import List, Tuple, Union, Optional

import numpy as np
from Projects.Project03_V2.line_segment_detector.rectangle import angle_difference


def second_moment(V: np.ndarray) -> np.ndarray:
    M = np.zeros((3, 3), dtype=np.float)
    for idx in range(V.shape[0]):
        v = np.squeeze(V[idx])
        M[0, 0] += v[0] * v[0]
        M[0, 1] += v[0] * v[1]
        M[0, 2] += v[0] * v[2]
        M[1, 0] += v[0] * v[1]
        M[1, 1] += v[1] * v[1]
        M[1, 2] += v[1] * v[2]
        M[2, 0] += v[0] * v[2]
        M[2, 1] += v[1] * v[2]
        M[2, 2] += v[2] * v[2]

    if np.isnan(M).any():
        a = 0

    return M


def homogenous_lines(V: np.ndarray) -> np.ndarray:
    homo_lines = np.zeros((V.shape[0], V.shape[1], 3))
    for idx in range(V.shape[0]):
        line_points = np.squeeze(V[idx])
        point0 = line_points[:3]
        point1 = line_points[3:]
        cross_ = np.cross(point0, point1)
        if cross_[2] == 0:
            return None
        cross_ = cross_ / cross_[2]
        homo_lines[idx] = cross_

    return homo_lines


# uniform scaling in x, y coordinates
def isotropic_scaling(row_vals: List[int], col_vals: List[int]) -> Tuple[float, float, float, float, np.ndarray]:
    col_bar = float(np.mean(col_vals))
    row_bar = float(np.mean(row_vals))
    rotation_val = 0
    for col_val, row_val in zip(col_vals, row_vals):
        rotation_val += np.sqrt(((col_val - col_bar) ** 2) + ((row_val - row_bar) ** 2))
    rotation_val /= len(col_vals) * np.sqrt(2)
    T = np.zeros((3, 3))
    T[0, 0] = 1 / rotation_val
    T[1, 1] = 1 / rotation_val
    T[0, 2] = -col_bar / rotation_val
    T[1, 2] = -row_bar / rotation_val
    T[2, 2] = 1

    return row_bar, col_bar, rotation_val, rotation_val, T


# non-uniform scaling in x, y coordinates
def anisotropic_scaling(
    row_vals: List[int], col_vals: List[int], img_shape: Tuple
) -> Tuple[float, float, float, float, np.ndarray]:
    col_bar = float(np.mean(col_vals))
    row_bar = float(np.mean(row_vals))
    col_scale = 0
    row_scale = 0
    for col_val, row_val in zip(col_vals, row_vals):
        col_scale += np.sqrt(((col_val - col_bar) ** 2) + ((row_val - (int(img_shape[0] / 2) - 1)) ** 2))
        row_scale += np.sqrt(((row_val - row_bar) ** 2) + ((col_val - (int(img_shape[1] / 2) - 1)) ** 2))
    col_scale /= len(col_vals)
    row_scale /= len(row_vals)
    T = np.zeros((3, 3))
    T[0, 0] = 1 / col_scale
    T[1, 1] = 1 / row_scale
    T[0, 2] = -col_bar / col_scale
    T[1, 2] = -row_bar / row_scale
    T[2, 2] = 1

    return row_bar, col_bar, row_scale, col_scale, T


def normalize_lines(
    line_group: np.ndarray, row_shift: float, col_shift: float, row_scale: float, col_scale: float
) -> np.ndarray:

    norm_line_group = np.zeros(line_group.shape)
    translation = np.asarray((row_shift, col_shift, 0))
    for lg_idx, endpoints in enumerate(line_group):
        norm_endpoints = np.zeros(endpoints.shape)
        for ep_idx, endpoint in enumerate(endpoints):
            temp = endpoint - translation
            temp[0] /= row_scale
            temp[1] /= col_scale
            temp[-1] = endpoint[-1]
            norm_endpoints[ep_idx] = temp
        norm_line_group[lg_idx] = norm_endpoints

    return norm_line_group


def euclidean_dst(pt_0: Union[np.ndarray, Tuple[int]], pt_1: Union[np.ndarray, Tuple[int]]) -> float:
    return np.sqrt(np.sum(np.square(pt_0 - pt_1)))


class NormalizeLinesTechnique(enum.Enum):
    ISOTROPIC: str = "isotropic"
    ANISOTROPIC: str = "anisotropic"


# def prep_lines_cv2_for_vanishing_points_by_normalization(
#     lines: np.ndarray,
#     insert_val: float = 1
# ) -> np.ndarray:
#     new_lines = np.zeros((lines.shape[0], lines.shape[1], lines.shape[2], 6))
#     for idx, parallel_lines in enumerate(lines):
#         new_parallel_lines = np.zeros((lines.shape[1], lines.shape[2], 6))
#         for pidx, line in enumerate(parallel_lines):
#             new_parallel_lines[pidx] = np.insert(line, [2, 4], insert_val)
#         new_lines[idx] = new_parallel_lines
#
#     return new_lines


def prep_lines_cv2_for_vanishing_points_by_normalization(lines: np.ndarray, insert_val: float = 1) -> np.ndarray:
    new_lines = np.zeros((lines.shape[0], lines.shape[1], lines.shape[2], 6))
    for idx, parallel_lines in enumerate(lines):
        new_parallel_lines = np.zeros((lines.shape[1], lines.shape[2], 6))
        for pidx, line in enumerate(parallel_lines):
            new_parallel_lines[pidx] = np.insert(line, [2, 4], insert_val)
        new_lines[idx] = new_parallel_lines

    return new_lines


def get_vanishing_points_by_normalization(
    grouped_lines: np.ndarray,
    row_mean: Optional[float] = None,
    col_mean: Optional[float] = None,
    row_scale: Optional[float] = None,
    col_scale: Optional[float] = None,
    transformation_matrix: Optional[np.ndarray] = None,
) -> np.ndarray:
    """

    :param grouped_lines:
    :param row_mean:
    :param col_mean:
    :param row_scale:
    :param col_scale:
    :param transformation_matrix:
    :return: vanishing points grouped by coordinate_dir
    """
    vanishing_points = np.zeros((len(grouped_lines), 1, 3))
    for idx, line_group in enumerate(grouped_lines):
        if transformation_matrix is not None:
            norm_line_group = normalize_lines(line_group, row_mean, col_mean, row_scale, col_scale)
        else:
            norm_line_group = line_group
        norm_homogenous_lines = homogenous_lines(norm_line_group)
        if norm_homogenous_lines is None:
            vanishing_points[idx] = None
        else:
            M = second_moment(norm_homogenous_lines)
            if transformation_matrix is not None:
                M = transformation_matrix.T @ M @ transformation_matrix
            eigvals, eigvects = np.linalg.eigh(M)
            smallest_eigval_idx = list(eigvals).index(np.min(eigvals))
            smallest_eigvect = eigvects[:, smallest_eigval_idx]

            vanishing_point = smallest_eigvect / smallest_eigvect[-1]
            vanishing_points[idx] = vanishing_point

    return vanishing_points


# using this as a reference 'https://www.ri.cmu.edu/pub_files/pub2/willson_reg_1993_1/willson_reg_1993_1.pdf'
def compute_score_for_vanishing_points(
    vanishing_points: np.ndarray,
    line_lengths: np.ndarray,
    line_angles: np.ndarray,
    angles_from_parallel_lines_to_vanishing_point: np.ndarray,
    sigma: float = 0.1,
) -> np.ndarray:
    vanishing_point_scores = np.zeros((vanishing_points.shape[0],))
    for idx in range(line_lengths.shape[0]):
        score = 0
        parallel_line_length = np.squeeze(line_lengths[idx])
        parallel_line_angles = np.squeeze(line_angles[idx])
        parallel_angles_to_vanishing_points = np.squeeze(angles_from_parallel_lines_to_vanishing_point[idx])
        for line_idx in range(line_lengths.shape[1]):
            line_length = abs(parallel_line_length[line_idx])
            line_angle = parallel_line_angles[line_idx]
            angle_to_vanishing_point = parallel_angles_to_vanishing_points[line_idx]

            abs_angle_diff = abs(angle_difference(line_angle, angle_to_vanishing_point))
            exp_pow = -abs_angle_diff / (2 * (sigma ** 2))

            s = line_length * (np.e ** exp_pow)
            score += s

        vanishing_point_scores[idx] = score

    return vanishing_point_scores
