__all__ = [
    "get_additional_point_far_away_with_slope",
    "SlopeOrientation",
    "find_intersection_of_two_lines_from_two_points",
    "find_intersection_from_two_points_with_orthogonal_slopes",
    "find_intersection_from_two_extents_and_one_excluded_point_with_orthogonal_slopes_and_update_extents",
    "check_line_segments",
    "determine_parallel_point",
    "find_distance_from_pt_to_line",
]
import enum
from dataclasses import dataclass
from typing import Tuple, Union, List, Optional

import numpy as np


@dataclass
class Extents:
    extents: np.ndarray

    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        return self.xmin, self.xmax, self.ymin, self.ymax

    @property
    def xmin(self) -> float:
        return np.min(self.extents[:, 0])

    @property
    def xmax(self) -> float:
        return np.max(self.extents[:, 0])

    @property
    def ymin(self) -> float:
        return np.min(self.extents[:, 1])

    @property
    def ymax(self) -> float:
        return np.max(self.extents[:, 1])

    def __truediv__(self, val: float) -> np.ndarray:
        return self.extents / val

    def __floordiv__(self, val: int) -> np.ndarray:
        return self.__truediv__(val).astype(np.int32)


@dataclass
class Line:
    center: np.ndarray
    extents: Optional[Extents]
    m: float  # slope
    b: float  # intercept

    @property
    def domain(self) -> Tuple[float, float]:
        return self.extents.xmin, self.extents.xmax

    @property
    def range(self) -> Tuple[float, float]:
        return self.extents.ymin, self.extents.ymax


def get_additional_point_far_away_with_slope(pt: np.ndarray, m: float) -> Tuple[float, float]:
    b = -(pt[1] - (m * pt[0]))
    col = -10
    row = (m * col) + b
    return (row, col)


@dataclass
class SlopeOrientation(enum.Enum):
    RECT = "rect"
    ORTHO = "ortho"


def find_intersection_from_two_points_with_orthogonal_slopes(
    pt0: np.ndarray, pt1: np.ndarray, m: float, m_ortho: float
) -> Tuple[np.ndarray, np.ndarray]:
    pt0_rect = get_additional_point_far_away_with_slope(pt0, m)
    pt1_ortho = get_additional_point_far_away_with_slope(pt1, m_ortho)
    rect_intersection = find_intersection_of_two_lines_from_two_points([pt0, pt0_rect], [pt1, pt1_ortho])

    pt0_ortho = get_additional_point_far_away_with_slope(pt0, m_ortho)
    pt1_rect = get_additional_point_far_away_with_slope(pt1, m)
    ortho_intersection = find_intersection_of_two_lines_from_two_points([pt0, pt0_ortho], [pt1, pt1_rect])

    return rect_intersection, ortho_intersection


def find_intersection_from_two_extents_and_one_excluded_point_with_orthogonal_slopes_and_update_extents(
    extent_pts: Tuple[np.ndarray, np.ndarray],
    extent_pt_idx_into_extents: Tuple[int, int],
    excluded_pt: np.ndarray,
    m: float,
    m_ortho: float,
    extents: np.ndarray,
) -> np.ndarray:
    # get points cast away from the extents and original excluded point in both directions of the rectangle
    # these points will be used to compute our homogenous lines and solve for the intersection of rectangle extension
    excluded_point_ortho = get_additional_point_far_away_with_slope(excluded_pt, m_ortho)
    excluded_point_rect = get_additional_point_far_away_with_slope(excluded_pt, m)
    for extent_pt, extent_pt_idx in zip(extent_pts, extent_pt_idx_into_extents):

        # solve for intersection between the ray from the extent in the direction of the rectangle and the orthogonal
        # ray of the excluded point
        extent_rect = get_additional_point_far_away_with_slope(extent_pt, m)
        rect_intersection = find_intersection_of_two_lines_from_two_points(
            [extent_pt, np.asarray(extent_rect)], [excluded_pt, np.asarray(excluded_point_ortho)]
        )

        # solve for intersection between the ray from the extent in the direction orthogonal to the rectangle and the
        # ray of the excluded point in the direction of the rectangle
        extent_ortho = get_additional_point_far_away_with_slope(extent_pt, m_ortho)
        ortho_intersection = find_intersection_of_two_lines_from_two_points(
            [extent_pt, np.asarray(extent_ortho)], [excluded_pt, np.asarray(excluded_point_rect)]
        )
        # if both rect_intersection and ortho_intersection are None, something is awry, b/c there has to be an
        # intersection among the two different rays
        if rect_intersection is None and ortho_intersection:
            raise RuntimeError(f"No intersection between two closest points from rotated rectangle...")

        # get the distances of the intersection and the existing extent to see which one makes more sense for
        # extending the rectangle bounds
        rect_dist = np.sqrt(np.sum((extent_pt - rect_intersection) ** 2))
        ortho_dist = np.sqrt(np.sum((extent_pt - ortho_intersection) ** 2))
        if rect_dist <= ortho_dist:
            extents[extent_pt_idx] = rect_intersection
        else:
            extents[extent_pt_idx] = ortho_intersection

    return extents


# try and determine if two points, with the same corresponding slope, have dissimilar y-intercepts
def determine_parallel_point(pt0: np.ndarray, pt1: np.ndarray, m: float) -> float:
    b0 = -(pt0[1] - (m * pt0[0]))
    b1 = -(pt1[1] - (m * pt1[0]))

    return abs((b0 - b1) / b0)


def check_line_segments(line_segment: Union[np.ndarray, List[float], Tuple[float, ...]]) -> np.ndarray:
    if len(line_segment) == 2:
        line_segment = list(line_segment)
        line_segment.append(1)
        line_segment = np.asarray(line_segment)
    elif len(line_segment) != 3:
        raise ValueError(
            f"Expect inputted line segments to either have 2 or 3 dimensions per points. Got '{len(line_segment)}'."
        )
    else:
        line_segment = np.asarray(line_segment)

    return line_segment


def find_intersection_of_two_lines_from_two_points(
    line_0_segments: List[np.ndarray], line_1_segments: List[np.ndarray]
) -> Optional[np.ndarray]:
    line_0_segment0 = check_line_segments(line_0_segments[0])
    line_0_segment1 = check_line_segments(line_0_segments[1])
    homo_line0 = np.cross(line_0_segment0, line_0_segment1)
    homo_line0 /= homo_line0[2]

    line_1_segment0 = check_line_segments(line_1_segments[0])
    line_1_segment1 = check_line_segments(line_1_segments[1])
    homo_line1 = np.cross(line_1_segment0, line_1_segment1)
    homo_line1 /= homo_line1[2]

    intersection = np.cross(homo_line0, homo_line1)
    if intersection[2] == 0:
        return None

    intersection /= intersection[2]
    return intersection[:2]


def find_distance_from_pt_to_line(pt: np.ndarray, m: float, b: float, pt_iters: int = 100) -> float:
    x_start = pt[0] - 5
    x_end = pt[0] + 5
    x = np.linspace(x_start, x_end, num=pt_iters)
    y = m * x + b
    test_pts = np.vstack((x, y)).T
    interm = (test_pts - pt) ** 2
    distances = np.sqrt(interm[:, 0] + interm[:, 1])

    return np.min(distances)
