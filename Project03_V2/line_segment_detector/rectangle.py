__all__ = ["rectangular_approximation", "rectApprox", "rectRegion", "Extents", "angle_difference"]
from dataclasses import dataclass
from typing import List, Tuple, NamedTuple, Optional

import numpy as np
from matplotlib.path import Path as mpPath

from Projects.Project03_V2.line_segment_detector.lines import (
    find_intersection_from_two_points_with_orthogonal_slopes,
    find_intersection_from_two_extents_and_one_excluded_point_with_orthogonal_slopes_and_update_extents,
    Extents,
)


rectRegion = NamedTuple("rectRegion", pixels=List[Tuple[int, int]], seed_pixel=Tuple[int, int], theta=float)
rectApprox = NamedTuple(
    "rectApprox", center=Tuple[float, float], width=float, length=float, extents=Extents, angle=float
)


def angle_difference(angle0: float, angle1: float) -> float:
    # calculate angle difference
    angle_diff = angle0 - angle1
    while angle_diff <= -np.pi:
        angle_diff += 2 * np.pi
    while angle_diff > np.pi:
        angle_diff -= 2 * np.pi

    if angle_diff < 0:
        return -angle_diff

    return angle_diff


def rectangular_approximation(
    region: rectRegion, mag_img: np.ndarray, tau: float = 22.5, verbose: bool = False
) -> Optional[rectApprox]:
    idx = np.squeeze(np.asarray(list(zip(region.pixels))))
    rows, cols = idx[:, 0], idx[:, 1]
    mags = mag_img[rows, cols]
    mags_sum = np.sum(mags)
    # get center of rectangle
    center_x = np.dot(mags, rows) / mags_sum
    center_y = np.dot(mags, cols) / mags_sum
    center = (center_x, center_y)

    if center[0] == 0 or center[0] == mag_img.shape[0] - 1 or center[1] == 0 or center[1] == mag_img.shape[1] - 1:
        if verbose:
            print("Estimated center of rectangle is at the border of the image. Returning 'None'.")
        return None

    rows_zero_mean = rows - center_x
    mxx = np.dot(mags, np.multiply(rows_zero_mean, rows_zero_mean)) / mags_sum

    cols_zero_mean = cols - center_y
    myy = np.dot(mags, np.multiply(cols_zero_mean, cols_zero_mean)) / mags_sum

    mxy = np.dot(mags, np.multiply(rows_zero_mean, cols_zero_mean)) / mags_sum

    # get second moment matrix?
    # this allows us to determine the eigenvectors and eigenvalues of the points for determining the direction of
    # the rectangle approximatino
    M = np.asarray([[mxx, mxy], [mxy, myy]])
    eigvals, eigvects = np.linalg.eigh(M)
    # smallest eigenvalue gives us the angle of the main rectangle
    smallest_eigval_idx = list(eigvals).index(np.min(eigvals))
    smallest_eigvect = eigvects[:, smallest_eigval_idx]
    # if smallest_eigvect[0] == 0:
    #     if region.theta < 0:
    #         angle = -np.pi / 2
    #     else:
    #         angle = np.pi / 2
    # else:
    # angle = np.arctan2(smallest_eigvect[1], smallest_eigvect[0])
    if abs(mxx) > abs(myy):
        angle = np.arctan2(np.min(eigvals) - mxx, mxy)
    else:
        angle = np.arctan2(mxy, np.min(eigvals) - myy)

    prec = np.pi * tau / 180
    # try and get correct orientation of line segment by using difference of rectangle and angle in order and moving
    # around unit circle appropriately
    if angle_difference(angle, region.theta) > prec:
        angle += np.pi

    # get angle orthogonal to main rectangle to help computing rectangle approximation
    ortho_angle = angle - (np.pi / 2)

    # # determine bounds for initial bounding rectangle around the points
    # ul_x_idx = list(rows).index(rows.min())
    # ul_y_idx = list(cols).index(cols.min())
    # ul_x = rows[ul_x_idx]
    # ul_y = cols[ul_y_idx]
    # ul_extents = (ul_x - 0.5, ul_y - 0.5)
    #
    # ur_x_idx = list(rows).index(rows.min())
    # ur_y_idx = list(cols).index(cols.max())
    # ur_x = rows[ur_x_idx]
    # ur_y = cols[ur_y_idx]
    # ur_extents = (ur_x - 0.5, ur_y + 0.5)
    #
    # lr_x_idx = list(rows).index(rows.max())
    # lr_y_idx = list(cols).index(cols.max())
    # lr_x = rows[lr_x_idx]
    # lr_y = cols[lr_y_idx]
    # lr_extents = (lr_x + 0.5, lr_y + 0.5)
    #
    # ll_x_idx = list(rows).index(rows.max())
    # ll_y_idx = list(cols).index(cols.min())
    # ll_x = rows[ll_x_idx]
    # ll_y = cols[ll_y_idx]
    # ll_extents = (ll_x + 0.5, ll_y - 0.5)
    #
    # extents = (ul_extents, ur_extents, lr_extents, ll_extents)
    # original_extents = extents
    #
    # # rotate rectangle to be oriented in the angle determined earlier
    # c = np.cos((np.pi / 2) - angle)
    # s = np.sin((np.pi / 2) - angle)
    # rotation_mat = np.array(((c, -s), (s, c)))
    # # subtract off where we want our rectangle to be rotated around, in our case, it is the center of the rectangle
    # extents_center_rot = extents - np.asarray(center)
    # # add back center to get back to being centered around our center
    # rotated_extents = (rotation_mat @ extents_center_rot.T).T + np.asarray(center)
    #
    # # determine slopes in rectangle orientations
    # m = np.tan(angle)
    # m_ortho = np.tan(ortho_angle)
    #
    # # check if all points supposed to be in rectangle are in rectangle
    # p = mpPath(rotated_extents)
    # points = np.vstack((rows, cols)).T
    # grid = p.contains_points(points)
    # # if not all pixels that are supposed to be within the extents of our now rotated rectangle, then iteratively solve
    # # for an updated rectangle that includes all points.
    # if not grid.all():
    #     excluded_points_idx = np.where(grid == False)[0]
    #     for idx in excluded_points_idx:
    #         # as we update the extents of the rectangle, check to see if we now contain all of the points
    #         p = mpPath(rotated_extents)
    #         grid = p.contains_points(points)
    #         if grid.all():
    #             break
    #
    #         # find the two extent points that are closest to our excluded point in order to adjust both height and width
    #         # of our rotated rectangle
    #         excluded_point = np.asarray((rows[idx], cols[idx]))
    #         diff = rotated_extents - excluded_point
    #         square = np.multiply(diff, diff)
    #         sum_ = square[:, 0] + square[:, 1]
    #         distances = np.sqrt(sum_)
    #         distances = list(distances)
    #         extent0_idx = distances.index(min(distances))
    #         distances[extent0_idx] = max(distances) + 1
    #         extent1_idx = distances.index(min(distances))
    #
    #         extent0 = rotated_extents[extent0_idx]
    #         extent1 = rotated_extents[extent1_idx]
    #
    #         # update our extents where there are intersections between the homogenous lines of our existing extents and
    #         # the excluded pts
    #         rotated_extents = find_intersection_from_two_extents_and_one_excluded_point_with_orthogonal_slopes_and_update_extents(
    #             extent_pts=(extent0, extent1),
    #             extent_pt_idx_into_extents=(extent0_idx, extent1_idx),
    #             excluded_pt=excluded_point,
    #             m=m,
    #             m_ortho=m_ortho,
    #             extents=rotated_extents,
    #         )
    #
    # # now we need to find the width and length of our rectangle by casting a ray from our center point in the m and
    # # m_ortho directions
    # rect_intersection, ortho_intersection = find_intersection_from_two_points_with_orthogonal_slopes(
    #     pt0=np.asarray(center), pt1=rotated_extents[0], m=m, m_ortho=m_ortho
    # )
    # if rect_intersection is None:
    #     raise RuntimeError(f"Got None for intersection in both directions from the center to an extent...")
    # length = np.sqrt(np.sum(np.square(np.asarray(center) - rect_intersection))) * 2
    #
    # if ortho_intersection is None:
    #     raise RuntimeError(f"Got None for intersection in ortho direction from the center to an extent...")
    # width = np.sqrt(np.sum(np.square(np.asarray(center) - ortho_intersection))) * 2

    dx = np.cos(angle)
    dy = np.sin(angle)
    length_min, length_max, width_min, width_max = 0, 0, 0, 0
    for pixel in region.pixels:
        length = ((pixel[0] - center_x) * dx) + ((pixel[1] - center_y) * dy)
        width = -((pixel[0] - center_x) * dy) + ((pixel[1] - center_y) * dx)

        if length > length_max:
            length_max = length
        elif length < length_min:
            length_min = length

        if width > width_max:
            width_max = width
        elif width < width_min:
            width_min = width

    width = width_max - width_min
    length = length_max - length_min

    x1 = center_x + (length_min * dy)
    y1 = center_y + (width_min * dx)
    x2 = center_x + (length_max * dy)
    y2 = center_y + (width_max * dx)
    # extents = [(x1 + 0.5, y1 + 0.5), (x1 + 0.5, y2 + 0.5), (x2 + 0.5, y2 + 0.5), (x2 + 0.5, y1 + 0.5)]
    extents = np.asarray([[x2 + 0.5, y2 + 0.5], [x2 + 0.5, y1 + 0.5], [x1 + 0.5, y1 + 0.5], [x1 + 0.5, y2 + 0.5]])
    # account for perfectly horizontal or vertical lines
    if width < 1:
        width = 1
    if length < 1:
        length = 1

    return rectApprox(center=center, width=width, length=length, extents=Extents(extents), angle=angle)
