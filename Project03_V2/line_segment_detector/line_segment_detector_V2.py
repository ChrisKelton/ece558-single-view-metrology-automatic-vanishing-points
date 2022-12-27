from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import math
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from tqdm import tqdm

from Projects.Project03_V2.filtering.filter_applications import compute_gradients
from Projects.Project03_V2.line_segment_detector.lines import Line
from Projects.Project03_V2.line_segment_detector.rectangle import (
    rectangular_approximation,
    rectApprox,
    rectRegion,
    angle_difference,
)
from Projects.Project03_V2.ransac.ransac import ransac_fit_line
from Projects.Project03_V2.utils.img_utils import load_image_as_grayscale, image_scaling
from Projects.Project03_V2.utils.serial_utils import save_pickle, load_pickle
from Projects.Project03_V2.utils.vanishing_points import *
from Projects.Project03_V2.utils.vrml import write_3d_vrml_file_with_default_options
from numerical import nfa_computation


@dataclass
class ParallelLines:
    lines: np.ndarray

    def two_longest_parallel_lines(self) -> np.ndarray:
        parallel_lines = np.zeros((2, 1, 4))
        dists = []
        for idx in range(len(self.lines)):
            dists.append(self.line_distance(idx))

        max_dist_idx = dists.index(np.nanmax(dists))
        parallel_lines[0] = self.lines[max_dist_idx]
        dists[max_dist_idx] = 0

        max_dist_idx = dists.index(np.nanmax(dists))
        parallel_lines[1] = self.lines[max_dist_idx]

        return parallel_lines

    def line_distance(self, idx) -> float:
        line = np.squeeze(self.lines[idx])
        cols_start = line[0]
        rows_start = line[1]
        cols_end = line[2]
        rows_end = line[3]

        return np.sqrt(((cols_end - cols_start) ** 2) + ((rows_start - rows_end) ** 2))


def gradient_threshold(mag_img: np.ndarray, tau: float = 22.5, q: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    # threshold gradient image
    th = abs(q / np.sin(tau * np.pi / 180))
    rows, cols = np.where(mag_img < th)
    mag_img[rows, cols] = 0
    status_img = np.zeros(mag_img.shape, dtype=bool)
    # where the gradient image is less than our threshold, mark these pixels as 'USED', so we don't use pixels that have
    # very little gradient magnitude
    status_img[rows, cols] = True

    return mag_img, status_img


def get_ordered_set_of_pixels(mag_img: np.ndarray, no_of_bins: int = 255) -> List[Tuple[int, int]]:
    bin_cnts, bins = np.histogram(mag_img, bins=no_of_bins)
    bins = list(bins)
    bins.pop(-1)
    bins.reverse()

    # get ordered set of pixels by reverse sorted bins intervals in order to assign rectangles starting with some seed
    # pixel that has a large gradient magnitude before other pixels with smaller gradient magnitudes collect pixels
    # in their rectangle and mark them as 'USED' before the more definitive regions can run
    ordered_pixel_idx = []
    mag_img_copy = mag_img.copy()
    for bin in bins:
        rows, cols = np.where(mag_img_copy > bin)
        ordered_pixel_idx.extend(list(zip(rows, cols)))
        mag_img_copy[rows, cols] = 0

    return ordered_pixel_idx


def get_pixel_idx_from_neighborhood8(pixel: Tuple[int, int], max_shape: Tuple[int, int]) -> List[Tuple[int, int]]:
    # get idx of pixels in 8-neighborhood around the center pixel
    neighborhood: List[Tuple[int, int]] = []
    pixel_prev_row = np.clip(pixel[0] - 1, 0, max_shape[0])
    pixel_next_row = np.clip(pixel[0] + 1, 0, max_shape[0])
    pixel_prev_col = np.clip(pixel[1] - 1, 0, max_shape[1])
    pixel_next_col = np.clip(pixel[1] + 1, 0, max_shape[1])
    neighborhood.append((pixel_prev_row, pixel_prev_col))  # i(x-1, y-1)
    neighborhood.append((pixel[0], pixel_prev_col))  # i(x, y-1)
    neighborhood.append((pixel_next_row, pixel_prev_col))  # i(x+1, y-1)
    neighborhood.append((pixel_prev_row, pixel[1]))  # i(x-1, y)
    neighborhood.append((pixel_next_row, pixel[1]))  # i(x+1, y)
    neighborhood.append((pixel_prev_row, pixel_next_col))  # i(x-1, y+1)
    neighborhood.append((pixel[0], pixel_next_col))  # i(x, y+1)
    neighborhood.append((pixel_next_row, pixel_next_col))  # i(x+1, y+1)

    return neighborhood


def region_growing(
    level_line_field: np.ndarray, seed_pixel: Tuple[int, int], status_img: np.ndarray, tau: float = 22.5
) -> Tuple[Optional[rectRegion], np.ndarray]:
    tau *= np.pi / 180
    region: List[Tuple[int, int]] = []
    region.append(seed_pixel)
    theta_region = level_line_field[seed_pixel]
    Sx = np.cos(theta_region)
    Sy = np.sin(theta_region)
    max_shape = (status_img.shape[0] - 1, status_img.shape[1] - 1)
    for pixel in region:
        # get pixels to test around seed pixel
        neighborhood_pixels = get_pixel_idx_from_neighborhood8(pixel, max_shape)
        for neighbor_pixel in neighborhood_pixels:
            # if neighboring pixel has not been used yet, then see if it is a good fit to grow a region with the seed
            # pixel
            if not status_img[neighbor_pixel]:
                lla_np = level_line_field[neighbor_pixel]
                # if the angle difference between the region angle (starting angle of the seed pixel) is smaller than
                # our tau, then append this pixel to our region and attempt to grow the region using that as our next
                # pixel in region
                if angle_difference(theta_region, lla_np) < tau:
                    region.append(neighbor_pixel)
                    # update our status image to show that this neighbor pixel is 'USED'
                    status_img[neighbor_pixel] = True
                    # update the angle orientation of the rectangle
                    Sx += np.cos(lla_np)
                    Sy += np.sin(lla_np)
                    theta_region = np.arctan2(Sy, Sx)

    # if the number of points within a region is too low, then discard the region
    if len(region) < 20:
        idx = np.squeeze(np.asarray(list(zip(region))))
        try:
            rows, cols = idx[:, 0], idx[:, 1]
        except IndexError:
            rows, cols = idx[0], idx[1]
        status_img[rows, cols] = False
        return None, status_img

    return rectRegion(pixels=region, seed_pixel=seed_pixel, theta=theta_region), status_img


def get_regions(
    level_line_field: np.ndarray,
    ordered_set_of_pixels: List[Tuple[int, int]],
    status_img: np.ndarray,
    tau: float = 22.5,
) -> List[rectRegion]:
    rect_regions: List[rectRegion] = []
    for seed_pixel in tqdm(ordered_set_of_pixels):
        if status_img.all():
            # if all pixels have been used, then break
            break
        if not status_img[seed_pixel]:
            # get rectangular regions from the level_line_field
            rect_region, status_img = region_growing(level_line_field, seed_pixel, status_img, tau)
            if rect_region is not None:
                rect_regions.append(rect_region)

    return rect_regions


def get_rectangular_approximations(rect_regions: List[rectRegion], mag_img: np.ndarray) -> List[Optional[rectApprox]]:
    rect_approxs: List[Optional[rectApprox]] = []
    for rect_region in tqdm(rect_regions):
        rect_approxs.append(rectangular_approximation(rect_region, mag_img))

    return rect_approxs


def mask_out_rectangles_that_failed_to_be_approximated(
    rect_approxs: List[rectApprox], rect_regions: List[rectRegion]
) -> Tuple[List[rectApprox], List[rectRegion]]:
    rect_mask = [False if rect_approx is None else True for rect_approx in rect_approxs]
    rect_approxs = [rect_approx for rect_approx, mask in zip(rect_approxs, rect_mask) if mask]
    rect_regions = [rect_region for rect_region, mask in zip(rect_regions, rect_mask) if mask]

    return rect_approxs, rect_regions


def calculate_aligned_point_density(rect_approx: rectApprox, rect_region: rectRegion) -> float:
    k = len(rect_region.pixels)
    length, width = rect_approx.length, rect_approx.width

    return k / (length * width)


def check_rectangle_approximations_using_aligned_point_density_and_cut_region_if_needed(
    rect_approxs: List[rectApprox],
    rect_regions: List[rectRegion],
    mag_img: np.ndarray,
    level_line_field: np.ndarray,
    status_img: np.ndarray,
    aligned_point_density_th: float = 0.7,
    original_tau: float = 22.5,
) -> Tuple[List[rectApprox], List[rectRegion], np.ndarray]:
    # want to check if there are any regions that could maybe be better split up into different regions due to some
    # gradual change of angles within the growing region originally
    new_tau = original_tau / 2
    num_regions_failed_aligned_point_density_th: int = 0
    indices_to_remove = set()
    new_rect_approxs_to_add = []
    new_rect_regions_to_add = []
    for rect_idx, (rect_approx, rect_region) in tqdm(
        enumerate(zip(rect_approxs, rect_regions)), total=len(rect_approxs)
    ):
        k = calculate_aligned_point_density(rect_approx, rect_region)
        tau_iteration = 1
        new_rect_region = None
        new_rect_approx = None
        while k < aligned_point_density_th:
            # print(f"Regrowing Region '{rect_idx}'")
            # num_regions_failed_aligned_point_density_th += 1
            # cut region and retry with smaller tau tolerance
            seed_pixel = rect_region.seed_pixel
            idx = np.squeeze(np.asarray(list(zip(rect_region.pixels))))
            rows, cols = idx[:, 0], idx[:, 1]
            # set status of pixels in region to False so they may be used in a different region
            status_img[rows, cols] = False
            new_rect_region, status_img = region_growing(
                level_line_field, seed_pixel, status_img, new_tau / tau_iteration
            )
            if new_rect_region is not None:
                # if new_rect_region is identical to the rect_region, then we don't need to re-approximate the
                # rectangle, b/c we already have it
                if new_rect_region.pixels != rect_region.pixels:
                    # approximate new rectangle
                    new_rect_approx = rectangular_approximation(new_rect_region, mag_img, new_tau)
                    if new_rect_approx is not None:
                        k = calculate_aligned_point_density(new_rect_approx, new_rect_region)
                        if tau_iteration > 4:
                            break
                        tau_iteration += 1
                    else:
                        indices_to_remove.add(rect_idx)
                        if tau_iteration > 4:
                            break
                        tau_iteration += 1
                else:
                    if tau_iteration > 4:
                        break
                    tau_iteration += 1
            else:
                indices_to_remove.add(rect_idx)
                # need to seed new region from different pixel within original region as those have now all been set back to 'UNUSED'
                if tau_iteration > 4:
                    break
                tau_iteration += 1
        if k > aligned_point_density_th and new_rect_approx is not None and new_rect_region is not None:
            rect_approxs[rect_idx] = new_rect_approx
            rect_regions[rect_idx] = new_rect_region

    indices_to_remove = list(indices_to_remove)
    for idx in sorted(indices_to_remove, reverse=True):
        rect_approxs.pop(idx)
        rect_regions.pop(idx)

    return rect_approxs, rect_regions, status_img


def compute_nfa_for_each_region(
    rect_approxs: List[rectApprox],
    rect_regions: List[rectRegion],
    level_line_field: np.ndarray,
    img_shape: Tuple[int, int],
    p_vals: List[float],
    gamma_: int = 11,
    tau: float = 22.5,
    epsilon: float = 1,
) -> Tuple[List[rectApprox], List[rectRegion], List[float]]:
    rect_idx_nfa_passed_regions = []
    initial_p = p_vals[0]
    p_vals.pop(0)
    rect_best_nfa_vals = []
    for rect_idx, (rect_approx, rect_region) in tqdm(
        enumerate(zip(rect_approxs, rect_regions)), total=len(rect_approxs)
    ):
        nfa_vals = []
        nfa_status: bool = False
        # calculate number of false alarms in each rectangle
        nfa_val = nfa_computation(
            rect_approx=rect_approx,
            rect_region=rect_region,
            img_shape=img_shape,
            gamma_=gamma_,
            p=initial_p,
            tau=tau,
            epsilon=epsilon,
        )
        best_p = initial_p
        nfa_vals.append(nfa_val)
        # if the nfa is below our threshold, then our rectangle is good
        if nfa_val < epsilon:
            nfa_status = True
            rect_idx_nfa_passed_regions.append(rect_idx)

        if not nfa_status:
            # try finer precisions
            for p in p_vals:
                nfa_val = nfa_computation(
                    rect_approx=rect_approx,
                    rect_region=rect_region,
                    img_shape=img_shape,
                    gamma_=gamma_,
                    p=p,
                    tau=tau,
                    epsilon=epsilon,
                )
                if nfa_val < epsilon:
                    nfa_status = True
                    rect_idx_nfa_passed_regions.append(rect_idx)
                    nfa_vals.append(nfa_val)
                    best_p = p
                    # just break p_vals iteration
                    break
                else:
                    if nfa_val < nfa_vals[-1]:
                        best_p = p
                    nfa_vals.append(nfa_val)
            # probably need to implement something to update the rectangle with the level of the best precision
        rect_best_nfa_vals.append(np.min(nfa_vals))

    # return rectangles that pass the number of false alarms threshold
    if len(rect_idx_nfa_passed_regions) > 0:
        nfa_rect_approxs = [rect_approxs[idx] for idx in rect_idx_nfa_passed_regions]
        nfa_rect_regions = [rect_regions[idx] for idx in rect_idx_nfa_passed_regions]
    else:
        raise RuntimeError(f"!!!ALL RECTANGLES FAILED NFA THRESHOLDING!!!")

    return nfa_rect_approxs, nfa_rect_regions, rect_best_nfa_vals


def plot_pixels_of_final_rectangles(rect_regions: List[rectRegion], img: np.ndarray):
    row_locations = []
    col_locations = []
    for rect_region in rect_regions:
        pixel_locs = np.squeeze(np.asarray(list(zip(rect_region.pixels))))
        rows, cols = pixel_locs[:, 0], pixel_locs[:, 1]
        row_locations.extend(list(rows))
        col_locations.extend(list(cols))

    plt.imshow(img)


def get_line_segments_for_rectangles(
    rect_approxs: List[rectApprox], rect_regions: List[rectRegion], scaling_factor: float = 0.8
) -> Tuple[List[Line], List[rectApprox], List[rectRegion]]:
    lines: List[Line] = []
    idx_of_rects_to_remove = []
    for rect_idx, (rect_approx, rect_region) in tqdm(
        enumerate(zip(rect_approxs, rect_regions)), total=len(rect_approxs)
    ):
        idx = np.squeeze(np.asarray(list(zip(rect_region.pixels))))
        rows, cols = idx[:, 0], idx[:, 1]
        best_fit_line = ransac_fit_line(cols, rows)
        if best_fit_line is None:
            idx_of_rects_to_remove.append(rect_idx)
        else:
            line = Line(center=rect_approx.center, extents=rect_approx.extents, m=best_fit_line[0], b=best_fit_line[1])
            lines.append(line)

    if len(idx_of_rects_to_remove) > 0:
        idx_of_rects_to_remove.sort(reverse=True)
        for idx in idx_of_rects_to_remove:
            rect_approxs.pop(idx)
            rect_regions.pop(idx)

    return lines, rect_approxs, rect_regions


def plot_line_segments_on_image(img: np.ndarray, line_segments: List[Line]):
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    plt.imshow(img)
    for line_segment in line_segments:
        extents = line_segment.extents
        cols = np.linspace(
            np.clip(round(extents.ymin), 0, img.shape[0]),
            np.clip(round(extents.ymax), 0, img.shape[0]),
            int(round(extents.ymax - extents.ymin)),
        )
        rows = np.clip(line_segment.m * cols + line_segment.b, 0, img.shape[1])

        plt.plot(cols.astype(np.int32), rows.astype(np.int32))

    plt.show()


def draw_line_segments_on_image_cv2(img: np.ndarray, lines: List[ParallelLines]) -> np.ndarray:
    plt.imshow(img)

    colors = cm.rainbow(np.linspace(0, 1, len(lines)))
    for line, color in zip(lines, colors):
        parallel_lines = line.two_longest_parallel_lines()
        for parallel_line in parallel_lines:
            parallel_line_ = np.squeeze(parallel_line)
            cv2.line(
                img,
                np.round(parallel_line_[:2]).astype(np.int32),
                np.round(parallel_line_[2:]).astype(np.int32),
                color=(0, 255, 0),
                thickness=2,
            )

    return img


def get_endpoints_of_line_segment(line_segment: Line) -> np.ndarray:
    extents = line_segment.extents
    col0 = extents.ymin
    col1 = extents.ymax
    if abs(col0 - col1) > 2:
        row0 = line_segment.m * col0 + line_segment.b
        row1 = line_segment.m * col1 + line_segment.b
    else:
        # vertical line
        row0 = extents.xmin
        row1 = extents.xmax
        col0 = extents.ymin
        col1 = extents.ymax

    # store as min to max
    endpoints = np.asarray((col0, row0, col1, row1))

    return np.expand_dims(endpoints, 0)


def get_parallel_lines(line_segments: List[Line], slope_th: float = 0.5) -> List[ParallelLines]:
    slopes = np.asarray([line.m for line in line_segments])
    intercepts = np.asarray([line.b for line in line_segments])
    parallel_lines = []
    for idx, (slope, intercept) in tqdm(enumerate(zip(slopes, intercepts)), total=len(slopes)):
        slope_diffs = slopes - slope
        slope_diffs = list(slope_diffs)
        slope_diffs.pop(idx)
        slope_diffs = np.asarray(slope_diffs)
        idx_where = np.where(slope_diffs < slope_th)[0]
        if len(idx_where) > 0:
            # account for popping self slope out of slope_diffs
            idx_where += 1
            lines_prll = np.zeros((len(idx_where) + 1, 1, 4))
            lines_prll[0] = get_endpoints_of_line_segment(line_segments[idx])
            for lines_idx, idx_ in enumerate(idx_where):
                # make sure we're not just grabbing the same lines (different y-intercepts)
                if (intercepts[lines_idx] - intercepts[idx_]) > 10:
                    lines_prll[lines_idx] = get_endpoints_of_line_segment(line_segments[idx])

            parallel_lines.append(ParallelLines(lines_prll))

    return parallel_lines


def get_parallel_lines_cv2(lines: np.ndarray, slope_th: float = 0.1) -> List[ParallelLines]:
    slopes = []
    intercepts = []
    for line in lines:
        line_ = line[0]
        cols_start = line_[0]
        rows_start = line_[1]
        cols_end = line_[2]
        rows_end = line_[3]
        # defining the x-y coordinate system of the image to have an origin of (0, 0) at the top left of the image
        # y-coordinate increases positively as the rows increase
        # x-coordinate increases positively as the cols increase
        m = (rows_end - rows_start) / (cols_end - cols_start)
        slopes.append(m)
        b = rows_end - (m * cols_end)
        intercepts.append(b)

    parallel_lines: List[ParallelLines] = []
    slopes = np.asarray(slopes)
    intercepts = np.asarray(intercepts)
    for idx, (slope, intercept) in enumerate(zip(slopes, intercepts)):
        slope_diffs = slopes - slope
        slope_diffs = list(slope_diffs)
        slope_diffs.pop(idx)
        slope_diffs = np.asarray(slope_diffs)
        idx_where = np.where(slope_diffs < slope_th)[0]
        if len(idx_where) > 0:
            # account for popping self slope out of slope_diffs
            idx_where += 1
            lines_prll = np.zeros((len(idx_where), 1, 4))
            lines_prll[0] = lines[idx]
            for lines_idx, idx_ in enumerate(idx_where):
                # make sure we're not just grabbing the same lines
                if (intercepts[lines_idx] - intercepts[idx_]) > 10:
                    lines_prll[lines_idx] = lines[idx_]

            parallel_lines.append(ParallelLines(lines_prll))

    return parallel_lines


def get_line_lengths_of_parallel_lines(parallel_lines: np.ndarray) -> np.ndarray:
    line_lengths = np.zeros((parallel_lines.shape[0], parallel_lines.shape[1]))
    for pl_idx in range(parallel_lines.shape[0]):
        parallel_line = np.squeeze(parallel_lines[pl_idx])
        for line_idx in range(parallel_line.shape[0]):
            points = parallel_line[line_idx]
            point0 = points[:2]
            point1 = points[2:]
            distance = np.sqrt(((point0[0] - point1[0]) ** 2) + ((point0[1] - point1[1]) ** 2))
            line_lengths[pl_idx, line_idx] = distance

    return line_lengths


def get_angles_from_parallel_lines_to_vanishing_points_and_of_lines(
    parallel_lines: np.ndarray, vanishing_points: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    angles = np.zeros((parallel_lines.shape[0], parallel_lines.shape[1]))
    line_angles = np.zeros((parallel_lines.shape[0], parallel_lines.shape[1]))
    for pl_idx in range(parallel_lines.shape[0]):
        parallel_line = np.squeeze(parallel_lines[pl_idx])
        vanishing_point = np.squeeze(vanishing_points[pl_idx])[:2]
        for line_idx in range(parallel_line.shape[0]):
            points = parallel_line[line_idx]
            test_point = points[:2]
            diff = vanishing_point[:2] - test_point
            # in terms of line geometry
            # rows = y
            # cols = x
            angle = np.arctan2(diff[1], diff[0])
            angles[pl_idx, line_idx] = angle

            line_diff = points[:2] - points[2:]
            line_angle = np.arctan2(line_diff[1], line_diff[0])
            line_angles[pl_idx, line_idx] = line_angle

    return angles, line_angles


def get_origin(vp0: np.ndarray, vp1: np.ndarray, vp2: np.ndarray) -> np.ndarray:
    # try and get intersection of all vanishing points
    cross01 = np.cross(vp0, vp1).reshape(3)
    cross02 = np.cross(vp0, vp2).reshape(3)
    cross12 = np.cross(vp1, vp2).reshape(3)

    crossed = np.column_stack((cross01, cross02, cross12))
    origin = np.mean(crossed, axis=1)

    return origin / origin[-1]


def lsd_cv2(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    lsd = cv2.createLineSegmentDetector(0)
    lines = lsd.detect(img)[0]
    drawn_img = lsd.drawSegments(img, lines)

    return lines, drawn_img


def check_if_paths_exist(paths: Union[Path, List[Path]]) -> bool:
    if not isinstance(paths, list):
        return paths.exists()

    for path_ in paths:
        if not path_.exists():
            return False

    return True


def ckelton_project03(
    img_path: Union[str, Path],
    output_path: Union[str, Path],
    pickle_path: Optional[Union[str, Path]] = None,
    use_cv2: bool = False,
):
    output_path.mkdir(exist_ok=True, parents=True)
    if pickle_path is not None:
        pickle_path = Path(pickle_path)
        pickle_path.mkdir(exist_ok=True, parents=True)
        rect_approxs_pickle_path = pickle_path / "rect-approxs.pickle"
        rect_regions_pickle_path = pickle_path / "rect-regions.pickle"
        mag_img_pickle_path = pickle_path / "mag-img.pickle"
        level_line_field_pickle_path = pickle_path / "level-line-field.pickle"
        status_img_pickle_path = pickle_path / "status-img.pickle"
        rect_approxs_after_nfa_pickle_path = pickle_path / "rect-approxs-after-nfa.pickle"
        rect_regions_after_nfa_pickle_path = pickle_path / "rect-regions-after-nfa.pickle"
        line_segments_pickle_path = pickle_path / "line-segments.pickle"
        rect_approxs_after_line_segment_pickle_path = pickle_path / "rect-approxs-after-line-segments.pickle"
        rect_regions_after_line_segment_pickle_path = pickle_path / "rect-regions-after-line-segments.pickle"
        parallel_lines_pickle_path = pickle_path / "parallel-lines.pickle"
        parallel_lines_arr_pickle_path = pickle_path / "parallel-lines-arr.pickle"
        parallel_lines_3d_pickle_path = pickle_path / "parallel-lines-3d.pickle"
        parallel_lines_pickle_path_cv2 = pickle_path / "parallel-lines-cv2.pickle"
        parallel_lines_arr_pickle_path_cv2 = pickle_path / "parallel-lines-arr-cv2.pickle"
        parallel_lines_3d_pickle_path_cv2 = pickle_path / "parallel-lines-3d-cv2.pickle"


    box_img = load_image_as_grayscale(img_path)
    box_img_color = cv2.imread(str(img_path.absolute()))
    tau = 22.5

    if not use_cv2:
        scaling_factor = 0.8
        scaled_img = image_scaling(box_img, scaling_factor)
        pickled_outputs_exist: bool = False
        if pickle_path is not None:
            pickled_outputs_exist = check_if_paths_exist([
                rect_approxs_pickle_path,
                rect_regions_pickle_path,
                mag_img_pickle_path,
                level_line_field_pickle_path,
                status_img_pickle_path,
                status_img_pickle_path,
            ])
        if not pickled_outputs_exist:
            derivative_filter = np.asarray([[-0.5, 0.5], [-0.5, 0.5]])
            # get gradient and level-line-field images
            mag_img, level_line_field = compute_gradients(scaled_img, derivative_filter, use_cv2=False)
            # threshold gradient magnitude image by some threshold
            mag_img, status_img = gradient_threshold(mag_img)
            # order the pixels from greatest gradient magnitude first, down to the lowest gradient magnitued
            ordered_set_of_pixels = get_ordered_set_of_pixels(mag_img.copy())
            # get initial regions from level_line_field
            print("***Region Growing***")
            rect_regions = get_regions(level_line_field, ordered_set_of_pixels, status_img)
            # get initial rectangular approximation for each region's points
            print("***Approximating Rectangles***")
            rect_approxs = get_rectangular_approximations(rect_regions, mag_img)
            # get rid of regions that failed during rectangular approximation
            rect_approxs, rect_regions = mask_out_rectangles_that_failed_to_be_approximated(rect_approxs, rect_regions)
            print("***CHECKING RECTANGLE APPROXIMATIONS USING ALIGNED POINT DENSITY AND CUTTING REGION IF NEEDED***")
            (
                rect_approxs,
                rect_regions,
                status_img,
            ) = check_rectangle_approximations_using_aligned_point_density_and_cut_region_if_needed(
                rect_approxs=rect_approxs,
                rect_regions=rect_regions,
                mag_img=mag_img,
                level_line_field=level_line_field,
                status_img=status_img,
                original_tau=tau,
            )
            # pickle our outputs
            if pickle_path is not None:
                save_pickle(rect_approxs, rect_approxs_pickle_path)
                save_pickle(rect_regions, rect_regions_pickle_path)
                save_pickle(mag_img, mag_img_pickle_path)
                save_pickle(level_line_field, level_line_field_pickle_path)
                save_pickle(status_img, status_img_pickle_path)
        else:
            print("Loading in pickled outputs...")
            # load in pickled outputs
            rect_approxs = load_pickle(rect_approxs_pickle_path)
            rect_regions = load_pickle(rect_regions_pickle_path)
            mag_img = load_pickle(mag_img_pickle_path)
            level_line_field = load_pickle(level_line_field_pickle_path)

        pickled_outputs_exist = False
        if pickle_path is not None:
            pickled_outputs_exist = check_if_paths_exist([
                rect_approxs_after_nfa_pickle_path,
                rect_regions_after_nfa_pickle_path,
            ])

        if not pickled_outputs_exist:
            # prepare our finer precisions for compute the number of false alarms per region
            tau_radians = tau * np.pi / 180
            initial_p = tau_radians / np.pi
            all_p_vals_step1 = [initial_p]
            p_vals_step1 = [(initial_p / (2 * mult)) for mult in np.linspace(1, 5, num=5)]
            all_p_vals_step1.extend(p_vals_step1)
            print("***CALCULATING NFA VALUES***")
            rect_approxs, rect_regions, rect_best_nfa_vals = compute_nfa_for_each_region(
                rect_approxs=rect_approxs,
                rect_regions=rect_regions,
                level_line_field=level_line_field,
                img_shape=mag_img.shape,
                gamma_=11,
                p_vals=all_p_vals_step1,
            )
            if pickle_path is not None:
                save_pickle(rect_approxs, rect_approxs_after_nfa_pickle_path)
                save_pickle(rect_regions, rect_regions_after_nfa_pickle_path)
        else:
            print("Loading in pickled outputs...")
            rect_approxs = load_pickle(rect_approxs_after_nfa_pickle_path)
            rect_regions = load_pickle(rect_regions_after_nfa_pickle_path)

        pickled_outputs_exist = False
        if pickle_path is not None:
            pickled_outputs_exist = check_if_paths_exist([
                line_segments_pickle_path,
                rect_approxs_after_line_segment_pickle_path,
                rect_regions_after_line_segment_pickle_path,
            ])
        if not pickled_outputs_exist:
            print("***GETTING LINE SEGMENTS FOR RECTANGLES***")
            line_segments, rect_approxs, rect_regions = get_line_segments_for_rectangles(rect_approxs, rect_regions)
            if pickle_path is not None:
                save_pickle(line_segments, line_segments_pickle_path)
                save_pickle(rect_approxs, rect_approxs_after_line_segment_pickle_path)
                save_pickle(rect_regions, rect_regions_after_line_segment_pickle_path)
        else:
            print("Loading in pickled outputs...")
            line_segments = load_pickle(line_segments_pickle_path)

        pickled_outputs_exist = False
        if pickle_path is not None:
            pickled_outputs_exist = check_if_paths_exist([
                parallel_lines_pickle_path,
                parallel_lines_3d_pickle_path,
                parallel_lines_arr_pickle_path,
            ])
        if not pickled_outputs_exist:
            print("***GETTING PARALLEL LINES***")
            parallel_lines = get_parallel_lines(line_segments)
            drawn_box_img = draw_line_segments_on_image_cv2(scaled_img, parallel_lines)
            drawn_box_path = output_path / "drawn-box-parallel-lines.jpg"
            cv2.imwrite(str(drawn_box_path.absolute()), drawn_box_img)
            # plot_line_segments_on_image(scaled_img, line_segments)
            parallel_lines_arr = np.asarray([par_lines.two_longest_parallel_lines() for par_lines in parallel_lines])
            parallel_lines_3d = prep_lines_cv2_for_vanishing_points_by_normalization(parallel_lines_arr)
            parallel_lines_3d_pickle_path = pickle_path / "parallel-lines-3d.pickle"

            if pickle_path is not None:
                save_pickle(parallel_lines, parallel_lines_pickle_path)
                save_pickle(parallel_lines_arr, parallel_lines_arr_pickle_path)
                save_pickle(parallel_lines_3d, parallel_lines_3d_pickle_path)
        else:
            print("Loading in pickled outputs...")
            parallel_lines_arr = load_pickle(parallel_lines_arr_pickle_path)
            parallel_lines_3d = load_pickle(parallel_lines_3d_pickle_path)
    else:
        pickled_outputs_exist: bool = False
        if pickle_path is not None:
            pickled_outputs_exist = check_if_paths_exist([
                parallel_lines_pickle_path_cv2,
                parallel_lines_arr_pickle_path_cv2,
                parallel_lines_3d_pickle_path_cv2,
            ])

        if not pickled_outputs_exist:
            line_segments, drawn_img = lsd_cv2(box_img)
            parallel_lines = get_parallel_lines_cv2(line_segments, slope_th=0.1)
            drawn_box_img = draw_line_segments_on_image_cv2(box_img_color, parallel_lines)
            drawn_box_path = output_path / "cv2-lsd-drawn-box-parallel-lines.jpg"
            cv2.imwrite(str(drawn_box_path.absolute()), drawn_box_img)
            parallel_lines_arr = np.asarray([par_lines.two_longest_parallel_lines() for par_lines in parallel_lines])
            parallel_lines_3d = prep_lines_cv2_for_vanishing_points_by_normalization(parallel_lines_arr)

            if pickle_path is not None:
                save_pickle(parallel_lines, parallel_lines_pickle_path_cv2)
                save_pickle(parallel_lines_arr, parallel_lines_arr_pickle_path_cv2)
                save_pickle(parallel_lines_3d, parallel_lines_3d_pickle_path_cv2)
        else:
            print("Loading in pickled outputs...")
            parallel_lines_arr = load_pickle(parallel_lines_arr_pickle_path_cv2)
            parallel_lines_3d = load_pickle(parallel_lines_3d_pickle_path_cv2)

    print("***GETTING ALL VIABLE VANISHING POINTS")
    vanishing_points = get_vanishing_points_by_normalization(parallel_lines_3d)
    print("***COMPUTING SCORES FOR VANISHING POINTS***")
    line_lengths = get_line_lengths_of_parallel_lines(parallel_lines_arr)
    angles_from_parallel_lines_to_vanishing_points, line_angles = get_angles_from_parallel_lines_to_vanishing_points_and_of_lines(
        parallel_lines_arr, vanishing_points
    )
    vanishing_point_scores_ = compute_score_for_vanishing_points(
        vanishing_points, line_lengths, line_angles, angles_from_parallel_lines_to_vanishing_points
    )
    vanishing_point_scores = np.asarray([vps if not math.isnan(vps) else 0 for vps in vanishing_point_scores_])

    vps_list = list(vanishing_point_scores)
    vp_idx_max0 = vps_list.index(np.nanmax(vps_list))
    vp0 = vanishing_points[vp_idx_max0]
    ref0 = np.squeeze(parallel_lines_3d[vp_idx_max0][0])[:3].reshape(1, 3)
    length0 = np.max(line_lengths[vp_idx_max0])
    vp_where_idx_max0 = np.where(vanishing_point_scores == np.nanmax(vps_list))[0]
    vanishing_point_scores[vp_where_idx_max0] = 0

    vps_list = list(vanishing_point_scores)
    vp_idx_max1 = vps_list.index(np.nanmax(vps_list))
    vp1 = vanishing_points[vp_idx_max1]
    ref1 = np.squeeze(parallel_lines_3d[vp_idx_max1][0])[:3].reshape(1, 3)
    length1 = np.max(line_lengths[vp_idx_max1])
    vp_where_idx_max1 = np.where(vanishing_point_scores == np.nanmax(vps_list))[0]
    vanishing_point_scores[vp_where_idx_max1] = 0

    vps_list = list(vanishing_point_scores)
    vp_idx_max2 = vps_list.index(np.nanmax(vps_list))
    vp2 = vanishing_points[vp_idx_max2]
    ref2 = np.squeeze(parallel_lines_3d[vp_idx_max2][0])[:3].reshape(1, 3)
    length2 = np.max(line_lengths[vp_idx_max2])
    vp_where_idx_max2 = np.where(vanishing_point_scores == np.nanmax(vps_list))[0]
    vanishing_point_scores[vp_where_idx_max2] = 0

    print("***PUTTING TOGETHER PROJECTION/HOMOGRAPHY MATRICES***")
    origin = get_origin(vp0, vp1, vp2)

    m0, _, _, _ = np.linalg.lstsq((vp0 - ref0).T, (ref0 - origin).T)
    m0 = m0[0][0] / length0

    m1, _, _, _ = np.linalg.lstsq((vp1 - ref1).T, (ref1 - origin).T)
    m1 = m1[0][0] / length1

    m2, _, _, _ = np.linalg.lstsq((vp2 - ref2).T, (ref2 - origin).T)
    m2 = m2[0][0] / length2

    projection_matrix = np.empty([3, 4])
    projection_matrix[:, 0] = vp0
    projection_matrix[:, 1] = vp1
    projection_matrix[:, 2] = vp2
    projection_matrix[:, 3] = origin

    projection_matrix[:, 0] *= m0
    projection_matrix[:, 1] *= m1
    projection_matrix[:, 2] *= m2

    homographic_matrix_xy = np.zeros((3, 3))
    homographic_matrix_yz = np.zeros((3, 3))
    homographic_matrix_zx = np.zeros((3, 3))

    # These homographic matrices will be transform matrices defined as
    # [[a1 a2 b1]
    #  [a3 a4 b2]
    #  [c1 c2 1]]
    # where [a1 a2] -> defines transformations such as rotation, scaling etc.
    #       [a3 a4]
    #
    # & [b1] -> defines translation vector
    #   [b2]
    #
    # & [c1 c2] -> projection vector

    homographic_matrix_xy[:, 0] = projection_matrix[:, 0]
    homographic_matrix_xy[:, 1] = projection_matrix[:, 1]
    homographic_matrix_xy[:, 2] = origin

    homographic_matrix_yz[:, 0] = projection_matrix[:, 1]
    homographic_matrix_yz[:, 1] = projection_matrix[:, 2]
    homographic_matrix_yz[:, 2] = origin

    homographic_matrix_zx[:, 0] = projection_matrix[:, 0]
    homographic_matrix_zx[:, 1] = projection_matrix[:, 2]
    homographic_matrix_zx[:, 2] = origin

    img = box_img_color.copy()
    rows, cols, channels = img.shape
    warped_xy = cv2.warpPerspective(img, homographic_matrix_xy, (rows, cols), flags=cv2.WARP_INVERSE_MAP)
    warped_yz = cv2.warpPerspective(img, homographic_matrix_yz, (rows, cols), flags=cv2.WARP_INVERSE_MAP)
    warped_zx = cv2.warpPerspective(img, homographic_matrix_zx, (rows, cols), flags=cv2.WARP_INVERSE_MAP)

    cv2.imshow("WarpedXY", warped_xy)
    cv2.imshow("WarpedYZ", warped_yz)
    cv2.imshow("WarpedZX", warped_zx)

    cv2.imwrite("XY.jpg", warped_xy)
    cv2.imwrite("YZ.jpg", warped_yz)
    cv2.imwrite("ZX.jpg", warped_zx)

    output_wrl_file = output_path / "box.wrl"
    write_3d_vrml_file_with_default_options(
        "XY.jpg", "YZ.jpg", "ZX.jpg", output_wrl_file
    )

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    img_path = Path("./cigar_box_large.jpg")
    pickle_path = Path("./pickle-outputs")
    output_path = Path("./outputs")
    ckelton_project03(img_path, output_path, pickle_path, use_cv2=False)


if __name__ == "__main__":
    main()
