__all__ = ["nfa_computation"]
import math
from typing import Tuple

import numpy as np
import scipy.special

from Projects.Project03_V2.line_segment_detector.rectangle import rectApprox, rectRegion

g = 7
# lanczos_coef = [
#     0.99999999999980993,
#     676.5203681218851,
#     -1259.1392167224028,
#     771.32342877765313,
#     -176.61502916214059,
#     12.507343278686905,
#     -0.13857109526572012,
#     9.9843695780195716e-6,
#     1.5056327351493116e-7,
# ]
# def gamma_fun(z) -> complex:
#     z = complex(z)
#     if z.real < 0.5:
#         return np.pi / (np.sin(np.pi * 2) * gamma_fun(1 - z.real))
#     else:
#         z -= 1
#         x = lanczos_coef[0] + np.sum([lanczos_coef[i] / (z.real + i) for i in range(1, g + 2)])
#         t = z.real + g + 0.5
#         try:
#             return np.sqrt(2 * np.pi) * (t ** (z.real + 0.5)) * np.exp(-t) * x
#         except OverflowError:
#             a = 0

lanczos_coef = [75122.6331530, 80916.6278952, 36308.2951477, 8687.24529705, 1168.92649479, 83.8676043424, 2.50662827511]


def gamma_lanczos(z: float) -> float:
    a = (z + 0.5) * np.log(z + 5.5) - (z + 5.5)
    b = 0

    for n in range(len(lanczos_coef)):
        a -= np.log(z + n)
        b += lanczos_coef[n] * (z ** n)

    return a + np.log(b)


# good approximation when z > 15
def gamma_windschitl(z: float) -> float:
    return 0.918938533204673 + (z - 0.5) * np.log(z) - z + 0.5 * z * np.log(z * np.sinh(1 / z) + 1 / (810 * (z ** 6)))


def log_gamma_fun(z: float) -> float:
    if z > 15:
        return gamma_windschitl(z)

    return gamma_lanczos(z)


def binomial(n: int, k: int) -> float:
    numerator = log_gamma_fun(n + 1)
    denominator_k = log_gamma_fun(k + 1)
    denominator_nk = log_gamma_fun(n - k + 1)

    binom = numerator / (denominator_k * denominator_nk)
    return binom


def nfa_computation(
    rect_approx: rectApprox,
    rect_region: rectRegion,
    img_shape: Tuple[int, int],
    gamma_: int,
    p: float,
    tau: float = 22.5,
    epsilon: float = 1,
) -> float:
    # calculate the maximum number of possible rectangles, which due to orientation, generates a maximum number of
    # (M x N)^2 possibilities with a factor of (M x N)^1/2 due to the maximum possible width of a rectangle in an image
    num_of_possible_rectangles = img_shape[0] * img_shape[1] ** (5 / 2)
    num_of_probs_with_rects = num_of_possible_rectangles * gamma_

    # get number of aligned points within the rectangel
    aligned_points = len(rect_region.pixels)

    # use the length and width of the rectangle to determine the maximum possible number of points within the rectangle
    length = rect_approx.length
    width = rect_approx.width
    number_of_points_in_rect = int(length * width)
    # this shouldn't be the case, due to a bug in the rectangle approximation. Possibly due to how it is updated due
    # to intersecting line points, but cases with sufficient number of points don't seem to see this issue.
    if number_of_points_in_rect < aligned_points:
        return epsilon + 1

    # get the probability of some number of aligned points given some number of possible points by using the binomial
    # distribution
    binomial_dist_sum = 0
    for j in range(aligned_points, number_of_points_in_rect):
        binom = binomial(number_of_points_in_rect, j)
        # binom = scipy.special.binom(number_of_points_in_rect, j)
        if math.isinf(binom):
            return epsilon + 1
        binom *= p ** j
        binom *= (1 - p) ** (number_of_points_in_rect - j)
        binomial_dist_sum += binom

    # calculate total number of false alarms within rectangle
    nfa = num_of_probs_with_rects * binomial_dist_sum

    return nfa
