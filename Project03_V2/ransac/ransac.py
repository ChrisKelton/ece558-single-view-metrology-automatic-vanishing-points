__all__ = ["ransac_fit_line"]
import math
from typing import Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np


def prep_linear_least_squares_fit_for_line(x_vals: np.ndarray, y_vals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    b = np.vstack(y_vals)
    A = np.column_stack((x_vals, np.ones(len(x_vals))))

    return A, b


def linear_least_squares_fit_line(x_vals: np.ndarray, y_vals: np.ndarray) -> np.ndarray:
    # put A in the form of our equations to solve
    # y = mx + b
    # A = [x_0  1], where we have ones in the last column in order to account for the y-intercept (b)
    #     [x_1  1]
    #     [x_2  1]
    #     [...  1]
    #     [x_n  1]
    #
    # Y = [y_0]
    #     [y_1]
    #     [...]
    #     [y_n]
    #
    # x = [m], m = slope
    #     [b], b = y-intercept
    #
    # And we are trying to solve the equation Ax = b (b here is not our y-intercept)
    # Ax = b, solving for x => x = A^-1 @ b
    #
    # Solving for A.T@A@x = A.T@b => x = (A.T@A)^-1@A.T@b
    A, b = prep_linear_least_squares_fit_for_line(x_vals, y_vals)
    x = linear_least_squares_fit(A, b)

    return np.squeeze(x)


def linear_least_squares_fit(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    ATA = A.T @ A
    ATb = A.T @ b
    x = np.linalg.inv(ATA) @ ATb

    return x


def ransac_fit_line(
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    p: float = 0.95,
    th: Optional[float] = None,
    std_divisor: float = 2,
    num_iters: int = 10000,
) -> Optional[np.ndarray]:
    # set initial number of iterations to be infinity
    num_it = math.inf
    samples_count = 0
    num_samples = 2
    max_inlier_count = 0
    best_line = None

    if th is None:
        th = np.std(y_vals) / std_divisor

    A, b = prep_linear_least_squares_fit_for_line(x_vals, y_vals)
    set_of_linear_equations = np.column_stack((A, b))

    num_of_singular_matrices = 0
    while num_it > samples_count and samples_count < num_iters and num_of_singular_matrices < num_iters:

        # get random points to sample
        np.random.shuffle(set_of_linear_equations)
        sample_data = set_of_linear_equations[:num_samples, :]

        if np.linalg.det(sample_data[:, :-1]) != 0:
            # solve for x in Ax = b; where x = [slope, y_intercept]
            estimated_line = linear_least_squares_fit(sample_data[:, :-1], sample_data[:, -1:])

            # calculate distances to estimated_line model from the original values
            distances = A.dot(estimated_line)
            error = np.abs(y_vals - distances.T)
            # errors within a threshold (typically std/2), are inliers
            distances_mask = np.squeeze(
                error <= th
            )  # want to make it <= to avoid possible errors with horizontal/vertical lines where they will be deemed not acceptably close to the line that is plotted on them
            inlier_cnt = len([mask for mask in distances_mask if mask])

            # if this estimated_line model has more inliers than the currently top stored one, then keep it
            if inlier_cnt > max_inlier_count:
                max_inlier_count = inlier_cnt
                best_line = estimated_line

            # adaptively update our number of iterations to perform by calculating the error percentage in regards to the
            # current number of outliers
            e = 1 - inlier_cnt / len(y_vals)
            # if 0, we'll get -infinity when calculating our math.log() below
            if e == 0:
                num_it = 0
            elif e == 1:
                num_it = math.log(1 - p) / math.log(0.9999)
            else:
                num_it = math.log(1 - p) / math.log(1 - ((1 - e) ** num_samples))

            # increase how many iterations we have performed
            samples_count += 1
        else:
            num_of_singular_matrices += 1

    if best_line is None:
        return None

    return np.squeeze(best_line)


def plot_arbitrary_line(m: float, intercept: float):
    axes = plt.gca()
    x_vals = np.asarray(axes.get_xlim())
    y_vals = intercept + m * x_vals
    plt.plot(x_vals, y_vals)


def main():
    samples = np.random.exponential(1, 1000)
    bins, hists = np.histogram(samples)
    x_vals = np.arange(len(bins))
    x_lls = linear_least_squares_fit_line(x_vals, bins)
    A_lls, b_lls = prep_linear_least_squares_fit_for_line(x_vals, bins)
    m_lls, y_intercept_lls = x_lls
    distances_lls = A_lls.dot(x_lls)
    err_lls = np.abs(bins - distances_lls)

    x_ransac = ransac_fit_line(x_vals, bins, std_divisor=2)
    A_ransac, b_ransac = prep_linear_least_squares_fit_for_line(x_vals, bins)
    m_ransac, y_intercept_ransac = x_ransac
    distances_ransac = A_ransac.dot(x_ransac)
    err_ransac = np.abs(bins - distances_ransac)

    plt.scatter(x_vals, bins)
    plot_arbitrary_line(m_lls, y_intercept_lls)
    plot_arbitrary_line(m_ransac, y_intercept_ransac)
    plt.legend(["LLS", "Ransac"])
    plt.show()
    a = 0


if __name__ == "__main__":
    main()
