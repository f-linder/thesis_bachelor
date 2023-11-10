from sklearn.neighbors import KernelDensity, NearestNeighbors
from scipy.special import gamma
import math
import numpy as np


# Kernel Density Estimator
class KDE:
    def __init__(self, kernel='exponential', bandwidth='silverman'):
        self.kernel = kernel
        self.bandwidth = bandwidth


# K-Nearest Neighbor Estimator
class KNN:
    def __init__(self, k=None):
        self.k = k


def pdf(estimator, returns):
    """
    Calculate the joint probability density function (PDF) for a given
    estimator and sample data.

    Parameters:
    - estimator (KDE or KNN): The estimator used.
    - returns (list of lists): List of return samples of the form [[x1, y1, ...], [x2, y2, ...], ...]

    Returns:
    - density_function (function): A function taking a list of values [x, y, ...] and calculating Pr(X=x, Y=y, ...)
    """
    if isinstance(estimator, KDE):
        kde = KernelDensity(kernel=estimator.kernel, bandwidth=estimator.bandwidth)
        kde.fit(returns)

        def density_function(values):
            log_density = kde.score([values])
            density = np.exp(log_density)
            return density

        return density_function

    elif isinstance(estimator, KNN):
        n = len(returns)
        d = len(returns[0])
        k = estimator.k if estimator.k else math.ceil(np.sqrt(n) / 2)
        knn = NearestNeighbors(n_neighbors=k).fit(returns)

        def density_function(values):
            distances, _ = knn.kneighbors([values])
            radius = np.max(distances)
            # volume of d-dimensional hypersphere
            volume = (np.pi ** (d / 2) / gamma(d / 2 + 1)) * (radius ** d)
            return (k - 1) / (n * volume)

        return density_function
