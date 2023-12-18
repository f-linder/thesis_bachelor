from sklearn.neighbors import KernelDensity, NearestNeighbors
import math
import numpy as np


# Kernel Density Estimator
class KDE:
    def __init__(self, kernel='exponential', bandwidth='silverman'):
        self.kernel = kernel
        self.bandwidth = bandwidth


# K-Nearest Neighbor Estimator
class KNN:
    def __init__(self, k=None, metric='euclidean'):
        # if k is not set pdf uses k=math.ceil(np.sqrt(num_samples))
        self.k = k
        self.metric = metric


def pdf(estimator, samples):
    """
    Calculate the joint probability density function (PDF) for a given
    estimator and sample data.

    Parameters:
    - estimator (KDE or KNN): The estimator used.
    - samples (np.ndarray): A 2D array where each row represents a sample and
    each column represents a different variable: [[x1, y1, z1, ...],
                                                  [x2, y2, z2, ...],
                                                  [x3, y3, z3, ...],
                                                  ...              ]

    Returns:
    - density_function (function): A function taking a list of values [x, y, ...] and calculating Pr(X=x, Y=y, ...)
    """
    assert samples.ndim == 2, 'Sample array must be 2 dimensional'

    if isinstance(estimator, KDE):
        kde = KernelDensity(kernel=estimator.kernel, bandwidth=estimator.bandwidth)
        kde.fit(samples)

        def density_function(values):
            log_density = kde.score([values])
            density = np.exp(log_density)
            return density

        return density_function

    elif isinstance(estimator, KNN):
        n_samples = samples.shape[0]
        # number of features
        dim = samples.shape[1]
        k = estimator.k if estimator.k else math.ceil(np.sqrt(n_samples) / 2)
        knn = NearestNeighbors(n_neighbors=k, metric=estimator.metric).fit(samples)

        def density_function(values):
            distances, _ = knn.kneighbors([values])
            radius = np.max(distances)
            # volume of dim-dimensional hypersphere
            volume = (np.pi ** (dim / 2) / math.gamma(dim / 2 + 1)) * (radius ** dim)
            return (k - 1) / (n_samples * volume)

        return density_function
