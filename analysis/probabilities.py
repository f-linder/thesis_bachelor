from sklearn.neighbors import KernelDensity, NearestNeighbors
import math
import numpy as np


class KDE:
    def __init__(self, _kernel='gaussian', _bandwidth='silverman'):
        self.kernel = _kernel
        self.bandwidth = _bandwidth


class KNN:
    def __init__(self, _k=None):
        self.k = _k


def get_KNN(returns, k):
    k = k if k else math.ceil(np.sqrt(len(returns[0])) / 2)
    knn = NearestNeighbors(n_neighbors=k).fit(returns)
    return knn


def get_KDE(returns, kernel, bandwidth):
    kde = KernelDensity(kernel=kernel, bandwidth=bandwidth).fit(returns)
    return kde


def pdf(returns, estimator):
    if isinstance(estimator, KDE):
        knn = get_KDE(returns, estimator.kernel, estimator.bandwidth)

        def density_function(values):
            log_density = knn.score_samples([values])
            density = np.exp(log_density)
            return density

        return density_function

    elif isinstance(estimator, KNN):
        kde = get_KNN(returns, estimator.k)
        n_samples = kde.n_samples_fit_
        k = estimator.k if estimator.k else math.ceil(np.sqrt(n_samples) / 2)

        def density_function(values):
            distances, _ = estimator.kneighbors([values])
            v = np.max(distances)
            return k / (n_samples * v)

        return density_function

# TODO: pdf with condition

