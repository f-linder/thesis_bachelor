from sklearn.neighbors import KernelDensity, NearestNeighbors
from scipy.special import gamma
import math
import numpy as np


class KDE:
    def __init__(self, kernel='gaussian', bandwidth='silverman'):
        self.kernel = kernel
        self.bandwidth = bandwidth


class KNN:
    def __init__(self, k=None):
        self.k = k


def pdf_joint(estimator, returns):
    # use kernel density estimator
    if isinstance(estimator, KDE):
        kde = KernelDensity(kernel=estimator.kernel, bandwidth=estimator.bandwidth).fit(returns)

        def density_function(*values):
            log_density = kde.score([list(values)])
            density = np.exp(log_density)
            return density

        return density_function

    # use k nearest neighbor estimator
    elif isinstance(estimator, KNN):
        n = len(returns)
        d = len(returns[0])
        k = estimator.k if estimator.k else math.ceil(np.sqrt(n) / 2)
        knn = NearestNeighbors(n_neighbors=k).fit(returns)

        def density_function(*values):
            distances, _ = knn.kneighbors([list(values)])
            radius = np.max(distances)
            # volume of d-dimensional hypersphere
            volume = (np.pi ** (d / 2) / gamma(d / 2 + 1)) * (radius ** d)
            return k / (n * volume)

        return density_function


def pdf(estimator, returns_joint, returns_cond=None):
    if returns_cond is not None:
        returns_all = np.hstack((returns_joint, returns_cond))

        density_all = pdf_joint(estimator, returns_all)
        density_cond = pdf_joint(estimator, returns_cond)

        # Pr(A|B) = Pr(A,B) / Pr(B)
        def density_function(*values):
            # one value is no tuple, therefore list(values) errors
            # values = [values] if not isinstance(values, tuple) else list(values)
            return density_all(*values) / density_cond(*values[len(returns_all[0]) - len(returns_cond[0]):])

        return density_function
    else:
        density = pdf_joint(estimator, returns_joint)

        def density_function(*values):
            return density(*values)

        return density_function
