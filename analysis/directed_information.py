import analysis.probabilities as prob
import analysis.utils as utils
import numpy as np
from enum import Enum
from sklearn.feature_selection import mutual_info_regression


def directed_information(x, y, z=None, order=1, subset_selection=None, estimator=prob.KNN()):
    """
    Estimate I(X -> Y || Z), the Directed Information between two time series X
    and Y causally conditioned on a set of time series Z with memory of size l
    (order l). If estimates turn out negative, they are corrected to be zero.

    I(X -> Y || Z) = 1/T * Sum_t=1^T I(Y_t, X_t-l^t-1 | Y_t-l^t-1, Z_t-l^t-1)
                   = 1/T * Sum_t=1^T [H(Y_t, Y_t-l^t-1, Z_t-l^t-1)
                                      + H(X_t-l^t-1, Y_t-l^t-1, Z_t-l^t-1)
                                      - H(Y_t, Y_t-l^t-1, X_t-l^t-1 Z_t-l^t-1)
                                      - H(Y_t-l^t-1, Z_t-l^t-1)]

    Parameters:
    - x (numpy.ndarray): The first time series [x1, x2, ...].
    - y (numpy.ndarray): The second time series [y1, y2, ...].
    - z (numpy.ndarray or None): A set of time series causally
    conditioned on of form [[a1, b1, ...],
                            [a2, b2, ...],
                            ...]
    - order (int): The memory (or lag) for DI estimation (default is 1).
    - subset_selection (SubsetSelection): Subset selection policy used to
    determine set causally conditioned on and reduce dimensionality.
    - estimator (KDE or KNN): An estimator for probability density functions.

    Returns:
    - di (float): The estimated Directed Information I(X -> Y || Z).
    """
    assert x.shape[0] == y.shape[0], 'Sample size of x and y must match'

    n_samples = x.shape[0]
    x = np.array([x]).transpose()
    y = np.array([y]).transpose()

    # (X_t-l^t-1)
    x_lagged = get_lagged_samples(x, [(1, order)])
    # (Y_t-l^t-1)
    y_lagged = get_lagged_samples(y, [(1, order)])

    # special case: I(X -> X || Z)
    if (x == y).all():
        y_lagged = np.array([[]] * (n_samples - order))

    # set z corresponding to context and subset_selection
    if z is None or len(z[0]) == 0:
        z = np.array([[]] * (n_samples - order))
    elif subset_selection is not None:
        z = select_subset(y, z, subset_selection, order)
    else:
        z = get_lagged_samples(z, [(1, order) for _ in range(len(z[0]))])

    # samples corresponding to entropy terms
    ytyz = np.hstack((y[order:], y_lagged, z))
    xyz = np.hstack((x_lagged, y_lagged, z))
    ytyxz = np.hstack((y[order:], y_lagged, x_lagged, z))
    yz = np.hstack((y_lagged, z))

    # density functions corresponding to entropy terms
    pdf_ytyz = prob.pdf(estimator, ytyz)
    pdf_xyz = prob.pdf(estimator, xyz)
    pdf_ytyxz = prob.pdf(estimator, ytyxz)
    pdf_yz = (lambda x: 1) if len(z[0]) == 0 and len(y_lagged[0]) == 0 else prob.pdf(estimator, yz)

    T = len(x) - order
    di = 0.0

    for t in range(T):
        # compute mutual information I(Y_t, X_t-l^t-1 | Y_t-l^t-1, Z_t-l^t-1)
        # using entropy terms
        h_ytyz = -np.log(pdf_ytyz(ytyz[t]))
        h_xyz = -np.log(pdf_xyz(xyz[t]))
        h_ytyxz = -np.log(pdf_ytyxz(ytyxz[t]))
        h_yz = -np.log(pdf_yz(yz[t]))

        mi = h_ytyz + h_xyz - h_ytyxz - h_yz

        di += mi

    # correcting negative estimates to be zero
    return di / T if di >= 0 else 0


def directed_information_graph(samples, labels=None, threshold=0.05, order=1, subset_selection=None, estimator=prob.KNN()):
    """"
    Compute directed information (DI) between all variables and plot results
    in a Direct Information Graph (DIG).

    Parameters:
    - samples (numpy.ndarray): A 2D array of time series data, where each row
    represents a sample and each column a variable [[x1, y1, z1, ...],
                                                    [x2, y2, z2, ...],
                                                    ...].
    - labels (list or None): A list of labels for each variable (optional).
    - threshold (float): The threshold for DI in graph (default is 0.05).
    - order (int): The memory order (lag) for DI estimation (default is 1).
    - subset_selection (object): Subset selection policy used to determine set
    causally conditioned on.
    - estimator (KDE or KNN): An estimator for probability density functions .

    Returns:
    - di_matrix (numpy.ndarray): A matrix containing DI values between
    all variables.
    - plot (graphviz.Digraph or None): Visual representation of DIG if labels
    are provided, else None.
    """
    assert samples.ndim == 2, 'Sample array must be 2-dimensional'

    n_vars = samples.shape[1]
    di_matrix = np.zeros((n_vars, n_vars))

    # compute directed information for every pair
    for i in range(n_vars):
        x = samples[:, i]

        for j in range(n_vars):
            y = samples[:, j]
            # exclude i, j from set causally conditioned on
            columns = np.ones(n_vars, dtype=np.bool_)
            columns[i] = False
            columns[j] = False
            z = samples[:, columns]

            di = directed_information(x, y, z, order, subset_selection, estimator)
            di_matrix[i, j] = di

    # plot if labels given
    if labels is not None:
        plot = utils.plot_directed_graph('dig', di_matrix, labels, threshold)
        return di_matrix, plot

    return di_matrix, None


def time_varying_di(window_size, step_size, x, y, z=None, order=1, subset_selection=None, estimator=prob.KNN()):
    """
    Estimate time-varying Directed Information using rolling window.

    Parameters:
    - window_size (int): The size of the rolling window.
    - step_size (int): The step size for moving the rolling window.
    - x (numpy.ndarray): The first time series of form [x1, x2, ...].
    - y (numpy.ndarray): The second time series of form [y1, y2, ...].
    - z (numpy.ndarray or None): A 2D array of time series data causally
    conditioned on of form [[a1, b1, ...],
                            [a2, b2, ...],
                            ...]
    - order (int): The memory order (lag) for DI estimation (default is 1).
    - subset_selection (object): Subset selection policy used to determine set
    causally conditioned on and reduce dimensionality.
    - estimator (KDE or KNN): An estimator for probability density functions.

    Returns:
    - di (numpy.ndarray): A list of time-varying Directed Information values
    for each window.
    """
    assert x.shape[0] == y.shape[0], 'Sample size of x and y must match'
    n_samples = x.shape[0]

    di = []
    num_steps = (n_samples - window_size) // step_size

    for i in range(num_steps):
        start = i * step_size
        end = start + window_size

        window_x = x[start:end]
        window_y = y[start:end]
        window_z = z[start:end] if z is not None else None

        di_window = directed_information(window_x, window_y, window_z, order, subset_selection, estimator)
        di.append(di_window)

    # including samples otherwise cut off
    start = num_steps * step_size

    window_x = x[start:]
    window_y = y[start:]
    window_z = z[start:] if z is not None else None

    di_window = directed_information(window_x, window_y, window_z, order, subset_selection, estimator)
    di.append(di_window)

    return np.array(di)


def time_varying_dig(samples, window_size, step_size, order=1, subset_selection=None, estimator=prob.KNN()):
    """
    Compute time-varying directed information (DI) between all variables using
    rolling window.

    Parameters:
    - samples (numpy.ndarray): A 2D array of time series data, where each row
    represents a sample and each column a variable [[x1, y1, ...],
                                                    [x2, y2, ...],
                                                    ...].
    - window_size (int): The size of the rolling window.
    - step_size (int): The step size for moving the rolling window.
    - order (int): The memory order (lag) for DI estimation (default is 1).
    - subset_selection (object): Subset selection policy used to determine set
    causally conditioned on and reduce dimensionality.
    - estimator (KDE or KNN): An estimator for probability density functions.

    Returns:
    - di_matrix (numpy.ndarray): A 3D list of time-varying DI
    values between all variables.

    """
    assert samples.ndim == 2, 'Sample array must be 2-dimensional'

    n_vars = samples.shape[1]
    di_matrix = []

    # compute directed information for every pair
    for i in range(n_vars):
        x = samples[:, i]
        di_from_x = []

        for j in range(n_vars):
            y = samples[:, j]

            # exclude i, j from z causally conditioned on
            columns = np.ones(n_vars, dtype=np.bool_)
            columns[i] = False
            columns[j] = False
            z = samples[:, columns]

            di = time_varying_di(window_size, step_size, x, y, z, order, subset_selection, estimator)
            di_from_x.append(di)

        di_matrix.append(di_from_x)

    return np.array(di_matrix)


class SubsetSelection:
    def __init__(self, n, policy, cut_off=0.05):
        self.n = n
        self.policy = policy
        self.cut_off = cut_off


class Policies(Enum):
    PAIRWISE = 1
    CORRELATION = 2
    MUTUAL_INFORMATION = 3
    PC_ALGORITHM = 4


def select_subset(target, features, subset_selection, order=1):
    """
    Selects a subset of features in regards to target given the subset
    selction policy.

    Parameters:
    - target (numpy.ndarray): A 2D array of samples (of one variable).
    - features (numpy.ndarray): A 2D array of samples (of multiple variables).
    - subset_selection (SubsetSelection): Policy used to determine subset of
    the features.

    Returns:
    - (numpy.ndarray): Subset of the feature variables selected with given
    selection policy.
    """
    assert target.ndim == features.ndim == 2, 'Target and feature arrays must be 2-dimensional'
    assert target.shape[0] == features.shape[0], 'Sample size of target and features must match'

    if subset_selection.policy == Policies.PAIRWISE:
        n_samples = target.shape[0]
        # return empty array
        return np.array([[]] * (n_samples - order))

    elif subset_selection.policy == Policies.CORRELATION:
        return subset_correlation(target, features, subset_selection.n, subset_selection.cut_off, order)

    elif subset_selection.policy == Policies.MUTUAL_INFORMATION:
        return subset_mutual_information(target, features, subset_selection.n, subset_selection.cut_off, order)

    elif subset_selection.policy == Policies.PC_ALGORITHM:
        return subset_PC(target, features, subset_selection.n, subset_selection.cut_off, order)


def subset_correlation(target, features, size, cut_off=0.05, order=1):
    """
    Selects a subset from features with the highest correlation to the
    target vector.

    Parameters:
    - target (numpy.ndarray): A 2D array of samples of the target variable.
    - features (numpy.ndarray): A 2D array of samples of the features to select
    from.
    - size (int): The desired subset size.
    - cut_off (float): The cutoff value for significant correlation.
    - order (int): The order or maximum lag to consider (default is 1).

    Returns:
    - np.ndarray: A subset of features based on correlation criteria.
    """
    assert target.ndim == features.ndim == 2, 'Target and feature arrays must be 2-dimensional'
    assert target.shape[0] == features.shape[0], 'Sample size of target and features must match'

    n_samples = target.shape[0]
    n_features = features.shape[1]

    cor_matrix = np.zeros((order, n_features))
    for o in range(1, order + 1):
        cor = np.corrcoef(target[o:], features[:-o], rowvar=False)
        cor = np.abs(cor[0, 1:])
        cor_matrix[o - 1, :] = cor

    cor_order_index = np.array([(value, o + 1, i) for o, row in enumerate(cor_matrix) for i, value in enumerate(row)])
    # discard insignificant correlations and order them by significance
    cor_order_index = cor_order_index[cor_order_index[:, 0] >= cut_off]
    cor_order_index = sorted(cor_order_index, key=lambda x: x[0])
    max_order_indices = [(int(o), int(i)) for (_, o, i) in cor_order_index][-size:]

    subset = [[] for _ in range(n_samples - order)]
    for o, i in max_order_indices:
        subset = np.hstack((subset, features[order-o:-o, [i]]))

    return np.array(subset)


def subset_mutual_information(target, features, size, cut_off=0.05, order=1):
    """
    Selects a subset from features with the highest mutual information
    with the target vector.

    Parameters:
    - target (numpy.ndarray): A 2D array of samples of the target variable.
    - features (numpy.ndarray): A 2D array of samples of the features to select
    from.
    - size (int): The desired subset size.
    - cut_off (float): The cutoff value for significant correlation.
    - order (int): The order or maximum lag to consider (default is 1).

    Returns:
    - np.ndarray: A subset of features based on mutual information criteria.
    """
    assert target.ndim == features.ndim == 2, 'Target and feature samples must be 2D arrays'
    assert target.shape[0] == features.shape[0], 'Sample size of target and features must match'

    n_samples = target.shape[0]
    n_features = features.shape[1]

    # compute mutual information between y and every lag and feature in z
    mi_matrix = np.zeros((order, n_features))
    for o in range(1, order + 1):
        mi = mutual_info_regression(features[:-o, :], target.transpose()[0, o:])
        mi_matrix[o - 1, :] = mi

    mi_order_index = np.array([(value, o + 1, i) for o, row in enumerate(mi_matrix) for i, value in enumerate(row)])
    # discard insignificant mutual information
    mi_order_index = mi_order_index[mi_order_index[:, 0] >= cut_off]
    mi_order_index = sorted(mi_order_index, key=lambda x: x[0])
    max_order_indices = [(int(o), int(i)) for (_, o, i) in mi_order_index][-size:]

    subset = [[] for _ in range(n_samples - order)]
    for o, i in max_order_indices:
        subset = np.hstack((subset, features[order-o:-o, [i]]))

    return np.array(subset)


def subset_PC(target, features, size, cut_off=0.05, order=1):
    """
    Iteratively builds a subset from features using parial correlation.
    In the first iteration the variabel with the highest normal correlation
    is selected and added to a set S. In the following iterations the variable
    with the highest correlation controlling for S will be selected and added.
    This ensures that the subset has low internal corrlation but high
    correlation with the target vector.

    Parameters:
    - target (numpy.ndarray): A 2D array of samples of the target variable.
    - features (numpy.ndarray): A 2D array of samples of the features to select
    from.
    - size (int): The desired subset size.
    - cut_off (float): The cutoff value for significant correlation.
    - order (int): The order or maximum lag to consider (default is 1).

    Returns:
    - np.ndarray: A subset of features based on mutual information criteria.
    """
    assert target.ndim == features.ndim == 2, 'Target and feature samples must be 2D arrays'
    assert target.shape[0] == features.shape[0], 'Sample size of target and features must match'

    n_samples = target.shape[0]
    n_features = features.shape[1]

    # list of selected (order, idx) tuples
    selected = []

    # iteratively build set causally conditioned on by selecting variables
    # that have highest partial correlation given the already selected variables
    for i in range(size):
        pcor_order_index = []
        control = np.array([[] for _ in range(n_samples - order)])
 
        for o in range(1, order + 1):
            # first step without controlling variable, use normal correlation
            if i == 0:
                cor = np.corrcoef(target[o:], features[:-o], rowvar=False)
                cor = np.abs(cor[0, 1:])
                
                for i, c in enumerate(cor):
                    pcor_order_index.append([c, o, i])
            # calculate partial correlation controlling for selected for each lag
            else:
                for i in range(n_features):
                    if (o, i) not in selected:
                        pcor = partial_correlation(target.transpose()[0][order:], features[order - o:-o, i], control)
                        pcor_order_index.append([abs(pcor), o, i])

        # cut insignificant partial correlation
        pcor_order_index = np.array(pcor_order_index)
        pcor_order_index = pcor_order_index[pcor_order_index[:, 0] >= cut_off]

        # no significant correlations left
        if len(pcor_order_index) == 0:
            break

        # add most significant to set controlled for in next iteration
        _, max_order, max_index = sorted(pcor_order_index, key=lambda x: x[0])[-1]
        max_order = int(max_order)
        max_index = int(max_index)

        selected.append((max_order, max_index))
        control = np.hstack((control, features[order - max_order:-max_order, [max_index]]))

    subset = [[] for _ in range(n_samples - order)]
    for o, i in selected:
        subset = np.hstack((subset, features[order - o:-o, [i]]))

    return np.array(subset)


def partial_correlation(x, y, control):
    """
    Calculate partial correlations between X and Y
    while controlling for the specified variables.

    Parameters:
    - x (numpy.ndarray): 1D array of first variable with n samples.
    - y (numpy.ndarray): 1D array of second variable with n samples.
    - control (numpy.ndarray): 2D array, n x p matrix of control variables.

    Returns:
    - partial_corrs (numpy.ndarray): 1D array, partial correlations for each
    feature.
    """
    assert x.ndim == y.ndim == 1, 'x and y must be one-dimensional'
    assert x.shape[0] == y.shape[0] == control.shape[0], 'Sample sizes of x, y and all control variables must match'

    x = np.array([x]).transpose()
    y = np.array([y]).transpose()
    matrix = np.hstack((x, y, control))
    corr = np.corrcoef(matrix, rowvar=False)
    corr_inv = np.linalg.inv(corr)

    return -corr_inv[0, 1] / np.sqrt(corr_inv[0, 0] * corr_inv[1, 1])


def get_lagged_samples(samples, lags):
    """
    Transform a list of samples with corresponding lag pairs into
    lagged time series data.

    Example:
    Turns (X_t, Y_t, Z_t, ...) with [(f0, l0), (f1, l1), (f2, l2), ...]
    into (X_t-l0^t-f0, Y_t-l1^t-f2, Z_t-l2^t-f2, ...) or in python notation
    (X[t-l0 : t-f0+1], Y[t-l1 : t-f1+1], Z[t-l2 : t-f2+1], ...)


    Parameters:
    - samples (numpy.ndarray): A 2D array, where each row represents a sample
    and each column a variable.
    - lags (list of tuples): List of lag pairs [(f0, l0), (f1, l1), (f2, l2), ...]
    with f_i < l_i, where f_i and l_i are first and last lag included for the
    i-th variable.

    Returns:
    - lagged_samples (numpy.ndarray): Lagged time series data.
    """
    n_samples = samples.shape[0]
    n_features = samples.shape[1]
    max_lag = max([t[1] for t in lags]) if lags != [] else 0

    assert n_features == len(lags)
    assert max_lag < samples.shape[0]

    lagged_samples = [[] for _ in range(n_samples - max_lag)]

    # extract values of feature for each lag pair and add to lagged_returns
    for feature, (first, furthest) in enumerate(lags):
        lagged_feature = np.array([samples[max_lag - k: -k if k > 0 else None, feature] for k in range(first, furthest + 1)])
        lagged_samples = np.hstack((lagged_samples, lagged_feature.transpose()))

    return lagged_samples
