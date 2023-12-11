import analysis.probabilities as prob
import analysis.utils as utils
import numpy as np
from enum import Enum
from sklearn.feature_selection import mutual_info_regression


def directed_information(x, y, z=None, order=1, subset_selection=None, estimator=prob.KNN()):
    """
    Estimate I(X -> Y || Z), the Directed Information between two time series X and Y 
    causally conditioned on a set of time series Z with memory of size l (order l).
    Directed information can't be negative. If estimates turn out negative,
    they are corrected to be zero.

    I(X -> Y || Z) = 1/T * Sum_t=1^T I(Y_t, X_t-l^t-1 | Y_t-l^t-1, Z_t-l^t-1)
                   = 1/T * Sum_t=1^T H(Y_t, Y_t-l^t-1, Z_t-l^t-1) + H(X_t-l^t-1, Y_t-l^t-1, Z_t-l^t-1)
                     - H(Y_t, Y_t-l^t-1, X_t-l^t-1 Z_t-l^t-1) - H(Y_t-l^t-1, Z_t-l^t-1)

    Parameters:
    - x (numpy.ndarray): The first time series (X).
    - y (numpy.ndarray): The second time series (Y).
    - z (numpy.ndarray or None): The set of time series causally conditioned on (Z).
    - order (int): The memory (or lag) for DI estimation (default is 1).
    - subset_selection (SubsetSelection): Subset selection policy and used to determine set causally conditioned on.
    - estimator (object): An estimator for probability density functions (e.g., KDE or KNN).

    Returns:
    - di (float): The estimated Directed Information I(X -> Y || Z).
    """
    # (X_t-l^t-1)
    x_lagged = get_lagged_returns(x, [(1, order)])
    # (Y_t-l^t-1)
    y_lagged = get_lagged_returns(y, [(1, order)])
    # special case: I(X -> X || Z)
    if (x == y).all():
        y_lagged = np.array([[] for _ in range(len(y) - order)])

    if z is None or len(z[0]) == 0:
        print('z not present: set to empty array []')
        z = np.array([[] for _ in range(len(x) - order)])
    elif subset_selection is not None:
        print('select subset')
        z = select_subset(y, z, subset_selection, order)
    else:
        print('z_present: get_lagged_returns')
        z = get_lagged_returns(z, [(1, order) for _ in range(len(z[0]))])

    # samples corresponding to entropy terms
    ytyz = np.hstack((y[order:], y_lagged, z))
    xyz = np.hstack((x_lagged, y_lagged, z))
    ytyxz = np.hstack((y[order:], y_lagged, x_lagged, z))
    yz = np.hstack((y_lagged, z))

    # density functions corresponding to entropy terms
    pdf_ytyz = prob.pdf(estimator, ytyz)
    pdf_xyz = prob.pdf(estimator, xyz)
    pdf_ytyxz = prob.pdf(estimator, ytyxz)
    pdf_yz = (lambda x: 1)if len(z[0]) == 0 and len(y_lagged[0]) == 0 else prob.pdf(estimator, yz)

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

    # ruling out negative estimates
    return di / T if di >= 0 else 0


def directed_information_graph(returns, labels=None, threshold=0.05, order=1, subset_selection=None, estimator=prob.KNN()):
    """"
    Compute directed information (DI) between all variables and plot results
    in a Direct Information Graph (DIG).

    Parameters:
    - returns (numpy.ndarray): A list of return samples of form [[x1, y1, ...], [x2, y2, ...], ...].
    - labels (list or None): A list of labels for each variable (optional).
    - threshold (float): The threshold for DI in graph (default is 0.05).
    - order (int): The memory order (lag) for DI estimation (default is 1).
    - subset_selection (object): Subset selection policy used to determine set causally conditioned on.
    - estimator (object): An estimator for probability density functions (KDE or KNN).

    Returns:
    - di_matrix (numpy.ndarray): A matrix containing DI values for all variables.
    - plot (graphviz.Digraph or None): Visual representation of DIG if labels are provided,
    else None.
    """
    n_vars = len(returns[0])
    di_matrix = np.zeros((n_vars, n_vars))

    # compute directed information for every pair
    for i in range(n_vars):
        x = returns[:, [i]]

        for j in range(n_vars):
            y = returns[:, [j]]
            # exclude i, j from z causally conditioned on
            cols = [r for r in range(n_vars) if r != i and r != j]
            z = returns[:, cols]

            print(f'from={i}, to={j}, on={cols}')
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
    - x (numpy.ndarray): The first time series (X).
    - y (numpy.ndarray): The second time series (Y).
    - z (numpy.ndarray or None): The third time series (Z), or None if not conditioned on Z.
    - order (int): The memory order (lag) for DI estimation (default is 1).
    - subset_selection (object): Subset selection policy used to determine set causally conditioned on.
    - estimator (object): An estimator for probability density functions (e.g., KDE or KNN).

    Returns:
    - di (numpy.ndarray): A list of time-varying Directed Information values for each window.
    """
    di = []
    num_steps = (len(x) - window_size) // step_size

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


def time_varying_dig(returns, window_size, step_size, order=1, subset_selection=None, estimator=prob.KNN()):
    """
    Compute time-varying directed information (DI) between all variables using rolling window.

    Parameters:
    - returns (numpy.ndarray): A list of return samples of form [[x1, y1, ...], [x2, y2, ...], ...].
    - window_size (int): The size of the rolling window.
    - step_size (int): The step size for moving the rolling window.
    - order (int): The memory order (lag) for DI estimation (default is 1).
    - subset_selection (object): Subset selection policy used to determine set causally conditioned on.
    - estimator (object): An estimator for probability density functions (e.g., KDE or KNN).

    Returns:
    - di_matrix (numpy.ndarray): A 3D list of time-varying Directed Information values between all variables.

    """
    n_vars = len(returns[0])
    di_matrix = []

    # compute directed information for every pair
    for i in range(n_vars):
        x = returns[:, [i]]

        di_from_x = []
        for j in range(n_vars):
            y = returns[:, [j]]
            # exclude i, j from z causally conditioned on
            cols = [r for r in range(n_vars) if r != i and r != j]
            z = returns[:, cols]

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


def select_subset(y, z, subset_selection, order=1):
    """
    Returns a subset of z according to the given subset_selection object.

    Parameters:
    - y (numpy.ndarray): List of return samples (of one variable).
    - z (numpy.ndarray): List of return samples (of multiple variables).
    - subset_selection (object): Policy used to determine subset of z.

    Returns:
    - subset_z (numpy.ndarray): List of return samples of a subset of the variables given in z.
    """
    assert y.shape[0] == z.shape[0], 'Sample size of y and z must match'

    n_samples = y.shape[0]

    # return empty array
    if subset_selection.policy == Policies.PAIRWISE:
        return np.array([[] for _ in range(n_samples - order)])

    # cor(Y_t, Z_t-l^t-1)
    elif subset_selection.policy == Policies.CORRELATION:
        # compute correlation between y and every lag and feature in z
        cor_matrix = np.zeros((order, len(z[0])))
        for o in range(1, order + 1):
            cor = np.corrcoef(y[o:], z[:-o], rowvar=False)
            cor = np.abs(cor[0, 1:])
            cor_matrix[o - 1, :] = cor

        cor_order_index = np.array([(value, o + 1, i) for o, row in enumerate(cor_matrix) for i, value in enumerate(row)])
        # discard insignificant correlations
        cor_order_index = cor_order_index[cor_order_index[:, 0] >= subset_selection.cut_off]
        cor_order_index = sorted(cor_order_index, key=lambda x: x[0])
        max_order_indices = [(int(o), int(i)) for (_, o, i) in cor_order_index][-subset_selection.n:]

        print(f'cor_matrix={cor_matrix}')
        print(f'max_order_indices = {max_order_indices}')
        subset_z = [[] for _ in range(n_samples - order)]
        for o, i in max_order_indices:
            subset_z = np.hstack((subset_z, z[order-o:-o, [i]]))

        return np.array(subset_z)

    # I(Y_t, Z_t-l^t-1)
    elif subset_selection.policy == Policies.MUTUAL_INFORMATION:
        # compute mutual information between y and every lag and feature in z
        mi_matrix = np.zeros((order, len(z[0])))
        for o in range(1, order + 1):
            mi = mutual_info_regression(z[:-o, :], y.transpose()[0, o:])
            mi_matrix[o - 1, :] = mi

        mi_order_index = np.array([(value, o + 1, i) for o, row in enumerate(mi_matrix) for i, value in enumerate(row)])
        # discard insignificant mutual information
        mi_order_index = mi_order_index[mi_order_index[:, 0] >= subset_selection.cut_off]
        mi_order_index = sorted(mi_order_index, key=lambda x: x[0])
        max_order_indices = [(int(o), int(i)) for (_, o, i) in mi_order_index][-subset_selection.n:]

        print(f'mi_matrix={mi_matrix}')
        print(f'max_order_indices = {max_order_indices}')
        subset_z = [[] for _ in range(n_samples - order)]
        for o, i in max_order_indices:
            subset_z = np.hstack((subset_z, z[order-o:-o, [i]]))

        return np.array(subset_z)

    # iteratively build set causally conditioned on by selecting variables
    # that have highest partial correlation given the already selected variables
    elif subset_selection.policy == Policies.PC_ALGORITHM:
        # list of selected order + indices
        selected = []

        for i in range(subset_selection.n):
            partial_cor = np.zeros((order, len(z[0])))
            for o in range(1, order + 1):
                # first step without controlling variable, use normal correlation
                if i == 0:
                    cor = np.corrcoef(y[o:], z[:-o], rowvar=False)
                    partial_cor[o - 1, :] = np.abs(cor[0, 1:])
                else:
                    target = y.transpose()[0][order:]
                    features_indices = np.array([(o, x) not in selected for x in range(len(z[0]))])
                    # no more features to condition on left for this lag
                    if (features_indices == False).all():
                        break
                    features = z[order - o:-o, features_indices]
                    control = [[] for _ in range(len(z) - order)]
                    for ord, idx in selected:
                        control = np.hstack((control, z[order - ord:-ord, [idx]]))

                    partial_cor[o - 1, :] = utils.partial_correlation(target, features, control)

            cor_order_index = np.array([(value, o + 1, i) for o, row in enumerate(partial_cor) for i, value in enumerate(row)])
            cor_order_index = cor_order_index[cor_order_index[:, 0] >= subset_selection.cut_off]
            # no significant correlations left
            if len(cor_order_index) == 0:
                break

            _, max_order, max_index = sorted(cor_order_index, key=lambda x: x[0])[-1]
            selected.append((int(max_order), int(max_index)))
            print(f'selected={selected}')

        subset_z = [[] for _ in range(n_samples - order)]
        for o, i in selected:
            subset_z = np.hstack((subset_z, z[order - o:-o, [i]]))

        print('-----------------------------------------')
        return np.array(subset_z)


def get_lagged_returns(returns, lags):
    """
    Transform a list of return samples with corresponding lag pairs into
    lagged time series data.

    Example:
    Turns (X_t, Y_t, Z_t, ...) with [(f0, l0), (f1, l1), (f2, l2), ...]
    into (X_t_l0^t-f0, Y_t-l1^t-f2, Z_t-l2^t-f2, ...) or in python notation
    (X[t-l0 : t-f0+1], Y[t-l1 : t-f1+1], Z[t-l2 : t-f2+1], ...)


    Parameters:
    - returns (numpy.ndarray): List of return samples.
    - lags (list of tuples): List of lag pairs [(f0, l0), (f1, l1), (f2, l2), ...] with f_i < l_i.

    Returns:
    - lagged_returns (numpy.ndarray): Lagged time series data.
    """
    n_features = len(returns[0])
    max_lag = max([t[1] for t in lags]) if lags != [] else 0

    assert n_features == len(lags)
    assert max_lag < len(returns)

    lagged_returns = [[] for _ in range(len(returns) - max_lag)]

    # extract values of feature for each lag pair and add to lagged_returns
    for feature, (first, furthest) in enumerate(lags):
        lagged_feature = np.array([returns[max_lag - k: -k if k > 0 else None, feature] for k in range(first, furthest + 1)])
        lagged_returns = np.hstack((lagged_returns, lagged_feature.transpose()))

    return lagged_returns

