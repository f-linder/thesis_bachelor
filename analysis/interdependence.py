import analysis.probabilities as prob
import analysis.utils as utils
import pandas as pd
import numpy as np
from enum import Enum


class SubsetSelection:
    def __init__(self, n, policy):
        self.n = n
        self.policy = policy


class Policies(Enum):
    CORRELATION = 1


def directed_information(x, y, z=None, order=1, subset_selection=None, estimator=prob.KNN()):
    """
    Estimate I(X -> Y || Z), the Directed Information between two time series X and Y 
    causally conditioned on a set of time series Z with memory of size l / order.

    I(X -> Y || Z) = 1/T * Sum_t=1^T I(Y_t, X_t-l^t-1 | Y_t-l^t-1, Z_t-l^t-1)
                   = 1/T * Sum_t=1^T H(Y_t, Y_t-l^t-1, Z_t-l^t-1) + H(X_t-l^t-1, Y_t-l^t-1, Z_t-l^t-1)
                     - H(Y_t, Y_t-l^t-1, X_t-l^t-1 Z_t-l^t-1) - H(Y_t-l^t-1, Z_t-l^t-1)

    Parameters:
    - x (numpy.ndarray): The first time series (X).
    - y (numpy.ndarray): The second time series (Y).
    - z (numpy.ndarray or None): The set of time series causally conditioned on (Z).
    - order (int): The memory (or lag) for DI estimation (default is 1).
    - estimator (object): An estimator for probability density functions (e.g., KDE or KNN).

    Returns:
    - di (float): The estimated Directed Information between X and
    """
    z = z if z is not None else [[] for _ in range(len(x) - order)]
    if subset_selection is not None:
        z = select_subset(y, z, subset_selection)

    # (X_t-l^t-1)
    x_lagged = get_lagged_returns(x, [(1, order)])
    y_lagged = get_lagged_returns(y, [(1, order)])
    z_lagged = get_lagged_returns(z, [(1, order) for _ in range(len(z[0]))])

    # samples corresponding to entropy terms
    ytyz = np.hstack((y[order:], y_lagged, z_lagged))
    xyz = np.hstack((x_lagged, y_lagged, z_lagged))
    ytyxz = np.hstack((y[order:], y_lagged, x_lagged, z_lagged))
    yz = np.hstack((y_lagged, z_lagged))

    # density functions corresponding to entropy terms
    pdf_ytyz = prob.pdf(estimator, ytyz)
    pdf_xyz = prob.pdf(estimator, xyz)
    pdf_ytyxz = prob.pdf(estimator, ytyxz)
    pdf_yz = prob.pdf(estimator, yz)

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

    return di / T


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
    - di_matrix (numpy.ndarray): A matrix containing DI values between all variables.
    """
    n_vars = len(returns[0])
    di_matrix = []

    # compute directed information for every pair
    for i in range(n_vars):
        x_influences = []
        x = returns[:, [i]]

        for j in range(n_vars):
            if i == j:
                x_influences.append(0)
                continue

            y = returns[:, [j]]
            # exclude i, j from set causally conditioned on
            cols = [r for r in range(n_vars) if r != i and r != j]
            z = returns[:, cols] if n_vars > 2 else None

            di = directed_information(x, y, z, order, subset_selection, estimator)
            x_influences.append(di)

        di_matrix.append(x_influences)

    # plot if labels given
    if labels is not None:
        utils.plot_dig(di_matrix, labels, threshold)

    return np.array(di_matrix)


def rolling_window(window_size, step_size, x, y, z=None, order=1, estimator=prob.KNN()):
    """
    Estimate time-varying Directed Information using a rolling window approach.

    Parameters:
    - window_size (int): The size of the rolling window.
    - step_size (int): The step size for moving the rolling window.
    - X (numpy.ndarray): The first time series (X).
    - Y (numpy.ndarray): The second time series (Y).
    - Z (numpy.ndarray or None): The third time series (Z), or None if not conditioned on Z.
    - order (int): The memory order (lag) for DI estimation (default is 1).
    - estimator (object): An estimator for probability density functions (e.g., KDE or KNN).

    Returns:
    - di (list): A list of time-varying Directed Information values estimated for each window.
    """
    di = []
    num_steps = (len(x) - window_size) // step_size

    for i in range(num_steps):
        start = i * step_size
        end = start + window_size

        window_X = x[start:end]
        window_Y = y[start:end]
        window_Z = z[start:end] if z is not None else None

        di_window = directed_information(window_X, window_Y, window_Z, order, estimator)
        di.append(di_window)

    # including samples otherwise cut off
    start = num_steps * step_size

    window_X = x[start:]
    window_Y = y[start:]
    window_Z = z[start:] if z is not None else None

    di_window = directed_information(window_X, window_Y, window_Z, order, estimator)
    di.append(di_window)

    return di


def select_subset(y, z, subset_selection):
    """
    Returns a subset of Z according to the given subset_selection.

    Parameters:
    - y (numpy.ndarray): List of return samples (of one variable)
    - z (numpy.ndarray): List of return samples (of multiple variables)

    Returns:
    - subset_z (numpy.ndarray): List of return samples of a subset of the variables given in z
    """
    if subset_selection.policy == Policies.CORRELATION:
        cor = np.corrcoef(y.tranpose()[0], z.transpose())[0, 1:]
        cor_index = [c for c in enumerate(cor)]
        sorted_cor_index = sorted(cor_index, key=lambda x: x[1])

        max_indices = [i for (i, _) in sorted_cor_index[-subset_selection.n:]]
        subset_z = z[max_indices].transpose()
        return subset_z

    return z


def get_lagged_returns(returns, lags):
    """
    Transform a list of return samples with corresponding lag pairs into
    lagged time series data.

    Example:
    Turns (X_t, Y_t, Z_t, ...) with [(f0, l0), (f1, l1), (f2, l2), ...]
    into (X_t-l0^t-f0, Y_t-l1^t-f2, Z_t-l2^t-f2, ...) or in python notation
    (X[t-l0 : t-f0+1], Y[t-l1 : t-f1+1], Z[t-l2 : t-f2+1], ...)


    Parameters:
    - returns (numpy.ndarray): List of return samples.
    - lags (list of tuples): List of lag pairs [(f0, l0), (f1, l1), (f2, l2), ...] with f_i < l_i.

    Returns:
    - lagged_returns (numpy.ndarray): Lagged time series data.
    """
    n_features = len(returns[0])
    max_lag = max([t[1] for t in lags]) if lags != [] else 0

    if n_features != len(lags) or max_lag >= len(returns):
        raise Exception('error get_lagged_returns()')

    lagged_returns = [[] for _ in range(len(returns) - max_lag)]

    # extract values of feature for each lag pair and add to lagged_returns
    for feature, (first, furthest) in enumerate(lags):
        lagged_feature = np.array([returns[max_lag - k: -k if k > 0 else None, feature] for k in range(first, furthest + 1)])
        lagged_returns = np.hstack((lagged_returns, lagged_feature.transpose()))

    return lagged_returns


def directed_information_tickers(ticker_source, ticker_target,
                                 tickers_cond=[], order=1, estimator=prob.KNN()):
    return 0
    # prepare data of tickers
    df_source = pd.read_csv(f'./data/{ticker_source}.csv')
    df_target = pd.read_csv(f'./data/{ticker_target}.csv')
    df_others = [pd.read_csv(f'./data/{o}.csv') for o in tickers_other]

    returns_source = np.array(df_source['Returns'])
    returns_target = np.array(df_target['Returns'])
    returns_others = np.array([df['Returns'] for df in df_others])

    # TODO: select subset of others with highest correlation
    # corr_matrix = [np.corrcoef(returns_target, o) for o in returns_others]
    cond = None

    return directed_information(returns_source.transpose(),
                                returns_target.transpose(), cond, estimator)


