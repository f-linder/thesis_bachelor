import analysis.probabilities as prob
import pandas as pd
import numpy as np


def directed_information(X, Y, Z=None, order=1, estimator=prob.KNN()):
    """
    Estimate I(X -> Y || Z), the Directed Information between two time series X and Y 
    causally conditioned on a set of time series Z with memory of size l / order.

    I(X -> Y || Z) = Sum_t=1^T I(Y_t, X_t-l^t-1 | Y_t-l^t-1, Z_t-l^t-1)
                   = Sum_t=1^T H(Y_t, Y_t-l^t-1, Z_t-l^t-1) + H(X_t-l^t-1, Y_t-l^t-1, Z_t-l^t-1)
                     - H(Y_t, Y_t-l^t-1, X_t-l^t-1 Z_t-l^t-1) - H(Y_t-l^t-1, Z_t-l^t-1)

    Parameters:
    - X (numpy.ndarray): The first time series (X).
    - Y (numpy.ndarray): The second time series (Y).
    - Z (numpy.ndarray or None): The set of time series causally conditioned on (Z).
    - order (int): The memory (or lag) for DI estimation (default is 1).
    - estimator (object): An estimator for probability density functions (e.g., KDE or KNN).

    Returns:
    - di (float): The estimated Directed Information between X and
    """

    Z = Z if Z is not None else [[] for _ in range(len(X) - order)]

    # (X_t-l^t-1)
    X_lagged = get_lagged_returns(X, [(1, order)])
    Y_lagged = get_lagged_returns(Y, [(1, order)])
    Z_lagged = get_lagged_returns(Z, [(1, order) for _ in range(len(Z[0]))])

    # samples corresponding to entropy terms
    ytyz = np.hstack((Y[order:], Y_lagged, Z_lagged))
    xyz = np.hstack((X_lagged, Y_lagged, Z_lagged))
    ytyxz = np.hstack((Y[order:], Y_lagged, X_lagged, Z_lagged))
    yz = np.hstack((Y_lagged, Z_lagged))

    # density functions corresponding to entropy terms
    pdf_ytyz = prob.pdf(estimator, ytyz)
    pdf_xyz = prob.pdf(estimator, xyz)
    pdf_ytyxz = prob.pdf(estimator, ytyxz)
    pdf_yz = prob.pdf(estimator, yz)

    T = len(X) - order
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

    return di


def rolling_window(window_size, step_size, X, Y, Z=None, order=1, estimator=prob.KNN()):
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
    num_steps = (len(X) - window_size) // step_size

    for i in range(num_steps):
        start = i * step_size
        end = start + window_size

        window_X = X[start:end]
        window_Y = Y[start:end]
        window_Z = Z[start:end] if Z is not None else None

        di_window = directed_information(window_X, window_Y, window_Z, order, estimator)
        di.append(di_window)

    # including samples otherwise cut off
    start = num_steps * step_size

    window_X = X[start:]
    window_Y = Y[start:]
    window_Z = Z[start:] if Z is not None else None

    di_window = directed_information(window_X, window_Y, window_Z, order, estimator)
    di.append(di_window)

    return di


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

