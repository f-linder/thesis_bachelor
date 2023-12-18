import os
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from graphviz import Digraph
from enum import Enum


'''
##################################################################
                    utils for acquiring data
##################################################################
'''


class Index(Enum):
    SP100 = 1
    SP500 = 2


def download_index(index, start_date, end_date, interval='1d'):
    """
    Download historical stock price data for a specified index.

    Parameters:
    - index (Index): An Index enum value (SP100 or SP500).
    - start_date (datetime): The start date for data download.
    - end_date (datetime): The end date for data download.
    - interval (str): The data interval (default is '1d' for daily).

    Returns:
    - tickers (dictionary): A dictionary mapping each ticker symbol to its
    returns in percent.
    """

    if index == Index.SP100:
        print('Selected S&P100 data set:')
        tickers = pd.read_csv('./data/sp100.csv')['Symbol']
        download_tickers(tickers, start_date, end_date, interval)
        return tickers

    elif index == Index.SP500:
        print('Selected S&P500 data set:')
        tickers = pd.read_csv('./data/sp500.csv')['Symbol']
        download_tickers(tickers, start_date, end_date, interval)
        return tickers


def download_tickers(tickers, start_date, end_date, interval='1d'):
    """
    Download historical stock price data for a list of ticker symbols
    using yfinance and calculate returns.

    Parameters:
    - tickers (list): A list of ticker symbols to download.
    - start_date (datetime): The start date for the data download.
    - end_date (datetime): The end date for the data download.
    - interval (str): The data interval (default is '1d' for daily).

    Returns:
    - ticker_returns (dictionary): A dictionary mapping each ticker symbol to
    its returns in percent.

    Note:
    - The function assumes that the input dates are valid and that the tickers
    exist in the data source.
    - If a ticker's data is empty or unavailable for yfinance, it will not be
    included in the returned dictionary.
    """
    print(f'Starting download for data of {len(tickers)} tickers...')

    ticker_returns = {}
    for i, ticker in enumerate(tickers):
        print(f'Downloading {ticker}: {i + 1}/{len(tickers)}')

        # [Date, Open, High, Low, Close, Adj Close, Volume]
        data = yf.download(ticker, interval=interval, start=start_date,
                           end=end_date, progress=False)

        data['Return Abs'] = data['Close'] - data['Open']
        data['Returns'] = data['Return Abs'] / data['Open'] * 100

        if not data.empty:
            data['Returns'].to_csv(f'./data/{ticker}.csv')
            ticker_returns[ticker] = data['Returns']

    print('Download complete')

    return ticker_returns


def clean_up():
    """
    Deleting downloaded data files, excluding specific files and directories.

    Parameters:
    - None

    Returns:
    - None
    """

    dir_path = './data'
    for filename in os.listdir(dir_path):
        # do not delete index .csv files
        if filename in ['sp100.csv', 'sp500.csv', 'simulations']:
            continue
        os.remove(os.path.join(dir_path, filename))

    dir_path = '.data/simulations'
    for filename in os.listdir(dir_path):
        os.remove(os.path.join(dir_path, filename))

    print('Clean up complete: removed all .csv files')


'''
##################################################################
            utils for simulations of VAR(p) models
##################################################################
'''


def reduce_to_VAR1(coefficients):
    """
    Reduce the coefficient matrix of a VAR(p) model to coefficients of a VAR(1)
    model, the companion form. The matrix is of shape (p x n_vars x n_vars).

    Parameters:
    - coefficients (numpy.ndarray): The 3D coefficient array of a VAR(p) model.

    Returns:
    - F (numpy.ndarray): The coefficient matrix for VAR(1) in companion form.
    """
    assert coefficients.ndim == 3, 'Coefficient matrix must be 3-dimensional'
    assert coefficients.shape[1] == coefficients.shape[2], 'Coefficient matrix must be of shape (p x n_vars x n_vars)'

    p = coefficients.shape[0]
    n_vars = coefficients.shape[1]

    if p == 1:
        return coefficients[0]

    F_upper = np.hstack(coefficients)
    F_lower = np.eye(n_vars * (p - 1), n_vars * p)
    F = np.vstack((F_upper, F_lower))

    return F


def is_stable(companion_matrix, threshold=0.9):
    """
    Checks whether the VAR model corresponding to the companion_matrix is
    stable. The maximum eigenvalue has to be non-zero and less or equal to
    the threshold.

    Parameters:
    - companion_matrix (numpy.ndarray): A 2D array containing the coefficients
    of every order.
    - threshold (float): The threshold for the largest eigenvalue (default is 0.9).

    Returns:
    - boolean: True if model is stable, False otherwise.
    """
    assert companion_matrix.ndim == 2, 'Companion matrix must be 2-dimensional'
    assert companion_matrix.shape[0] == companion_matrix.shape[1], 'Companion matrix must be of shape (m x m)'
    assert threshold < 1, 'Stability not guaranteed for a threshold >= 1.'

    eigenvalues = np.linalg.eigvals(companion_matrix)
    max = np.max(np.abs(eigenvalues))

    return True if max != 0 and max <= 0.9 else False


'''
##################################################################
                        visualization
##################################################################
'''


def plot_directed_graph(draw, data, labels, threshold=0.05):
    """
    Generate and visualize a directed graph based on the given data and labels.

    Parameters:
    - draw (str): Specifies the type of graph to draw.
    Can be one of ['dig', 'var', 'nvar'].
    - data (numpy.ndarray): The directed information or influence data.
    For a DIG it is the 2D adjecency matrix with DI values. For VAR is the 3D
    array of coefficients for each lag. For NVAR it is the 3D string
    representation array of the functions for each lag.
    - labels (list): List of labels for nodes in the graph.
    - threshold (float): The threshold for edge inclusion in the graph (default
    is 0.05).

    Returns:
    - graphviz.Digraph: The generated directed graph.
    """
    assert draw in ['dig', 'var', 'nvar'], 'Draw must be one of ["dig", "var", "nvar"]'
    assert len(labels) == data.shape[1]

    n_vars = data.shape[1]
    graph = Digraph(format='svg', engine='fdp', graph_attr={'color': 'lightblue', 'rankdir': 'LR', 'splines': 'true'})

    if draw == 'dig':
        for i in range(n_vars):
            for j in range(n_vars):
                influence = data[i, j]
                if influence >= threshold:
                    graph.node(labels[i], label=labels[i])
                    graph.node(labels[j], label=labels[j])
                    graph.edge(labels[i], labels[j], label=f'{influence:.3f}')
    else:
        order = len(data)
        for i in range(n_vars):
            # time t
            name_t = f'{labels[i]}_t'
            graph.node(name_t, label=name_t, pos=f'{n_vars * 2},{n_vars - i}!', color='blue')

            for o in range(1, order + 1):
                # time t-1
                name = f'{labels[i]}_t-{o}'
                graph.node(name, labels=name, pos=f'{(n_vars - o) * 2},{n_vars - i}!', color='lightblue')

        # add edges
        for i in range(n_vars):
            name_to = f'{labels[i]}_t'
            for j in range(n_vars):
                for o in range(order):
                    name_from = f'{labels[j]}_t-{o + 1}'
                    influence = data[o, i, j]

                    if draw == 'nvar' and influence != '':
                        graph.edge(name_from, name_to, label=f'{influence}')
                    elif draw == 'var' and np.abs(influence) > 0:
                        graph.edge(name_from, name_to, label=f'{influence:.3f}')

    return graph


def plot_3D(xs, ys, fun, titel):
    """
    Create and display a 3D surface plot of a function.

    Parameters:
    - xs (numpy.ndarray): 1D array of x-values.
    - ys (numpy.ndarray): 1D array of y-values.
    - fun (function): The function to be plotted. It takes a list [x, y] as input and returns a scalar value.
    - title (str): The title for the plot.

    Returns:
    - None
    """

    x, y = np.meshgrid(xs, ys)
    z = np.array([[fun([x, y]) for x in xs] for y in ys])

    fig = plt.figure()

    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(x, y, z, cmap='viridis')
    ax1.set_title(titel)

    plt.show()
