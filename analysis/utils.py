import os
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats, linalg
from graphviz import Digraph
from enum import Enum


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
    - tickers (dictionary): A dictionary mapping each ticker symbol of the index 
    to its returns in percent.
    """

    if index == Index.SP100:
        print('Selected S&P100 data set')
        tickers = pd.read_csv('./data/sp100.csv')['Symbol']
        download_tickers(tickers, start_date, end_date, interval)
        return tickers

    elif index == Index.SP500:
        print('Selected S&P500 data set')
        tickers = pd.read_csv('./data/sp500.csv')['Symbol']
        download_tickers(tickers, start_date, end_date, interval)
        return tickers


def download_tickers(tickers, start_date, end_date, interval='1d'):
    """
    Download historical stock price data for a list of ticker symbols and calculate returns.

    Parameters:
    - tickers (list): A list of ticker symbols to download.
    - start_date (datetime): The start date for the data download.
    - end_date (datetime): The end date for the data download.
    - interval (str): The data interval (default is '1d' for daily).

    Returns:
    - ticker_returns (dictionary): A dictionary mapping each ticker symbol to its returns in percent.

    Note:
    - The function assumes that the input dates are valid and that the tickers exist in the data source.
    - If a ticker's data is empty or unavailable, it will not be included in the returned dictionary.
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
    Clean up downloaded data files, excluding specific files and directories.

    Parameters:
    - None

    Returns:
    - None
    """

    dir_path = './data'
    for filename in os.listdir(dir_path):
        if filename in ['sp100.csv', 'sp500.csv', 'simulations']:
            continue
        os.remove(os.path.join(dir_path, filename))

    dir_path = '.data/simulations'
    for filename in os.listdir(dir_path):
        os.remove(os.path.join(dir_path, filename))

    print('Clean up complete: removed all .csv files')


def partial_correlation(target, features, control):
    """
    Calculate partial correlations between the target and each feature
    while controlling for the specified variables.

    Parameters:
    - target (numpy.ndarray): 1D array, target variable with n samples.
    - features (numpy.ndarray): 2D array, n x d matrix of features.
    - control (numpy.ndarray): 2D array, n x p matrix of control variables.

    Returns:
    - partial_corrs (numpy.ndarray): 1D array, partial correlations for each feature.
    """
    assert target.ndim == 1, 'Target vector must be one-dimensional'
    assert target.shape[0] == features.shape[0], 'Sample size of target and features must match'
    assert target.shape[0] == control.shape[0], 'Sample size of target and control must match'

    matrix = np.hstack((features, np.array([target]).transpose(), control))
    corr = np.corrcoef(matrix, rowvar=False)
    corr_inv = np.linalg.inv(corr)

    dim = features.shape[1]
    partial_corrs = np.zeros(dim)

    # j = dim is index of target vector
    for i in range(dim):
        p_ij = corr_inv[i, dim]
        p_ii = corr_inv[i, i]
        p_jj = corr_inv[dim, dim]

        partial_corrs[i] = -p_ij / np.sqrt(p_ii * p_jj)

    return partial_corrs


def plot_directed_graph(draw, data, labels, threshold=0.05):
    """
    Generate and visualize a directed graph based on the given data and labels.

    Parameters:
    - draw (str): Specifies the type of graph to draw. Can be one of ['dig', 'var', 'nvar'].
    - data (numpy.ndarray): The directed information or influence data.
    - labels (list): List of labels for nodes in the graph.
    - threshold (float): The threshold for edge inclusion in the graph (default is 0.05).

    Returns:
    - graphviz.Digraph: The generated directed graph.
    """
    assert draw in ['dig', 'var', 'nvar']

    n_vars = len(data[0])
    graph = Digraph(format='svg', graph_attr={'color': 'lightblue2', 'rankdir': 'LR'})

    if draw == 'dig':
        for label in labels:
            graph.node(label, label=label)
        for i in range(n_vars):
            for j in range(n_vars):
                influence = data[0, i, j]
                if influence >= threshold:
                    graph.edge(labels[i], labels[j], label=f'{influence:.3f}')
    else:
        order = len(data)
        # add nodes of orders
        for o in range(order, 0, -1):
            with graph.subgraph(name=f't-{o}') as sub:
                sub.attr(rank='same')
                for i in range(n_vars):
                    name = f'{labels[i]}_t-{o}'
                    sub.node(name, label=name)
        # add nodes for time t
        with graph.subgraph(name='t') as sub:
            sub.attr(rank='same')
            for i in range(n_vars):
                name = f'{labels[i]}_t'
                sub.node(name, label=name)

        # add edges
        for i in range(n_vars):
            name_to = f'{labels[i]}_t'
            for j in range(n_vars):
                for o in range(order):
                    name_from = f'{labels[j]}_t-{o + 1}'
                    influence = data[o, i, j]

                    if draw == 'nvar' and influence != '':
                        graph.edge(name_from, name_to, label=f'{influence}')
                    elif draw == 'var' and influence > 0:
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
