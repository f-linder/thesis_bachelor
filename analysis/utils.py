import os
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
    - tickers (list): A list of ticker symbols downloaded for specified index.
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
    Download historical stock price data for a list of ticker symbols.

    Parameters:
    - tickers (list): A list of ticker symbols to download.
    - start_date (datetime): The start date for data download.
    - end_date (datetime): The end date for data download.
    - interval (str): The data interval (default is '1d' for daily).

    Returns:
    - None
    """

    print(f'Starting download for data of {len(tickers)} tickers...')

    for i, ticker in enumerate(tickers):
        print(f'Downloading {ticker}: {i + 1}/{len(tickers)}')

        # [Date, Open, High, Low, Close, Adj Close, Volume]
        data = yf.download(ticker, interval=interval, start=start_date,
                           end=end_date, progress=False)

        data['Return Abs'] = data['Close'] - data['Open']
        data['Returns'] = data['Return Abs'] / data['Open'] * 100

        if not data.empty:
            data['Returns'].to_csv(f'./data/{ticker}.csv')

    print('Download complete')


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
                for i in range(n_vars):
                    name = f'{labels[i]}_t-{o}'
                    sub.node(name, label=name)
                sub.attr(rank='same')
        # add nodes for time t
        with graph.subgraph(name='t') as sub:
            for i in range(n_vars):
                name = f'{labels[i]}_t'
                sub.node(name, label=name)
            sub.attr(rank='same')

        # add edges
        for i in range(n_vars):
            name_to = f'{labels[i]}_t'
            for j in range(n_vars):
                for o in range(order):
                    name_from = f'{labels[j]}_t-{o + 1}'
                    influence = data[o, i, j]

                    if draw == 'nvar' and influence != 's':
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
