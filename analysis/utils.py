import os
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
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


def plot_directed_graph(adjacency_matrix, labels, threshold=0.05):
    """
    Create and display a directed graph to visualize causal relationships based on DI values.

    Parameters:
    - adjacency_matrix (numpy.ndarray): Adjacency matrix containing weights of edges between all variables.
    - labels (list): A list of labels for each variable.
    - threshold (float): The threshold for weights of edges values to appear in graph (default is 0.05).

    Returns:
    - None
    """
    n_vars = len(adjacency_matrix)
    dig = nx.DiGraph()

    for i in range(n_vars):
        for j, di in enumerate(adjacency_matrix[i]):
            if di >= threshold:
                dig.add_edge(labels[i], labels[j], weight=di)

    pos = nx.planar_layout(dig)
    edge_labels = nx.get_edge_attributes(dig, 'weight')
    edge_labels = {e: f'{w:.3f}' for e, w in edge_labels.items()}  # limit decimals

    nx.draw(dig, pos, arrows=True, with_labels=True, node_color="lightblue")
    nx.draw_networkx_edge_labels(dig, pos, edge_labels=edge_labels)


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
