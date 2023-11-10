import os
import yfinance as yf
import pandas as pd
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
