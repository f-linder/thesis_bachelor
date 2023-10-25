import os
import yfinance as yf
import pandas as pd
from enum import Enum


class Index(Enum):
    SP100 = 1
    SP500 = 2


def download_index(index, start_date, end_date, interval='1d'):
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
    dir_path = './data'

    for filename in os.listdir(dir_path):
        if filename == 'sp100.csv' or filename == 'sp500.csv':
            continue
        os.remove(os.path.join(dir_path, filename))

    print('Clean up complete: removed all ticker csv files')
