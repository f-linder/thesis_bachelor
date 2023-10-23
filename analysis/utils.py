from enum import Enum
from datetime import datetime
import os
import pandas as pd
import yfinance as yf


class Index(Enum):
    SP100 = 1
    SP500 = 2


def download_all(index, interval='1d', start_date=datetime(2015, 10, 15),
                 end_date=datetime(2023, 10, 15)):
    input_file = ''
    if index == Index.SP100:
        print('Starting download for all S&P100 constituents...')
        input_file = './data/sp100.csv'
    elif index == Index.SP500:
        print('Starting download for all S&P500 constituents...')
        input_file = './data/sp500.csv'
    else:
        raise Exception('index not found')

    df = pd.read_csv(input_file)
    tickers = df['Symbol'].tolist()

    for i, ticker in enumerate(tickers):
        print(f'Downloading {ticker}: {i + 1}/{len(tickers)}')
        download(ticker, interval, start_date, end_date)



def download(ticker, interval='1d', start_date=datetime(2015, 10, 15),
             end_date=datetime(2023, 10, 15)):
    # [Date, Open, High, Low, Close, Adj Close, Volume]
    data = yf.download(ticker, interval=interval, start=start_date,
                       end=end_date, progress=False)

    data['Return Abs'] = data['Close'] - data['Open']
    data['Return Rel'] = data['Return Abs'] / data['Open'] * 100

    data.drop(['High', 'Low', 'Volume', 'Adj Close'], axis=1, inplace=True)

    if not data.empty:
        data.to_csv(f'./data/{ticker}.csv')


def clean_up():
    dir_path = './data'

    for filename in os.listdir(dir_path):
        if filename == 'sp100.csv' or filename == 'sp500.csv':
            continue
        os.remove(os.path.join(dir_path, filename))

    print('Clean up complete: removed all ticker csv files')
