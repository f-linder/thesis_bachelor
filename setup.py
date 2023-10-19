# downloads the data of all contintuents of the S&P500
# 'pip install yfinance' required

from datetime import datetime
import pandas as pd
import yfinance as yf
import os

start_date = datetime(2015, 10, 15)
end_date = datetime(2023, 10, 15)

input_file = 'data/sp100.csv'
# input_file = 'data/sp500.csv'

df = pd.read_csv(input_file)
tickers = df['Symbol'].tolist()

if not os.path.exists('data'):
    os.makedirs('data')

for ticker in tickers:
    # returns pandas dataframe
    # [Date, Open, High, Low, Close, Adj Close, Volume]
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    data.drop(['High', 'Low', 'Volume', 'Adj Close'], axis=1, inplace=True)

    data['Daily Return Abs'] = data['Close'] - data['Open']
    data['Daily Return Percent'] = data['Daily Return Abs'] / data['Open'] * 100

    if not data.empty:
        data.to_csv(f'data/{ticker}.csv')
        print(f"Downloaded data for {ticker}")


print('Download complete.')
