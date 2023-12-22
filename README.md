# Interdependency Analysis of Stock Market Data - A Time Series Approach
Abstract

## Functionality
* Estimation of causally conditioned directed information: I(X -> Y || Z)
* Estimation of time-varying directed information using rolling window
* Subset selection policicies to reduce dimensionality of data
* Simulation of randomly generated, stationary, linear and non-linear multivariate autoregressive models
* Visualization of directed information graphs
* Download of historical stock market data

Check out the Jupyter Notebooks `demo_simulation.ipynb` and `demo_real.ipynb` for a demonstration of the functionality.

## Requirements
* Python (tested on 3.10)
* [Graphviz](https://graphviz.org/)

### Packages
* numpy
* pandas
* sklearn
* matplotlib
* graphviz
* yfinance

## Data
The estimators are designed for continuous time series data (such as stock returns).

You can use your own datasets or acquire percentage stock returns using the functionality provided:

```python 
from analysis.utils import download_tickers, download_index

# format: YYYY-DD-MM
start_date = '2023-15-10' 
end_date = '2024-15-02'
interval = '1d'

# download stock returns of your interest, e.g. apple, google, blackrock
tickers = ['AAPL', 'GOOGL', 'BLK']
returns = download_tickers(tickers, start_date, end_date, interval)

# download every constituent to index: S&P100 or S&P500
index = 'S&P100'
# index = 'S&P500'
returns = download_index(index, start_date, end_date, interval)
```

## Paper
Author: Felix Linder\
Supervisor: Prof. Dr. Jalal Etesami\
Submission Date: 15.02.2024
