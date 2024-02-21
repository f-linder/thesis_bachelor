# Interdependency Analysis of Stock Market Data - A Time Series Approach
Understanding the inter-connectivity of financial institutions is crucial for decision making and risk management. It enables individuals to construct a well-diversified portfolio, while also helping central banks and governance to establish policies aiming at the stabilization of the financial market, by serving as indicator for systemic risk. In general, there are two categories for measuring interdependence between financial institutions. Network-based approaches construct mathematical models using information about cross-holding, liabilities and contractual agreements. Market-based approaches apply statistical analysis to financial series. Belonging to the latter, this work introduces an information-theoretic concept, directed information, for measuring interdependence between financial institutions. In contrast to approaches in literature, it is able of capturing non-linear, causal links without assuming a certain model. The estimation of directed information is solely reliant on the accuracy of the underlying probability distribution. We evaluated kernel density estimation and k-nearest neighbor as estimation techniques for different sample and network sizes. Furthermore, we analyzed the daily returns of the S\&P100 from 2016 to 2023 using directed information. The results are consistent with other research, showing a significant increase of inter-connectivity in the years of the COVID-19 pandemic.

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

You can use your own datasets or download percentage stock returns using the functionality provided:

```python 
from analysis.utils import download_tickers, download_index

# format YYYY-MM-DD
start_date = '2023-10-15' 
end_date = '2024-02-15'
interval = '1d'

# companies of your interest, e.g. apple, google, blackrock
tickers = ['AAPL', 'GOOGL', 'BLK']
returns = download_tickers(tickers, start_date, end_date, interval)

# download every constituent of index: S&P100 or S&P500
index = 'S&P100'
returns = download_index(index, start_date, end_date, interval)
```

Once downloaded, the data is stored in the `data/<ticker>.csv`:

```python 
import pandas as pd 

tickers = ['AAPL', 'GOOGL', 'BLK']
returns = {ticker: pd.read_csv(f'data/{ticker}.csv')['Returns'] for ticker in tickers}
```

## Paper
Author: Felix Linder\
Supervisor: Prof. Dr. Jalal Etesami\
Submission Date: 15.02.2024
