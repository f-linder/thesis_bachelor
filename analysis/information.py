import analysis.probabilities as prob
import pandas as pd
import numpy as np


def directed_information_tickers(ticker_source, ticker_target,
                                 tickers_other=[], estimator=prob.KNN()):
    return 0
    # prepare data of tickers
    df_source = pd.read_csv(f'./data/{ticker_source}.csv')
    df_target = pd.read_csv(f'./data/{ticker_target}.csv')
    df_others = [pd.read_csv(f'./data/{o}.csv') for o in tickers_other]

    returns_source = np.array(df_source['Returns'])
    returns_target = np.array(df_target['Returns'])
    returns_others = np.array([df['Returns'] for df in df_others])

    # TODO: select subset of others with highest correlation
    # corr_matrix = [np.corrcoef(returns_target, o) for o in returns_others]
    cond = None

    return directed_information(returns_source.transpose(),
                                returns_target.transpose(), cond, estimator)


# I(X -> Y || Z) = 1/T sum_{t=1}^T I(X^t; Y^t | Z^t, X^t)
# average mutual information over time horizon
# TODO: ???
def directed_information(source, target, cond=None, estimator=prob.KNN()):
    pass
#     T = len(source)
#     di = 0.0
#
#     for t in range(500, T, 100):
#         x = source[:t]
#         y = target[:t]
#         z = np.hstack ((cond[:t], y)) if cond is not None else None
#
#         mi = mutual_information(x, y, z)
#         print(f'mi={mi}')
#         di += mi
#
    # return di / ((T - 500) / 100)


# based on Sricharan et al. (2011)
# I(X;Y|Z) = H(X|Z) - H(X|Y,Z)
#          = H(X,Z) + H(Y,Z) − H(X,Y,Z) − H(Z)
#          = 1/N sum_{x,y,z} (log(Pr(x,y,z)) + log(Pr(z)) - log(Pr(x,z)) - log(Pr(y,z)) <- used
def mutual_information(source, target, cond=None, estimator=prob.KNN()):
    N = len(source)
    cond_returns = cond if cond is not None else [[] for _ in range(N)]

    xyz = np.hstack((source, target, cond_returns))
    xz = np.hstack((source, cond_returns))
    yz = np.hstack((target, cond_returns))

    pdf_xyz = prob.pdf(estimator, xyz)
    pdf_z = prob.pdf(estimator, cond) if cond is not None else lambda *x: 1
    pdf_xz = prob.pdf(estimator, xz)
    pdf_yz = prob.pdf(estimator, yz)

    mi = 0.0
    for x, y, *z in xyz:
        mi += np.log2(pdf_xyz(x, y, *z)) + np.log2(pdf_z(*z)) - np.log2(pdf_xz(x, *z)) - np.log2(pdf_yz(y, *z))

    return mi / N
