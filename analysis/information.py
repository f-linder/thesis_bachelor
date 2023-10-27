from probabilities import *
from scipy.integrate import nquad
import pandas as pd
import numpy as np


def di_tickers(source, target, others=None, estimator=KNN()):
    # prepare data of tickers
    df_source = pd.read_csv(f'./data/{source}.csv')
    df_target = pd.read_csv(f'./data/{target}.csv')
    df_others = [pd.read_csv(f'./data/{o}.csv') for o in others]

    returns_source = np.array(df_source['Returns'])
    returns_target = np.array(df_target['Returns'])
    # returns_others = np.array([df['Returns'] for df in df_others])

    # TODO: select subset of others with highest correlation
    # corr_matrix = [np.corrcoef(returns_target, o) for o in returns_others]
    cond = None

    return di(returns_source.transpose(), returns_target.transpose(), cond, estimator)


def di(source, target, cond=None, estimator=KNN()):
    T = len(source)
    di = 0.0

    for t in range(1, T):
        source_past = source[:t]
        target_present = target[t]
        cond_past = cond[:t]

        di += mi(source_past, target_present, cond_past)

    return di / T

def mi(source, target, cond=None, estimator=KNN()):
    n_cond = 0 if cond is None else len(conditions[0])
    # TODO: function  
    # I(X;Y|Z) = int_z int_y int_x Pr(x,y|z) * log(Pr(x,y|z) / (Pr(x|z) * Pr(y|z))) dxdydz
    # mi, _ = nquad(func, [(-np.inf, np.inf) for _ in range(2 + n_cond)])

    return 0
