from probabilities import Estimator
import pandas as pd
import numpy as np

def di(source, target, others=None, estimator=Estimator.KDE):
    df_source = pd.read_csv(f'./data/{source}.csv')
    df_target = pd.read_csv(f'./data/{target}.csv')
    df_others = [pd.read_csv(f'./data/{o}.csv') for o in others]

    returns_source = np.array(df_source['Returns'])
    returns_target = np.array(df_target['Returns'])
    returns_others = np.array([df['Returns'] for df in df_others])

    # TODO: select subset of others with highest correlation
    # corr_matrix = [np.corrcoef(returns_target, o) for o in returns_others]

    return di(returns_source.transpose(), returns_target.transpose(),
              returns_others.transpose(), estimator)

def directed_information(source, target, conditions, estimator):
    T = len(source)
    di = 0.0

    for t in range(1, T):
        source_past = source[:t]
        target_present = target[t]
        conditions_past = conditions[:t]

        di += mutual_information(source_past, target_present, conditions_past)

    return di / T

def mutual_information(source, target, conditions, estimator):
    return 0
