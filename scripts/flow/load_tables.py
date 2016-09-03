import pandas as pd
import numpy as np
import parameters as p
from math import log1p

## Function to load one or several weeks
# Uses indexes as argument
def load_train_week(weeks, indexes):

    data = pd.DataFrame()
    for week in weeks:
        data = pd.concat([data, pd.read_csv('{}/pipeline/by_week/semana_{}.csv'.format(p.DIR_ROOT, week),
                                            dtype=p.DtypeMain)], axis=0)
        print('Length of table after week {}: {}'.format(week, len(data)))

    # Keep only given indexes
    data = data.iloc[indexes]
    print('Length of table after sampling: {}'.format(len(data)))
    return data

## Function to load one week at the time
# Uses a fraction of the data as argument
def load_week(week, percent):

    data = pd.read_csv('{}/pipeline/by_week/semana_{}.csv'.format(p.DIR_ROOT, week),
                       dtype=p.DtypeMain)

    # if percent == 1, return the full week
    if percent != 1:
        data = data.sample(frac=percent, replace=False)
    print('Length of table: {}'.format(len(data)))
    return data

## Build features and return target column and features matrix
def get_features_target(table, features_only=False):

    use = {}
    dtype = {}

    # Start with features, that are independent from ref_tables: Lag
    # We do not have a Lag_1w_venta as we cannot get this data
    # for w11 from w10
    features = ['Lag_1w', 'Lag_2w', 'Lag_3w', 'Lag_4w', 'Lag_5w',
                'Lag_2w_venta', 'Lag_3w_venta', 'Lag_4w_venta', 'Lag_5w_venta']

    # Create a few new features based on Lag...
    print('\nCreate new Lag features first')

    # Variation over 1,2,3 and 4 weeks
    for var in range(1,5):
        print('Get variation over {} week(s)'.format(var))
        for x in range(1, 6-var):
            table.loc[:,'Lag_{}w-{}w'.format(x, x + var)] = table.loc[:, 'Lag_{}w'.format(
                x)].values - table.loc[:, 'Lag_{}w'.format(x + var)].values
            features += ['Lag_{}w-{}w'.format(x, x + var)]

    # Variation over 2,3 and weeks venta
    for var in range(1,4):
        print('Get variation over {} week(s) for venta'.format(var))
        for x in range(2, 6 - var):
            table.loc[:,'Lag_{}w-{}w_venta'.format(x, x + var)] = table.loc[:, 'Lag_{}w_venta'.format(x)].values - \
                                                             table.loc[:, 'Lag_{}w_venta'.format(x + var)].values
            features += ['Lag_{}w-{}w_venta'.format(x, x + var)]

    # Price per unit lag
    for x in range(2, 6):
        mask = table['Lag_{}w'.format(x)] == 0
        table.loc[~mask, 'Lag_ppu_{}w'.format(x)] = \
            table.loc[~mask, 'Lag_{}w_venta'.format(
                x)] / table.loc[~mask, 'Lag_{}w'.format(x)]
        table.loc[mask, 'Lag_ppu_{}w'.format(x)] = 0.0
        table.loc[:, 'Lag_ppu_{}w'.format(x)] = table.loc[
                                                   :, 'Lag_ppu_{}w'.format(x)].astype(np.float32)
        features += ['Lag_ppu_{}w'.format(x)]

    # Sum of all lags
    table.loc[:, 'Lag_Total'] = table['Lag_1w'] + table['Lag_2w'] + \
                                   table['Lag_3w'] + table['Lag_4w'] + table['Lag_5w']
    features += ['Lag_Total']

    # Loading features from reference tables
    # One reference table at the time
    print('Getting features: ', end="")
    for x in ['product', 'cliente', 'agencia', 'canal', 'ruta']:

        print('{}'.format(x), end="")

        # Create use list and dtype
        # Containing all features to load
        use[x] = [p.Id[x]] + p.OtherFeat[x] + p.GeneratedFeat[x]
        dtype[x] = dict(list({p.Id[x]: p.DtypeMain[p.Id[x]]}.items()) +
                        list(p.DtypeOtherFeat[x].items()) +
                        list(p.DtypeGeneratedFeat[x].items()))

        # Remove features from FeaturesToKill
        removed = 0
        for feat in p.FeaturesToKill:
            if feat in use[x]:
                use[x].remove(feat)
                removed += 1
        print('(-{}) '.format(removed), end="")

        features += use[x]
        features.remove(p.Id[x])

        # Load table
        ref = pd.read_csv('{}/pipeline/modified_ref_tables/{}.csv'.format(p.DIR_ROOT, x),
                          dtype=dtype[x], usecols=use[x])

        # Merge table
        table = pd.merge(table, ref, on=p.Id[
                         x], how='left').drop(p.Id[x], axis=1)
        del ref

    # Change types to
    # Decrease memory usage
    for col in table.columns.values:
        if table[col].max() < np.finfo(np.float16).max:
            table.loc[:, col] = table[col].astype(np.float16)

    # Return only features matrix or features matrix and target column
    print('\nNumber of features: {}'.format(len(features)))
    if features_only:
        return table.loc[:, features]
    else:
        ## Apply log1p to target to move to log space
        # This will allow usage of rmse eval metric for our model
        # Then we move back to demanda_uni_equil space and get the rmsle
        target = table.Demanda_uni_equil.apply(log1p).astype(np.float32)
        return table.loc[:, features], target