import pandas as pd
import parameters as p
import numpy as np

## Generate Lag features
## Getting the average demand for a given product/cliente during the past weeks
def get_lag_features(data, prev_week, lag, target):

    # Apply a suffix to the feature name in case we use venta_hoy and not Demanda_uni_equil
    if target == 'Venta_hoy':
        suffix = '_venta'
    else:
        suffix = ''

    # Get average demand from last week and rename the target column
    mean_demanda = prev_week[target].mean()
    prev_week.columns = prev_week.columns.str.replace(target, 'Demanda_prev')
    # prev_week['Demanda_prev'] = prev_week[target]
    # prev_week = prev_week.drop(target, axis=1)

    # Generate a few groupby tables for easy lookup later
    mean_by_cliente = prev_week.loc[:, ['Cliente_ID', 'Demanda_prev']].groupby(
        by=['Cliente_ID'], as_index=False).mean()
    mean_by_prod_ruta = prev_week.loc[:, ['Producto_ID', 'Ruta_SAK', 'Demanda_prev']].groupby(by=['Producto_ID', 'Ruta_SAK'],
                                                                                              as_index=False).mean()
    mean_by_prod_agencia = prev_week.loc[:, ['Producto_ID', 'Agencia_ID', 'Demanda_prev']].groupby(
        by=['Producto_ID', 'Agencia_ID'], as_index=False).mean()

    # Now we can remove features no more needed from the previous week data
    prev_week = prev_week.drop(['Ruta_SAK', 'Agencia_ID', 'Canal_ID'], axis=1)

    # Load reference tables ...
    products = pd.read_csv('{}/pipeline/modified_ref_tables/product.csv'.format(p.DIR_ROOT),
                           usecols=['Producto_ID', 'Brand', 'Product'])
    clients = pd.read_csv('{}/pipeline/modified_ref_tables/cliente.csv'.format(p.DIR_ROOT),
                          usecols=['Cliente_ID', 'NombreCliente', 'ClienteGroup'])

    # ... and merge them with the prev_week & data
    prev_week = pd.merge(prev_week, clients, on='Cliente_ID').drop('Cliente_ID', axis=1)
    prev_week = pd.merge(prev_week, products, on='Producto_ID')
    data = pd.merge(data, clients, on='Cliente_ID')
    data = pd.merge(data, products, on='Producto_ID')
    del clients, products

    # First attempt at finding lag features, need same product_ID + cliente:
    ## On 'Producto_ID', 'NombreCliente', 'ClienteGroup', 'Brand', 'Product'

    prev_week = prev_week.groupby(by=['Producto_ID', 'NombreCliente', 'ClienteGroup', 'Brand', 'Product'],
                                  as_index=False).mean()
    data['Lag_{}w{}'.format(lag, suffix)] = pd.merge(data, prev_week,
                                                        on=['Producto_ID', 'NombreCliente',
                                                            'ClienteGroup', 'Brand', 'Product'],
                                                        how='left')['Demanda_prev'].values
    still_to_fill = len(data[data['Lag_{}w{}'.format(lag, suffix)].isnull()])
    print('Still need {} to fill'.format(still_to_fill))

    # Second attempt, only need the same product + cliente:
    # On 'NombreCliente', 'ClienteGroup', 'Brand', 'Product'
    if still_to_fill > 0:

        prev_week = prev_week.groupby(by=['NombreCliente', 'ClienteGroup', 'Brand', 'Product'],
                                      as_index=False).mean()
        data.loc[data['Lag_{}w{}'.format(lag, suffix)].isnull(), 'Lag_{}w{}'.format(lag, suffix)] = pd.merge(
            data.loc[data['Lag_{}w{}'.format(
                lag, suffix)].isnull()], prev_week,
            on=['NombreCliente', 'ClienteGroup', 'Brand', 'Product'],
            how='left')['Demanda_prev'].values
        still_to_fill = len(data.loc[data['Lag_{}w{}'.format(lag, suffix)].isnull()])
        print('Still need {} to fill'.format(still_to_fill))

    # Third attempt:
    # Lookup Prod/Ruta_SAK
    if still_to_fill > 0:
        print('Try to find Prod/Ruta_SAK...')
        data.loc[data['Lag_{}w{}'.format(lag, suffix)].isnull(), 'Lag_{}w{}'.format(lag, suffix)] = pd.merge(
            data.loc[data['Lag_{}w{}'.format(
                lag, suffix)].isnull()], mean_by_prod_ruta,
            on=['Producto_ID', 'Ruta_SAK'],
            how='left')['Demanda_prev'].values
        still_to_fill = len(data[data['Lag_{}w{}'.format(lag, suffix)].isnull()])
        print('Still need {} to fill'.format(still_to_fill))

    # Fourth attempt:
    # Prod/Agencia_ID
    if still_to_fill > 0:
        print('Try to find Prod/Agencia_ID..')
        data.loc[data['Lag_{}w{}'.format(lag, suffix)].isnull(), 'Lag_{}w{}'.format(lag, suffix)] = pd.merge(
            data.loc[data['Lag_{}w{}'.format(
                lag, suffix)].isnull()], mean_by_prod_agencia,
            on=['Producto_ID', 'Agencia_ID'],
            how='left')['Demanda_prev'].values
        still_to_fill = len(data[data['Lag_{}w{}'.format(lag, suffix)].isnull()])
        print('Still need {} to fill'.format(still_to_fill))

    # Fifth attempt:
    # ClienteGroup and product
    if still_to_fill > 0:
        # On 'ClienteGroup', 'Brand', 'Product'
        prev_week = prev_week.groupby(
            by=['ClienteGroup', 'Brand', 'Product'], as_index=False).mean()
        data.loc[data['Lag_{}w{}'.format(lag, suffix)].isnull(), 'Lag_{}w{}'.format(lag, suffix)] = pd.merge(
            data.loc[data['Lag_{}w{}'.format(
                lag, suffix)].isnull()], prev_week,
            on=['ClienteGroup', 'Brand', 'Product'],
            how='left')['Demanda_prev'].values
        still_to_fill = len(data[data['Lag_{}w{}'.format(lag, suffix)].isnull()])
        print('Still need {} to fill'.format(still_to_fill))

    # Sixth attempt:
    # ClienteGroup and Brand
    if still_to_fill > 0:
        # On ClienteGroup Brand
        prev_week = prev_week.groupby(
            by=['ClienteGroup', 'Brand'], as_index=False).mean()
        data.loc[data['Lag_{}w{}'.format(lag, suffix)].isnull(), 'Lag_{}w{}'.format(lag, suffix)] = pd.merge(
            data.loc[data['Lag_{}w{}'.format(
                lag, suffix)].isnull()], prev_week,
            on=['ClienteGroup', 'Brand'],
            how='left')['Demanda_prev'].values
        still_to_fill = len(data[data['Lag_{}w{}'.format(lag, suffix)].isnull()])
        print('Still need {} to fill'.format(still_to_fill))

    # Seventh attempt:
    # Only same brand...
    if still_to_fill > 0:
        # On Brand
        prev_week = prev_week.groupby(by=['Brand'], as_index=False).mean()
        data.loc[data['Lag_{}w{}'.format(lag, suffix)].isnull(), 'Lag_{}w{}'.format(lag, suffix)] = pd.merge(
            data.loc[data['Lag_{}w{}'.format(
                lag, suffix)].isnull()], prev_week,
            on=['Brand'],
            how='left')['Demanda_prev'].values
        still_to_fill = len(data[data['Lag_{}w{}'.format(lag, suffix)].isnull()])
        print('Still need {} to fill'.format(still_to_fill))

    # Eights attempt:
    # Only same Cliente
    if still_to_fill > 0:
        print('Try to find Cliente_ID...')
        data.loc[data['Lag_{}w{}'.format(lag, suffix)].isnull(), 'Lag_{}w{}'.format(lag, suffix)] = pd.merge(
            data.loc[data['Lag_{}w{}'.format(
                lag, suffix)].isnull()], mean_by_cliente,
            on=['Cliente_ID'],
            how='left')['Demanda_prev'].values
        still_to_fill = len(data[data['Lag_{}w{}'.format(lag, suffix)].isnull()])
        print('Still need {} to fill'.format(still_to_fill))

    # Giving up and taking the mean
    if still_to_fill > 0:
        print('Give up and take the mean... If that happens too often try more steps before...')
        # Finally ...
        data.loc[data['Lag_{}w{}'.format(lag, suffix)].isnull(),
                 'Lag_{}w{}'.format(lag, suffix)] = mean_demanda
        still_to_fill = len(data[data['Lag_{}w{}'.format(lag, suffix)].isnull()])
        print('Still need {} to fill'.format(still_to_fill))

    return data.drop(['NombreCliente', 'ClienteGroup', 'Brand', 'Product'], axis=1)

## Build the weekly data with all lag features
if __name__ == "__main__":

    # Load train data
    table = pd.read_csv('{}/inputs/train.csv'.format(p.DIR_ROOT), dtype=p.DtypeMain)

    # Start by processing weeks 3 to 10 (aka train data)
    for x in range(3, 10):

        print('Building week {}'.format(x))
        week = table.loc[table.Semana == x].copy()

        # We need lag features only for week
        # 8 and 9 as the others will not be used
        # for training
        if x > 7:

            # Get up to five weeks of lag
            for lag in [1, 2, 3, 4, 5]:

                print('Getting Lag_{}w feature - Demanda_uni_equil'.format(lag))

                # Load previous week with Demanda_uni_equil
                prev_week = pd.read_csv('{}/pipeline/by_week/semana_{}.csv'.format(p.DIR_ROOT, x - lag),
                                        usecols=['Cliente_ID', 'Producto_ID', 'Ruta_SAK', 'Agencia_ID',
                                                 'Canal_ID', 'Demanda_uni_equil'], dtype=p.DtypeMain)

                # Get lag features for Demanda_uni_equil
                week = get_lag_features(week, prev_week, lag, 'Demanda_uni_equil')

                # get lag 2-5weeks for Venta_hoy
                if lag > 1:
                    print('Getting Lag_{}w_venta feature - Venta_hoy'.format(lag))

                    # Load previous week with venta_hoy
                    prev_week = pd.read_csv('{}/pipeline/by_week/semana_{}.csv'.format(p.DIR_ROOT, x - lag),
                                            usecols=['Cliente_ID', 'Producto_ID', 'Ruta_SAK', 'Agencia_ID',
                                                     'Canal_ID', 'Venta_hoy'],dtype=p.DtypeMain)
                    week = get_lag_features(week, prev_week, lag, 'Venta_hoy')

        # Write down the week data
        print('Writing down week {}'.format(x))
        week.to_csv('{}/pipeline/by_week/semana_{}.csv'.format(p.DIR_ROOT, x), index=False)

    del table

    # Then build features for test tables
    # We will not be able to get the 1week lag for week 11 as we need to
    # predict Demanda_uni_equil for week 10 first ...
    print('Getting test tables')
    test = pd.read_csv('{}/inputs/test.csv'.format(p.DIR_ROOT), dtype=p.DtypeMain)

    for x in range(10, 12):

        print('\nBuilding week {}'.format(x))
        week = test.loc[test.Semana == x].copy()

        # for week 10 we can do 1 week lag, not for 11
        if x == 10:
            lag_list = [1, 2, 3, 4, 5]
        elif x == 11:
            lag_list = [2, 3, 4, 5]

        for lag in lag_list:
            print('Getting Lag_{}w feature - Demanda_uni_equil'.format(lag))

            # Add lag previsions from previous week
            prev_week = pd.read_csv('{}/pipeline/by_week/semana_{}.csv'.format(p.DIR_ROOT, x - lag),
                                    usecols=['Cliente_ID', 'Producto_ID', 'Ruta_SAK', 'Agencia_ID',
                                             'Canal_ID', 'Demanda_uni_equil'], dtype=p.DtypeMain)
            week = get_lag_features(week, prev_week, lag, 'Demanda_uni_equil')

            if lag > 1:

                print('Getting Lag_{}w_venta feature - Venta_hoy'.format(lag))
                # Add lag previsions from previous week
                prev_week = pd.read_csv('{}/pipeline/by_week/semana_{}.csv'.format(p.DIR_ROOT, x - lag),
                                        usecols=['Cliente_ID', 'Producto_ID', 'Ruta_SAK', 'Agencia_ID',
                                                 'Canal_ID', 'Venta_hoy'], dtype=p.DtypeMain)

                # Get lag features
                week = get_lag_features(week, prev_week, lag, 'Venta_hoy')

        # Writing down test week
        print('Writing down week {}'.format(x))
        week.to_csv('{}/pipeline/by_week/semana_{}.csv'.format(p.DIR_ROOT, x), index=False)
    del test