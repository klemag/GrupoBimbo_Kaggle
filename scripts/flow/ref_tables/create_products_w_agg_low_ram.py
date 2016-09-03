#### This script is a low memory version of the create_products_w_agg script
## It will create the products.csv file with only the mean features as float
## It deals with memory limitation by loading features one by one instead of
## all at once. If memory isn't an issue, use the other script

import pandas as pd
import numpy as np
from reftables_functions import get_brand, get_prod, get_weight, build_dict_products
from reftables_functions import load_weeks, get_new_features
import sys
sys.path.append('../../flow')
import parameters as p

products = pd.read_csv(p.DIR_ROOT + '/inputs/producto_tabla.csv')

# Group all unknown products together
products.loc[products.NombreProducto ==
             'NO IDENTIFICADO 0', 'NombreProducto'] = 'Unknown'

# Get Brand
products['Brand'] = products.NombreProducto
products.loc[~products.Brand.isnull(), 'Brand'] = products.loc[~products.NombreProducto.isnull(), 'NombreProducto'
                                                               ].apply(get_brand)
products.loc[products.Brand.isnull(), 'Brand'] = 'Unknown'

# Get Product Name
dict_prods = build_dict_products(products.NombreProducto.unique())
products['Product'] = products.NombreProducto
products.loc[~products.Product.isnull(), 'Product'] = products.loc[~products.NombreProducto.isnull(), 'NombreProducto'
                                                                   ].apply(get_prod, dict_prods=dict_prods)
products.loc[products.Product.isnull(), 'Product'] = 'Unknown'

# Get Weights
products['Weight'] = products.NombreProducto
products.loc[~products.Weight.isnull(), 'Weight'] = products.loc[~products.NombreProducto.isnull(), 'NombreProducto'
                                                                ].apply(get_weight)
products.loc[products.Weight.isnull(), 'Weight'] = products.Weight.mean()
products['Weight'] = products.Weight.astype(np.float16)

# Drop NombreProducto column
products = products.drop('NombreProducto', axis=1)

# Start building dicts
# Load training data to get stats for each product

print(products.info())
col_feat = p.AggFeatures_init.copy()

# Run a load_weeks to update col_feat and have all the generated features
tab = load_weeks([p.Id['product']], col_feat, p.DtypeMain)
del tab

# Use another method here in order
# to be able to fit everything in the memory
# we re-load the table for every column we need to group ...
for type_gp in p.ToGroup['product']:

    for col in col_feat:

        col_feat_new = p.AggFeatures_init.copy()

        # Only keep the ID and the column we will aggregate
        table = load_weeks([p.Id['product']], col_feat_new, p.DtypeMain).loc[:,[p.Id['product']] + [col]]
        print(table.info())

        print('Get the groupby table')
        dict_gp = pd.merge(table, products, on=p.Id['product'], how='left').loc[
            :, [type_gp, col]].groupby(by=type_gp, as_index=True).mean().loc[:, [col]].to_dict()[col]
        del table

        # Build the new features (can't reuse get_new_features here...)
        print('Build new features')
        new_feat = pd.DataFrame()
        key = col + '_mean'
        name = type_gp + '_' + key

        # Map with groupby dict to create a new feature
        new_feat[name] = products[type_gp].map(dict_gp)
        del dict_gp
        m = new_feat.loc[~new_feat[name].isnull(), name].mean()
        new_feat.loc[new_feat[name].isnull(), name] = m

        # Add new feature to products
        products = pd.concat([products, new_feat], axis=1)
        del new_feat

# Write down to CSV
products.to_csv(
    p.DIR_ROOT + '/pipeline/modified_ref_tables/product.csv', index=False)