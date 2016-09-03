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

## Create new features

# Define the features to aggregate
col_feat = p.AggFeatures_init.copy()

# Load data
table=load_weeks([p.Id['product']], col_feat, p.DtypeMain)
print()
print(table.info())
print(products.info())

# Get all grouped by tables for each column defined in ToGroup
dict_gp={}
for type_gp in p.ToGroup['product']:
    dict_gp[type_gp] = pd.merge(table, products, on=p.Id['product'], how='left').loc[
        :, [type_gp] + col_feat].groupby(by=type_gp, as_index=True)
del table

# Build new features to add to products using the grouped by tables
new_features=get_new_features(dict_gp, p.ToGroup['product'], products, col_feat)

# Add new features to product
products = pd.concat([products, new_features], axis=1)

# Add dummies for Brand
dummies_brand = pd.get_dummies(products['Brand'], prefix='Brand')
products = pd.concat([products, dummies_brand], axis=1)

# Write down to CSV
products.to_csv(p.DIR_ROOT+'/pipeline/modified_ref_tables/product.csv', index=False)