import pandas as pd
import numpy as np
from reftables_functions import load_weeks, get_new_features, build_dict_cliente, get_cliente_group
import sys
sys.path.append('../../flow')
import parameters as p

# Load Cliente table
cliente = pd.read_csv(p.DIR_ROOT + '/inputs/cliente_tabla.csv', dtype={'Cliente_ID': np.uint32})

# Drop Duplicated Cliente_ID
cliente = cliente.drop_duplicates(subset='Cliente_ID', keep='first')

# Group all unknown Cliente
cliente.loc[cliente.NombreCliente == 'SIN NOMBRE', 'NombreCliente'] = 'Unknown'
cliente.loc[cliente.NombreCliente == 'NO IDENTIFICADO', 'NombreCliente'] = 'Unknown'

# Get Cliente Group
dict_cliente = build_dict_cliente(cliente.NombreCliente.unique())
cliente['ClienteGroup'] = cliente.NombreCliente
cliente.loc[~cliente.ClienteGroup.isnull(), 'ClienteGroup'] = cliente.loc[~cliente.NombreCliente.isnull(), 'NombreCliente'
                                                                          ].apply(get_cliente_group, dict_cliente=dict_cliente)
cliente.loc[cliente.ClienteGroup.isnull(), 'ClienteGroup'] = 'Unknown'

# Start building dicts
# Load training data to get stats for each cliente

print(cliente.info())
col_feat = p.AggFeatures_init.copy()

# Run a load_weeks to update col_feat and have all the generated features
tab = load_weeks([p.Id['cliente']], col_feat, p.DtypeMain)
del tab

# Use another method here in order
# to be able to fit everything in the memory
# we re-load the table for every column we need to group ...
for type_gp in p.ToGroup['cliente']:

    for col in col_feat:

        col_feat_new = p.AggFeatures_init.copy()

        # Only keep the ID and the column we will aggregate
        table = load_weeks([p.Id['cliente']], col_feat_new, p.DtypeMain).loc[:,[p.Id['cliente']] + [col]]
        print(table.info())

        print('Get the groupby table')
        dict_gp = pd.merge(table, cliente, on=p.Id['cliente'], how='left').loc[
            :, [type_gp, col]].groupby(by=type_gp, as_index=True).mean().loc[:, [col]].to_dict()[col]
        del table

        # Build the new features (can't reuse get_new_features here...)
        print('Build new features')
        new_feat = pd.DataFrame()
        key = col + '_mean'
        name = type_gp + '_' + key

        # Map with groupby dict to create a new feature
        new_feat[name] = cliente[type_gp].map(dict_gp)
        del dict_gp
        m = new_feat.loc[~new_feat[name].isnull(), name].mean()
        new_feat.loc[new_feat[name].isnull(), name] = m

        # Add new feature to cliente
        cliente = pd.concat([cliente, new_feat], axis=1)
        del new_feat

# Write down to CSV
cliente.to_csv(
    p.DIR_ROOT + '/pipeline/modified_ref_tables/cliente.csv', index=False)