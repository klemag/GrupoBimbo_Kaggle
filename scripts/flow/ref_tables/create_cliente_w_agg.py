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

# Define the features to aggregate
col_feat = p.AggFeatures_init.copy()

# Load data
table=load_weeks([p.Id['cliente']], col_feat, p.DtypeMain)
print()
print(table.info())
print(cliente.info())

# Get all grouped by tables for each column defined in ToGroup
dict_gp={}
for type_gp in p.ToGroup['cliente']:
    dict_gp[type_gp] = pd.merge(table, cliente, on=p.Id['cliente'], how='left').loc[
        :, [type_gp] + col_feat].groupby(by=type_gp, as_index=True)
del table

# Build new features to add to cliente using the grouped by tables
new_features=get_new_features(dict_gp, p.ToGroup['cliente'], cliente, col_feat)

# Add new features to cliente
cliente = pd.concat([cliente, new_features], axis=1)

# Write down to CSV
cliente.to_csv(p.DIR_ROOT+'/pipeline/modified_ref_tables/cliente.csv', index=False)