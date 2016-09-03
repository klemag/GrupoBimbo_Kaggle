import pandas as pd
import numpy as np
import sys
from reftables_functions import load_weeks, get_new_features
sys.path.append('../../flow')
import parameters as p

# Load Agencia table
agencia = pd.read_csv(p.DIR_ROOT+'/inputs/town_state.csv',dtype={p.Id['agencia']: np.uint32})

# Define the features to aggregate
col_feat = p.AggFeatures_init.copy()

# Load data
table=load_weeks([p.Id['agencia']], col_feat, p.DtypeMain)
print()
print(table.info())
print(agencia.info())

# Get all grouped by tables for each column defined in ToGroup
dict_gp={}
for type_gp in p.ToGroup['agencia']:
    dict_gp[type_gp] = pd.merge(table, agencia, on=p.Id['agencia'], how='left').loc[
        :, [type_gp] + col_feat].groupby(by=type_gp, as_index=True)
del table

# Build new features to add to agencia using the grouped by tables
new_features=get_new_features(dict_gp, p.ToGroup['agencia'], agencia, col_feat)

# Add new features to agencia
agencia = pd.concat([agencia, new_features], axis=1)

# Write down to CSV
agencia.to_csv(p.DIR_ROOT+'/pipeline/modified_ref_tables/agencia.csv', index=False)