import pandas as pd
from reftables_functions import load_weeks, get_new_features
import sys
sys.path.append('../../flow')
import parameters as p

## No stand alone files were given for Canal_ID
# Need to create a dataframe from train and test data

# Define the features to aggregate
col_feat = p.AggFeatures_init.copy()

# Load data
table=load_weeks([p.Id['canal']], col_feat, p.DtypeMain)

# Create a new DataFrame that will contain all the Canal_ID
canal = pd.DataFrame()
# Get Canal_ID from test as well
test = pd.read_csv(p.DIR_ROOT+'/inputs/test.csv', usecols=['Canal_ID'])
canal['Canal_ID'] = pd.concat([table, test], axis=0).Canal_ID.unique()
del test

print()
print(table.info())
print(canal.info())

# Get all grouped by tables for each column defined in ToGroup
dict_gp={}
for type_gp in p.ToGroup['canal']:
    dict_gp[type_gp] = table.loc[:, [type_gp] + col_feat].groupby(by=type_gp, as_index=True)
del table

# Build new features to add to canal using the grouped by tables
new_features=get_new_features(dict_gp, p.ToGroup['canal'], canal, col_feat)

# Add new features to canal
canal = pd.concat([canal, new_features], axis=1)

# Write down to CSV
canal.to_csv(p.DIR_ROOT+'/pipeline/modified_ref_tables/canal.csv', index=False)