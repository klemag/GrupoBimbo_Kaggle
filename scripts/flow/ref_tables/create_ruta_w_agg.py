import pandas as pd
from reftables_functions import load_weeks, get_new_features
import sys
sys.path.append('../../flow')
import parameters as p

## No stand alone files were given for Ruta_SAK
# Need to create a dataframe from train and test data

# Define the features to aggregate
col_feat = p.AggFeatures_init.copy()

# Load data
table = load_weeks([p.Id['ruta']], col_feat, p.DtypeMain)

# Create a new DataFrame that will contain all the Ruta_SAK
ruta = pd.DataFrame()
# Get Ruta_SAK from test as well
test = pd.read_csv(p.DIR_ROOT + '/inputs/test.csv', usecols=['Ruta_SAK'])
ruta['Ruta_SAK'] = pd.concat([table, test], axis=0).Ruta_SAK.unique()
del test

print()
print(table.info())
print(ruta.info())

# Get all grouped by tables for each column defined in ToGroup
dict_gp = {}
for type_gp in p.ToGroup['ruta']:
    dict_gp[type_gp] = table.loc[:, [type_gp] + col_feat].groupby(by=type_gp, as_index=True)
del table

# Build new features to add to ruta using the grouped by tables
new_features = get_new_features(dict_gp, p.ToGroup['ruta'], ruta, col_feat)

# Add new features to ruta
ruta = pd.concat([ruta, new_features], axis=1)

# Write down to CSV
ruta.to_csv(p.DIR_ROOT + '/pipeline/modified_ref_tables/ruta.csv', index=False)