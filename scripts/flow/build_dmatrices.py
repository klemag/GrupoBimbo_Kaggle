import pandas as pd
import numpy as np
from training_functions import *
import parameters as p

# Find length of the two weeks combined
w8 = pd.read_csv(p.DIR_ROOT + '/pipeline/by_week/semana_8.csv', usecols=['Canal_ID'], dtype={'Canal_ID':np.int8})
w9 = pd.read_csv(p.DIR_ROOT + '/pipeline/by_week/semana_9.csv', usecols=['Canal_ID'], dtype={'Canal_ID':np.int8})
df = pd.concat([w8,w9], axis=0)
length_2w = pd.Series(range(len(df)))
del df, w8, w9

# Get random indexes
msk = np.random.rand(len(length_2w)) < p.SAMPLE_SIZE

# Generate two Series containing indexes to pass to get_DMatrix
ltrain = length_2w.iloc[msk]
ltest = length_2w.iloc[~msk]

print('Train dataset size: {}\nTest dataset size: {}'.format(len(ltrain), len(ltest)))

# Get DMatrix and save it
# If PCA is on it will also generate and save a PCA model
get_DMatrix([8, 9], ltrain, ltest)