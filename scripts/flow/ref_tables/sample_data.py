import pandas as pd
import numpy as np
import sys
sys.path.append('../../flow')
import parameters as p

## Use only a fraction of the data to speed up tests/debug

## Sample size, fraction of the data to use
SAMPLE_SIZE=0.01

## First move the file train/test.csv to train/test_all.csv ..., train/test.csv will contain the sample
for table in ['train', 'test']:
    print('Get sample of size {}, for table: {}'.format(SAMPLE_SIZE,table))
    df = pd.read_csv(p.DIR_ROOT+'/inputs/{}_all.csv'.format(table), dtype=p.DtypeMain)
    l = len(df)
    msk = np.random.rand(l) < SAMPLE_SIZE
    df = df.iloc[msk]
    df.to_csv(p.DIR_ROOT+'/inputs/{}.csv'.format(table), index=False)
    del df