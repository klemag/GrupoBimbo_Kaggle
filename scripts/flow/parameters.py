import numpy as np
import os

#-------------------------------------------------------------------------
# Directory root
#-------------------------------------------------------------------------

DIR_ROOT = os.path.dirname(os.path.realpath(__file__)) + '/../..'

#-------------------------------------------------------------------------
# List of features to generate from aggregation
#-------------------------------------------------------------------------

### Types of aggregation to use
### Choose from: ['mean', 'med', 'max', 'min', 'std']
AggFeatures_types = ['mean']

### Native features to use for aggregation
AggFeatures_init = ['Demanda_uni_equil', 'Venta_uni_hoy', 'Venta_hoy',
                    'Dev_uni_proxima', 'Dev_proxima']

### New features to use for aggregation
AggFeatures_new = ['Venta_ppu', 'Dev_ppu', 'Diff_ppu_prox_hoy',
                   'Diff_uni_dem_hoy', 'Diff_uni_prox_dem', 'Mult_uni_dem_hoy']

AggFeatures_all = AggFeatures_init + AggFeatures_new

#-------------------------------------------------------------------------
# Training variables
#-------------------------------------------------------------------------

# Choose training mode
# Can be 'Fraction' or 'Ensemble'

# In fraction mode, the data is divided in two matrices, test and train
# it saves both matrices to a binary file, which allows faster loading
# It also includes a possibility to use PCA and early_stopping_round to
# stop training when test errors reaches its minimum

# Ensemble mode divides the data in N_SAMPLES matrices. A model is trained on each
# sample and tested against another one.

TRAINING_MODE = 'Fraction'
#TRAINING_MODE = 'Ensemble'

# Fraction mode only variables
# Use PCA, and choose number of features
PCA_ON = False
# PCA_N_FEAT needs to be <= than number of features
PCA_N_FEAT = 40

# Fraction of the data to use for training
# Needs to be < 1 to ensure we have some data to test
SAMPLE_SIZE = 0.9

# Number of rounds before early stopping
# Early stopping occurs when the score has not improved for XGB_STOP_ROUNDS rounds
XGB_STOP_ROUNDS = 10

# Ensemble mode only variables
# N_SAMPLES has to be a power of 2: 2,4,8,16 (we limit ourselves to 16 different samples)
N_SAMPLES = 4

# Global training variables
# Model parameters
XGB_PARAM = {'max_depth': 10,
         'eta': 0.1,
         'min_child_weight': 1,
         'subsample': 0.85,
         'colsample_bytree': 0.7,
         'silent': 0,
         'objective': 'reg:linear',
         'nthread': 8,
         'eval_metric': 'rmse'}

# Number of boost rounds
XGB_N_ROUNDS = 200

# List of previous week10 predictions to inject into future predictions
# Each time a week10 is predicted, it will be averaged with all files
# in that list, ensuring a more stable prediction of week 11, as week 11
# contains features directly taken from week10 (lag_1w)
# Only add the filename, all files should be kept in pipeline/week10_reuse
WEEK10_OLD_PRED = ['ensemble_prediction.csv']

#-------------------------------------------------------------------------
# Other parameters
#-------------------------------------------------------------------------

### Remove outliers
### Values > mean*OutliersThreshold are set to mean*OutliersThreshold
RemoveOutliersAggFeat=True
OutliersThreshold = 10000

### Generated features maximum value after normalization
### All features will be integers in [-MaxValueGenerated, MaxValueGenerated]
NormalizeAggFeat=True
MaxValueGenerated = 10000

### Features to remove from the model
FeaturesToKill = []

#----------------------------------------------------------------------------
# Main table types
#-------------------------------------------------------------------------

DtypeMain = {
    'id': np.uint32,
    'Semana': np.uint8,
    'Agencia_ID': np.uint32,
    'Canal_ID': np.uint8,
    'Ruta_SAK': np.uint32,
    'Cliente_ID': np.uint32,
    'Producto_ID': np.uint32,
    'Venta_uni_hoy': np.int32,
    'Venta_hoy': np.float32,
    'Dev_uni_proxima': np.int32,
    'Dev_proxima': np.float32,
    'Lag_1w': np.float32,
    'Lag_2w': np.float32,
    'Lag_3w': np.float32,
    'Lag_4w': np.float32,
    'Lag_5w': np.float32,
    'Lag_ppu_2w': np.float32,
    'Lag_ppu_3w': np.float32,
    'Lag_ppu_4w': np.float32,
    'Lag_ppu_5w': np.float32,
    'Lag_2w_venta': np.float32,
    'Lag_3w_venta': np.float32,
    'Lag_4w_venta': np.float32,
    'Lag_5w_venta': np.float32,
    'Lag_1w-2w': np.float32,
    'Lag_2w-3w': np.float32,
    'Lag_3w-4w': np.float32,
    'Lag_4w-5w': np.float32,
    'Lag_1w-3w': np.float32,
    'Lag_2w-4w': np.float32,
    'Lag_3w-5w': np.float32,
    'Lag_1w-4w': np.float32,
    'Lag_2w-5w': np.float32,
    'Lag_1w-5w': np.float32,
    'Lag_2w-3w_venta': np.float32,
    'Lag_3w-4w_venta': np.float32,
    'Lag_4w-5w_venta': np.float32,
    'Lag_2w-4w_venta': np.float32,
    'Lag_3w-5w_venta': np.float32,
    'Lag_2w-5w_venta': np.float32,
    'Lag_Total': np.float32,
    'Demanda_uni_equil': np.int32
}

#-------------------------------------------------------------------------
# Defines dict containing the features
#-------------------------------------------------------------------------

Id = {}
ToGroup = {}
GeneratedFeat = {}
OtherFeat = {}
DtypeGeneratedFeat = {}
DtypeOtherFeat = {}

#-------------------------------------------------------------------------
# Product table details
#-------------------------------------------------------------------------

USE_DUMMY_BRANDS=False

if USE_DUMMY_BRANDS:
    dummies_Brand = ['Brand_AM', 'Brand_AV', 'Brand_BAR', 'Brand_BIM', 'Brand_BRE', 'Brand_BRL',
                  'Brand_CAR', 'Brand_CC', 'Brand_CHK', 'Brand_COR', 'Brand_DH',
                  'Brand_DIF', 'Brand_EMB', 'Brand_GBI', 'Brand_GV', 'Brand_JMX',
                  'Brand_KOD', 'Brand_LAR', 'Brand_LC', 'Brand_LON', 'Brand_MCM',
                  'Brand_MLA', 'Brand_MP', 'Brand_MR', 'Brand_MSK', 'Brand_MTB',
                  'Brand_NAI', 'Brand_NEC', 'Brand_NES', 'Brand_ORO', 'Brand_PUL',
                  'Brand_RIC', 'Brand_SAN', 'Brand_SKD', 'Brand_SL', 'Brand_SUA',
                  'Brand_SUN', 'Brand_THO', 'Brand_TR', 'Brand_TRI', 'Brand_Unknown',
                  'Brand_VER', 'Brand_VR', 'Brand_WON']
else:
    dummies_Brand = []

# Id
Id['product'] = 'Producto_ID'

# Features to generate
ToGroup['product'] = ['Producto_ID']

# List of features
GeneratedFeat['product'] = [x + '_' + y + '_' + z for x in ToGroup['product']
                             for y in AggFeatures_all
                             for z in AggFeatures_types]

OtherFeat['product'] = [] + dummies_Brand

# Dtypes
DtypeGeneratedFeat['product'] = {
    x: np.float16 for x in GeneratedFeat['product']}
DtypeOtherFeat['product'] = dict(list({'Weight': np.float16}.items()) +
                                   list({x: np.bool for x in dummies_Brand}.items()))

#-------------------------------------------------------------------------
# Cliente table details
#-------------------------------------------------------------------------

# ID
Id['cliente'] = 'Cliente_ID'

# Features to generate
ToGroup['cliente'] = ['NombreCliente']

# List of features
GeneratedFeat['cliente'] = [x + '_' + y + '_' + z for x in ToGroup['cliente']
                             for y in AggFeatures_all
                             for z in AggFeatures_types]

OtherFeat['cliente'] = []

# Dtypes
DtypeGeneratedFeat['cliente'] = {
    x: np.float32 for x in GeneratedFeat['cliente']}
DtypeOtherFeat['cliente'] = {}

#-------------------------------------------------------------------------
# Canal table details
#-------------------------------------------------------------------------

# ID
Id['canal'] = 'Canal_ID'

# Features to generate
ToGroup['canal'] = ['Canal_ID']

# List of features
GeneratedFeat['canal'] = [x + '_' + y + '_' + z for x in ToGroup['canal']
                           for y in AggFeatures_all
                           for z in AggFeatures_types]

OtherFeat['canal'] = []

# Dtypes
DtypeGeneratedFeat['canal'] = {
    x: np.float16 for x in GeneratedFeat['canal']}
DtypeOtherFeat['canal'] = {}

#-------------------------------------------------------------------------
# Agencia table details
#-------------------------------------------------------------------------

# ID
Id['agencia'] = 'Agencia_ID'

# Features to generate
ToGroup['agencia'] = ['Agencia_ID']

# List of features
GeneratedFeat['agencia'] = [x + '_' + y + '_' + z for x in ToGroup['agencia']
                             for y in AggFeatures_all
                             for z in AggFeatures_types]

OtherFeat['agencia'] = []

# Dtypes
DtypeGeneratedFeat['agencia'] = {
    x: np.float32 for x in GeneratedFeat['agencia']}
DtypeOtherFeat['agencia'] = {}

#-------------------------------------------------------------------------
# Ruta table details
#-------------------------------------------------------------------------

# ID
Id['ruta'] = 'Ruta_SAK'

# Features to generate
ToGroup['ruta'] = ['Ruta_SAK']

# List of features
GeneratedFeat['ruta'] = [x + '_' + y + '_' + z for x in ToGroup['ruta']
                          for y in AggFeatures_all
                          for z in AggFeatures_types]

OtherFeat['ruta'] = []

# Dtypes
DtypeGeneratedFeat['ruta'] = {x: np.float32 for x in GeneratedFeat['ruta']}
DtypeOtherFeat['ruta'] = {}