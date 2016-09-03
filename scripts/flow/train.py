import pandas as pd
from training_functions import *
import parameters as p
from xgboost.sklearn import XGBRegressor
from datetime import datetime
import os

# Make sure the training mode is properly set
if p.TRAINING_MODE not in ['Fraction', 'Ensemble']:
    raise ValueError('TRAINING_MODE should be Fraction or Ensemble')

# Get a time stamp and create a new directory for the submission
time = datetime.now()
stp = time.strftime('%y%m%d%H%M')
del time
os.makedirs('{}/outputs/submission_{}'.format(p.DIR_ROOT, stp))
SUB_DIR = '{}/outputs/submission_{}/submissions'.format(p.DIR_ROOT, stp)
os.makedirs(SUB_DIR)
FEAT_DIR = '{}/outputs/submission_{}/best_features'.format(p.DIR_ROOT, stp)
os.makedirs(FEAT_DIR)

if p.TRAINING_MODE == 'Fraction':

    print('Training mode selected is Fraction, will load the latest dmatrices saved')

    if p.PCA_ON:
        print('PCA is on, will use features from the latest PCA model saved')

    # Load DMatrices and create a watchlist
    dtrain = xgb.DMatrix(
        '{}/pipeline/dmatrices/train_matrix.buffer'.format(p.DIR_ROOT))
    dtest = xgb.DMatrix(
        '{}/pipeline/dmatrices/test_matrix.buffer'.format(p.DIR_ROOT))
    evallist = [(dtest, 'eval')]

    print('\n\nTraining starting')

    # Load training week
    model = xgb.train(p.XGB_PARAM, dtrain, p.XGB_N_ROUNDS,
                      evallist, early_stopping_rounds=p.XGB_STOP_ROUNDS)
    del dtrain, dtest

    # Get predictions
    # Generates two predictions, a normal one first.
    # Then we pass this normal prediction + other old predictions of week 10 if needed
    # to the get_predictions functions. This will average the week 10 with other predictions
    # in order to generate both a better week10 and week11 prediction
    # this is because the prediction of week11 depends on the prediction of
    # week10
    print('\nGetting predictions')
    basic_prediction = get_predictions(model, [])
    id = basic_prediction.id
    basic_prediction.to_csv(
        '{}/basic_prediction.csv'.format(SUB_DIR), index=False)

    # get IDs from week 10 (need to load week 10...)
    week10 = []
    data_10_ids = list(load_week(10, 1).id)

    # Insert previous week10 predictions to be averaged with latest week10
    # prediction
    for w10_pred in p.WEEK10_OLD_PRED:
        old = pd.read_csv(
            '{}/pipeline/week10_reuse/{}'.format(p.DIR_ROOT, w10_pred))
        # Keep only the week 10...
        old = old.loc[old.id.isin(data_10_ids)]
        week10.append(old)

    # Generate new predictions, with the old week10s this time
    pred = get_predictions(model, week10)
    pred.to_csv('{}/ensemble_prediction.csv'.format(SUB_DIR), index=False)


# In the ensemble mode we use the sklearn API for XGB that works natively
# with pandas dataframes
elif p.TRAINING_MODE == 'Ensemble':

    # Find length of the two weeks combined
    w8 = pd.read_csv(p.DIR_ROOT + '/pipeline/by_week/semana_8.csv',
                     usecols=['Canal_ID'], dtype={'Canal_ID': np.int8})
    w9 = pd.read_csv(p.DIR_ROOT + '/pipeline/by_week/semana_9.csv',
                     usecols=['Canal_ID'], dtype={'Canal_ID': np.int8})
    df = pd.concat([w8, w9], axis=0)
    length_2w = pd.Series(range(len(df)))
    del df, w8, w9

    # Create a XGB model using sklearn API
    model = XGBRegressor(
        learning_rate=p.XGB_PARAM['eta'],
        n_estimators=p.XGB_N_ROUNDS,
        max_depth=p.XGB_PARAM['max_depth'],
        min_child_weight=p.XGB_PARAM['min_child_weight'],
        subsample=p.XGB_PARAM['subsample'],
        colsample_bytree=p.XGB_PARAM['colsample_bytree'],
        objective=p.XGB_PARAM['objective'],
        nthread=p.XGB_PARAM['nthread'],
        silent=False)

    # indexes is a list of list
    # items are each sub dataset defined by its indexes
    indexes = []
    if p.N_SAMPLES not in [2, 4, 8, 16]:
        raise ValueError('N_samples size not supported, must be in [2,4,8,16]')

    # Start by spliting the dataset in two samples
    msk = np.random.rand(len(length_2w)) < 0.5
    l1 = length_2w.iloc[msk]
    l2 = length_2w.iloc[~msk]
    list_prev_div = [l1, l2]

    # Divides each sample by two until we get to the good amount of samples
    n_iter = p.N_SAMPLES / 2 + 1
    while n_iter > 1:
        list_cur_div = []
        for lx in list_prev_div:
            msk = np.random.rand(len(lx)) < 0.5
            list_cur_div.append(lx.iloc[msk])
            list_cur_div.append(lx.iloc[~msk])
        list_prev_div = list_cur_div.copy()
        n_iter -= 1

    # Add all the new datasets to the list of indexes
    for lx in list_prev_div:
        indexes.append(list(lx))

    for sample in range(p.N_SAMPLES):
        print('Sample {} size: {}'.format(sample, len(indexes[sample])))

    # List containing all predictions for each sample
    predictions = []
    # Train each sample individually and average the predictions
    for sample in range(p.N_SAMPLES):
        print('\n\nTraining sample {}'.format(sample))

        # Train sample from weeks 8 and 9
        train_week([8, 9], indexes[sample], model)

        # Test sample is the next one in the list
        sample_to_test = (sample + 1) % p.N_SAMPLES
        # Get best features and save them
        pd.Series(model.booster().get_fscore()).sort_values(ascending=False). \
            to_csv('{}/best_features_sample_{}.csv'.format(FEAT_DIR, sample))

        # Test
        print('\nTesting on sample {}'.format(sample_to_test))
        rmsle = test_week([8, 9], indexes[sample_to_test], model)
        print('\n--> RMSLE = {}'.format(rmsle))

        # Get predictions, average week10 with previous ones if the list is not
        # empty
        print('\nGetting predictions')
        week10 = []
        if p.WEEK10_OLD_PRED:
            # get IDs from week 10 (need to load week 10...)
            data_10_ids = list(load_week(10, 1).id)

            # Insert previous (more stable) week10 predictions to be averaged
            # with current week10 prediction
            for w10_pred in p.WEEK10_OLD_PRED:
                old = pd.read_csv(
                    '{}/pipeline/week10_reuse/{}'.format(p.DIR_ROOT, w10_pred))
                # Keep only the week 10...
                old = old.loc[old.id.isin(data_10_ids)]
                week10.append(old)

        pred = get_predictions(model, week10)
        id = pred.id
        # Append the predicted value to the list of predictions
        # To be averaged later when all samples have been trained
        predictions.append(pred.Demanda_uni_equil)
        pred.to_csv(
            '{}/prediction_sample_{}.csv'.format(SUB_DIR, sample), index=False)
        del pred

    # Average all predictions
    predictions = pd.concat(predictions, axis=1).mean(axis=1)
    print('\nWriting down averaged submission')
    submission = pd.DataFrame({'id': id, 'Demanda_uni_equil': predictions})
    submission.to_csv(
        '{}/ensemble_prediction.csv'.format(SUB_DIR), index=False)
