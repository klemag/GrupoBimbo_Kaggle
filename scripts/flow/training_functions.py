from load_tables import *
from sklearn.metrics import mean_squared_error
import parameters as p
from math import sqrt, expm1
from create_semanas import get_lag_features
import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn.externals import joblib

## If using fraction of the data save train data to xgb Dmatrix format
# This mode comes with:
# Faster training/testing flow (no need to rebuild matrices, only load them from bin file)
# Possibility to use PCA
# early_stopping_round and test watchlist whilst training
def get_DMatrix(weeks_number, indexes, indexes_test):

    # Load training week
    week = load_train_week(weeks_number, indexes)
    train, target = get_features_target(week)
    del week

    # Load test week
    week_test = load_train_week(weeks_number, indexes_test)
    test, test_target = get_features_target(week_test)
    del week_test
    print(test.info(verbose=True))

    # PCA transform
    if p.PCA_ON:
        pca = PCA(p.PCA_N_FEAT)
        pca.fit(train)
        joblib.dump(pca, p.DIR_ROOT + '/pipeline/dmatrices/pca.pkl')
        train = pca.transform(train)
        test = pca.transform(test)

    # Build DMatrix and save it
    dtrain = xgb.DMatrix(train, label=target)
    del train, target
    dtrain.save_binary(p.DIR_ROOT + "/pipeline/dmatrices/train_matrix.buffer")
    dtest = xgb.DMatrix(test, label=test_target)
    del test, test_target
    dtest.save_binary(p.DIR_ROOT + "/pipeline/dmatrices/test_matrix.buffer")

## Function for training a sample
# Used only with ensemble mode
def train_week(weeks_number, indexes, model):

    # Load training week
    week = load_train_week(weeks_number, indexes)
    train, target = get_features_target(week)
    del week

    # Train model
    print(train.info(verbose=True))
    model.fit(train, target)
    best_feat = pd.Series(model.booster().get_fscore()).sort_values(ascending=False)[:20]
    worst_feat = pd.Series(model.booster().get_fscore()).sort_values(ascending=False)[-20:]
    print('\nBest 20 features:')
    print(best_feat)
    print('\nWorst 20 features:')
    print(worst_feat)
    del train, target

def test_week(weeks_number, indexes, model):

    # Load test week
    week = load_train_week(weeks_number, indexes)
    test, test_target = get_features_target(week)
    del week

    print(test.info(verbose=True))
    # Get predictions
    pred = model.predict(test)
    del test

    # Calculate RMSLE (the prediction and test_target are both in log space
    # from get_features_target function)
    rmsle = sqrt(mean_squared_error(test_target, pred))
    del test_target
    return rmsle

# Function to generate predictions. It can take a list of
# old week10 predictions as an argument, these old predictions will
# be averaged with the current week10 prediction in order to have a
# more stable model and a better prediction of week11
def get_predictions(model, week10):

    # Load PCA model
    if p.PCA_ON and p.TRAINING_MODE=='Fraction':
        pca = joblib.load(p.DIR_ROOT + '/pipeline/dmatrices/pca.pkl')

    data_10 = load_week(10, 1)
    data_11 = load_week(11, 1)
    id = pd.concat([data_10, data_11], axis=0).id
    L_10 = len(data_10)
    L_11 = len(data_11)
    pred_10 = []
    pred_11 = []
    # Splits each week in N_bins to reduce memory usage
    # Useful if too many features that it can't fit in the memory
    N_bins = 1

    # First predict week 10
    for x in range(N_bins):
        test = get_features_target(data_10.iloc[int(
            x * L_10 / N_bins):int((x + 1) * L_10 / N_bins)], features_only=True)

        if p.TRAINING_MODE=='Fraction':
            if p.PCA_ON:
                test = pca.transform(test)
            # Create DMatrix
            test = xgb.DMatrix(test)

        pred_10.append(pd.DataFrame(model.predict(test)))

    # Concat all the predictions for week 10 and use expm1 to get back to
    # 'Demanda_uni_equil space (so far we were predicting in log space
    pred_10 = pd.concat(pred_10, axis=0)[0].apply(expm1)

    # Add the predictions to data_10 and update 11 to get the lag_1w
    data_10['Demanda_uni_equil'] = pred_10.values

    # If we pass a list of other week10, average it with them
    if week10:
        print('Averaging with old week10 predictions')
        new_prediction = data_10.loc[:, ['id', 'Demanda_uni_equil']]
        for we in week10:
            new_prediction = pd.merge(new_prediction, we, on=['id'], how='left')
        new_prediction = new_prediction.drop('id', axis=1).mean(axis=1)
        data_10['Demanda_uni_equil'] = new_prediction
        # Update pred_10 to take be the averaged data
        pred_10 = data_10['Demanda_uni_equil']

    # Finally get lag features for week11
    data_11 = get_lag_features(data_11, data_10, 1, 'Demanda_uni_equil')
    del data_10

    # Now predict week 11
    for x in range(N_bins):
        test = get_features_target(data_11.iloc[int(
            x * L_11 / N_bins):int((x + 1) * L_11 / N_bins)], features_only=True)

        if p.TRAINING_MODE=='Fraction':
            if p.PCA_ON:
                test = pca.transform(test)
            # Get DMatrix
            test = xgb.DMatrix(test)

        pred_11.append(pd.DataFrame(model.predict(test)))

    del data_11
    pred_11 = pd.concat(pred_11, axis=0)[0].apply(expm1)

    # Build new dataframce with pred_10 and pred_11
    pred = pd.concat([pred_10, pred_11], axis=0)
    del pred_10, pred_11

    submission = pd.DataFrame({'id': id, 'Demanda_uni_equil': pred.values})
    del pred, id
    return submission