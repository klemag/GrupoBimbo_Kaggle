from training_functions import *
import parameters as p

def write_to_log(s):
    print()
    print(s)
    file = open('model_select.txt', 'a')
    file.write(s)
    file.close()

if __name__ == "__main__":

    # Need to run build_dmatrices.py before to create the DMatrices for xgboost
    print('Model selection will load the latest dmatrices saved')

    if p.PCA_ON:
        print('PCA is on, will use features from the latest PCA model saved')

    # Load DMatrices and create a watchlist
    dtrain = xgb.DMatrix('{}/pipeline/dmatrices/train_matrix.buffer'.format(p.DIR_ROOT))
    dtest = xgb.DMatrix('{}/pipeline/dmatrices/test_matrix.buffer'.format(p.DIR_ROOT))
    evallist = [(dtest, 'eval')]

    # Start with a simple model
    param = {'max_depth': 6,
             'eta': 0.3,
             'min_child_weight': 1,
             'subsample': 0.65,
             'colsample_bytree': 0.65,
             'silent': 0,
             'objective': 'reg:linear',
             'nthread': 8,
             'eval_metric': 'rmse'}

    MAX_ROUNDS = 5

    # Training start
    print('\n\nTraining starting')
    # Initialize best rmsle
    rmsle_prev = 100
    rmsle_now = 99

    # Create log file
    file = open('model_select.txt', 'w+')
    file.close()
    write_to_log('First, find optimal max depth')

    # Start with depth
    x = 0
    score = {}
    while rmsle_now < rmsle_prev:
        rmsle_prev = rmsle_now
        depth = 7 + x
        # Update param
        param['max_depth'] = depth
        print('\nMax depth {}'.format(depth))

        # Train model
        model = xgb.train(param, dtrain, MAX_ROUNDS, evallist, early_stopping_rounds=p.XGB_STOP_ROUNDS,
                          evals_result=score)

        # Getting RMSLE
        rmsle_now = float(min(score['eval']['rmse']))
        write_to_log('\n--> RMSLE for depth {} = {}'.format(depth, rmsle_now))
        x += 1

    # Optimal max_depth is the previous depth used
    # Update param to the optimal depth
    param['max_depth'] = depth - 1
    write_to_log('\nOptimal max_depth is {}'.format(depth - 1))


    ### Then optimize min_child_weight
    write_to_log('\nThen, find optimal min_child_weight')

    # The best rmsle so far was the prev one, we start from that
    rmsle_now = rmsle_prev
    rmsle_prev = 100
    x = 0
    score = {}
    while rmsle_now < rmsle_prev:
        rmsle_prev = rmsle_now
        weight = 1 + x
        # Update param
        param['min_child_weight'] = weight
        print('\nMin weight {}'.format(weight))

        # Train model
        model = xgb.train(param, dtrain, MAX_ROUNDS, evallist, early_stopping_rounds=p.XGB_STOP_ROUNDS,
                          evals_result=score)

        # Getting RMSLE
        rmsle_now = float(min(score['eval']['rmse']))
        write_to_log('\n--> RMSLE for weight {} = {}'.format(weight, rmsle_now))
        x += 1

    # Optimal min_weight is the previous weight used
    # Update param
    param['min_child_weight'] = max(1,weight - 1)
    write_to_log('\nOptimal min_weight is {}'.format(max(1,weight - 1)))


    ### Then optimize subsample
    write_to_log('\nThen, find optimal subsample')

    # The best rmsle so far was the prev one, we start from that
    rmsle_now = rmsle_prev
    rmsle_prev = 100
    x = 0
    score = {}
    while rmsle_now < rmsle_prev:
        rmsle_prev = rmsle_now
        sub = 0.65 + x
        # Update param
        param['subsample'] = sub
        print('\nSubsample {}'.format(sub))

        # Train model
        model = xgb.train(param, dtrain, MAX_ROUNDS, evallist, early_stopping_rounds=p.XGB_STOP_ROUNDS,
                          evals_result=score)

        # Getting RMSLE
        rmsle_now = float(min(score['eval']['rmse']))
        write_to_log('\n--> RMSLE for subsample {} = {}'.format(sub, rmsle_now))
        x += 0.05

    # Optimal subsample is the previous
    # Update param
    param['subsample'] = sub - 0.05
    write_to_log('\nOptimal subsample is {}'.format(sub - 0.05))


    # Finally optimize the colsample
    write_to_log('\nFinally, optimize colsample')

    # The best rmsle so far was the prev one, we start from that
    rmsle_now = rmsle_prev
    rmsle_prev = 100
    x = 0
    score = {}
    while rmsle_now < rmsle_prev:
        rmsle_prev = rmsle_now
        col = 0.75 + x
        # Update param
        param['colsample_bytree'] = col
        print('\nColsample_bytree {}'.format(col))

        # Train model
        model = xgb.train(param, dtrain, MAX_ROUNDS, evallist, early_stopping_rounds=p.XGB_STOP_ROUNDS,
                          evals_result=score)

        # Getting RMSLE
        rmsle_now = float(min(score['eval']['rmse']))
        write_to_log('\n--> RMSLE for colsample_bytree {} = {}'.format(col, rmsle_now))
        x += 0.05

    # Optimal colsample is the previous
    # Update param
    param['colsample_bytree'] = col - 0.05
    write_to_log('\nOptimal colsample_bytree is {}'.format(col - 0.05))

    write_to_log('\nFinal model:\n\
                 Optimal max_depth {}\n\
                 Optimal min_child_weight {}\n\
                 Optimal subsample {}\n\
                 Optimal colsample {}'.format(
                     param['max_depth'],
                     param['min_child_weight'],
                     param['subsample'],
                     param['colsample_bytree']))