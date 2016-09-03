# GrupoBimbo_Kaggle

This workflow was built to create a model for the Kaggle competition GrupoBimbo
Link to the competition: https://www.kaggle.com/c/grupo-bimbo-inventory-demand

GrupoBimbo is a Mexican bakery product manufacturing company.
Bimbo has one of the widest distribution networks in the world, surpassing 52,000 routes. Bimbo operates under a scheme of recurrent sales channels views, making three daily visits to the same establishment.

The goal of the competition was to build a predictive model to forecast the inventory demand for each customer/product for two weeks.
GrupoBimbo provided 6 weeks of product orders data.

The workflow allows to automate the generation of features and the training of an XGB model.
This flow was used to generate the submission that lead me to the 229th place out of 1969 competitors. It was done
using an ensemble of predictions, generated by tuning the different parameters/generating different features.

Memory and training time can be an important limitation, I used 16Go of RAM and could not use all features as the same time.
Some low_ram scripts are available but this will not be enough in some cases - A trade-off is needed between adding more features and using more data.

## Usage:

1 - First download the data from Kaggle. It can be done manually or using ./inputs/dl_data.sh (this will require a cookies.txt file with your Kaggle login saved)

2 - Generate additional features from the reference tables. This will generate csv files in ./pipeline/modified_ref_tables
The large amount of possible values for categorical features (product, clients) made it hard to vectorize. In order to still take them into account I extracted more generic categorical features (such as Brand, ClienteGroup). I also created new features from aggregation of the categorical features. The categorical features and the type of aggregation to use (by default only mean is created) can be set in ./scripts/flow/parameters.py (see AggFeatures). Use ./scripts/flow/ref_tables/run.py to generate those tables (or use the scripts one by one/low_ram scripts). Set OutliersThreshold and MaxValueGenerated in parameters.py in order to ignore outliers in the dataset and normalized the data.

3 - Generate additional features from the weekly data. This will create files by week in ./pipeline/by_week. The generated features take into account the demand observed for a given product/client on the previous weeks. Use ./scripts/flow/create_semanas.py to generate those tables.

4 - Generate DMatrix for later use in XGB. This will load all training data (I used weeks 8 and 9) and join it with the reference tables in order to build a train and a test dataset with all the generated features. Those datasets are saved in ./pipeline/dmatrices. Use build_dmatrices.py to generate the DMatrix.
If PCA_ON is True in parameters.py, a PCA model will also be generated. Tune the fraction of data that is used for training by setting SAMPLE_SIZE in parameters.py. Use the variable FeaturesToKill to remove features from the training data.

5 - Find good parameters for the model. Use model_selection.py to find optimal (well, good...) parameters to train the xgb model. Needs dmatrices to be generated before. Will return a txt file with the best parameters found.

6 - Ready to train. Two training mode are available:
    
    a) Fraction mode: This will reuse the DMatrix generated with build_dmatrices.py. This mode allows early_stopping_rounds and watchlist. Meaning that whilst training the model, after each round the RMSLE will be calculated and displayed. If the RMSLE is not improved for a given number of rounds the training stops, avoiding overfitting. This mode also comes with the possibility to use PCA for dimension reduction.
    
    b) Ensemble mode: This will split the data into N_SAMPLES (from parameters.py) and create a new model for each. All predictions will then be averaged to get a more stable prediction. PCA and watchlist are not supported with this mode.
    
    Select the TRAINING_MODE in parameters.py. Both modes come with the possibility to inject one or several old prediction(s) whilst generating a new one. This will average the current week 10 with the old ones and predict the week 11 using better previous week demand data. Add files to inject into WEEK10_OLD_PRED.
    
7 - Submissions are saved in ./outputs/submission_DATETIME
