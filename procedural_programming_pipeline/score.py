import preprocessing_functions as pf
import config

# =============== scoring pipeline ===============

def predict(data):
    # imput numerical variable
    data[config.numerical_to_imput] = pf.impute_na(data, 
    config.numerical_to_imput, replacement=config.median_age)
    
    # imput categorical variables
    for var in config.categorical_to_imput:
        data[var] = pf.impute_na(data, var, replacement='Missing')


    # log transform numerical variables
    for var in config.numerical_log:
        data[var] = pf.log_transform(data, var)


    # remove rare labels from categorical variables
    for var in config.categorical_encode:
        data[var] = pf.remove_rare_labels(data, 
        var, config.frequent_labels[var])


    # encode categorical variables
    for var in config.categorical_encode:
        data[var] = pf.encode_categorical(data, 
        var, config.encoding_mappings[var])


    # scale features
    data = pf.scale_features(data[config.features], 
    config.output_scaler_path)

    # make predictions
    predictions = pf.predict(data, config.output_model_path)

    return predictions


# =============================================================

# small test to ensure scripts are working

if __name__ == '__main__':

    from math import sqrt
    import numpy as np

    from sklearn.metrics import mean_squared_error, roc_auc_score, f1_score, precision_score

    import warnings
    warnings.simplefilter(action='ignore')

    # load data
    data = pf.load_data(config.path_to_dataset)
    X_train, X_test, y_train, y_test = pf.split_train_test(data, config.target)

    pred= predict(X_test)

    # determine mse and rmse
    print('test mse: {}'.format(mean_squared_error(y_test, pred)))
    print('test rmse: {}'.format(sqrt(mean_squared_error(y_test, pred))))
    print('test f1_score: {}'.format(f1_score(y_test, pred)))
    print('test roc_auc_score: {}'.format(roc_auc_score(y_test, pred)))
    print('test precision_score: {}'.format(precision_score(y_test, pred)))

