import numpy as np 

import preprocessing_functions as pf 
import config

import warnings
warnings.simplefilter(action='ignore')

# ================================================
# training the model

# load data
data = pf.load_data(config.path_to_dataset)

# split the data
X_train, X_test, y_train, y_test = pf.split_train_test(data, config.target)

# imput numerical variable
X_train[config.numerical_to_imput] = pf.impute_na(X_train, 
config.numerical_to_imput, replacement=config.median_age)


# imput categorical variables
for var in config.categorical_to_imput:
    X_train[var] = pf.impute_na(X_train, var, replacement='Missing')


# log transform numerical variables
for var in config.numerical_log:
    X_train[var] = pf.log_transform(X_train, var)


# remove rare labels from categorical variables
for var in config.categorical_encode:
    X_train[var] = pf.remove_rare_labels(X_train, 
    var, config.frequent_labels[var])


# encode categorical variables
for var in config.categorical_encode:
    X_train[var] = pf.encode_categorical(X_train, 
    var, config.encoding_mappings[var])


# train scaler and save
scaler = pf.train_scaler(X_train[config.features], 
config.output_scaler_path)

# scale train set
X_train = scaler.transform(X_train[config.features])

# train and save model
pf.train_model(X_train, y_train, config.output_model_path)

print('Finished training')