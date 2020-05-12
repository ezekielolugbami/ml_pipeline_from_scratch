import pandas as pd
import numpy as np 

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression

import joblib

# individual pre-processing and training functions
# ================================================

# function to load training data
def load_data(df_path):
    return pd.read_csv(df_path)


# fucntion to split into train and test
def split_train_test(df, target):
    X_train, X_test, y_train, y_test = train_test_split(df,
    df[target], test_size=0.1, random_state=0)
    return X_train, X_test, y_train, y_test


# function to fill na with value entered or default to 'missing' string
def impute_na(df, var, replacement='Missing'):
    df[var+ '_na'] = np.where(df[var].isnull(), 1, 0)
    return df[var].fillna(replacement)


# log transform skewed variables
def log_transform(df, var):
    return np.log1p(df[var])


# replace rare labels with 'rare' string 
def remove_rare_labels(df, var, frequent_labels):
    return np.where(df[var].isin(frequent_labels), df[var], 'Rare')


# replace strings by mapping dictionary numbers
def encode_categorical(df, var, mappings):
    return df[var].map(mappings)


# define and persist scaler
def train_scaler(df, output_path):
    scaler = MinMaxScaler()
    scaler.fit(df)
    joblib.dump(scaler, output_path)
    return scaler


def scale_features(df, scaler):
    scaler = joblib.load(scaler)
    return scaler.transform(df)


def train_model(df, target, output_path):
    # initialize model 
    log_reg = LogisticRegression(random_state=0)
    # train model 
    log_reg.fit(df, target)
    # save model 
    joblib.dump(log_reg, output_path)
    return None 

def predict(df, model):
    model = joblib.load(model)
    return model.predict(df)
