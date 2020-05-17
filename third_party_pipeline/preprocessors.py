import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin

# numerical missing value imputer
class NumericalImputer(BaseEstimator, TransformerMixin):
    
    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables
            
        def fit(self, X, y=None):
            #  persist median in a dictionary
            self.imputer_dict_ = {}
            
            for feature in self.variables:
                self.imputer_dict_[feature] = X[feature].median()[0]
            return self
        
        def transform(self, X):
            
            X = X.copy()
            for feature in self.variables:
                X[feature].fillna(self.imputer_dict_[feature], inplace=True)
            return X
        
        
        
class CategoricalImputer(BaseEstimator, TransformerMixin):
    
    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables
            
    def fit(self, X, y=None):
        # we need the fit statement to accomodate the sklearn pipeline
        return self
    
    def transform(self, X):
        X = X.copy()
        for feature in self.variables:
            X[feature] = X[feature].fillna('Missing')
            
        return X
        

            