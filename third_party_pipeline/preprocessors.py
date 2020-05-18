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
            X[feature+'_na'] = np.where(X[feature].isnull(), 1, 0)
            self.imputer_dict_[feature] = X[feature].median()[0]
        return self

    def transform(self, X):

        X = X.copy()
        for feature in self.variables:
            X[feature].fillna(self.imputer_dict_[feature], inplace=True)
        return X
#     self.variable.to_csv('X.csv', index=False)
        
        
        
# # Temporal variable calculator
# class TemporalVariableEstimator(BaseEstimator, TransformerMixin):

#     def __init__(self, variables=None, reference_variable=None):
        
#         if not isinstance(variables, list):
#             self.variables = [variables]
#         else:
#             self.variables = variables

#         self.reference_variables = reference_variable

#     def fit(self, X, y=None):
#         # we need this step to fit the sklearn pipeline
#         return self

#     def transform(self, X):
#         X = X.copy()
#         for feature in self.variables:
#             X[feature+'_na'] = np.where(X[self.reference_variables].isnull(), 1, 0)

#         return X        
        
        
        
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
        
        
# logarithm transformer
class LogTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):
        # to accomodate the pipeline
        return self

    def transform(self, X):
        X = X.copy()

        for feature in self.variables:
            X[feature] = np.log1p(X[feature])

        return X
    
    
# frequent label categorical encoder
class RareLabelCategoricalEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, tol=0.005, variables=None):
        
        self.tol = tol
        
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):

        # persist frequent labels in dictionary
        self.encoder_dict_ = {}

        for var in self.variables:
            # the encoder will learn the most frequent categories
            t = pd.Series(X[var].value_counts() / np.float(len(X)))
            # frequent labels:
            self.encoder_dict_[var] = list(t[t >= self.tol].index)

        return self

    def transform(self, X):
        X = X.copy()
        for feature in self.variables:
            X[feature] = np.where(X[feature].isin(self.encoder_dict_[
                    feature]), X[feature], 'Rare')

        return X
    
    
    
# string to numbers categorical encoder
class CategoricalEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y):
        temp = pd.concat([X, y], axis=1)
        temp.columns = list(X.columns) + ['target']

        # persist transforming dictionary
        self.encoder_dict_ = {}

        for var in self.variables:
            t = temp.groupby([var])['target'].mean().sort_values(ascending=True).index
            self.encoder_dict_[var] = {k: i for i, k in enumerate(t, 0)}

        return self

    def transform(self, X):
        # encode labels
        X = X.copy()
        for feature in self.variables:
            X[feature] = X[feature].map(self.encoder_dict_[feature])

        return X
    
 
    
class DropUnecessaryFeatures(BaseEstimator, TransformerMixin):

    def __init__(self, variables_to_drop=None):
        
        self.variables = variables_to_drop

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # encode labels
        X = X.copy()
        X = X.drop(self.variables, axis=1)

        return X