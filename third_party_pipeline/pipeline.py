from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

import preprocessors as pp
import config


survive_pipe = Pipeline(
    [
    
        
        ('numerical_inputer',
            pp.NumericalImputer(variables=config.numerical_vars_with_na)),
        
        ('categorical_imputer',
            pp.CategoricalImputer(variables=config.categorical_vars_with_na)),
         
         ('log_transformer',
            pp.LogTransformer(variables=config.numerical_log_vars)),
         
        ('rare_label_encoder',
            pp.RareLabelCategoricalEncoder(
                tol=0.005,
                variables=config.categorical_vars)),
         
        ('categorical_encoder',
            pp.CategoricalEncoder(variables=config.categorical_vars)),
         
        ('drop_features',
            pp.DropUnecessaryFeatures(variables_to_drop=config.drop_features)),
         
        ('scaler', MinMaxScaler()),
        ('Linear_model', LogisticRegression(random_state=0))
    ]
)