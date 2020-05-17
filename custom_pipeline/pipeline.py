import pandas as pd 
import numpy as np 

from preprocessors import Pipeline
import config

pipeline = Pipeline(target = config.target,
    numerical_to_impute = config.numerical_to_impute,
    categorical_to_impute = config.categorical_to_impute,
    numerical_log = config.numerical_log,
    categorical_encode = config.categorical_encode,
    features = config.features
)


if __name__ == '__main__':
    
    # load data
    data = pd.read_csv(config.path_to_dataset)

    pipeline.fit(data)
    # determine mse and rmse
    print('model performance')
    pipeline.evaluate_model()
    print()
    print('Some predictions:')
    predictions = pipeline.predict(data)
    print(predictions)