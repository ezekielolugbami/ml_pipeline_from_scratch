import pandas as pd 
import numpy as np 

from preprocessing import pipeline
import config

pipeline = Pipeline(target = config.target,
    numerical_to_imput = config.numerical_to_imput,
    categorical_to_imput = config.categorical_to_imput,
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