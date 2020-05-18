import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
import joblib

import pipeline
import config


def run_training():
    """Train the model."""

    # read training data
    data = pd.read_csv(config.training_data_file)

    # divide train and test
    X_train, X_test, y_train, y_test = train_test_split(
        data[config.features],
        data[config.target],
        test_size=0.1,
        random_state=0)  # we are setting the seed here

    pipeline.survive_pipe.fit(X_train[config.features], y_train)
    joblib.dump(pipeline.survive_pipe, config.pipeline_name)


if __name__ == '__main__':
    run_training()