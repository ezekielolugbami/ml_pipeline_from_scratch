import pandas as pd

import joblib
import config


def make_prediction(input_data):
    
    _survive_price = joblib.load(filename=config.pipeline_name)
    
    results = _survive_price.predict(input_data)

    return results
   
if __name__ == '__main__':
    
    # test pipeline
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, f1_score, roc_auc_score, precision_score

    data = pd.read_csv(config.training_data_file)

    X_train, X_test, y_train, y_test = train_test_split(
        data[config.features],
        data[config.target],
        test_size=0.1,
        random_state=0)
    
    pred = make_prediction(X_test)
    
    # determine mse and rmse
    print('test mse: {}'.format(mean_squared_error(self.y_test, pred)))
    print('test rmse: {}'.format(np.sqrt(mean_squared_error(self.y_test, pred))))
    print('test f1_score: {}'.format(f1_score(self.y_test, pred)))
    print('test roc_auc_score: {}'.format(roc_auc_score(self.y_test, pred)))
    print('test precision_score: {}'.format(precision_score(self.y_test, pred)))
    print()
    
  
    