import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, roc_auc_score, f1_score, precision_score


class Pipeline:
    ''' When we call the FeaturePreprocessor for the first time
    we initialise it with the data set we use to train the model,
    plus the different groups of variables to which we wish to apply
    the different engineering procedures'''

    def __init__(self, target, numerical_to_impute, categorical_to_impute, numerical_log, categorical_encode, features, test_size =0.1, random_state =0, percentage=0.005):
        
        # data set
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        # engineering parameters to be learnt from the data
        self.imputing_dict = {}
        self.frequent_category_dict = {}
        self.encoding_dict = {}


        # models
        self.scaler = MinMaxScaler()
        self.model = LogisticRegression(random_state=random_state)


        # variables to be engineered
        self.target = target 
        self.numerical_to_impute = numerical_to_impute
        self.categorical_to_impute = categorical_to_impute
        self.numerical_log = numerical_log
        self.categorical_encode =  categorical_encode 
        self.features = features

        # more parameters
        self.test_size = test_size
        self.random_state = random_state
        self.percentage = percentage


        # ====================functions to learn parameters from train set =============

    def missing_value_indicator(self, df):
        ''' engineer missing value indicator'''
        df = df.copy()
        for variable in self.numerical_to_impute:
            df[variable+'_na'] = np.where(df[variable].isnull(), 1, 0)
        return df
              
        
    def find_imputation_replacement(self):
        '''find value to be used for imputattion'''

        for variable in self.numerical_to_impute:
            replacement = self.X_train[variable].median()
            self.imputing_dict[variable] = replacement
        return self 


    def find_frequent_categories(self):
        ''' find list of frequent categories in categorical variables'''

        for variable in self.categorical_encode:
            tmp = self.X_train.groupby(variable)[self.target].count() /len(self.X_train)
            self.frequent_category_dict[variable] = tmp[tmp > self.percentage].index
        return self 



    def find_categorical_mappings(self):
        ''' create category to integer mappings for categorical encoding'''

        for variable in self.categorical_encode:
            ordered_labels = self.X_train.groupby(variable)[self.target].sum().sort_values().index 
            ordinal_labels = {k: i for i, k in enumerate(ordered_labels, 0)}
            self.encoding_dict[variable] = ordinal_labels
        return self 


    # functions to transform data
    def remove_rare_labels(self, df):
        ''' group infrequent labels in group Rare'''

        df = df.copy()
        for variable in self.categorical_encode:
            df[variable] = np.where(df[variable].isin(self.frequent_category_dict[variable]), df[variable], 'Rare')
        return df
    

    def encode_categorical_variables(self, df): 
        ''' replace categories by numbers in categorical variables'''
        df = df.copy()
        for variable in self.categorical_encode:
            df[variable] = df[variable].map(self.encoding_dict[variable])
        return df


    # master function 
    def fit(self, data):
        '''pipeline to learn parameters from data, fit the scaler and lasso'''
        #             engineer missing value indicator
        data = self.missing_value_indicator(data)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(data, data[self.target], test_size=self.test_size, random_state=self.random_state)

        # find imputation parameters
        self.find_imputation_replacement()

        # imput missing values
        # numerical
        self.X_train[self.numerical_to_impute]= self.X_train[self.numerical_to_impute].fillna(
            self.imputing_dict[self.numerical_to_impute[0]])

        self.X_test[self.numerical_to_impute] = self.X_test[self.numerical_to_impute].fillna(self.imputing_dict[self.numerical_to_impute[0]])

        # categorical
        self.X_train[self.categorical_to_impute] = self.X_train[self.categorical_to_impute].fillna('Missing')

        self.X_test[self.categorical_to_impute] = self.X_test[self.categorical_to_impute].fillna('Missing')

        # transform numerical variable
        self.X_train[self.numerical_log] = np.log1p(self.X_train[self.numerical_log])
        self.X_test[self.numerical_log] = np.log1p(self.X_test[self.numerical_log])

        # find frequent labels
        self.find_frequent_categories()                    

        # remove rare labels
        self.X_train = self.remove_rare_labels(self.X_train)
        self.X_test = self.remove_rare_labels(self.X_test)

        # find categorical mapping
        self.find_categorical_mappings()


        # encode categorical variables                
        self.X_train = self.encode_categorical_variables(self.X_train)    
        self.X_test = self.encode_categorical_variables(self.X_test)    

        #  train scaler
        self.scaler.fit(self.X_train[self.features])


        # scale variables
        self.X_train = self.scaler.transform(self.X_train[self.features])
        self.X_test = self.scaler.transform(self.X_test[self.features])
        print(self.X_train.shape, self.X_test.shape)

        # train model
        self.model.fit(self.X_train, self.y_train)

        return self


    def transform(self, data):
        ''' transforms the raw data into engineered features'''

        data = data.copy()

        #             engineer missing value indicator
        data = self.missing_value_indicator(data)

#             impute numerical 
        data[self.numerical_to_impute] = data[self.numerical_to_impute].fillna(
        self.imputing_dict[self.numerical_to_impute[0]])

#             impute categorical
        data[self.categorical_to_impute] = data[self.categorical_to_impute].fillna(
            'Missing')

#             transform numerical variables
        data[self.numerical_log] = np.log1p(data[self.numerical_log])

#             remove rare labels
        data = self.remove_rare_labels(data)

#             encode categorical variables
        data = self.encode_categorical_variables(data)

#             scale variables
        data = self.scaler.transform(data[self.features])

        return data


    def predict(self, data):
        ''' obtain predictions'''
        data = self.transform(data)
        predictions = self.model.predict(data)
        return predictions


    def evaluate_model(self):
        '''evaluates trained model on train and test sets'''
        pred = self.model.predict(self.X_train)
        # determine mse and rmse
        print('train mse: {}'.format(mean_squared_error(self.y_train, pred)))
        print('train rmse: {}'.format(np.sqrt(mean_squared_error(self.y_train, pred))))
        print('train f1_score: {}'.format(f1_score(self.y_train, pred)))
        print('train roc_auc_score: {}'.format(roc_auc_score(self.y_train, pred)))
        print('train precision_score: {}'.format(precision_score(self.y_train, pred)))

        pred = self.model.predict(self.X_test)

        # determine mse and rmse
        print('test mse: {}'.format(mean_squared_error(self.y_test, pred)))
        print('test rmse: {}'.format(np.sqrt(mean_squared_error(self.y_test, pred))))
        print('test f1_score: {}'.format(f1_score(self.y_test, pred)))
        print('test roc_auc_score: {}'.format(roc_auc_score(self.y_test, pred)))
        print('test precision_score: {}'.format(precision_score(self.y_test, pred)))









             



























                 









































































 



































































































































































































     






















































































 




























































































































































































































































































































































 