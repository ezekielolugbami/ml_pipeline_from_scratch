from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, roc_auc_score, f1_score, precision_score


class Pipeline:

    def __init__(self, target, numerical_to_imput, categorical_to_imput, numerical_log, 
    categorical_encode, features, test_size =0.1, random_state =0, percentage=0.005):

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
        self.numerical_to_imput = numerical_to_imput
        self.categorical_to_imput = categorical_to_imput
        self.numerical_log = numerical_log
        self.categorical_encode =  categorical_encode 
        self.features = features

        # more parameters
        self.test_size = test_size
        self.random_state = random_state
        self.percentage = percentage


        # functions to learn parameters from train set 

        def find_imputation_replacement(self):
            for variable in self.numerical_to_imput:
                replacement = self.X_train[variable].median()
                self.imputing_dict[variable] = replacement
            return self 


        def find_frequent_categories(self):
            for variable in self.categorical_encode:
                tmp = self.X_train.groupby(variable)[self.target].count() /len(self.X_train)
                self.frequent_category_dict[variable] = tmp[tmp > self.percentage].index
            return self 



        def find_categorical_mappings(self):
            for variable in self.categorical_encode:
                ordered_labels = self.X_train.groupby(variable)[self.target].sum().sort_values().index 
                ordinal_labels = {k: i for i, k in enumerate(ordered_labels, 0)}
                self.encoding_dict[variable] = ordinal_labels
            return self 


        # functions to transform data
        def remove_rare_labels(self, df):
            df = df.copy()
            for variable in self.categorical_encode:
                df[variable] = np.where(df[variable].isin(self.frequent_category_dict[variable]), df[variable], 'Rare')
            return df

        def encode_categorical_variables(self, df): 
            df = df.copy()
            for variable in self.categorical_encode:
                df[variable] = df[variable].map(self.encoding_dict[variable])
            return df


        # master function 
        def fit(self, data):
            self.X_train, self.X_test, self.y_train, self.y_test =
            train_test_split(data, data[self.target], test_size=self.test_size, random_state=self.random_state)

            # find imputation parameters
            self.find_imputation_replacement()

            # imput missing values
            # numerical
            self.X_train[self.numerical_to_imput] = 
            self.X_train[self.numerical_to_imput].fillna(self.imputing_dict[self.numerical_to_imput])

            self.X_test[self.numerical_to_imput] = 
            self.X_test[self.numerical_to_imput].fillna(self.imputing_dict[self.numerical_to_imput])
            
            # categorical
            self.X_train[self.categorical_to_imput] = 
            self.X_train[self.categorical_to_imput].fillna('Missing')

            self.X_test[self.categorical_to_imput] = 
            self.X_test[self.categorical_to_imput].fillna('Missing')

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
            self.x_test = self.encode_categorical_variables(self.x_test)    

            #  train scaler
            self.scaler.fit(self.X_train[self.features])


            # scale variables
            self.X_train = self.scaler.transform(self.X_train[self.features])
            self.X_test = self.scaler.transform(self.X_test[self.features])
            print(self.X_train.shape, self.X_test)

            # train model
            self.model.fit(self.X_train, self.y_train)

            return self











             



























                 









































































 



































































































































































































     






















































































 




























































































































































































































































































































































 