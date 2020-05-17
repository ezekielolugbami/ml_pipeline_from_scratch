#data
training_data_file = '../titanic.csv'
pipeline_name = 'logistic_regression'

target = 'Survived'


# input variables
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Ticket', 'Cabin', 'Embarked', 'Age_na']

# numerical variables with na in train set
numerical_vars_with_na = ['Age']

# categorical variables with na in train set
categorical_vars_with_na = ['Cabin', 'Embarked']

# variables to log transform
numerical_log_vars = ['Age', 'Fare']

# categorical variables to encode
categorical_vars = ['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked']