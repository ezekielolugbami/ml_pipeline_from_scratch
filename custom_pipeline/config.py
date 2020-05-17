path_to_dataset = '../titanic.csv'

target = 'Survived'

categorical_to_impute = ['Cabin', 'Embarked']
numerical_to_impute = ['Age']

# variables to transform
numerical_log = ['Age', 'Fare']


# variables to encode
categorical_encode = ['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked']


# slected features for training
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Ticket', 'Cabin', 'Embarked', 'Age_na']