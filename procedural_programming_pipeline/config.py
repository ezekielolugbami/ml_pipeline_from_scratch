# ====== Paths =========================

path_to_dataset = 'titanic.csv'
output_scaler_path = 'scaler.pkl'
output_model_path = 'logistic_regression.pkl'

# ========= parameters ====================

# imputation parameters
median_age = 28.5

# encoding parameters
frequent_labels = {
    'Name': [],
    'Sex' : ['female', 'male'],
    'Ticket' : ['1601', '3101295', '347082', 'CA 2144', 
    'CA. 2343'],
    'Cabin' : ['Missing'],
    'Embarked' : ['C', 'Q', 'S']
}

encoding_mappings = {
    'Name' : {'Rare': 0},
    'Sex' : {'male': 0, 'female': 1},
    'Ticket' : {'3101295': 0, '347082': 1, 'CA 2144': 2, 'CA. 2343': 3, '1601': 4, 'Rare': 5},
    'Cabin' : {'Rare': 0, 'Missing': 1},
    'Embarked' : {'Rare': 0, 'Q': 1, 'C': 2, 'S': 3}
}

# ========== feature groups =====================
# variable groups for engineering steps
target = 'Survived'

categorical_to_imput = ['Cabin', 'Embarked']
numerical_to_imput = 'Age'

# variables to transform
numerical_log = ['Age', 'Fare']

# variables to encode
categorical_encode = ['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked']

# slected features for training
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Ticket', 'Cabin', 'Embarked', 'Age_na']