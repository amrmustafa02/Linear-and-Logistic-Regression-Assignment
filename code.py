import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np
import os

GREEN = '\033[92m'
BLUE = '\033[94m'
RESET = '\033[0m'

# -------------------------------
#  importing data set
train_data_file_path = "loan_old.csv"
predict_data_file_path = "loan_new.csv"

data_set = pd.read_csv(train_data_file_path)
predict_data_set = pd.read_csv(predict_data_file_path)

os.makedirs("./output", exist_ok=True)

# -------------------------------
#  features and targets
all_features: pd.DataFrame
all_targets: pd.DataFrame
features_train: pd.DataFrame
features_test: pd.DataFrame
targets_train: pd.DataFrame
targets_test: pd.DataFrame

# -------------------------------
#  linear regression model
model = LinearRegression()

# -------------------------------
#  logistic regression model
weights = []
bias = 0
learning_rate = 0.03
costs = []


def check_if_features_have_same_scale():
    numerical_columns = all_features.select_dtypes(include=['number'])

    is_normalized = (numerical_columns.min() >= 0).all() and (numerical_columns.max() <= 1).all()

    is_standardized = (numerical_columns.mean().abs() < 1e-10).all() and (
            numerical_columns.std().abs() - 1 < 1e-10).all()

    return is_normalized or is_standardized


def encode_targets():
    print("\nTargets before encoded...\n")

    print("\n------------------ Targets Train ------------------\n")
    print(targets_train)

    print("\n------------------ Targets Test ------------------\n")
    print(targets_test)

    train_target_encoded = LabelEncoder().fit_transform(targets_train['Loan_Status'])
    test_target_encoded = LabelEncoder().fit_transform(targets_test['Loan_Status'])

    targets_train['Loan_Status'] = train_target_encoded
    targets_test['Loan_Status'] = test_target_encoded

    print("\nTargets after encoded...\n")

    print("------------------ Targets Train------------------")
    print(targets_train)

    print("------------------ Targets Test------------------")
    print(targets_test)


def encode_categories_columns(column_name):
    global features_train, features_test

    encoder = OneHotEncoder(sparse_output=False)

    property_area_encoder_in_train_data = encoder.fit_transform(features_train[[column_name]])
    property_area_encoder_in_test_data = encoder.fit_transform(features_test[[column_name]])

    encoded_col_names = encoder.get_feature_names_out([column_name])

    encoded_col_in_train_data = pd.DataFrame(property_area_encoder_in_train_data, columns=encoded_col_names)
    encoded_col_in_test_data = pd.DataFrame(property_area_encoder_in_test_data, columns=encoded_col_names)

    features_train = pd.concat([features_train.drop(columns=[column_name]), encoded_col_in_train_data], axis=1)
    features_test = pd.concat([features_test.drop(columns=[column_name]), encoded_col_in_test_data], axis=1)


def encode_features():
    global features_train, features_test
    print("\nFeatures before encoded...\n")

    print("\n------------------ Features Train------------------")
    print(features_train)

    print("\n------------------ Features Test------------------")
    print(features_test)

    features_train['Gender'] = LabelEncoder().fit_transform(features_train['Gender'])
    features_test['Gender'] = LabelEncoder().fit_transform(features_test['Gender'])

    features_train['Education'] = LabelEncoder().fit_transform(features_train['Education'])
    features_test['Education'] = LabelEncoder().fit_transform(features_test['Education'])

    features_train['Married'] = LabelEncoder().fit_transform(features_train['Married'])
    features_test['Married'] = LabelEncoder().fit_transform(features_test['Married'])

    encode_categories_columns('Property_Area')
    encode_categories_columns('Dependents')

    print("\nFeatures after encoded...\n")

    print("------------------ Features Train------------------")
    print(features_train)

    print("------------------ Features Test------------------")
    print(features_test)


def encode_predict_data_categories_columns(column_name):
    global predict_data_set
    encoder = OneHotEncoder(sparse_output=False)

    property_area_encoder_in_predict_data = encoder.fit_transform(predict_data_set[[column_name]])

    encoded_col_names = encoder.get_feature_names_out([column_name])

    encoded_col_in_predict_data = pd.DataFrame(property_area_encoder_in_predict_data, columns=encoded_col_names)

    predict_data_set = pd.concat([predict_data_set.drop(columns=[column_name]), encoded_col_in_predict_data], axis=1)
    return


def encode_predict_categorical_features():
    global predict_data_set
    predict_data_set['Gender'] = LabelEncoder().fit_transform(predict_data_set['Gender'])
    predict_data_set['Education'] = LabelEncoder().fit_transform(predict_data_set['Education'])
    predict_data_set['Married'] = LabelEncoder().fit_transform(predict_data_set['Married'])
    encode_predict_data_categories_columns('Property_Area')
    encode_predict_data_categories_columns('Dependents')


def standardize_numerical_features():
    global features_train, features_test

    print("\nFeatures before standardized...\n")

    print("\n------------------ Features Train------------------")
    print(features_train)

    print("\n------------------ Features Test------------------")
    print(features_test)

    sc = StandardScaler()

    features_train["Income"] = sc.fit_transform(features_train[["Income"]])
    features_test["Income"] = sc.fit_transform(features_test[["Income"]])

    features_train["Coapplicant_Income"] = sc.fit_transform(features_train[["Coapplicant_Income"]])
    features_test["Coapplicant_Income"] = sc.fit_transform(features_test[["Coapplicant_Income"]])

    features_train["Loan_Tenor"] = sc.fit_transform(features_train[["Loan_Tenor"]])
    features_test["Loan_Tenor"] = sc.fit_transform(features_test[["Loan_Tenor"]])

    print("\nFeatures after standardized...\n")

    print("------------------ Features Train------------------")
    print(features_train)

    print("------------------ Features Test------------------")
    print(features_test)


def standardize_predict_data_numerical_features():
    global predict_data_set
    sc = StandardScaler()
    predict_data_set["Income"] = sc.fit_transform(predict_data_set[["Income"]])
    predict_data_set["Coapplicant_Income"] = sc.fit_transform(predict_data_set[["Coapplicant_Income"]])
    predict_data_set["Loan_Tenor"] = sc.fit_transform(predict_data_set[["Loan_Tenor"]])


def remove_rows_have_missing_values(_data_set):
    _data_set = _data_set.dropna()
    return _data_set


def split_data_to_train_and_test():
    global features_train, features_test, targets_train, targets_test

    features_train, features_test, targets_train, targets_test = train_test_split(all_features, all_targets,
                                                                                  train_size=0.7,
                                                                                  random_state=1)

    features_train.reset_index(drop=True, inplace=True)
    features_test.reset_index(drop=True, inplace=True)
    targets_train.reset_index(drop=True, inplace=True)
    targets_test.reset_index(drop=True, inplace=True)

    print("------------------ Features Train------------------")
    print(features_train)

    print("------------------ Features Test------------------")
    print(features_test)

    print("------------------ Targets Train------------------")
    print(targets_train)

    print("------------------ Targets Test------------------")
    print(targets_test)


def perform_analysis():
    # ------------------------------------------
    # check whether there are missing values
    # ------------------------------------------
    print(GREEN + "1 - Check whether there are missing values..." + RESET)
    time.sleep(2)

    if data_set.isnull().values.any():
        print("There are missing values in the dataset.")
        print(data_set.isnull().sum())
    else:
        print("There are no missing values in the dataset.")

    # ------------------------------------------
    # check the type of each feature (categorical or numerical)
    # ------------------------------------------
    print(GREEN + "\n2 - Check the type of each feature (categorical or numerical)..." + RESET)
    time.sleep(2)

    for column in all_features.columns:
        col_type = all_features[column].dtype
        if pd.api.types.is_numeric_dtype(col_type):
            print(f"'{column}': numerical.")
        else:
            print(f"'{column}': categorical.")

    # ------------------------------------------
    #  check whether numerical features have the same scale
    # ------------------------------------------
    print(GREEN + "\n3 - Check whether numerical features have the same scale..." + RESET)
    time.sleep(2)
    features_are_scaled = check_if_features_have_same_scale()

    if features_are_scaled:
        print("Features have same scale")
    else:
        print("Features not have same scale")

    # ------------------------------------------
    #  visualize a pairplot between numercial columns
    # ------------------------------------------
    print(GREEN + "\n4 - visualize a pairplot between numercial columns..." + RESET)
    time.sleep(2)

    numercial_columns = data_set.select_dtypes(include="number")
    sns.pairplot(numercial_columns)
    plt.show()


def data_preprocessing():
    global data_set, all_features, all_targets, features_train, features_test, targets_train, targets_test

    #  records containing missing values are removed
    print(GREEN + "1- Records containing missing values are removed..." + RESET)
    time.sleep(2)

    print("------------------ Dataset before drop missing values ------------------")
    print(data_set)

    data_set.dropna(inplace=True)
    data_set.reset_index(drop=True, inplace=True)

    print("------------------ Dataset after drop missing values ------------------")
    print(data_set)

    #  The features and targets are separated
    print(GREEN + "\n2 - Separate features and targets..." + RESET)
    time.sleep(2)

    separate_features_targets()

    print("------------------ Features ------------------")
    print(all_features)

    print("------------------ Targets ------------------")
    print(all_targets)

    #  the data is shuffled and split into training and testing sets
    print(GREEN + "\n3 - the data is shuffled and split into training and testing sets..." + RESET)
    time.sleep(2)

    split_data_to_train_and_test()

    #  categorical features are encoded
    print(GREEN + "\n4 -  categorical features are encoded.." + RESET)
    time.sleep(2)

    encode_features()

    #  categorical targets are encoded
    print(GREEN + "\n5 - categorical targets are encoded.." + RESET)
    time.sleep(2)

    encode_targets()

    #  numerical features are standardized
    print(GREEN + "\n6 - numerical features are standardized..." + RESET)
    time.sleep(2)

    standardize_numerical_features()

    # write new files have data preprocessed
    with pd.ExcelWriter("output/features_train.xlsx") as writer:
        features_train.to_excel(writer, index=False)

    with pd.ExcelWriter("output/features_test.xlsx") as writer:
        features_test.to_excel(writer, index=False)

    with pd.ExcelWriter("output/targets_train.xlsx") as writer:
        targets_train.to_excel(writer, index=False)

    with pd.ExcelWriter("output/targets_test.xlsx") as writer:
        targets_test.to_excel(writer, index=False)


def linear_regression():
    global model

    loan_amount_train = targets_train['Max_Loan_Amount']
    loan_amount_test = targets_test['Max_Loan_Amount']

    model = LinearRegression()
    model.fit(features_train, loan_amount_train)

    loan_amount_predicted = model.predict(features_test)

    r2_score_value = r2_score(loan_amount_test, loan_amount_predicted)

    print(f"R2 : {r2_score_value}")


def separate_features_targets():
    global all_targets, all_features
    all_features = data_set.iloc[:, :-2]
    all_targets = data_set.iloc[:, -2:]

    all_features.drop(columns=['Loan_ID'], inplace=True)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def calculate_hypothesis(features):
    h = np.dot(features, weights) + bias
    return h


def calculate_cost(m, h):
    cost = np.sum(- targets_train["Loan_Status"] * np.log(h) - (1 - targets_train["Loan_Status"]) * np.log(1 - h)) / m
    return cost


def logistic_regression(iterations):
    global weights, bias

    samples_number = features_train.shape[0],
    features_number = features_train.shape[1]

    weights = np.zeros(features_number)

    for i in range(iterations):
        #  step 1 - > linear equation
        h = calculate_hypothesis(features_train)

        #  step 2 -> sigmoid
        predicted = sigmoid(h)

        #  step 3 -> gradient descent
        gradient_weights = np.dot(features_train.T, predicted - targets_train["Loan_Status"]) / samples_number
        gradient_bias = np.sum(predicted - targets_train["Loan_Status"]) / samples_number

        #  step 4 -> update weights and bias
        weights -= gradient_weights * learning_rate
        bias -= gradient_bias * learning_rate

    print(f"weights\n {weights}")
    print(f"bias\n {bias}")


def calculate_accuracy():
    global features_test

    h = calculate_hypothesis(features_test)

    predicted = sigmoid(h)

    predicted = np.where(predicted > 0.5, 1, 0)

    accuracy = np.sum(predicted == targets_test["Loan_Status"]) / len(targets_test["Loan_Status"])

    return accuracy


def predict_data_preprocessing():
    global predict_data_set

    predict_data_set = remove_rows_have_missing_values(predict_data_set)

    predict_data_set.reset_index(drop=True, inplace=True)

    standardize_predict_data_numerical_features()

    encode_predict_categorical_features()


def new_predictions():
    global predict_data_set

    predict_data_preprocessing()

    predict_data_set.drop(columns=['Loan_ID'], inplace=True)

    with pd.ExcelWriter("output/predictions_data_preprocessed.xlsx") as writer:
        predict_data_set.to_excel(writer, index=False)

    amount_predictions = model.predict(predict_data_set)

    h = calculate_hypothesis(predict_data_set)

    predicted_probabilities = sigmoid(h)

    loan_status_predictions = np.where(predicted_probabilities > 0.5, 1, 0)

    predictions_df = pd.DataFrame(
        {'Loan_Status': loan_status_predictions, "Max_Loan_Amount": amount_predictions})

    print("\n Predictions \n")
    print(predictions_df)

    with pd.ExcelWriter("output/new_predictions.xlsx") as writer:
        predictions_df.to_excel(writer, index=False)


# main
separate_features_targets()

print(BLUE + "Step 1: Perform analysis" + RESET)
print("----------------------------------------------------")
perform_analysis()

print("----------------------------------------------------")
print(BLUE + "\nStep 2: Data Preprocessing" + RESET)
print("----------------------------------------------------")
data_preprocessing()

print("----------------------------------------------------")
print(BLUE + "\nStep 3: Evaluate the linear regression model using sklearn R2 score." + RESET)
print("----------------------------------------------------")
linear_regression()

print("----------------------------------------------------")
print(BLUE + "\nStep 4: Logistic regression (gradient descent)." + RESET)
print("----------------------------------------------------")
logistic_regression(5000)

print("----------------------------------------------------")
print(BLUE + "\nStep 5: Accuracy." + RESET)
print("----------------------------------------------------")
print(f"Accuracy:  {calculate_accuracy() * 100}")

print("----------------------------------------------------")
print(BLUE + "\nStep 6: New predictions." + RESET)
print("----------------------------------------------------")
new_predictions()
