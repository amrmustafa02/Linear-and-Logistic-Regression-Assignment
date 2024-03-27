import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

GREEN = '\033[92m'
BLUE = '\033[94m'
RESET = '\033[0m'

train_data_file_path = "data/loan_old.csv"
predict_data_file_path = "data/loan_new.csv"
# test_data_file_path = "data/Data - Copy.csv"
test_data_file_path = "data/Data.csv"

data_set = pd.read_csv(test_data_file_path)

all_features = []
all_targets = []
features_train = []
features_test = []
targets_train = []
targets_test = []


def check_if_features_have_same_scale():
    numerical_columns = all_features.select_dtypes(include=['number'])
    is_normalized = (numerical_columns.min() >= 0).all() and (numerical_columns.max() <= 1).all()

    is_standardized = (numerical_columns.mean().abs() < 1e-10).all() and (
            numerical_columns.std().abs() - 1 < 1e-10).all()

    return is_normalized or is_standardized


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
    for column in data_set.columns:
        col_type = data_set[column].dtype
        if pd.api.types.is_numeric_dtype(col_type):
            print(f"Feature '{column}': numerical.")
        else:
            print(f"Feature '{column}': categorical.")
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

    # ------------------------------------------
    #  records containing missing values are removed
    # ------------------------------------------
    print(GREEN + "1- Records containing missing values are removed..." + RESET)
    print("\nData before remove records containing missing values")
    print(data_set)
    time.sleep(2)
    data_set = data_set.dropna()
    print("\nData after remove records containing missing values")
    print(data_set)
    time.sleep(1)
    # ------------------------------------------
    #  The features and targets are separated
    # ------------------------------------------
    print(GREEN + "\n2 - Separate features and targets..." + RESET)
    time.sleep(2)
    separate_features_targets()
    print("------------------ Features ------------------")
    print(all_features)
    print("------------------ Targets ------------------")
    print(all_targets)
    # ------------------------------------------
    #  the data is shuffled and split into training and testing sets
    # ------------------------------------------
    print(GREEN + "\n3 - the data is shuffled and split into training and testing sets..." + RESET)
    time.sleep(2)
    features_train, features_test, targets_train, targets_test = train_test_split(all_features, all_targets,
                                                                                  train_size=0.8,
                                                                                  random_state=1)
    print("------------------ Features Train------------------")
    print(features_train)
    print("------------------ Features Test------------------")
    print(features_test)
    print("------------------ Targets Train------------------")
    print(targets_train)
    print("------------------ Targets Test------------------")
    print(targets_test)
    # ------------------------------------------
    #  categorical features are encoded
    # ------------------------------------------
    print(GREEN + "\n4 - the data is shuffled and split into training and testing sets..." + RESET)

    # ------------------------------------------
    #  categorical targets are encoded
    # ------------------------------------------
    print(GREEN + "\n5 - the data is shuffled and split into training and testing sets..." + RESET)

    # ------------------------------------------
    #  numerical features are standardized
    # ------------------------------------------
    print(GREEN + "\n6 - the data is shuffled and split into training and testing sets..." + RESET)


def separate_features_targets():
    global all_targets, all_features
    all_features = data_set.iloc[:, :-1]
    all_targets = data_set.iloc[:, -1:]


separate_features_targets()
print(BLUE + "Step 1: Perform analysis" + RESET)
print("----------------------------------------------------")
perform_analysis()
print("----------------------------------------------------")
time.sleep(1)
print(BLUE + "\nStep 2: Data Preprocessing" + RESET)
print("----------------------------------------------------")
data_preprocessing()
print("----------------------------------------------------")

# print(all_features)
# ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder="passthrough")
# X = np.array(ct.fit_transform(all_features))
# print(X)
