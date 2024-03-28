import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

GREEN = '\033[92m'
BLUE = '\033[94m'
RESET = '\033[0m'

train_data_file_path = "data/loan_old.csv"
predict_data_file_path = "data/loan_new.csv"
# test_data_file_path = "data/Data - Copy.csv"
test_data_file_path = "data/Data.csv"

data_set = pd.read_csv(train_data_file_path)

all_features: pd.DataFrame
all_targets: pd.DataFrame
features_train: pd.DataFrame
features_test: pd.DataFrame
targets_train: pd.DataFrame
targets_test: pd.DataFrame


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

    print("\nData after encoded...\n")
    print("------------------ Targets Train------------------")
    print(targets_train)
    print("------------------ Targets Test------------------")
    print(targets_test)


def encode_special_columns(column_name):
    global features_train, features_test
    encoder = OneHotEncoder(sparse_output=False)

    property_area_encoder_in_train_data = encoder.fit_transform(features_train[[column_name]])
    property_area_encoder_in_test_data = encoder.fit_transform(features_test[[column_name]])

    encoded_col_names = encoder.get_feature_names_out([column_name])

    encoded_col_in_train_data = pd.DataFrame(property_area_encoder_in_train_data, columns=encoded_col_names)
    encoded_col_in_test_data = pd.DataFrame(property_area_encoder_in_test_data, columns=encoded_col_names)

    features_train = pd.concat([features_train.drop(columns=[column_name]), encoded_col_in_train_data], axis=1)
    features_test = pd.concat([features_test.drop(columns=[column_name]), encoded_col_in_test_data], axis=1)
    return


def encode_features():
    global features_train, features_test
    print("\nFeatures before encoded...\n")
    print("\n------------------ Features Train------------------")
    print(features_train)
    print("\n------------------ Features Test------------------")
    print(features_test)

    train_feature_encoded = LabelEncoder().fit_transform(features_train['Gender'])
    test_feature_encoded = LabelEncoder().fit_transform(features_test['Gender'])

    features_train['Gender'] = train_feature_encoded
    features_test['Gender'] = test_feature_encoded

    train_feature_encoded = LabelEncoder().fit_transform(features_train['Education'])
    test_feature_encoded = LabelEncoder().fit_transform(features_test['Education'])

    features_train['Education'] = train_feature_encoded
    features_test['Education'] = test_feature_encoded

    train_feature_encoded = LabelEncoder().fit_transform(features_train['Married'])
    test_feature_encoded = LabelEncoder().fit_transform(features_test['Married'])

    features_train['Married'] = train_feature_encoded
    features_test['Married'] = test_feature_encoded

    encode_special_columns('Property_Area')
    encode_special_columns('Dependents')

    print("\nFeatures after encoded...\n")
    print("------------------ Features Train------------------")
    print(features_train)
    print("------------------ Features Test------------------")
    print(features_test)


def standardize_numerical_features():
    global features_train, features_test
    print("\nFeatures before standardized...\n")
    print("\n------------------ Features Train------------------")
    print(features_train["Income"])
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


def remove_rows_have_missing_values():
    global data_set
    print("\nData before remove records containing missing values")
    print(data_set)
    data_set = data_set.dropna()
    print("\nData after remove records containing missing values")
    print(data_set)
    time.sleep(1)


def split_data_to_train_and_test():
    global features_train, features_test, targets_train, targets_test
    features_train, features_test, targets_train, targets_test = train_test_split(all_features, all_targets,
                                                                                  train_size=0.8,
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
    for column in data_set.columns:
        col_type = data_set[column].dtype
        if pd.api.types.is_numeric_dtype(col_type):
            print(f"'{column}': numerical.")
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

    #  records containing missing values are removed
    print(GREEN + "1- Records containing missing values are removed..." + RESET)
    time.sleep(2)
    remove_rows_have_missing_values()

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
    print(GREEN + "\n6 - the data is shuffled and split into training and testing sets..." + RESET)
    time.sleep(2)
    standardize_numerical_features()
    with pd.ExcelWriter("output/features_train.xlsx") as writer:
        features_train.to_excel(writer, index=False)
    with pd.ExcelWriter("output/features_test.xlsx") as writer:
        features_test.to_excel(writer, index=False)
    with pd.ExcelWriter("output/targets_train.xlsx") as writer:
        targets_train.to_excel(writer, index=False)
    with pd.ExcelWriter("output/targets_test.xlsx") as writer:
        targets_test.to_excel(writer, index=False)


def separate_features_targets():
    global all_targets, all_features
    all_features = data_set.iloc[:, :-2]
    all_targets = data_set.iloc[:, -2:]

    all_features.drop(columns=['Loan_ID'], inplace=True)


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
print(BLUE + "\nStep 3: Evaluate the linear regression model using sklearn R2 score." + RESET)
print("----------------------------------------------------")
