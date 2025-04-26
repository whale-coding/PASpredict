import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler


def split_features_and_labels(dataset, label_col='Label'):
    X = dataset.drop(columns=[label_col])
    y = dataset[label_col]
    return X, y


def preprocess_features(X, impute=True, scale=True):
    if impute:
        imputer = KNNImputer(n_neighbors=5)
        X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    if scale:
        scaler = MinMaxScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    return X


def preprocess_data(dataset_path, label_col='Label'):
    dataset = pd.read_csv(dataset_path)

    dataset = dataset.sample(frac=1, random_state=2025).reset_index(drop=True)

    X, y = split_features_and_labels(dataset, label_col)
    X_processed = preprocess_features(X)
    return X_processed, y


def load_data(train_file_path, test_file_path, label_col='Label'):
    X_train, y_train = preprocess_data(train_file_path, label_col)
    X_test, y_test = preprocess_data(test_file_path, label_col)
    print("data loading and processing have been completed")
    return X_train, X_test, y_train, y_test
