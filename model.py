import argparse

import numpy as np
import pandas as pd
from joblib import dump
from xgboost import XGBClassifier
from data_loader import load_data
from test import test_model
from utils import grid_search_tuning, sequential_backward_selection


def train_and_test(train_path, test_path, model_save_path, output_dir):
    print("Starting XGBoost model training...")

    X_train, X_test, y_train, y_test = load_data(train_path, test_path)

    xgboost = XGBClassifier(objective='binary:logistic', random_state=42)

    X_train_feature_indices, X_train_selected, y_train = sequential_backward_selection(X_train, y_train, xgboost)

    X_train_selected_df = pd.DataFrame(X_train_selected, columns=X_train.columns[X_train_feature_indices])
    y_train_df = pd.DataFrame(y_train, columns=['Label'])
    X_train_selected_with_labels_df = pd.concat([X_train_selected_df, y_train_df], axis=1)
    X_train_selected_with_labels_df.to_csv(f"{output_dir}/XGBoost_train_selected_label.csv", index=False)

    X_test_selected = X_test.iloc[:, X_train_feature_indices]
    y_test_df = pd.DataFrame(y_test, columns=['Label'])
    X_test_selected_with_labels_df = pd.concat([X_test_selected, y_test_df], axis=1)
    X_test_selected_with_labels_df.to_csv(f"{output_dir}/XGBoost_test_selected_label.csv", index=False)

    param_grid = {
        'n_estimators': np.random.randint(50, 500, 5),
        'max_depth': np.random.randint(3, 10, 1),
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    }

    best_grid_search_xgboost = grid_search_tuning(xgboost, param_grid, X_train_selected, y_train)
    best_model_xgboost = best_grid_search_xgboost.best_estimator_

    print(f"Model saved to {model_save_path}")
    dump(best_model_xgboost, './model_hub/XGBoost_model.joblib')
    
    print(f"Start model test")
    test_model(best_model_xgboost, X_test_selected.values, y_test)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate XGBoost model with feature selection and tuning.")
    parser.add_argument('--train', type=str, required=True, help="Path to training feature CSV file.")
    parser.add_argument('--test', type=str, required=True, help="Path to test feature CSV file.")
    parser.add_argument('--model_out', type=str, default='./model_hub/XGBoost_model.joblib', help="Path to save trained model.")
    parser.add_argument('--output_dir', type=str, default='./data_hub', help="Directory to save selected features and results.")

    args = parser.parse_args()

    train_and_test(args.train, args.test, args.model_out, args.output_dir)
