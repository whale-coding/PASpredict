import argparse

import pandas as pd
from joblib import load
from data_loader import preprocess_features
from utils import calculate_metrics, format_metrics


def test_model(best_model, X_test, y_test):
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    results_df = pd.DataFrame({
        'True_Label': y_test,
        'Predict_Label': y_pred,
        'Probability': y_proba
    })

    results_df.to_csv("model_results.txt", sep="\t", index=False)

    Recall, Specificity, Precision, F1, MCC, Acc, AUC, AUPR = calculate_metrics(y_pred, y_proba, y_test)

    print(format_metrics(Recall, Specificity, Precision, F1, MCC, Acc, AUC, AUPR))

    metrics_df = pd.DataFrame({
        'Recall': [Recall],
        'Specificity': [Specificity],
        'Precision': [Precision],
        'F1': [F1],
        'MCC': [MCC],
        'Accuracy': [Acc],
        'AUC': [AUC],
        'AUPR': [AUPR]
    })
    metrics_df.to_csv("metrics.txt", sep="\t", index=False, float_format="%.3f")


def test(model_path, data_path):
    print(f"Loading model from {model_path}...")
    model = load(model_path)
    print(f"Reading data from {data_path}...")
    test_data = pd.read_csv(data_path)

    X_test = test_data.iloc[:, :-1]
    y_test = test_data['Label']
    X_test = preprocess_features(X_test)

    test_model(model, X_test.values, y_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate a trained PAS prediction model.")
    parser.add_argument('--model', type=str, required=True, help="Path to the trained model file (e.g., ./model/PASpredict.joblib)")
    parser.add_argument('--data', type=str, required=True, help="Path to the test data CSV file (with labels)")

    args = parser.parse_args()

    test(args.model, args.data)
