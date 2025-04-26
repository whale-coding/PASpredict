import pandas as pd
import argparse
from joblib import load
from data_loader import preprocess_features


def evaluate(model, X_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    results_df = pd.DataFrame({
        'Predict_Label': y_pred,
        'Probability': y_proba
    })

    results_df.to_csv("model_pred_result.csv", sep=",", index=False)
    print("Prediction results saved to model_pred_result.csv")


def predict(model_path, data_path):
    print(f"Loading model from {model_path}...")
    model = load(model_path)

    print(f"Reading data from {data_path}...")
    X = pd.read_csv(data_path)
    X = preprocess_features(X)

    print("Start to predict...")
    evaluate(model, X.values)
    print("End to predict...")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Predict PAS using a trained model.")
    parser.add_argument('--model', type=str, required=True,
                        help="Path to the trained model (e.g., ./model/PASpredict.joblib)")
    parser.add_argument('--data', type=str, required=True,
                        help="Path to the input feature CSV file (e.g., feature.csv)")

    args = parser.parse_args()

    predict(args.model, args.data)
