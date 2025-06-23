import pandas as pd
import mlflow
import mlflow.sklearn
import argparse
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def main(data_path):
    # Load dataset
    df = pd.read_csv(data_path)

    # Pisahkan fitur dan label
    X = df.drop('HeartDisease', axis=1)
    y = df['HeartDisease']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Aktifkan autologging
    mlflow.sklearn.autolog()

    # MLflow tracking
    with mlflow.start_run():
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {acc}")

        # Save model artifact locally
        os.makedirs("artifacts/model", exist_ok=True)
        mlflow.sklearn.save_model(model, "artifacts/model")
        print("Model saved to artifacts/model")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", 
        type=str, 
        default="heart_failure_preprocessing/heart_processed.csv",
        help="Path ke dataset hasil preprocessing"
    )
    args = parser.parse_args()
    main(args.data_path)
