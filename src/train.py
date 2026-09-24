import os
import json
import yaml
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

def main():
    with open("params.yaml", "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    model_params = params["model"].copy()
    model_params.pop("type", None)

    X_train = pd.read_csv("data/processed/X_train.csv")
    y_train = pd.read_csv("data/processed/y_train.csv")["target"].values

    clf = RandomForestClassifier(**model_params)
    clf.fit(X_train, y_train)

    os.makedirs("models", exist_ok=True)
    joblib.dump(clf, "models/model.pkl")

    y_pred = clf.predict(X_train)
    metrics = {
        "train_accuracy": float(accuracy_score(y_train, y_pred)),
        "train_f1_macro": float(f1_score(y_train, y_pred, average="macro")),
    }
    with open("metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("[OK] train terminé — models/model.pkl + metrics.json")

if __name__ == "__main__":
    main()
