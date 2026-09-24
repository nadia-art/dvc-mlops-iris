import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, classification_report,
)
from sklearn.preprocessing import label_binarize

CLASSES = ["setosa", "versicolor", "virginica"]


def main():
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_test = pd.read_csv("data/processed/y_test.csv")["target"].values
    model = joblib.load("models/model.pkl")

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    # ---- 1. Métriques détaillées ----
    evaluation = {
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "test_precision_macro": float(precision_score(y_test, y_pred, average="macro")),
        "test_recall_macro": float(recall_score(y_test, y_pred, average="macro")),
        "test_f1_macro": float(f1_score(y_test, y_pred, average="macro")),
        "test_roc_auc_ovr": float(roc_auc_score(y_test, y_proba, multi_class="ovr")),
    }
    with open("evaluation.json", "w", encoding="utf-8") as f:
        json.dump(evaluation, f, indent=2)

    os.makedirs("reports", exist_ok=True)
    report = classification_report(y_test, y_pred, target_names=CLASSES, output_dict=True)
    with open("reports/classification_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # ---- 2. Données des plots au format JSON ----
    os.makedirs("plots", exist_ok=True)

    # Matrice de confusion : une ligne par prédiction
    confusion = [
        {"actual": CLASSES[a], "predicted": CLASSES[p]}
        for a, p in zip(y_test, y_pred)
    ]
    with open("plots/confusion_matrix.json", "w", encoding="utf-8") as f:
        json.dump(confusion, f, indent=2)

    # Importance des variables
    importance = sorted(
        [
            {"feature": name, "importance": float(v)}
            for name, v in zip(X_test.columns, model.feature_importances_)
        ],
        key=lambda d: d["importance"],
        reverse=True,
    )
    with open("plots/feature_importance.json", "w", encoding="utf-8") as f:
        json.dump(importance, f, indent=2)

    # Courbe ROC (moyenne micro, one-vs-rest)
    y_bin = label_binarize(y_test, classes=[0, 1, 2])
    fpr, tpr, _ = roc_curve(y_bin.ravel(), y_proba.ravel())
    roc = [{"fpr": float(a), "tpr": float(b)} for a, b in zip(fpr, tpr)]
    with open("plots/roc_curve.json", "w", encoding="utf-8") as f:
        json.dump(roc, f, indent=2)

    print("[OK] evaluate terminé — evaluation.json + plots/*.json")


if __name__ == "__main__":
    main()
    from mlflow_utils import log_to_mlflow
    log_to_mlflow()
