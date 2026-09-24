"""Envoi des résultats du pipeline DVC vers MLflow (serveur DagsHub).

Le suivi est activé seulement si MLFLOW_TRACKING_URI est défini
(dans le CI GitHub Actions). Sinon, le pipeline fonctionne normalement.
"""
import json
import os

import yaml


def _flatten(d, prefix=""):
    out = {}
    for key, value in d.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(_flatten(value, name + "."))
        else:
            out[name] = value
    return out


def _read_json(path):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def log_to_mlflow():
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        print("[INFO] MLFLOW_TRACKING_URI non défini : suivi MLflow ignoré")
        return

    try:
        import mlflow

        with open("params.yaml", encoding="utf-8") as f:
            params = _flatten(yaml.safe_load(f))
        metrics = {**_read_json("metrics.json"), **_read_json("evaluation.json")}

        mlflow.set_experiment("iris-dvc-pipeline")
        run_name = (os.environ.get("GITHUB_HEAD_REF")
                    or os.environ.get("GITHUB_REF_NAME") or "local")

        with mlflow.start_run(run_name=run_name):
            mlflow.log_params({k: str(v) for k, v in params.items()})
            mlflow.log_metrics({k: float(v) for k, v in metrics.items()
                                if isinstance(v, (int, float))})
            mlflow.set_tags({
                "git_commit": os.environ.get("GITHUB_SHA", "local"),
                "pipeline": "dvc repro",
            })
            for path in ["metrics.json", "evaluation.json",
                         "reports/classification_report.json", "models/model.pkl"]:
                if os.path.exists(path):
                    mlflow.log_artifact(path)
            if os.path.isdir("plots"):
                mlflow.log_artifacts("plots", artifact_path="plots")

        print("[OK] run MLflow enregistré sur", os.environ["MLFLOW_TRACKING_URI"])
    except Exception as e:
        print("[AVERTISSEMENT] échec de l'envoi MLflow :", e)