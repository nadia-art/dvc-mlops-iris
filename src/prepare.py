import os
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def main():
    with open("params.yaml", "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    df = pd.read_csv("data/iris_raw.csv")
    X = df.drop(columns=["target"])
    y = df["target"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=float(params["data"]["test_size"]),
        random_state=int(params["data"]["random_state"]),
        stratify=y,
    )

    os.makedirs("data/processed", exist_ok=True)
    pd.DataFrame(X_train, columns=X.columns).to_csv("data/processed/X_train.csv", index=False)
    pd.DataFrame(X_test, columns=X.columns).to_csv("data/processed/X_test.csv", index=False)
    pd.DataFrame({"target": y_train}).to_csv("data/processed/y_train.csv", index=False)
    pd.DataFrame({"target": y_test}).to_csv("data/processed/y_test.csv", index=False)
    print("[OK] prepare terminé")

if __name__ == "__main__":
    main()
