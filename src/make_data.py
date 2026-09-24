import os
from sklearn.datasets import load_iris
import pandas as pd

def main():
    os.makedirs("data", exist_ok=True)
    iris = load_iris(as_frame=True)
    iris.frame.to_csv("data/iris_raw.csv", index=False)
    print("[OK] data/iris_raw.csv généré")

if __name__ == "__main__":
    main()
