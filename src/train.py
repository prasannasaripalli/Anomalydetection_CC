from pathlib import Path
import pickle
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

from src.utils import FEATURES

DATA_PATH = Path("data/raw/creditcard.csv")  # change if your file is elsewhere
MODEL_OUT = Path("models/isolation_forest_pipeline.pkl")
CONTAMINATION = 0.005


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES]
    y = df["Class"]

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    X_train_normal = X_train[y_train == 0]
    if len(X_train_normal) == 0:
        raise ValueError("No normal rows (Class=0) found.")

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", IsolationForest(
            n_estimators=300,
            contamination=CONTAMINATION,
            random_state=42,
            n_jobs=-1
        ))
    ])

    model.fit(X_train_normal)

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_OUT, "wb") as f:
        pickle.dump(model, f)

    print(f"Saved: {MODEL_OUT}")


if __name__ == "__main__":
    main()