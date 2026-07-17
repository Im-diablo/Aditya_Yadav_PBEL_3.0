"""
src/data_preprocessing.py
Preprocessing pipeline for the ML (XGBoost) module.

Handles:
  - ULB Credit Card Fraud dataset loading
  - Null checking, feature scaling (Amount + Time)
  - Stratified train / val / test splitting
  - Synthetic data fallback when creditcard.csv is absent

Note: SMOTE is intentionally NOT used. With only 344 real fraud samples,
SMOTE generates 199k synthetic frauds (580x oversampling) that don't
reflect real fraud patterns. XGBoost's scale_pos_weight handles imbalance
natively and produces better-calibrated probabilities.
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

from utils import get_logger, ensure_dirs, save_scaler

logger = get_logger(__name__)


# ── Synthetic fallback ────────────────────────────────────────────────────────

def _generate_synthetic_ulb(n_legit: int = 5000, n_fraud: int = 100,
                             random_state: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic ULB-structured dataset when the real CSV is unavailable.

    Produces 30 feature columns (Time, V1–V28, Amount) + Class label.
    Fraudulent samples have shifted means so XGBoost can learn a signal.

    Args:
        n_legit:       Number of legitimate transaction rows.
        n_fraud:       Number of fraudulent transaction rows.
        random_state:  RNG seed for reproducibility.

    Returns:
        DataFrame with ULB schema, ready for preprocessing.
    """
    rng = np.random.RandomState(random_state)
    feature_cols = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

    # Legit transactions: near-zero PCA values, typical amounts
    legit = pd.DataFrame(
        rng.randn(n_legit, 30) * np.array([50_000] + [0] * 28 + [100]),
        columns=feature_cols
    )
    legit["Amount"] = np.abs(rng.exponential(scale=100, size=n_legit))
    legit["Class"] = 0

    # Fraud transactions: shifted PCA values (especially V14, V17 — known in ULB)
    fraud = pd.DataFrame(rng.randn(n_fraud, 30), columns=feature_cols)
    fraud["V14"] -= 5      # Known fraud indicator in ULB dataset
    fraud["V17"] -= 3
    fraud["V4"]  += 3
    fraud["Amount"] = np.abs(rng.exponential(scale=300, size=n_fraud))
    fraud["Class"] = 1

    df = pd.concat([legit, fraud], ignore_index=True).sample(
        frac=1, random_state=random_state
    ).reset_index(drop=True)

    logger.info(
        f"[SYNTHETIC] Generated {n_legit} legit + {n_fraud} fraud samples "
        f"({n_fraud / (n_legit + n_fraud) * 100:.2f}% fraud)."
    )
    return df


# ── Main preprocessing function ───────────────────────────────────────────────

def load_and_preprocess_ulb(
    filepath: str = "data/raw/creditcard.csv",
    scaler_path: str = "models/scaler.pkl"
):
    """
    Load and preprocess the ULB Credit Card Fraud dataset.

    Pipeline:
      1. Load CSV (falls back to synthetic data if file not found).
      2. Check for nulls.
      3. Scale 'Amount' and 'Time' — V1-V28 are already PCA-scaled.
      4. Stratified 70 / 15 / 15 split.
      5. Persist the scaler for use in app.py.

    No SMOTE — XGBoost uses scale_pos_weight to handle imbalance natively.

    Args:
        filepath:    Path to creditcard.csv.
        scaler_path: Where to save the fitted scaler.

    Returns:
        Tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    ensure_dirs("models", "data/processed")

    # ── Load or synthesise ────────────────────────────────────────────────────
    if os.path.exists(filepath):
        logger.info(f"Loading dataset from {filepath} …")
        df = pd.read_csv(filepath)
        logger.info(f"Loaded {len(df):,} rows × {df.shape[1]} columns.")
    else:
        logger.warning(
            f"Dataset not found at '{filepath}'. "
            "Using SYNTHETIC data for pipeline validation. "
            "Download creditcard.csv from Kaggle for real training."
        )
        df = _generate_synthetic_ulb()

    # ── Null check ────────────────────────────────────────────────────────────
    nulls = df.isnull().sum().sum()
    logger.info(f"Null values in dataset: {nulls}")

    fraud_pct = df["Class"].mean() * 100
    logger.info(
        f"Fraud cases: {df['Class'].sum():,} / {len(df):,} ({fraud_pct:.3f}%)"
    )

    # ── Scaling Amount and Time ───────────────────────────────────────────────
    scaler = StandardScaler()
    df[["Time", "Amount"]] = scaler.fit_transform(df[["Time", "Amount"]])

    save_scaler(scaler, scaler_path)

    # ── Features / Target ─────────────────────────────────────────────────────
    X = df.drop("Class", axis=1)
    y = df["Class"]

    # ── Stratified split: 70% train, 15% val, 15% test ───────────────────────
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=0.176,   # 0.176 × 0.85 ≈ 0.15 of total
        random_state=42, stratify=y_temp
    )

    logger.info(
        f"Split -> Train: {len(X_train):,} | Val: {len(X_val):,} | "
        f"Test: {len(X_test):,}"
    )
    logger.info(f"Fraud in train: {y_train.sum():,} / {len(y_train):,}")

    # ── Persist processed splits ──────────────────────────────────────────────
    pd.concat([X_train, y_train], axis=1).to_csv(
        "data/processed/ml_train.csv", index=False
    )
    pd.concat([X_test, y_test], axis=1).to_csv(
        "data/processed/ml_test.csv", index=False
    )
    logger.info("Processed splits saved to data/processed/")

    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_preprocess_ulb()
    logger.info("Preprocessing complete.")
