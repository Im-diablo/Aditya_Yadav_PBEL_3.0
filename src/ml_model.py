"""
src/ml_model.py
XGBoost Transaction Fraud Classifier — Module 1.

Anti-overfitting measures:
  - Shallow trees   (max_depth=4)
  - Slow learning   (learning_rate=0.05)
  - Row sampling    (subsample=0.8)
  - Feature sampling(colsample_bytree=0.8)
  - L1 + L2 reg    (reg_alpha=0.1, reg_lambda=1.0)
  - Early stopping  (early_stopping_rounds=20)
  - SMOTE balancing (applied during preprocessing)
  - scale_pos_weight=577 — native imbalance handling (284,315 legit / 492 fraud ≈ 577)
  - No SMOTE — avoids extreme synthetic oversampling from only 344 real fraud samples

Explainability:
  - SHAP TreeExplainer provides per-feature contribution scores.
"""

import os
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score, f1_score,
    precision_recall_curve
)
import matplotlib.pyplot as plt
import joblib

from utils import get_logger, ensure_dirs, save_metrics, plot_confusion_matrix

logger = get_logger(__name__)


# ── Training ──────────────────────────────────────────────────────────────────

def train_xgboost(X_train, y_train, X_val, y_val) -> XGBClassifier:
    """
    Train an XGBoost classifier with robust anti-overfitting parameters.

    Key parameters:
      max_depth=6              — Deeper trees capture complex fraud patterns.
      learning_rate=0.05       — Small steps give better generalisation.
      n_estimators=1000        — Upper ceiling; early stopping will halt sooner.
      min_child_weight=1       — Allow splits with few samples (only 344 real frauds).
      subsample=0.8            — Each tree uses 80% of rows (random selection).
      colsample_bytree=0.8     — Each tree uses 80% of features.
      reg_alpha=0.1            — L1 regularisation on leaf weights.
      reg_lambda=1.0           — L2 regularisation on leaf weights.
      scale_pos_weight=577     — 284,315 legit / 492 fraud ≈ 577; handles imbalance natively.
      eval_metric='aucpr'      — PR-AUC is better than ROC for imbalanced data.
      early_stopping_rounds=50 — Stop when val PR-AUC stagnates for 50 rounds.

    Args:
        X_train: Training features (post-SMOTE).
        y_train: Training labels.
        X_val:   Validation features (no SMOTE applied).
        y_val:   Validation labels.

    Returns:
        Fitted XGBClassifier.
    """
    model = XGBClassifier(
        max_depth=6,
        learning_rate=0.05,
        n_estimators=1000,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=577,  # 284,315 legit / 492 fraud ≈ 577; no SMOTE so native weighting applies
        eval_metric="aucpr",
        early_stopping_rounds=50,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
        use_label_encoder=False,
    )

    logger.info("Training XGBoost classifier …")
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=50,
    )

    logger.info(f"Best iteration: {model.best_iteration}")
    return model


# ── Evaluation ────────────────────────────────────────────────────────────────

def _best_threshold(y_true, y_prob, beta: float = 2.0) -> float:
    """Find threshold maximising F-beta (default beta=2: recall-weighted).

    beta=2 is appropriate for fraud detection — a missed fraud (FN) costs
    more than a false alarm (FP), so recall is weighted twice over precision.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    beta2 = beta ** 2
    scores = np.where(
        (beta2 * precision + recall) == 0, 0,
        (1 + beta2) * precision * recall / (beta2 * precision + recall)
    )
    return float(thresholds[np.argmax(scores[:-1])])


def evaluate_model(
    model: XGBClassifier,
    X_test,
    y_test,
    X_val=None,
    y_val=None,
    feature_names: list = None,
    save_dir: str = "plots"
) -> dict:
    """
    Full evaluation of the trained XGBoost model.

    Threshold is tuned on X_val/y_val (if provided) to maximise F1,
    then applied to X_test. Falls back to 0.5 if val set is not given.

    Args:
        model:         Trained XGBClassifier.
        X_test:        Test features.
        y_test:        Test ground-truth labels.
        X_val:         Validation features for threshold tuning (optional).
        y_val:         Validation labels for threshold tuning (optional).
        feature_names: List of column names (for display).
        save_dir:      Directory to save confusion matrix plot.

    Returns:
        Dict of evaluation metrics.
    """
    ensure_dirs(save_dir)

    y_prob = model.predict_proba(X_test)[:, 1]

    # Tune threshold on val set; fall back to 0.5
    if X_val is not None and y_val is not None:
        val_prob = model.predict_proba(X_val)[:, 1]
        threshold = _best_threshold(y_val, val_prob)
        logger.info(f"Optimal F2 threshold (from val set): {threshold:.4f}")
    else:
        threshold = 0.5
        logger.info("No val set provided — using default threshold 0.5")

    y_pred = (y_prob >= threshold).astype(int)

    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc  = average_precision_score(y_test, y_prob)
    f1      = f1_score(y_test, y_pred)

    print("\n" + "=" * 55)
    print("CLASSIFICATION REPORT — XGBoost ML Module")
    print("=" * 55)
    print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))

    print(f"Threshold: {threshold:.4f}  (tuned on val set for max F2, beta=2)")
    print(f"ROC-AUC:  {roc_auc:.4f}   (target > 0.95)")
    print(f"PR-AUC:   {pr_auc:.4f}   (target > 0.75)")
    print(f"F1 Score: {f1:.4f}   (target > 0.85)")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"  True Negatives  (legit -> legit):   {cm[0, 0]:,}")
    print(f"  False Positives (legit -> fraud):   {cm[0, 1]:,}")
    print(f"  False Negatives (fraud -> legit):   {cm[1, 0]:,}  <- most costly!")
    print(f"  True Positives  (fraud -> fraud):   {cm[1, 1]:,}")

    plot_confusion_matrix(cm, ["Legit", "Fraud"], "XGBoost_ML", save_dir)

    metrics = {
        "module": "XGBoost_ML",
        "roc_auc":   round(roc_auc,   4),
        "pr_auc":    round(pr_auc,    4),
        "f1":        round(f1,        4),
        "threshold": round(threshold, 4),
    }
    save_metrics(metrics, "models/ml_metrics.json")
    return metrics


# ── SHAP Explainability ───────────────────────────────────────────────────────

def explain_predictions(model: XGBClassifier, X_test, feature_names: list,
                         n_samples: int = 100, save_dir: str = "plots") -> None:
    """
    Use SHAP TreeExplainer to explain model predictions.

    Generates:
      - Summary plot: which features matter most overall.
      - Waterfall plot: explains a single specific fraud prediction.

    Args:
        model:         Trained XGBClassifier.
        X_test:        Test features.
        feature_names: Column names.
        n_samples:     Number of samples to explain (default 100).
        save_dir:      Where to save plots.
    """
    try:
        import shap
        ensure_dirs(save_dir)

        logger.info("Generating SHAP explanations …")
        # SHAP 0.49+ / XGBoost 3.x fix: set feature names on the booster
        # before passing to TreeExplainer. This forces SHAP to use the
        # native C++ path instead of the broken text-dump parser.
        sample = X_test[:n_samples]
        sample_np = sample.values if hasattr(sample, "values") else sample

        booster = model.get_booster()
        booster.feature_names = (
            feature_names if feature_names
            else [f"f{i}" for i in range(sample_np.shape[1])]
        )
        explainer   = shap.TreeExplainer(booster)
        shap_values = explainer(sample_np).values

        # Summary plot
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(
            shap_values, sample_np,
            feature_names=feature_names, show=False
        )
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "shap_summary.png"), dpi=150,
                    bbox_inches="tight")
        plt.close()
        logger.info(f"SHAP summary plot saved -> {save_dir}/shap_summary.png")

    except ImportError:
        logger.warning("SHAP not installed. Skipping explainability. "
                       "Run: pip install shap")
    except Exception as exc:
        logger.warning(f"SHAP explanation failed: {exc}")


# ── Model persistence ─────────────────────────────────────────────────────────

MODEL_PATH = "models/xgboost_fraud.ubj"  # native XGBoost binary format


def save_model(model: XGBClassifier, path: str = MODEL_PATH) -> None:
    """Persist the trained XGBoost model using native .ubj format.

    .ubj (Universal Binary JSON) is XGBoost's own format — version-portable
    and avoids the pickle compatibility warning from joblib.
    """
    ensure_dirs(os.path.dirname(path))
    model.save_model(path)
    logger.info(f"Model saved -> {path}")


# ── Model cache (loaded once per process, reused on every request) ────────────
_ML_CACHE: dict = {}  # keys: "model", "path"


def load_model(path: str = MODEL_PATH) -> XGBClassifier:
    """Load a previously saved XGBoost model, caching after the first load.

    Handles native .ubj format, explicit .pkl format, and falls back to .pkl
    if the .ubj file is not found.

    Args:
        path: Path to the model file (.ubj or .pkl).

    Returns:
        Loaded XGBClassifier (cached after first call).

    Raises:
        FileNotFoundError: If neither .ubj nor .pkl model file exists.
    """
    # ── Return cached model if already loaded ─────────────────────────────────
    if _ML_CACHE.get("path") == path:
        logger.debug("[ML] Using cached XGBoost model.")
        return _ML_CACHE["model"]

    logger.info(f"[ML] Loading XGBoost model from disk: {path}")

    if path.endswith(".pkl"):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Trained XGBoost pickle model not found at '{path}'.")
        model = joblib.load(path)
    else:
        # Backwards-compat fallback for old pickle models
        pkl_path = path.replace(".ubj", ".pkl")
        if not os.path.exists(path) and os.path.exists(pkl_path):
            logger.warning(
                f".ubj model not found, falling back to pickle: {pkl_path}. "
                "Re-run retrain_ml.py to generate the .ubj format."
            )
            model = joblib.load(pkl_path)
        elif not os.path.exists(path):
            raise FileNotFoundError(
                f"Trained XGBoost model not found at '{path}'. "
                "Run retrain_ml.py to generate the model weights."
            )
        else:
            try:
                model = XGBClassifier()
                model.load_model(path)
            except Exception as e:
                try:
                    model = joblib.load(path)
                except Exception:
                    raise e

    _ML_CACHE["model"] = model
    _ML_CACHE["path"]  = path
    logger.info("[ML] Model cached — subsequent calls will be fast.")
    return model


# ── Prediction helper (for fusion layer) ─────────────────────────────────────

def predict_transaction(features, model_path: str = MODEL_PATH,
                        metrics_path: str = "models/ml_metrics.json",
                        scaler_path: str = "models/scaler.pkl") -> dict:
    """
    Predict fraud probability for a single transaction feature vector.

    Args:
        features:      List or numpy array of 30 transaction features
                       (same ordering as ULB dataset: Time, V1-V28, Amount).
        model_path:    Path to saved model weights.
        metrics_path:  Path to ml_metrics.json (contains tuned threshold).
        scaler_path:   Path to scaler.pkl.

    Returns:
        Dict with 'fraud_probability' (float), 'verdict' (str).
    """
    import json
    from utils import load_scaler
    model = load_model(model_path)

    threshold = 0.5
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            threshold = json.load(f).get("threshold", 0.5)

    # Copy features to prevent side-effects on original input
    features_copy = list(features)

    # Scale Time (index 0) and Amount (index 29) if scaler exists
    if os.path.exists(scaler_path):
        try:
            scaler = load_scaler(scaler_path)
            # Check if scaler is fitted on 2 features (Time, Amount)
            if hasattr(scaler, "n_features_in_") and scaler.n_features_in_ == 2:
                scaled = scaler.transform([[features_copy[0], features_copy[29]]])[0]
                features_copy[0] = float(scaled[0])
                features_copy[29] = float(scaled[1])
            elif hasattr(scaler, "n_features_in_") and scaler.n_features_in_ == 1:
                # Legacy behavior: scaler is only fitted on 1 feature (Time)
                scaled_time = float(scaler.transform([[features_copy[0]]])[0][0])
                features_copy[0] = scaled_time
                # Amount remains unscaled (as per legacy training bug setup)
        except Exception as e:
            logger.warning(f"Failed to scale features during prediction: {e}")

    feature_names = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
    df = pd.DataFrame([features_copy], columns=feature_names)

    prob = float(model.predict_proba(df)[0][1])

    # ── Normalize probability around 0.5 decision boundary ─────────────────────
    # If the threshold is e.g. 0.30, map 0.30 to 0.50 so that the fusion layer's
    # hardcoded 0.50 decision boundary is aligned with the selected threshold.
    if prob < threshold:
        normalized_prob = (prob / threshold) * 0.5 if threshold > 0 else 0.0
    else:
        normalized_prob = 0.5 + ((prob - threshold) / (1.0 - threshold)) * 0.5 if threshold < 1.0 else 1.0

    return {
        "fraud_probability": round(normalized_prob, 4),
        "raw_probability":   round(prob, 4),
        "verdict":           "FRAUD" if prob >= threshold else "LEGIT",
    }


# ── Entry-point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from data_preprocessing import load_and_preprocess_ulb

    X_train, X_val, X_test, y_train, y_val, y_test = load_and_preprocess_ulb()

    model = train_xgboost(X_train, y_train, X_val, y_val)
    feature_names = X_test.columns.tolist() if hasattr(X_test, "columns") else None
    evaluate_model(model, X_test, y_test, X_val=X_val, y_val=y_val, feature_names=feature_names)
    explain_predictions(model, X_test, feature_names=feature_names)
    save_model(model)

    logger.info("ML module training complete.")
