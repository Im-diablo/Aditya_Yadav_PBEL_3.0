"""
train_all.py
Sequential training orchestrator for all 3 fraud detection modules.

Run this script once to train and save all model weights:
  python train_all.py

Model weights are saved to the models/ directory:
  models/xgboost_fraud.pkl        — XGBoost (Module 1)
  models/distilbert_fraud/        — DistilBERT (Module 2)
  models/siamese_resnet18.pt      — Siamese ResNet18 (Module 3)
  models/scaler.pkl               — StandardScaler for Amount/Time

After training completes, launch the dashboard with:
  streamlit run app.py
"""

import os
import sys
import time
import traceback

# Add src to path so all imports resolve correctly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from utils import get_logger, ensure_dirs, verify_project_structure

logger = get_logger("train_all")


def train_module_1():
    """Train the XGBoost ML fraud classifier."""
    logger.info("=" * 60)
    logger.info("MODULE 1 — XGBoost Transaction Fraud Classifier")
    logger.info("=" * 60)
    t0 = time.time()

    from data_preprocessing import load_and_preprocess_ulb
    from ml_model import train_xgboost, evaluate_model, explain_predictions, save_model

    X_train, X_val, X_test, y_train, y_val, y_test = load_and_preprocess_ulb()
    model = train_xgboost(X_train, y_train, X_val, y_val)

    feature_names = (
        X_test.columns.tolist()
        if hasattr(X_test, "columns")
        else [f"V{i}" for i in range(X_test.shape[1])]
    )

    metrics = evaluate_model(model, X_test, y_test, X_val=X_val, y_val=y_val, feature_names=feature_names)
    explain_predictions(model, X_test, feature_names=feature_names)
    save_model(model)

    elapsed = time.time() - t0
    logger.info(f"Module 1 complete in {elapsed:.1f}s | ROC-AUC: {metrics.get('roc_auc', 'N/A')}")
    return metrics


def train_module_2():
    """Fine-tune DistilBERT NLP phishing text detector."""
    logger.info("=" * 60)
    logger.info("MODULE 2 — DistilBERT NLP Phishing Text Detector")
    logger.info("=" * 60)
    t0 = time.time()

    from nlp_model import train_nlp_model

    model, tokenizer = train_nlp_model()

    elapsed = time.time() - t0
    logger.info(f"Module 2 complete in {elapsed:.1f}s")
    return model, tokenizer


def train_module_3():
    """Train Siamese ResNet18 signature forgery detector."""
    logger.info("=" * 60)
    logger.info("MODULE 3 — Siamese ResNet18 Signature Forgery Detector")
    logger.info("=" * 60)
    t0 = time.time()

    from cv_model import train_siamese

    model = train_siamese()

    elapsed = time.time() - t0
    logger.info(f"Module 3 complete in {elapsed:.1f}s")
    return model


def main():
    """Run the complete training pipeline for all modules."""
    total_start = time.time()

    logger.info("╔══════════════════════════════════════════════════════════╗")
    logger.info("║  AI-Powered Financial Fraud Detection System             ║")
    logger.info("║  Full Training Pipeline                                   ║")
    logger.info("╚══════════════════════════════════════════════════════════╝")

    # Verify project structure
    status = verify_project_structure(".")
    for d, exists in status.items():
        logger.info(f"  {'✓' if exists else '✗  [MISSING]'} {d}")

    ensure_dirs("models", "plots", "data/processed")

    results = {}

    # ── Module 1: XGBoost ─────────────────────────────────────────────────────
    try:
        metrics = train_module_1()
        results["ml"] = {"status": "SUCCESS", "metrics": metrics}
    except Exception as e:
        logger.error(f"Module 1 failed: {e}")
        logger.error(traceback.format_exc())
        results["ml"] = {"status": "FAILED", "error": str(e)}

    # ── Module 2: DistilBERT ──────────────────────────────────────────────────
    try:
        train_module_2()
        results["nlp"] = {"status": "SUCCESS"}
    except Exception as e:
        logger.error(f"Module 2 failed: {e}")
        logger.error(traceback.format_exc())
        results["nlp"] = {"status": "FAILED", "error": str(e)}

    # ── Module 3: Siamese ResNet18 ────────────────────────────────────────────
    try:
        train_module_3()
        results["cv"] = {"status": "SUCCESS"}
    except Exception as e:
        logger.error(f"Module 3 failed: {e}")
        logger.error(traceback.format_exc())
        results["cv"] = {"status": "FAILED", "error": str(e)}

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed_total = time.time() - total_start

    logger.info("\n" + "=" * 60)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 60)
    for module, info in results.items():
        status = info["status"]
        icon   = "✓" if status == "SUCCESS" else "✗"
        logger.info(f"  {icon} Module {module.upper():3s} — {status}")
        if "metrics" in info:
            m = info["metrics"]
            logger.info(
                f"       ROC-AUC: {m.get('roc_auc', 'N/A')} | "
                f"PR-AUC: {m.get('pr_auc', 'N/A')} | "
                f"F1: {m.get('f1', 'N/A')}"
            )
        if "error" in info:
            logger.info(f"       Error: {info['error']}")

    logger.info(f"\nTotal training time: {elapsed_total / 60:.1f} minutes")
    logger.info("\nNext step: streamlit run app.py")


if __name__ == "__main__":
    main()
