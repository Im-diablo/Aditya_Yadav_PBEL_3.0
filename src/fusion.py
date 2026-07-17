"""
src/fusion.py
Multi-Modal Fraud Score Fusion Layer.

Combines predictions from all three modules:
  - Module 1 (ML):  XGBoost tabular score   -> weight 0.50
  - Module 2 (NLP): DistilBERT text score    -> weight 0.30
  - Module 3 (CV):  Siamese signature score  -> weight 0.20

Each module is OPTIONAL. When fewer than 3 inputs are supplied,
weights are renormalised to sum to 1.0 automatically.

Decision: final_score > 0.5 -> FRAUD, else LEGITIMATE.
"""

import os
import numpy as np
from utils import get_logger

logger = get_logger(__name__)

# ── Default fusion weights ────────────────────────────────────────────────────
# Reflect reliability and coverage of each module:
#   ML  — most reliable; covers ALL digital transactions.
#   NLP — covers text-based social engineering fraud.
#   CV  — covers cheque/document fraud only.

WEIGHTS = {
    "ml":  0.50,
    "nlp": 0.30,
    "cv":  0.20,
}


# ── Main fusion function ──────────────────────────────────────────────────────

def predict_fraud(
    transaction_features=None,   # list / numpy array of 30 ULB-format features
    transaction_text: str = None, # SMS / email body / transaction description
    signature_paths: tuple = None, # (ref_image_path, test_image_path)
    weights: dict = WEIGHTS,
    ml_model_path:  str = "models/xgboost_fraud.ubj",
    nlp_model_path: str = "models/distilbert_fraud",
    cv_model_path:  str = "models/siamese_resnet18.pt",
) -> dict:
    """
    Unified multi-modal fraud prediction.

    Runs whichever modules have valid inputs and combines their
    fraud probability scores using a weighted average.

    Args:
        transaction_features: Array of 30 transaction features (ML module).
        transaction_text:     Raw text string (NLP module).
        signature_paths:      Tuple of (genuine_path, test_path) (CV module).
        weights:              Dict of base weights per module (will be renormalised).
        ml_model_path:        Path to saved XGBoost weights.
        nlp_model_path:       Path to saved DistilBERT directory.
        cv_model_path:        Path to saved Siamese ResNet18 weights.

    Returns:
        Dict with keys:
          final_score    — Weighted fraud probability (0.0–1.0)
          final_label    — 0 (legit) or 1 (fraud)
          verdict        — "FRAUD" | "LEGITIMATE"
          confidence     — Confidence percentage of the verdict
          weights_used   — Renormalised weights applied
          module_results — Per-module score and verdict
    """
    scores         = {}
    active_weights = {}
    results        = {}

    # ── Module 1 — ML ─────────────────────────────────────────────────────────
    if transaction_features is not None:
        try:
            from ml_model import predict_transaction
            ml_result = predict_transaction(transaction_features, ml_model_path)
            scores["ml"]         = ml_result["fraud_probability"]
            active_weights["ml"] = weights["ml"]
            results["ml"]        = {
                "score":   round(ml_result["fraud_probability"], 4),
                "verdict": ml_result["verdict"],
            }
            logger.info(
                f"ML module: score={scores['ml']:.4f} "
                f"({results['ml']['verdict']})"
            )
        except FileNotFoundError as e:
            logger.warning(f"ML model unavailable: {e}")
        except Exception as exc:
            logger.error(f"ML module failed: {exc}")

    # ── Module 2 — NLP ────────────────────────────────────────────────────────
    if transaction_text is not None and transaction_text.strip():
        try:
            from nlp_model import predict_text
            nlp_result = predict_text(transaction_text, nlp_model_path)
            scores["nlp"]         = nlp_result["fraud_probability"]
            active_weights["nlp"] = weights["nlp"]
            results["nlp"]        = {
                "score":   round(nlp_result["fraud_probability"], 4),
                "verdict": nlp_result["verdict"],
            }
            logger.info(
                f"NLP module: score={scores['nlp']:.4f} "
                f"({results['nlp']['verdict']})"
            )
        except FileNotFoundError as e:
            logger.warning(f"NLP model unavailable: {e}")
        except Exception as exc:
            logger.error(f"NLP module failed: {exc}")

    # ── Module 3 — CV ─────────────────────────────────────────────────────────
    if signature_paths is not None:
        try:
            from cv_model import predict_signature
            ref_path, test_path = signature_paths
            cv_result = predict_signature(ref_path, test_path,
                                          model_path=cv_model_path)
            scores["cv"]         = cv_result["fraud_probability"]
            active_weights["cv"] = weights["cv"]
            results["cv"]        = {
                "score":    round(cv_result["fraud_probability"], 4),
                "verdict":  cv_result["verdict"],
                "distance": cv_result["distance"],
            }
            logger.info(
                f"CV module: distance={cv_result['distance']:.4f} "
                f"({results['cv']['verdict']})"
            )
        except FileNotFoundError as e:
            logger.warning(f"CV model unavailable: {e}")
        except Exception as exc:
            logger.error(f"CV module failed: {exc}")

    # ── Require at least one active module ────────────────────────────────────
    if not scores:
        raise ValueError(
            "No module produced a result. "
            "Provide at least one of: transaction_features, "
            "transaction_text, or signature_paths. "
            "Ensure the corresponding trained model exists."
        )

    # ── Renormalise weights to sum to 1.0 ─────────────────────────────────────
    total_weight = sum(active_weights.values())
    norm_weights = {k: v / total_weight for k, v in active_weights.items()}

    # ── Weighted average ──────────────────────────────────────────────────────
    final_score = sum(scores[k] * norm_weights[k] for k in scores)
    final_label = 1 if final_score > 0.5 else 0
    verdict     = "FRAUD" if final_label == 1 else "LEGITIMATE"
    # Confidence = probability of the winning class
    confidence  = final_score if final_label == 1 else (1 - final_score)

    logger.info(
        f"Fusion result: score={final_score:.4f} -> {verdict} "
        f"({confidence * 100:.1f}% confidence)"
    )

    return {
        "final_score":    round(final_score, 4),
        "final_label":    final_label,
        "verdict":        verdict,
        "confidence":     round(confidence * 100, 1),
        "weights_used":   {k: round(v, 4) for k, v in norm_weights.items()},
        "module_results": results,
    }


# ── Entry-point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))

    # Example 1 — Transaction only
    print("=" * 55)
    print("Example 1: Transaction features only")
    try:
        result = predict_fraud(
            transaction_features=[0.0] * 28 + [150.0, 0.4]   # 30 features
        )
        print(f"Verdict: {result['verdict']} | "
              f"Score: {result['final_score']} | "
              f"Confidence: {result['confidence']}%")
    except Exception as e:
        print(f"  (Model not trained yet — run train_all.py first) {e}")

    # Example 2 — Text only
    print("\nExample 2: Phishing text")
    try:
        result = predict_fraud(
            transaction_text=(
                "URGENT: Your bank account has been compromised. "
                "Click here immediately: http://fake-bank.com"
            )
        )
        print(f"Verdict: {result['verdict']} | "
              f"Score: {result['final_score']} | "
              f"Confidence: {result['confidence']}%")
        print(f"Modules: {result['module_results']}")
    except Exception as e:
        print(f"  (Model not trained yet — run train_all.py first) {e}")
