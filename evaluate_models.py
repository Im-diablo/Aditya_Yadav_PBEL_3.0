"""
evaluate_models.py
Accuracy and performance evaluation script for the AI-Powered Fraud Detection System.

This script loads the trained models for ML, NLP, and CV modules, prepares their 
respective test splits (falling back to synthetic data generators if needed), 
and calculates key classification metrics (Accuracy, Precision, Recall, F1-Score) 
formatted as percentages.
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)

# Insert src into Python path for importing modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def evaluate_ml():
    """
    Evaluate the XGBoost ML model.
    Loads models/xgboost_fraud.pkl and tests it against the test set split.
    Uses the tuned F-beta threshold from ml_metrics.json (not hardcoded 0.5).
    """
    print("\n" + "=" * 60)
    print("Evaluating Module 1: XGBoost Transaction Fraud Classifier")
    print("=" * 60)
    
    from ml_model import load_model
    
    # 1. Load model
    model = load_model()

    # 2. Load tuned threshold (set during training via F-beta=2 on val set)
    threshold = 0.5
    metrics_path = "models/ml_metrics.json"
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            threshold = json.load(f).get("threshold", 0.5)
    print(f"[ML] Using tuned threshold: {threshold:.4f}")
    
    # 3. Re-create or load test splits
    if not os.path.exists("data/processed/ml_test.csv"):
        print("[ML] Processed test CSV not found. Running pre-processing pipeline...")
        from data_preprocessing import load_and_preprocess_ulb
        _, _, X_test, _, _, y_test = load_and_preprocess_ulb()
    else:
        print("[ML] Loading test split from data/processed/ml_test.csv...")
        test_df = pd.read_csv("data/processed/ml_test.csv")
        X_test = test_df.drop("Class", axis=1)
        y_test = test_df["Class"]
        
    # 4. Inference with tuned threshold
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    
    # 5. Metrics
    acc     = accuracy_score(y_test, y_pred) * 100
    prec    = precision_score(y_test, y_pred, zero_division=0) * 100
    rec     = recall_score(y_test, y_pred, zero_division=0) * 100
    f1      = f1_score(y_test, y_pred, zero_division=0) * 100
    roc_auc = roc_auc_score(y_test, y_prob) * 100
    pr_auc  = average_precision_score(y_test, y_prob) * 100
    
    print(f"ML Test Accuracy:  {acc:.2f}%")
    print(f"ML Test Precision: {prec:.2f}%")
    print(f"ML Test Recall:    {rec:.2f}%")
    print(f"ML Test F1-Score:  {f1:.2f}%")
    print(f"ML ROC-AUC:        {roc_auc:.2f}%")
    print(f"ML PR-AUC:         {pr_auc:.2f}%")
    
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc
    }


def evaluate_nlp(device):
    """
    Evaluate the DistilBERT NLP model.
    Loads models/distilbert_fraud and tests it against the test split.
    """
    print("\n" + "=" * 60)
    print("Evaluating Module 2: DistilBERT NLP Phishing Text Detector")
    print("=" * 60)
    
    from nlp_model import load_nlp_data, FraudTextDataset
    from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
    from sklearn.model_selection import train_test_split
    
    # 1. Load data & re-create split
    print("[NLP] Loading NLP text datasets...")
    df = load_nlp_data()
    df = df.dropna(subset=["label", "text"])
    df["label"] = df["label"].astype(int)
    # Keep texts as object array to avoid numpy fixed-width unicode allocation
    # (one corrupted long row causes numpy to allocate TiB of memory)
    df["text"] = df["text"].astype(str).str[:2048]  # cap at 2048 chars
    texts  = df["text"].values  # object dtype — no fixed-width allocation
    labels = df["label"].values.astype(np.int64)

    X_train_txt, X_temp, y_train, y_temp = train_test_split(
        texts, labels,
        test_size=0.30, random_state=42, stratify=labels
    )
    X_val_txt, X_test_txt, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )
    
    # 2. Load model and tokenizer
    model_path = "models/distilbert_fraud"
    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"DistilBERT model folder not found at '{model_path}'. Run training first.")
        
    print(f"[NLP] Loading fine-tuned DistilBERT from {model_path}...")
    tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    model = model.to(device)
    model.eval()
    
    # 3. Create Dataset and DataLoader
    test_ds = FraudTextDataset(X_test_txt, y_test, tokenizer)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)
    
    # 4. Predict
    test_preds = []
    test_true = []
    print("[NLP] Running batch inference on test texts...")
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"]
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = outputs.logits.argmax(dim=1).cpu().numpy()
            
            test_preds.extend(preds)
            test_true.extend(labels.numpy())
            
    # 5. Metrics calculation
    acc = accuracy_score(test_true, test_preds) * 100
    prec = precision_score(test_true, test_preds, zero_division=0) * 100
    rec = recall_score(test_true, test_preds, zero_division=0) * 100
    f1 = f1_score(test_true, test_preds, zero_division=0) * 100
    
    print(f"NLP Test Accuracy:  {acc:.2f}%")
    print(f"NLP Test Precision: {prec:.2f}%")
    print(f"NLP Test Recall:    {rec:.2f}%")
    print(f"NLP Test F1-Score:  {f1:.2f}%")
    
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1
    }


def evaluate_cv(device):
    """
    Evaluate the Siamese ResNet18 Signature Verifier.
    Loads models/siamese_resnet18.pt and optimizes threshold on validation set
    before calculating final test performance metrics.
    """
    print("\n" + "=" * 60)
    print("Evaluating Module 3: Siamese ResNet18 Signature Forgery Detector")
    print("=" * 60)
    
    from cv_model import (
        SiamesePairDataset, val_transform, train_transform, 
        PAIRS_PER_CLASS, _generate_synthetic_signature_dirs,
        SiameseResNet, EMBEDDING_DIM
    )
    from torch.utils.data import random_split
    
    genuine_dir = "data/raw/signatures/genuine"
    forged_dir = "data/raw/signatures/forged"
    kaggle_base = "data/raw/sign_data/train"
    
    # 1. Assemble signature dataset (replicating training logic)
    if os.path.exists(kaggle_base):
        print(f"[CV] Loading Kaggle signatures dataset from {kaggle_base}...")
        genuine_files, forged_files = [], []
        for root, dirs, _ in os.walk(kaggle_base):
            for d in dirs:
                full_dir = os.path.join(root, d)
                if "forg" in d.lower():
                    forged_files.extend([os.path.join(full_dir, f) for f in os.listdir(full_dir) if f.lower().endswith((".png", ".jpg"))])
                else:
                    genuine_files.extend([os.path.join(full_dir, f) for f in os.listdir(full_dir) if f.lower().endswith((".png", ".jpg"))])
        
        class FlatListDataset(SiamesePairDataset):
            def __init__(self, gen_files, forg_files, transform=None):
                self.transform = transform
                self.pairs, self.labels = [], []
                rng = np.random.RandomState(42)
                if len(gen_files) < 2 or len(forg_files) < 1:
                    return
                for _ in range(PAIRS_PER_CLASS):
                    i1, i2 = rng.choice(len(gen_files), 2, replace=False)
                    self.pairs.append((gen_files[i1], gen_files[i2]))
                    self.labels.append(0)
                for _ in range(PAIRS_PER_CLASS):
                    i1 = rng.randint(len(gen_files))
                    i2 = rng.randint(len(forg_files))
                    self.pairs.append((gen_files[i1], forg_files[i2]))
                    self.labels.append(1)
                    
        full_dataset = FlatListDataset(genuine_files, forged_files, transform=train_transform)
        
    elif not (os.path.exists(genuine_dir) and os.path.exists(forged_dir)):
        print("[CV] Signature directories not found. Running synthetic generator...")
        genuine_dir, forged_dir = _generate_synthetic_signature_dirs()
        full_dataset = SiamesePairDataset(genuine_dir, forged_dir, transform=train_transform)
    else:
        print("[CV] Loading signature directories...")
        full_dataset = SiamesePairDataset(
            genuine_dir, forged_dir,
            transform=train_transform,
            pairs_per_class=PAIRS_PER_CLASS,
        )
        
    n = len(full_dataset)
    train_size = int(0.70 * n)
    val_size = int(0.15 * n)
    test_size = n - train_size - val_size
    
    # Exact split matching cv_model.py
    train_set, val_set, test_set = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )
    
    # Apply evaluation transform
    val_set.dataset.transform = val_transform
    test_set.dataset.transform = val_transform
    
    val_loader = DataLoader(val_set, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=16, shuffle=False)
    
    # 2. Load model
    print("[CV] Loading Siamese ResNet18 model weight parameters...")
    model = SiameseResNet(embedding_dim=EMBEDDING_DIM)
    model.load_state_dict(torch.load("models/siamese_resnet18.pt", map_location=device))
    model = model.to(device)
    model.eval()
    
    # 3. Extract embeddings and distances
    def get_distances_and_labels(loader):
        all_distances = []
        all_labels = []
        with torch.no_grad():
            for img1, img2, labels in loader:
                img1, img2 = img1.to(device), img2.to(device)
                emb1, emb2 = model(img1, img2)
                distances = torch.nn.functional.pairwise_distance(emb1, emb2)
                all_distances.extend(distances.cpu().numpy())
                all_labels.extend(labels.numpy())
        return np.array(all_distances), np.array(all_labels)
        
    print("[CV] Analyzing validation set to optimize Euclidean distance threshold...")
    val_distances, val_labels = get_distances_and_labels(val_loader)
    
    # Grid search best classification threshold on validation set
    thresholds = np.linspace(0.1, 1.5, 141)
    best_thresh = 0.5
    best_val_acc = 0.0
    for t in thresholds:
        preds = (val_distances > t).astype(int)
        acc = accuracy_score(val_labels, preds)
        if acc > best_val_acc:
            best_val_acc = acc
            best_thresh = t
            
    print(f"[CV] Optimized distance threshold: {best_thresh:.3f} (Val Accuracy: {best_val_acc * 100:.2f}%)")
    
    # 4. Inference on test set
    print("[CV] Extracting and classifying test signature pairs...")
    test_distances, test_labels = get_distances_and_labels(test_loader)
    test_preds = (test_distances > best_thresh).astype(int)
    
    # 5. Metrics calculation
    acc = accuracy_score(test_labels, test_preds) * 100
    prec = precision_score(test_labels, test_preds, zero_division=0) * 100
    rec = recall_score(test_labels, test_preds, zero_division=0) * 100
    f1 = f1_score(test_labels, test_preds, zero_division=0) * 100
    
    print(f"CV Test Accuracy:  {acc:.2f}%")
    print(f"CV Test Precision: {prec:.2f}%")
    print(f"CV Test Recall:    {rec:.2f}%")
    print(f"CV Test F1-Score:  {f1:.2f}%")
    
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "optimal_threshold": float(best_thresh)
    }


def print_report(ml_res, nlp_res, cv_res):
    """
    Format all calculated metrics into a beautiful ASCII table dashboard.
    """
    print("\n" + "=" * 80)
    print("                      🛡️  AI FRAUD DETECTION SYSTEM REPORT                      ")
    print("=" * 80)
    print(f"  Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 80)
    print(f" {'Module Name':<35} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-" * 80)
    
    if ml_res:
        print(f"  📈 ML Module (XGBoost)            | {ml_res['accuracy']:>8.2f}% | {ml_res['precision']:>8.2f}% | {ml_res['recall']:>8.2f}% | {ml_res['f1']:>8.2f}%")
    if nlp_res:
        print(f"  💬 NLP Module (DistilBERT)        | {nlp_res['accuracy']:>8.2f}% | {nlp_res['precision']:>8.2f}% | {nlp_res['recall']:>8.2f}% | {nlp_res['f1']:>8.2f}%")
    if cv_res:
        print(f"  ✍️  CV Module (Siamese ResNet18)   | {cv_res['accuracy']:>8.2f}% | {cv_res['precision']:>8.2f}% | {cv_res['recall']:>8.2f}% | {cv_res['f1']:>8.2f}%")
        
    print("-" * 80)
    print("  Note: All metrics above are computed on independent test splits.")
    if cv_res and 'optimal_threshold' in cv_res:
        print(f"  * CV Module evaluated with optimized decision threshold of {cv_res['optimal_threshold']:.3f}.")
    print("=" * 80)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating models using device: {device}")
    
    ml_res = None
    nlp_res = None
    cv_res = None
    
    # 1. Run ML Evaluation
    try:
        ml_res = evaluate_ml()
    except Exception as e:
        print(f"\n[ERROR] Failed evaluating ML module: {e}")
        
    # 2. Run NLP Evaluation
    try:
        nlp_res = evaluate_nlp(device)
    except Exception as e:
        print(f"\n[ERROR] Failed evaluating NLP module: {e}")
        
    # 3. Run CV Evaluation
    try:
        cv_res = evaluate_cv(device)
    except Exception as e:
        print(f"\n[ERROR] Failed evaluating CV module: {e}")
        
    # 4. Print Summary Report
    print_report(ml_res, nlp_res, cv_res)
    
    # 5. Save results to detailed_accuracies.json
    results = {
        "timestamp": time.strftime('%Y-%m-%dT%H:%M:%S'),
        "device": str(device),
        "ml": ml_res,
        "nlp": nlp_res,
        "cv": cv_res
    }
    
    with open("models/detailed_accuracies.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved comprehensive report to: models/detailed_accuracies.json")


if __name__ == "__main__":
    main()
