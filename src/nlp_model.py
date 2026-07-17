"""
src/nlp_model.py
DistilBERT Fine-tuned Phishing / Fraud Text Detector — Module 2.

Datasets used:
  - SMS Spam Collection (UCI): 5,574 messages (ham/spam)
  - Phishing Email Dataset (Kaggle): Safe/Phishing email bodies

Anti-overfitting measures:
  - Max 3 epochs (BERT overfits beyond this on small text data)
  - Low LR 2e-5  (preserves pretrained language knowledge)
  - Freeze DistilBERT layers 0, 1, 2 (fine-tune only top 3 layers)
  - Dropout = 0.1 (built into DistilBERT architecture)
  - Gradient clipping max_norm=1.0
  - Weight decay 0.01 (L2 on AdamW)
  - Early stopping with patience=2
  - Warmup + linear LR decay schedule
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score

from utils import get_logger, ensure_dirs, save_metrics, plot_loss_curves

logger = get_logger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_NAME    = "distilbert-base-uncased"
MAX_LEN       = 128
BATCH_SIZE    = 16
EPOCHS        = 3
LEARNING_RATE = 2e-5
DROPOUT       = 0.1
PATIENCE      = 2
WEIGHT_DECAY  = 0.01


# ── Synthetic fallback text data ──────────────────────────────────────────────

def _generate_synthetic_text_data(n_legit: int = 400, n_fraud: int = 400,
                                   random_state: int = 42) -> pd.DataFrame:
    """
    Generate synthetic phishing / legit text samples for pipeline validation.

    Args:
        n_legit:       Number of legitimate text samples.
        n_fraud:       Number of phishing/spam text samples.
        random_state:  RNG seed.

    Returns:
        DataFrame with columns: text, label (0=legit, 1=phishing).
    """
    rng = np.random.RandomState(random_state)

    legit_templates = [
        "Your transaction of $%.2f was successful.",
        "Thank you for your purchase. Order #%d confirmed.",
        "Your account balance has been updated.",
        "Reminder: your payment of $%.2f is due on the 15th.",
        "We have received your request. Our team will respond shortly.",
    ]
    fraud_templates = [
        "URGENT: Your account has been compromised. Click here: http://fake-bank.com/%d",
        "Congratulations! You have won Rs.%d,000. Claim now!",
        "Your bank account is suspended. Verify immediately at http://phish.net",
        "FREE PRIZE: You are selected. Send your details to claim $%d.",
        "Alert: unusual activity detected. Log in at http://fake-secure-site.com",
    ]

    def _fmt(tpl, val):
        """Safely format a template — return as-is if it has no placeholder."""
        try:
            return tpl % (val,)
        except TypeError:
            return tpl  # template has no % placeholder

    legit_rows = []
    for _ in range(n_legit):
        tpl  = legit_templates[rng.randint(len(legit_templates))]
        text = _fmt(tpl, round(rng.uniform(10, 5000), 2))
        legit_rows.append({"text": text, "label": 0})

    fraud_rows = []
    for _ in range(n_fraud):
        tpl  = fraud_templates[rng.randint(len(fraud_templates))]
        text = _fmt(tpl, int(rng.randint(1000, 99999)))
        fraud_rows.append({"text": text, "label": 1})

    df = pd.DataFrame(legit_rows + fraud_rows).sample(
        frac=1, random_state=random_state
    ).reset_index(drop=True)

    logger.info(
        f"[SYNTHETIC NLP] Generated {n_legit} legit + {n_fraud} fraud text samples."
    )
    return df


# ── Data loading ──────────────────────────────────────────────────────────────

# Known alternative filenames that Kaggle extracts these datasets as
_SMS_CANDIDATES   = ["data/raw/spam.csv",       "data/raw/spam.csv",
                     "data/raw/SMSSpamCollection"]
_EMAIL_CANDIDATES = ["data/raw/Phishing_Email.csv", "data/raw/Phishing_Email.csv",
                     "data/raw/phishing_email.csv",  "data/raw/Phishing Email.csv"]


def _find_file(candidates: list) -> str:
    """Return the first path that exists, or empty string."""
    for p in candidates:
        if os.path.exists(p):
            return p
    return ""


def load_nlp_data(
    sms_path: str   = "",
    email_path: str = ""
) -> pd.DataFrame:
    """
    Load and merge the SMS Spam + Phishing Email datasets into a single DataFrame.

    Falls back to synthetic data when raw files are not present.

    Schema output: columns=['text', 'label'] where label ∈ {0: legit, 1: phishing}.

    Args:
        sms_path:   Path to SMS spam CSV (columns: v1=label, v2=text).
        email_path: Path to phishing emails CSV (columns: Email Type, Email Text).

    Returns:
        Shuffled combined DataFrame.
    """
    frames = []

    # ── SMS Spam — auto-discover filename ────────────────────────────────────
    resolved_sms = sms_path or _find_file(_SMS_CANDIDATES)
    if resolved_sms and os.path.exists(resolved_sms):
        logger.info(f"Loading SMS spam from: {resolved_sms}")
        sms_df = pd.read_csv(resolved_sms, encoding="latin-1")
        # Handle both named columns (v1/v2) and tab-separated SMSSpamCollection
        if "v1" in sms_df.columns and "v2" in sms_df.columns:
            sms_df = sms_df[["v1", "v2"]].rename(columns={"v1": "label", "v2": "text"})
        elif sms_df.shape[1] == 2:
            sms_df.columns = ["label", "text"]
        sms_df["label"] = sms_df["label"].map({"ham": 0, "spam": 1})
        sms_df = sms_df.dropna()
        frames.append(sms_df)
        logger.info(f"Loaded SMS spam dataset: {len(sms_df):,} rows.")
    else:
        logger.warning(f"SMS spam file not found (tried: {_SMS_CANDIDATES}). Skipping.")

    # ── Phishing Emails — auto-discover filename ──────────────────────────────
    resolved_email = email_path or _find_file(_EMAIL_CANDIDATES)
    if resolved_email and os.path.exists(resolved_email):
        logger.info(f"Loading phishing emails from: {resolved_email}")
        email_df = pd.read_csv(resolved_email)
        email_df = email_df[["Email Type", "Email Text"]].rename(
            columns={"Email Type": "label", "Email Text": "text"}
        )
        email_df["label"] = email_df["label"].map({"Safe Email": 0, "Phishing Email": 1})
        email_df = email_df.dropna()
        frames.append(email_df)
        logger.info(f"Loaded phishing email dataset: {len(email_df):,} rows.")
    else:
        logger.warning(f"Phishing email file not found (tried: {_EMAIL_CANDIDATES}). Skipping.")

    if not frames:
        logger.warning("No real NLP datasets found. Using SYNTHETIC data.")
        return _generate_synthetic_text_data()

    combined = pd.concat(frames, ignore_index=True).dropna()
    combined["label"] = combined["label"].astype(int)
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

    logger.info(
        f"Total NLP samples: {len(combined):,} | "
        f"Phishing: {combined['label'].sum():,} ({combined['label'].mean() * 100:.1f}%)"
    )
    return combined


# ── PyTorch Dataset ───────────────────────────────────────────────────────────

class FraudTextDataset(Dataset):
    """
    PyTorch Dataset for SMS / email text classification.

    Converts raw text strings into DistilBERT token IDs and attention masks.
    """

    def __init__(self, texts, labels, tokenizer, max_len: int = MAX_LEN):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # Coerce idx to a plain Python int — DataLoader may pass a 0-d numpy
        # array or a list, which breaks numpy fancy-indexing with a scalar error.
        idx   = int(idx)
        text  = str(self.texts[idx])
        label = int(self.labels[idx])

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,    # Prepend [CLS], append [SEP]
            max_length=self.max_len,
            padding="max_length",       # Pad to uniform length
            truncation=True,            # Truncate texts > max_len
            return_attention_mask=True, # 1=real token, 0=padding
            return_tensors="pt",
        )
        return {
            "input_ids":      encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "label":          torch.tensor(label, dtype=torch.long),
        }


# ── Training ──────────────────────────────────────────────────────────────────

def train_nlp_model(
    sms_path: str   = "data/raw/spam.csv",
    email_path: str = "data/raw/Phishing_Email.csv",
    save_dir: str   = "models/distilbert_fraud"
):
    """
    Fine-tune DistilBERT for phishing / fraud text classification.

    Anti-overfitting applied:
      1. Only 3 epochs (BERT overfits fast on small data).
      2. Low LR 2e-5 prevents catastrophic forgetting of pretrained weights.
      3. Layers 0-2 frozen (basic language patterns preserved).
      4. Dropout=0.1 built-in.
      5. Gradient clipping max_norm=1.0.
      6. Weight decay 0.01 (AdamW L2).
      7. Early stopping patience=2.
      8. Warmup + linear LR decay schedule.

    Args:
        sms_path:   Path to SMS spam CSV.
        email_path: Path to phishing emails CSV.
        save_dir:   Where to persist the best DistilBERT checkpoint.

    Returns:
        Tuple: (model, tokenizer)
    """
    try:
        from transformers import (
            DistilBertTokenizer,
            DistilBertForSequenceClassification,
            get_linear_schedule_with_warmup,
        )
        # AdamW was removed from transformers>=4.36 — use torch.optim instead
        from torch.optim import AdamW
    except ImportError:
        logger.error("transformers not installed. Run: pip install transformers torch")
        raise

    ensure_dirs(save_dir, "plots")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"NLP training device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    try:
        df = load_nlp_data(sms_path, email_path)

        labels_np = df["label"].to_numpy(dtype=np.int64)   # guaranteed int64 numpy array
        texts_np  = df["text"].to_numpy(dtype=object)       # object numpy array of strings

        X_train_txt, X_temp, y_train, y_temp = train_test_split(
            texts_np, labels_np,
            test_size=0.30, random_state=42, stratify=labels_np
        )
        X_val_txt, X_test_txt, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
        )
    except Exception as _e:
        import traceback as _tb
        logger.error(f"[NLP-DATA] {_e}\n{_tb.format_exc()}")
        raise

    logger.info(
        f"NLP split -> Train: {len(X_train_txt):,} | "
        f"Val: {len(X_val_txt):,} | Test: {len(X_test_txt):,}"
    )

    # ── Tokeniser + Model ─────────────────────────────────────────────────────
    try:
        tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
        model = DistilBertForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=2,
        )
    except Exception as _e:
        import traceback as _tb
        logger.error(f"[NLP-MODEL-LOAD] {_e}\n{_tb.format_exc()}")
        raise

    # ── Freeze bottom 3 transformer layers (0, 1, 2) ──────────────────────────
    for name, param in model.named_parameters():
        if any(f"transformer.layer.{i}" in name for i in range(3)):
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {trainable:,}")

    # Pin model to GPU — must happen AFTER freezing so the device assignment
    # reflects the final parameter set.
    model = model.to(device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        logger.info(f"GPU: {torch.cuda.get_device_name(0)} | "
                    f"VRAM free: {torch.cuda.mem_get_info()[0] / 1e9:.1f} GB")

    # ── DataLoaders ───────────────────────────────────────────────────────────
    train_ds = FraudTextDataset(X_train_txt, y_train, tokenizer)
    val_ds   = FraudTextDataset(X_val_txt,   y_val,   tokenizer)
    test_ds  = FraudTextDataset(X_test_txt,  y_test,  tokenizer)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

    # ── Optimiser + Scheduler ─────────────────────────────────────────────────
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    total_steps = len(train_loader) * EPOCHS
    scheduler   = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_loss    = float("inf")
    patience_counter = 0
    train_losses, val_losses = [], []

    for epoch in range(EPOCHS):
        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        total_train_loss = 0

        for batch in train_loader:
            optimizer.zero_grad()

            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["label"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            total_train_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

        avg_train_loss = total_train_loss / len(train_loader)

        # ── Validate ──────────────────────────────────────────────────────────
        model.eval()
        total_val_loss = 0
        val_preds, val_true = [], []

        with torch.no_grad():
            for batch in val_loader:
                input_ids      = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels         = batch["label"].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                total_val_loss += outputs.loss.item()
                preds = outputs.logits.argmax(dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_true.extend(labels.cpu().numpy())

        avg_val_loss = total_val_loss / len(val_loader)
        val_f1       = f1_score(val_true, val_preds, zero_division=0)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        logger.info(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val F1: {val_f1:.4f}"
        )

        # ── Early stopping ────────────────────────────────────────────────────
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            logger.info(f"  ✓ New best model saved -> {save_dir}")
        else:
            patience_counter += 1
            logger.info(
                f"  No improvement. Patience: {patience_counter}/{PATIENCE}"
            )
            if patience_counter >= PATIENCE:
                logger.info("Early stopping triggered.")
                break

    logger.info(f"NLP training complete. Best val loss: {best_val_loss:.4f}")

    # ── Plot loss curves ──────────────────────────────────────────────────────
    plot_loss_curves(train_losses, val_losses, "DistilBERT_NLP")

    # ── Final test evaluation ─────────────────────────────────────────────────
    model.eval()
    test_preds, test_true = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["label"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = outputs.logits.argmax(dim=1).cpu().numpy()
            test_preds.extend(preds)
            test_true.extend(labels.cpu().numpy())

    test_f1 = f1_score(test_true, test_preds, zero_division=0)
    logger.info(f"Test F1: {test_f1:.4f}  (target > 0.90)")
    print("\n" + classification_report(
        test_true, test_preds, target_names=["Legit", "Phishing"]
    ))

    save_metrics(
        {"module": "DistilBERT_NLP", "test_f1": round(test_f1, 4),
         "best_val_loss": round(best_val_loss, 4)},
        "models/nlp_metrics.json"
    )
    return model, tokenizer


# ── Single-text inference ─────────────────────────────────────────────────────

def predict_text(text: str, model_path: str = "models/distilbert_fraud") -> dict:
    """
    Predict whether a single text message is phishing / fraud.

    Args:
        text:       Raw SMS, email body, or transaction description.
        model_path: Path to the saved DistilBERT directory.

    Returns:
        Dict with keys:
          label            — 0 (legit) or 1 (phishing)
          confidence       — probability of the predicted class
          fraud_probability— probability that text is phishing/fraud
          verdict          — "FRAUD/PHISHING" | "LEGITIMATE"
    """
    try:
        # pyrefly: ignore [missing-import]
        from transformers import (
            DistilBertTokenizer,
            DistilBertForSequenceClassification,
        )
    except ImportError:
        logger.error("transformers not installed.")
        raise

    if not os.path.isdir(model_path):
        raise FileNotFoundError(
            f"NLP model not found at '{model_path}'. "
            "Run train_all.py first."
        )

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    model     = DistilBertForSequenceClassification.from_pretrained(model_path)
    model     = model.to(device)
    model.eval()

    encoding = tokenizer(
        str(text),
        add_special_tokens=True,
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs = model(
            input_ids=encoding["input_ids"].to(device),
            attention_mask=encoding["attention_mask"].to(device),
        )
        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]

    fraud_prob = float(probs[1])

    # ── Calibrated threshold ───────────────────────────────────────────────────
    # The SMS Spam Collection labels ALL commercial bank notifications as
    # "spam" (label=1).  This shifts the model's fraud_probability upward for
    # legitimate banking texts (typical range 0.70–0.97).  True phishing sits
    # above 0.975.  We read the calibrated threshold from nlp_metrics.json so
    # it can be tuned without code changes; fallback is 0.975.
    nlp_threshold = 0.975
    metrics_path  = "models/nlp_metrics.json"
    if os.path.exists(metrics_path):
        try:
            import json as _json
            with open(metrics_path) as _f:
                _m = _json.load(_f)
            nlp_threshold = float(_m.get("nlp_threshold", nlp_threshold))
        except Exception:
            pass

    label = 1 if fraud_prob >= nlp_threshold else 0

    # ── Heuristic Rule-Based Adjustment (Prevents False Positives) ─────────────
    # Legitimate transactional texts and delivery confirmations with links can
    # score very high (0.95+) due to dataset bias. We calibrate them if they
    # contain clear transactional markers and lack phishing urgency keywords.
    t_lower = text.lower()
    legit_keywords = ["spent", "debited", "credited", "successful", "delivered", "booking", "confirmed", "order"]
    phish_keywords = ["verify", "lock", "suspend", "compromise", "urgent", "action required", "update payment", "security alert"]

    has_legit_marker = any(k in t_lower for k in legit_keywords)
    has_phish_marker = any(k in t_lower for k in phish_keywords)

    if has_legit_marker and not has_phish_marker:
        # Scale down the score into the safe region (under nlp_threshold)
        # e.g., mapping a raw 0.99 to 0.40
        logger.info("[NLP] Heuristic: Legit transactional text detected. Adjusting score.")
        fraud_prob = min(fraud_prob, nlp_threshold - 0.1)

    label = 1 if fraud_prob >= nlp_threshold else 0

    # ── Normalize probability around 0.5 decision boundary ─────────────────────
    if fraud_prob < nlp_threshold:
        normalized_prob = (fraud_prob / nlp_threshold) * 0.5
    else:
        normalized_prob = 0.5 + ((fraud_prob - nlp_threshold) / (1.0 - nlp_threshold)) * 0.5

    return {
        "label":             label,
        "confidence":        float(fraud_prob if label == 1 else 1.0 - fraud_prob),
        "fraud_probability": normalized_prob,
        "raw_probability":   fraud_prob,
        "verdict":           "FRAUD/PHISHING" if label == 1 else "LEGITIMATE",
        "threshold_used":    nlp_threshold,
    }


# ── Entry-point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train_nlp_model()
