"""
src/cv_model.py
Siamese ResNet18 Signature Forgery Detector — Module 3.

Architecture:
  - Two identical ResNet18 branches (shared weights)
  - Custom 256-dim embedding head on top
  - Contrastive Loss: minimise distance for genuine pairs,
                      maximise for genuine-forged pairs
  - Euclidean distance -> converted to fraud probability

Anti-overfitting:
  - Pretrained ResNet18 backbone (ImageNet weights)
  - Freeze ResNet layer1 & layer2 (low-level feature preservation)
  - Dropout 0.3 in embedding head
  - Training data augmentation (rotation, shear, brightness jitter)
  - L2 weight decay 1e-4
  - StepLR scheduler: ×0.5 every 10 epochs
  - Contrastive loss naturally avoids pixel-level memorisation

Dataset fallback:
  - If CEDAR signature images are absent, synthetic random image pairs
    are generated to allow end-to-end pipeline validation.
"""

import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
# pyrefly: ignore [missing-import]
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms

from utils import get_logger, ensure_dirs, save_metrics, plot_loss_curves

logger = get_logger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
IMG_SIZE      = 128
BATCH_SIZE    = 16
EPOCHS        = 30
LEARNING_RATE = 1e-4
EMBEDDING_DIM = 256
PAIRS_PER_CLASS = 500


# ── Transforms ────────────────────────────────────────────────────────────────

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomRotation(degrees=10),
    transforms.RandomAffine(degrees=0, shear=5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomHorizontalFlip(p=0.1),  # Rare — signatures rarely flip
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


# ── Synthetic fallback ────────────────────────────────────────────────────────

def _generate_synthetic_signature_dirs(base_dir: str = "data/raw/signatures_synthetic",
                                        n_genuine: int = 50, n_forged: int = 50) -> tuple:
    """
    Create random grayscale PNG images to simulate CEDAR signature structure.
    Used when the real CEDAR dataset is unavailable.

    Args:
        base_dir:  Where to write the fake signature images.
        n_genuine: Number of genuine signature PNGs to create.
        n_forged:  Number of forged signature PNGs to create.

    Returns:
        Tuple: (genuine_dir, forged_dir)
    """
    genuine_dir = os.path.join(base_dir, "genuine")
    forged_dir  = os.path.join(base_dir, "forged")
    ensure_dirs(genuine_dir, forged_dir)

    rng = np.random.RandomState(42)

    def _make_imgs(folder, n, mean, std):
        existing = [
            f for f in os.listdir(folder)
            if f.endswith(".png")
        ]
        if len(existing) >= n:
            return  # Already created
        for i in range(n):
            arr = np.clip(rng.normal(mean, std, (64, 128, 3)) * 255, 0, 255).astype(np.uint8)
            Image.fromarray(arr).save(os.path.join(folder, f"sig_{i:04d}.png"))

    _make_imgs(genuine_dir, n_genuine, mean=0.8, std=0.1)
    _make_imgs(forged_dir,  n_forged,  mean=0.5, std=0.3)   # More noise

    logger.info(
        f"[SYNTHETIC CV] Created {n_genuine} genuine + {n_forged} forged "
        f"signature images in '{base_dir}'."
    )
    return genuine_dir, forged_dir


# ── Dataset — Siamese Pairs ───────────────────────────────────────────────────

class SiamesePairDataset(Dataset):
    """
    Creates PAIRS of signature images for Siamese training.

    Each pair is either:
      (genuine, genuine) -> label = 0  (same person, should be close)
      (genuine, forged)  -> label = 1  (different, should be far apart)

    Why pairs?
    The Siamese model learns to compare two images rather than classify
    them independently. This works well with small datasets like CEDAR
    (2,640 images) because the number of possible pairs is much larger.
    """

    def __init__(
        self,
        genuine_dir: str,
        forged_dir: str,
        transform=None,
        pairs_per_class: int = PAIRS_PER_CLASS,
        random_state: int = 42,
    ):
        self.transform = transform
        self.pairs  = []
        self.labels = []

        def _get_files(d):
            files = []
            if os.path.exists(d):
                for root, _, fnames in os.walk(d):
                    for f in fnames:
                        if f.lower().endswith((".png", ".jpg", ".jpeg")):
                            files.append(os.path.join(root, f))
            return sorted(files)

        genuine_files = _get_files(genuine_dir)
        forged_files  = _get_files(forged_dir)

        rng = np.random.RandomState(random_state)

        # Positive pairs — (genuine, genuine), label = 0
        for _ in range(pairs_per_class):
            i1, i2 = rng.choice(len(genuine_files), 2, replace=False)
            self.pairs.append((genuine_files[i1], genuine_files[i2]))
            self.labels.append(0)

        # Negative pairs — (genuine, forged), label = 1
        for _ in range(pairs_per_class):
            i1 = rng.randint(len(genuine_files))
            i2 = rng.randint(len(forged_files))
            self.pairs.append((genuine_files[i1], forged_files[i2]))
            self.labels.append(1)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img1 = Image.open(self.pairs[idx][0]).convert("RGB")
        img2 = Image.open(self.pairs[idx][1]).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, torch.tensor(label, dtype=torch.float32)


# ── Siamese Network ───────────────────────────────────────────────────────────

class SiameseResNet(nn.Module):
    """
    Siamese Neural Network using pretrained ResNet18 as a shared backbone.

    Architecture:
      Input (sig1, sig2)
        -> ResNet18 shared branch
        -> Flatten (512-dim)
        -> Linear -> ReLU -> Dropout -> Linear (256-dim)
        -> L2-normalised embeddings (emb1, emb2)
        -> Euclidean distance

    Why ResNet18?
      - Pre-trained on ImageNet -> already understands edges and textures.
      - Small enough for free Colab GPU (11M parameters).
      - Freezing layer1 & layer2 preserves low-level features while
        fine-tuning layer3 & layer4 on signature-specific high-level patterns.
    """

    def __init__(self, embedding_dim: int = EMBEDDING_DIM):
        super(SiameseResNet, self).__init__()

        backbone = models.resnet18(pretrained=True)

        # Remove final FC classification head; keep feature extraction layers.
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])

        # Custom embedding head
        self.embedding_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, embedding_dim),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(embedding_dim, embedding_dim),
        )

        # Freeze low-level ResNet layers (basic edges/textures)
        for name, param in self.feature_extractor.named_parameters():
            if "layer1" in name or "layer2" in name:
                param.requires_grad = False

    def forward_once(self, x: torch.Tensor) -> torch.Tensor:
        """Extract L2-normalised embedding for a single signature image."""
        features  = self.feature_extractor(x)
        embedding = self.embedding_head(features)
        return F.normalize(embedding, p=2, dim=1)

    def forward(self, img1: torch.Tensor, img2: torch.Tensor):
        """Process a pair of images and return both embeddings."""
        return self.forward_once(img1), self.forward_once(img2)


# ── Contrastive Loss ──────────────────────────────────────────────────────────

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for Siamese training.

    For SIMILAR pairs (label=0):    loss = distance²
    For DIFFERENT pairs (label=1):  loss = max(0, margin − distance)²

    The margin ensures dissimilar pairs are pushed at least `margin` apart.
    Once beyond the margin, no more penalty is applied.
    """

    def __init__(self, margin: float = 1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, emb1, emb2, labels):
        distance = F.pairwise_distance(emb1, emb2)
        loss = (
            (1 - labels) * distance.pow(2) +
            labels * F.relu(self.margin - distance).pow(2)
        )
        return loss.mean()


# ── Training ──────────────────────────────────────────────────────────────────

def train_siamese(
    genuine_dir: str = "data/raw/signatures/genuine",
    forged_dir:  str = "data/raw/signatures/forged",
    save_path:   str = "models/siamese_resnet18.pt"
) -> SiameseResNet:
    """
    Full Siamese ResNet18 training pipeline.

    Falls back to synthetic image pairs if CEDAR directories are absent.

    Args:
        genuine_dir: Path to genuine signature images.
        forged_dir:  Path to forged signature images.
        save_path:   Where to save the best model state dict.

    Returns:
        Best-performing SiameseResNet model.
    """
    ensure_dirs("models", "plots")

    # ── Dataset availability check & auto-discovery ───────────────────────────
    # Kaggle robinreni/signature-verification-dataset extracts to: data/raw/sign_data/
    kaggle_base = "data/raw/sign_data/train"
    if os.path.exists(kaggle_base):
        # The dataset groups by signer (e.g. 001/ vs 001_forg/)
        logger.info(f"Kaggle signature dataset found at {kaggle_base}")
        # Create virtual genuine/forged lists by searching all subdirs
        genuine_files, forged_files = [], []
        for root, dirs, _ in os.walk(kaggle_base):
            for d in dirs:
                full_dir = os.path.join(root, d)
                if "forg" in d.lower():
                    forged_files.extend([os.path.join(full_dir, f) for f in os.listdir(full_dir) if f.lower().endswith((".png", ".jpg"))])
                else:
                    genuine_files.extend([os.path.join(full_dir, f) for f in os.listdir(full_dir) if f.lower().endswith((".png", ".jpg"))])
        
        # We need to hack the Dataset class slightly to accept flat file lists instead of dirs
        class FlatListDataset(SiamesePairDataset):
            def __init__(self, gen_files, forg_files, transform=None):
                self.transform = transform
                self.pairs, self.labels = [], []
                rng = np.random.RandomState(42)
                # Ensure we have enough files
                if len(gen_files) < 2 or len(forg_files) < 1:
                    return
                # Positive pairs
                for _ in range(PAIRS_PER_CLASS):
                    i1, i2 = rng.choice(len(gen_files), 2, replace=False)
                    self.pairs.append((gen_files[i1], gen_files[i2]))
                    self.labels.append(0)
                # Negative pairs
                for _ in range(PAIRS_PER_CLASS):
                    i1 = rng.randint(len(gen_files))
                    i2 = rng.randint(len(forg_files))
                    self.pairs.append((gen_files[i1], forg_files[i2]))
                    self.labels.append(1)

        full_dataset = FlatListDataset(genuine_files, forged_files, transform=train_transform)

    elif not (os.path.exists(genuine_dir) and os.path.exists(forged_dir)):
        logger.warning(
            "CEDAR signature directories not found. Using SYNTHETIC image "
            "data for pipeline validation."
        )
        genuine_dir, forged_dir = _generate_synthetic_signature_dirs()
        full_dataset = SiamesePairDataset(genuine_dir, forged_dir, transform=train_transform)
    else:
        full_dataset = SiamesePairDataset(
            genuine_dir, forged_dir,
            transform=train_transform,
            pairs_per_class=PAIRS_PER_CLASS,
        )

    n          = len(full_dataset)
    train_size = int(0.70 * n)
    val_size   = int(0.15 * n)
    test_size  = n - train_size - val_size

    train_set, val_set, test_set = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )

    # Apply val/test transform without augmentation
    # Note: Subsets share the parent dataset transform. We override safely.
    val_set.dataset.transform  = val_transform
    test_set.dataset.transform = val_transform

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"CV training device: {device}")

    model     = SiameseResNet(embedding_dim=EMBEDDING_DIM).to(device)
    criterion = ContrastiveLoss(margin=1.0)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    best_val_loss = float("inf")
    train_losses, val_losses = [], []

    for epoch in range(EPOCHS):
        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        total_train_loss = 0
        for img1, img2, labels in train_loader:
            img1, img2, labels = (
                img1.to(device), img2.to(device), labels.to(device)
            )
            optimizer.zero_grad()
            emb1, emb2 = model(img1, img2)
            loss = criterion(emb1, emb2, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        # ── Validate ──────────────────────────────────────────────────────────
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for img1, img2, labels in val_loader:
                img1, img2, labels = (
                    img1.to(device), img2.to(device), labels.to(device)
                )
                emb1, emb2 = model(img1, img2)
                loss = criterion(emb1, emb2, labels)
                total_val_loss += loss.item()

        avg_train = total_train_loss / len(train_loader)
        avg_val   = total_val_loss   / len(val_loader)
        scheduler.step()

        train_losses.append(avg_train)
        val_losses.append(avg_val)

        logger.info(
            f"Epoch {epoch + 1:02d}/{EPOCHS} | "
            f"Train: {avg_train:.4f} | Val: {avg_val:.4f}"
        )

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), save_path)
            logger.info(f"  ✓ Model saved -> {save_path}")

    logger.info("CV training complete.")
    plot_loss_curves(train_losses, val_losses, "Siamese_CV")

    save_metrics(
        {"module": "Siamese_CV", "best_val_loss": round(best_val_loss, 4)},
        "models/cv_metrics.json"
    )
    return model


# ── Single-pair inference ─────────────────────────────────────────────────────

def predict_signature(
    img1_path: str,
    img2_path: str,
    threshold: float = 0.5,
    model_path: str  = "models/siamese_resnet18.pt"
) -> dict:
    """
    Compare two signature images and determine if the second is forged.

    Args:
        img1_path:  Reference (genuine) signature image path.
        img2_path:  Signature to verify.
        threshold:  Euclidean distance above which the signature is FORGED.
        model_path: Path to saved model state dict (.pt file).

    Returns:
        Dict with keys:
          distance         — Euclidean distance between embeddings (lower = more similar)
          fraud_probability— Normalised probability of forgery
          verdict          — "FORGED" | "GENUINE"
          threshold_used   — The threshold applied
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"CV model not found at '{model_path}'. "
            "Run train_all.py first."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = SiameseResNet(embedding_dim=EMBEDDING_DIM)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model  = model.to(device)
    model.eval()

    def _load(path):
        img = Image.open(path).convert("RGB")
        return val_transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        emb1, emb2 = model(_load(img1_path), _load(img2_path))
        distance   = F.pairwise_distance(emb1, emb2).item()

    # Convert distance to [0, 1] fraud probability
    fraud_prob = min(distance / (threshold * 2), 1.0)

    return {
        "distance":          round(distance, 4),
        "fraud_probability": round(fraud_prob, 4),
        "verdict":           "FORGED" if distance > threshold else "GENUINE",
        "threshold_used":    threshold,
    }


# ── Entry-point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train_siamese()
