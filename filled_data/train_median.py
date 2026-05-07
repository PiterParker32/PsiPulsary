"""
Pulsar MLP — Full From-Scratch PyTorch Pipeline
=================================================
Data expected in the same directory:
  - train_filled.xlsx   : 12,528 labelled rows (target_class column)
  - test_filled.xlsx    : 5,370 unlabelled rows (no target_class)

Pipeline:
  1. Load & clip physically impossible imputed values
  2. Missingness flag as extra feature
  3. Mixup augmentation on pulsars
  4. StratifiedKFold CV
     - StandardScaler fit on train fold only
     - Jitter oversampling on train fold only (no imblearn)
     - WeightedRandomSampler for balanced batches
     - MLP: 64→32, BatchNorm, ReLU, Dropout
     - AdamW + OneCycleLR scheduler
     - BCEWithLogitsLoss with pos_weight=9
     - Gradient clipping
     - Early stopping on PR-AUC
  5. Threshold tuning via PR curve (OOF predictions)
  6. Pseudo-labelling of unlabelled test set
  7. Final model retrained on labelled + pseudo-labelled data
  8. Predictions saved to pulsar_predictions.csv

Dependencies: numpy, pandas, torch, scikit-learn  (no imblearn)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    average_precision_score, roc_auc_score,
    precision_recall_curve, classification_report,
)

# ─────────────────────────────────────────────────────────────
# CONFIG  — tweak these without touching the rest of the script
# ─────────────────────────────────────────────────────────────
CFG = dict(
    train_path        = "train_filled.xlsx",
    test_path         = "test_filled.xlsx",
    output_path       = "pulsar_predictions.csv",

    # Clipping bounds for physically impossible imputed values
    clip_rules        = {
        "Standard deviation of the DM-SNR curve": (7.37,  None),
        "Skewness of the DM-SNR curve":           (-1.98, None),
    },

    # Mixup augmentation (pulsar pairs interpolated)
    n_synthetic       = 400,
    mixup_alpha       = 0.4,

    # Jitter oversampling (replaces SMOTE — pure numpy, no imblearn)
    oversample_ratio  = 0.3,   # target minority/majority ratio after oversampling
    jitter_std        = 0.05,  # gaussian noise std added to repeated pulsars (scaled space)

    # Sampler
    pos_sample_weight = 9.0,       # how much more likely to sample a pulsar

    # Model
    hidden_layers     = (64, 32),
    dropout           = (0.3, 0.2),

    # Loss
    pos_weight        = 9.0,       # BCEWithLogitsLoss pos_weight

    # Optimiser
    lr                = 1e-3,
    weight_decay      = 1e-4,

    # Scheduler (OneCycleLR)
    max_lr            = 1e-2,
    pct_start         = 0.3,       # fraction of training used for warmup

    # Training loop
    epochs            = 200,
    batch_size        = 64,
    grad_clip         = 1.0,
    early_stop_patience = 15,      # epochs without PR-AUC improvement

    # CV
    n_folds           = 5,
    random_state      = 42,

    # Pseudo-labelling
    pseudo_pos_thresh = 0.90,      # confidence to accept as pulsar
    pseudo_neg_thresh = 0.05,      # confidence to accept as noise
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ─────────────────────────────────────────────────────────────
# 1. DATA LOADING & CLIPPING
# ─────────────────────────────────────────────────────────────
def load_and_clip(cfg: dict):
    train = pd.read_excel(cfg["train_path"])
    test  = pd.read_excel(cfg["test_path"])
    raw_train = pd.read_csv(
        "../pulsar_data_train.csv",
        skipinitialspace=True,
        usecols=lambda c: not c.startswith("Unnamed"),
    )
    raw_test = pd.read_csv(
        "../pulsar_data_test.csv",
        skipinitialspace=True,
        usecols=lambda c: not c.startswith("Unnamed"),
    )

    # ── Clip physically impossible imputed values
    for df in [train, test]:
        for col, (lo, hi) in cfg["clip_rules"].items():
            if col in df.columns:
                df[col] = df[col].clip(lower=lo, upper=hi)

    feature_cols = [c for c in train.columns if c != "target_class"]

    X       = train[feature_cols].values.astype(np.float32)
    y       = train["target_class"].values.astype(np.float32)
    X_unlab = test[feature_cols].values.astype(np.float32)

    # ── Sanity-check that column names now align between raw and filled
    raw_feature_cols_tr = [c for c in feature_cols if c in raw_train.columns]
    raw_feature_cols_te = [c for c in feature_cols if c in raw_test.columns]

    missing_tr = set(feature_cols) - set(raw_train.columns)
    missing_te = set(feature_cols) - set(raw_test.columns)
    if missing_tr:
        raise ValueError(
            f"These feature columns are absent from raw_train CSV "
            f"(likely a name mismatch after stripping spaces):\n  {missing_tr}"
        )
    if missing_te:
        raise ValueError(
            f"These feature columns are absent from raw_test CSV:\n  {missing_te}"
        )

    # ── Row-count guard before building the flag
    assert len(raw_train) == len(train), (
        f"Train row mismatch: raw CSV has {len(raw_train)} rows, "
        f"filled xlsx has {len(train)}"
    )
    assert len(raw_test) == len(test), (
        f"Test row mismatch: raw CSV has {len(raw_test)} rows, "
        f"filled xlsx has {len(test)}"
    )

    # ── Missingness flag: 1 if ANY feature column was NaN in the raw file
    miss_flag       = (raw_train[raw_feature_cols_tr].isnull().any(axis=1)
                       .values.astype(np.float32).reshape(-1, 1))
    miss_flag_unlab = (raw_test[raw_feature_cols_te].isnull().any(axis=1)
                       .values.astype(np.float32).reshape(-1, 1))

    X       = np.hstack([X,       miss_flag])
    X_unlab = np.hstack([X_unlab, miss_flag_unlab])

    n_miss_tr = int(miss_flag.sum())
    n_miss_te = int(miss_flag_unlab.sum())
    print(f"Train : {X.shape}  Pulsars: {int(y.sum())} ({y.mean()*100:.1f}%)")
    print(f"  └─ rows with any imputed feature : {n_miss_tr} ({n_miss_tr/len(X)*100:.1f}%)")
    print(f"Unlabelled test: {X_unlab.shape}")
    print(f"  └─ rows with any imputed feature : {n_miss_te} ({n_miss_te/len(X_unlab)*100:.1f}%)")

    return X, y, X_unlab


# ─────────────────────────────────────────────────────────────
# 2. MIXUP AUGMENTATION (pulsars only)
# ─────────────────────────────────────────────────────────────
def mixup_pulsars(X: np.ndarray, y: np.ndarray, cfg: dict) -> tuple:
    """
    Interpolate between random pairs of real pulsars.
    """
    rng    = np.random.RandomState(cfg["random_state"])
    X_pos  = X[y == 1]
    n      = cfg["n_synthetic"]
    lam    = rng.beta(cfg["mixup_alpha"], cfg["mixup_alpha"], size=(n, 1))
    idx_a  = rng.randint(0, len(X_pos), n)
    idx_b  = rng.randint(0, len(X_pos), n)
    X_syn  = lam * X_pos[idx_a] + (1 - lam) * X_pos[idx_b]
    y_syn  = np.ones(n, dtype=np.float32)

    X_aug  = np.vstack([X,  X_syn])
    y_aug  = np.concatenate([y, y_syn])
    print(f"After Mixup — Total: {len(X_aug)}  Pulsars: {int(y_aug.sum())}")
    return X_aug, y_aug


# ─────────────────────────────────────────────────────────────
# 3. JITTER OVERSAMPLING
# ─────────────────────────────────────────────────────────────
def jitter_oversample(X: np.ndarray, y: np.ndarray, cfg: dict,
                      rng: np.random.RandomState) -> tuple:
    """
    Oversample the minority class (pulsars) to reach `oversample_ratio`
    by repeating real pulsar rows and adding small Gaussian noise.

    Why jitter instead of SMOTE:
      - SMOTE interpolates between a sample and one of its K nearest neighbours,
        which can create samples in sparse regions or across class boundaries.
      - Jitter stays tightly around real pulsars - safer for a small minority class
        where neighbours are already rare.
      - `jitter_std=0.05` in standardised space around 5% of one standard deviation,
        enough for diversity without distorting the pulsar distribution.

    Must be called AFTER StandardScaler so noise magnitude is meaningful.
    Must be called on the TRAINING FOLD ONLY — never on validation.
    """
    X_pos    = X[y == 1]
    X_neg    = X[y == 0]
    n_pos    = len(X_pos)
    n_neg    = len(X_neg)

    # How many synthetic pulsars do we need?
    target_n_pos = int(n_neg * cfg["oversample_ratio"])
    n_extra      = max(0, target_n_pos - n_pos)

    if n_extra == 0:
        return X, y

    # Sample with replacement from existing pulsars, add Gaussian jitter
    idx      = rng.randint(0, n_pos, n_extra)
    noise    = rng.normal(0, cfg["jitter_std"], size=(n_extra, X_pos.shape[1]))
    X_extra  = X_pos[idx] + noise.astype(np.float32)
    y_extra  = np.ones(n_extra, dtype=np.float32)

    X_out = np.vstack([X, X_extra])
    y_out = np.concatenate([y, y_extra])
    return X_out, y_out


# ─────────────────────────────────────────────────────────────
# 3. PYTORCH DATASET
# ─────────────────────────────────────────────────────────────
class PulsarDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray | None = None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]


def make_loader(X: np.ndarray, y: np.ndarray, cfg: dict,
                weighted: bool = True) -> DataLoader:
    """
    weighted=True  → WeightedRandomSampler (for training batches)
    weighted=False → sequential, no sampler (for validation / inference)
    """
    dataset = PulsarDataset(X, y)
    if weighted:
        w       = np.where(y == 1, cfg["pos_sample_weight"], 1.0)
        sampler = WeightedRandomSampler(
            weights     = torch.tensor(w, dtype=torch.float32),
            num_samples = len(y),
            replacement = True,
        )
        return DataLoader(dataset, batch_size=cfg["batch_size"],
                          sampler=sampler)
    return DataLoader(dataset, batch_size=cfg["batch_size"],
                      shuffle=False)


# ─────────────────────────────────────────────────────────────
# 4. MODEL
# ─────────────────────────────────────────────────────────────
class PulsarMLP(nn.Module):
    """
    Two hidden layers with BatchNorm → ReLU → Dropout.
    Output is a raw logit (no sigmoid) - BCEWithLogitsLoss handles that.

    Architecture choices:
      - BatchNorm: stabilises training on tabular data with varied feature scales
      - Dropout decreasing (0.3→0.2): more regularisation where representation is richer
      - No sigmoid output: BCEWithLogitsLoss is numerically more stable and allows for punishing missing pulsars more
    """
    def __init__(self, input_dim: int, hidden: tuple, dropout: tuple):
        super().__init__()
        h1, h2 = hidden
        d1, d2 = dropout
        self.net = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.BatchNorm1d(h1),
            nn.ReLU(),
            nn.Dropout(d1),

            nn.Linear(h1, h2),
            nn.BatchNorm1d(h2),
            nn.ReLU(),
            nn.Dropout(d2),

            nn.Linear(h2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


def build_model(input_dim: int, cfg: dict) -> PulsarMLP:
    return PulsarMLP(
        input_dim = input_dim,
        hidden    = cfg["hidden_layers"],
        dropout   = cfg["dropout"],
    ).to(DEVICE)


# ─────────────────────────────────────────────────────────────
# 5. TRAINING & EVALUATION HELPERS
# ─────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, scheduler, criterion,
                    grad_clip: float) -> float:
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss   = criterion(logits, y_batch)
        loss.backward()
        # Gradient clipping: prevents exploding gradients caused by pos_weight=9
        # spiking the loss on pulsar-heavy batches early in training
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        scheduler.step()   # OneCycleLR steps per BATCH, not per epoch
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def predict_proba(model, loader) -> np.ndarray:
    model.eval()
    probas = []
    for batch in loader:
        X_batch = batch[0].to(DEVICE) if isinstance(batch, (list, tuple)) else batch.to(DEVICE)
        logits  = model(X_batch)
        probas.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probas)


def train_fold(X_tr, y_tr, X_val, y_val, cfg: dict) -> tuple:
    """
    Full training run for one CV fold.
    Returns (best_model_state, val_proba).
    """
    rng = np.random.RandomState(cfg["random_state"])

    # Jitter oversample - only on training fold, never on validation
    X_tr_r, y_tr_r = jitter_oversample(X_tr, y_tr, cfg, rng)

    train_loader = make_loader(X_tr_r, y_tr_r, cfg, weighted=True)
    val_loader   = make_loader(X_val,  y_val,  cfg, weighted=False)

    input_dim  = X_tr.shape[1]
    model      = build_model(input_dim, cfg)
    criterion  = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([cfg["pos_weight"]], device=DEVICE)
    )
    optimizer  = torch.optim.AdamW(
        model.parameters(),
        lr           = cfg["lr"],
        weight_decay = cfg["weight_decay"],
    )
    # OneCycleLR: ramps lr up for pct_start of training then decays
    scheduler  = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr           = cfg["max_lr"],
        steps_per_epoch  = len(train_loader),
        epochs           = cfg["epochs"],
        pct_start        = cfg["pct_start"],
    )

    best_prauc  = 0.0
    best_state  = None
    patience_ct = 0

    for epoch in range(cfg["epochs"]):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler,
            criterion, cfg["grad_clip"]
        )
        val_proba = predict_proba(model, val_loader)
        prauc     = average_precision_score(y_val, val_proba)

        if prauc > best_prauc:
            best_prauc  = prauc
            best_state  = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ct = 0
        else:
            patience_ct += 1
            if patience_ct >= cfg["early_stop_patience"]:
                print(f"    Early stop at epoch {epoch+1}  best PR-AUC={best_prauc:.4f}")
                break

    model.load_state_dict(best_state)
    val_proba = predict_proba(model, val_loader)
    return model, val_proba


# ─────────────────────────────────────────────────────────────
# 6. STRATIFIED K-FOLD CROSS-VALIDATION
# ─────────────────────────────────────────────────────────────
def cross_validate(X_aug: np.ndarray, y_aug: np.ndarray,
                   cfg: dict) -> np.ndarray:
    skf     = StratifiedKFold(n_splits=cfg["n_folds"], shuffle=True,
                               random_state=cfg["random_state"])
    oof     = np.zeros(len(X_aug))

    print(f"\n{'─'*50}")
    print("Cross-Validation")
    print(f"{'─'*50}")

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_aug, y_aug)):
        X_tr, X_val = X_aug[tr_idx], X_aug[val_idx]
        y_tr, y_val = y_aug[tr_idx], y_aug[val_idx]

        # Scaler fit on training fold only !!!
        scaler = StandardScaler()
        X_tr   = scaler.fit_transform(X_tr)
        X_val  = scaler.transform(X_val)

        _, val_proba = train_fold(X_tr, y_tr, X_val, y_val, cfg)
        oof[val_idx] = val_proba

        roc   = roc_auc_score(y_val, val_proba)
        prauc = average_precision_score(y_val, val_proba)
        print(f"  Fold {fold+1}: ROC-AUC={roc:.4f}  PR-AUC={prauc:.4f}")

    print(f"\n  OOF ROC-AUC : {roc_auc_score(y_aug, oof):.4f}")
    print(f"  OOF PR-AUC  : {average_precision_score(y_aug, oof):.4f}  ← primary metric")
    return oof


# ─────────────────────────────────────────────────────────────
# 7. THRESHOLD TUNING
# ─────────────────────────────────────────────────────────────
def tune_threshold(y_true: np.ndarray, oof_proba: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, oof_proba)
    f1          = 2 * precision * recall / (precision + recall + 1e-8)
    best_idx    = f1[:-1].argmax()
    best_thresh = float(thresholds[best_idx])

    print(f"\n{'─'*50}")
    print("Threshold Tuning (OOF predictions)")
    print(f"{'─'*50}")
    print(f"  Best threshold : {best_thresh:.4f}  (maximises F1)")
    print(f"  Precision      : {precision[best_idx]:.4f}")
    print(f"  Recall         : {recall[best_idx]:.4f}")
    print(f"  F1             : {f1[best_idx]:.4f}")
    print()
    print(classification_report(
        y_true, (oof_proba >= best_thresh).astype(int),
        target_names=["Noise", "Pulsar"]
    ))
    return best_thresh


# ─────────────────────────────────────────────────────────────
# 8. TRAIN FULL MODEL (on all augmented labelled data)
# ─────────────────────────────────────────────────────────────
def train_full_model(X: np.ndarray, y: np.ndarray,
                     cfg: dict) -> tuple:
    """Returns (model, scaler) trained on the entire dataset."""
    scaler  = StandardScaler()
    X_sc    = scaler.fit_transform(X)

    rng     = np.random.RandomState(cfg["random_state"])
    X_r, y_r = jitter_oversample(X_sc, y, cfg, rng)

    loader  = make_loader(X_r, y_r, cfg, weighted=True)
    input_dim = X.shape[1]
    model   = build_model(input_dim, cfg)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([cfg["pos_weight"]], device=DEVICE)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr          = cfg["max_lr"],
        steps_per_epoch = len(loader),
        epochs          = cfg["epochs"],
        pct_start       = cfg["pct_start"],
    )

    for epoch in range(cfg["epochs"]):
        train_one_epoch(model, loader, optimizer, scheduler,
                        criterion, cfg["grad_clip"])

    return model, scaler


# ─────────────────────────────────────────────────────────────
# 9. PSEUDO-LABELLING
# ─────────────────────────────────────────────────────────────
def pseudo_label(model, scaler, X_unlab: np.ndarray,
                 cfg: dict) -> tuple:
    """
    Run model on unlabelled test data, keep only high-confidence predictions.
    Uncertain samples (between thresholds) are discarded - better to ignore
    than to add noisy labels.
    """
    loader      = DataLoader(
        PulsarDataset(scaler.transform(X_unlab)),
        batch_size=cfg["batch_size"], shuffle=False
    )
    proba       = predict_proba(model, loader)

    mask_pos    = proba >= cfg["pseudo_pos_thresh"]
    mask_neg    = proba <= cfg["pseudo_neg_thresh"]
    mask_unsure = ~mask_pos & ~mask_neg

    print(f"\n{'─'*50}")
    print("Pseudo-Labelling")
    print(f"{'─'*50}")
    print(f"  Confident pulsars : {mask_pos.sum()}")
    print(f"  Confident noise   : {mask_neg.sum()}")
    print(f"  Discarded (unsure): {mask_unsure.sum()}")

    X_pseudo = np.vstack([X_unlab[mask_pos], X_unlab[mask_neg]])
    y_pseudo = np.concatenate([
        np.ones(mask_pos.sum(),  dtype=np.float32),
        np.zeros(mask_neg.sum(), dtype=np.float32),
    ])
    return X_pseudo, y_pseudo, proba


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    torch.manual_seed(CFG["random_state"])
    np.random.seed(CFG["random_state"])

    # ── 1. Load
    X, y, X_unlab = load_and_clip(CFG)
    n_original = len(X)          # ← remember boundary BEFORE mixup

    # ── 2. Mixup augmentation (appends synthetic rows at the END)
    X_aug, y_aug = mixup_pulsars(X, y, CFG)

    # ── 3. Cross-validate — OOF array covers ALL rows (real + synthetic)
    oof_proba = cross_validate(X_aug, y_aug, CFG)

    # ── 4. Threshold tuning on ORIGINAL rows only (no synthetic leakage)
    oof_orig  = oof_proba[:n_original]   # first n_original entries = real data
    y_orig    = y_aug[:n_original]       # identical to original y
    best_thresh = tune_threshold(y_orig, oof_orig)

    # ── 5. Train full model on all augmented labelled data
    print(f"\n{'─'*50}")
    print("Training full model on all labelled data...")
    model_full, scaler_full = train_full_model(X_aug, y_aug, CFG)

    # ── 6. Pseudo-label unlabelled test set
    X_pseudo, y_pseudo, test_proba_round1 = pseudo_label(
        model_full, scaler_full, X_unlab, CFG
    )

    # ── 7. Retrain on labelled + pseudo-labelled data
    X_final = np.vstack([X_aug, X_pseudo])
    y_final = np.concatenate([y_aug, y_pseudo])
    print(f"\nFinal dataset: {X_final.shape}  Pulsars: {int(y_final.sum())} ({y_final.mean()*100:.1f}%)")
    print("Retraining final model on labelled + pseudo-labelled data...")
    model_final, scaler_final = train_full_model(X_final, y_final, CFG)

    # ── 8. Evaluate final model on ORIGINAL labelled data only
    #        (no mixup rows, no jitter rows, no pseudo-labels)
    print(f"\n{'─'*50}")
    print("Final Model Evaluation — original labelled data only")
    print(f"{'─'*50}")

    eval_loader = make_loader(
        scaler_final.transform(X),   # original X, scaled with final scaler
        y,                            # original labels
        CFG,
        weighted=False,               # sequential, no sampler
    )
    eval_proba = predict_proba(model_final, eval_loader)
    eval_pred  = (eval_proba >= best_thresh).astype(int)

    roc   = roc_auc_score(y, eval_proba)
    prauc = average_precision_score(y, eval_proba)
    print(f"  ROC-AUC : {roc:.4f}")
    print(f"  PR-AUC  : {prauc:.4f}")
    print()
    print(classification_report(
        y, eval_pred,
        target_names=["Noise", "Pulsar"],
        digits=4,
    ))

    # ── 9. Generate test predictions
    test_loader = DataLoader(
        PulsarDataset(scaler_final.transform(X_unlab)),
        batch_size=CFG["batch_size"], shuffle=False
    )
    test_proba = predict_proba(model_final, test_loader)
    test_pred  = (test_proba >= best_thresh).astype(int)

    print(f"\n{'─'*50}")
    print("Final Predictions (unlabelled test set)")
    print(f"{'─'*50}")
    print(f"  Predicted pulsars : {test_pred.sum()} / {len(test_pred)} ({test_pred.mean()*100:.1f}%)")

    pd.DataFrame({
        "predicted_proba": test_proba,
        "predicted_class": test_pred,
    }).to_csv(CFG["output_path"], index=False)
    print(f"  Saved → {CFG['output_path']}")


if __name__ == "__main__":
    main()