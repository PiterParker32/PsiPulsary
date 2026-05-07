import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

# --- Configuration ---
cfg = {
    "batch_size": 64,
    "pseudo_pos_thresh": 0.85,
    "pseudo_neg_thresh": 0.15,
    "pos_sample_weight": 2.0,
    "lr": 0.001,
    "epochs": 50,
    "random_state": 42,
    "n_synthetic": 500,
    "mixup_alpha": 0.2,
    "oversample_ratio": 0.4,
    "jitter_std": 0.05,
}
rng = np.random.RandomState(cfg["random_state"])


# --- Data & Model Classes ---
class PulsarDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = (
            torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
            if y is not None
            else None
        )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (self.X[idx], self.y[idx]) if self.y is not None else self.X[idx]


class PulsarMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


# --- Logic Utilities ---
def jitter_oversample(X, y, cfg, rng):
    X_pos, X_neg = X[y == 1], X[y == 0]
    n_extra = max(0, int(len(X_neg) * cfg["oversample_ratio"]) - len(X_pos))
    if n_extra <= 0:
        return X, y
    idx = rng.randint(0, len(X_pos), n_extra)
    noise = rng.normal(0, cfg["jitter_std"], size=(n_extra, X_pos.shape[1])).astype(
        np.float32
    )
    return np.vstack([X, X_pos[idx] + noise]), np.concatenate(
        [y, np.ones(n_extra, dtype=np.float32)]
    )


def mixup_pulsars(X, y, cfg):
    X_pos = X[y == 1]
    n = cfg["n_synthetic"]
    lam = rng.beta(cfg["mixup_alpha"], cfg["mixup_alpha"], size=(n, 1)).astype(
        np.float32
    )
    idx_a, idx_b = rng.randint(0, len(X_pos), n), rng.randint(0, len(X_pos), n)
    X_syn = lam * X_pos[idx_a] + (1 - lam) * X_pos[idx_b]
    return np.vstack([X, X_syn]), np.concatenate([y, np.ones(n, dtype=np.float32)])


def make_loader(X, y, cfg, weighted=True):
    dataset = PulsarDataset(X, y)
    if weighted:
        w = np.where(y == 1, cfg["pos_sample_weight"], 1.0)
        sampler = WeightedRandomSampler(
            torch.tensor(w, dtype=torch.float32), len(y), replacement=True
        )
        return DataLoader(dataset, batch_size=cfg["batch_size"], sampler=sampler)
    return DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=False)


def predict_proba(model, loader):
    model.eval()
    probas = []
    with torch.no_grad():
        for batch in loader:
            X_batch = batch[0] if isinstance(batch, (list, tuple)) else batch
            probas.append(model(X_batch).cpu().numpy())
    return np.vstack(probas).flatten()


def train_model(loader, epochs, cfg):
    model = PulsarMLP()
    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"])
    criterion = nn.BCELoss()
    for _ in range(epochs):
        model.train()
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            criterion(model(X_batch), y_batch).backward()
            optimizer.step()
    return model


def evaluate(model, X_scaled, y_true, label):
    probas = predict_proba(model, make_loader(X_scaled, y_true, cfg, weighted=False))
    precision, recall, thresholds = precision_recall_curve(y_true, probas)
    f1_curve = 2 * precision * recall / (precision + recall + 1e-8)
    best_thresh = float(thresholds[f1_curve[:-1].argmax()])
    preds = (probas >= best_thresh).astype(int)
    print(f"\n--- {label} ---\nROC-AUC: {roc_auc_score(y_true, probas):.4f}")
    print(
        classification_report(y_true, preds, target_names=["Noise", "Pulsar"], digits=4)
    )
    return best_thresh


# --- Execution ---

# 1. Load & Split (Fixing Error 1: Split before imputation)
train_df = pd.read_excel("train_filled.xlsx")
test_df = pd.read_excel("test_filled.xlsx")
target_col = "target_class"

train_set, val_set = train_test_split(
    train_df,
    test_size=0.15,
    stratify=train_df[target_col],
    random_state=cfg["random_state"],
)

# 2. Impute using training-only statistics
medians = train_set.drop(target_col, axis=1).median()
X_train_raw = train_set.drop(target_col, axis=1).fillna(medians).values
y_train = train_set[target_col].values.astype(np.float32)
X_val_raw = val_set.drop(target_col, axis=1).fillna(medians).values
y_val = val_set[target_col].values.astype(np.float32)
X_test_raw = test_df.drop(target_col, axis=1, errors="ignore").fillna(medians).values

# 3. Scale based on training-only statistics
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_val = scaler.transform(X_val_raw)
X_test = scaler.transform(X_test_raw)

# 4. Augment (Fixing Error 3: Mixup on scaled data)
X_mix, y_mix = mixup_pulsars(X_train, y_train, cfg)
X_jit, y_jit = jitter_oversample(X_mix, y_mix, cfg, rng)

# 5. Round 1 Training
model = train_model(make_loader(X_jit, y_jit, cfg, weighted=True), cfg["epochs"], cfg)
evaluate(model, X_val, y_val, "Round 1 - Val Set")

# 6. Pseudo-labelling
probas_unlab = predict_proba(
    model, DataLoader(PulsarDataset(X_test), batch_size=cfg["batch_size"])
)
mask_pos, mask_neg = (
    probas_unlab >= cfg["pseudo_pos_thresh"],
    probas_unlab <= cfg["pseudo_neg_thresh"],
)
X_pseudo = X_test[mask_pos | mask_neg]
y_pseudo = np.concatenate([np.ones(mask_pos.sum()), np.zeros(mask_neg.sum())]).astype(
    np.float32
)

# 7. Round 2 Training
X_final_mix, y_final_mix = mixup_pulsars(
    np.vstack([X_train, X_pseudo]), np.concatenate([y_train, y_pseudo]), cfg
)
X_final_jit, y_final_jit = jitter_oversample(X_final_mix, y_final_mix, cfg, rng)
model_final = train_model(
    make_loader(X_final_jit, y_final_jit, cfg, weighted=True), cfg["epochs"], cfg
)

# 8. Final Eval & Test Predictions
best_thresh = evaluate(model_final, X_val, y_val, "Round 2 - Val Set")
final_probas = predict_proba(
    model_final, DataLoader(PulsarDataset(X_test), batch_size=cfg["batch_size"])
)
test_preds = (final_probas >= best_thresh).astype(int)
print(f"\nFinal Test Pulsars: {test_preds.sum()} / {len(test_preds)}")
