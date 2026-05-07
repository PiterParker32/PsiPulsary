import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# --- 1. Data Loading ---
train_df = pd.read_excel('train_filled.xlsx')
test_df = pd.read_excel('test_filled.xlsx')

train_medians = train_df.median()
train_df = train_df.fillna(train_medians)
test_df = test_df.fillna(train_medians)

target_col = 'target_class' 
X_train_raw = train_df.drop(target_col, axis=1).values
y_train_raw = train_df[target_col].values.astype(np.float32)
X_test_raw = test_df.drop(target_col, axis=1, errors='ignore').values

# --- 2. Configuration & Functions ---
cfg = {
    "batch_size": 64,
    "pseudo_pos_thresh": 0.85,
    "pseudo_neg_thresh": 0.15,
    "pos_sample_weight": 2.0,  # Lowered slightly as jitter also handles balance
    "lr": 0.001,
    "epochs": 50,
    "random_state": 42,
    "n_synthetic": 500,        # Mixup count
    "mixup_alpha": 0.2,
    "oversample_ratio": 0.4,   # Target ratio of pulsars to noise
    "jitter_std": 0.05         # Noise magnitude for oversampling
}

rng = np.random.RandomState(cfg["random_state"])

def jitter_oversample(X: np.ndarray, y: np.ndarray, cfg: dict,
                     rng: np.random.RandomState) -> tuple:
    X_pos, X_neg = X[y == 1], X[y == 0]
    n_pos, n_neg = len(X_pos), len(X_neg)

    target_n_pos = int(n_neg * cfg["oversample_ratio"])
    n_extra = max(0, target_n_pos - n_pos)

    if n_extra == 0:
        return X, y

    idx = rng.randint(0, n_pos, n_extra)
    noise = rng.normal(0, cfg["jitter_std"], size=(n_extra, X_pos.shape[1]))
    X_extra = X_pos[idx] + noise.astype(np.float32)
    y_extra = np.ones(n_extra, dtype=np.float32)

    return np.vstack([X, X_extra]), np.concatenate([y, y_extra])

def mixup_pulsars(X: np.ndarray, y: np.ndarray, cfg: dict) -> tuple:
    X_pos = X[y == 1]
    n = cfg["n_synthetic"]
    lam = rng.beta(cfg["mixup_alpha"], cfg["mixup_alpha"], size=(n, 1))
    idx_a, idx_b = rng.randint(0, len(X_pos), n), rng.randint(0, len(X_pos), n)
    X_syn = lam * X_pos[idx_a] + (1 - lam) * X_pos[idx_b]
    y_syn = np.ones(n, dtype=np.float32)
    return np.vstack([X, X_syn]), np.concatenate([y, y_syn])

def make_loader(X: np.ndarray, y: np.ndarray, cfg: dict, weighted: bool = True) -> DataLoader:
    dataset = PulsarDataset(X, y)
    if weighted:
        w = np.where(y == 1, cfg["pos_sample_weight"], 1.0)
        sampler = WeightedRandomSampler(
            weights=torch.tensor(w, dtype=torch.float32),
            num_samples=len(y),
            replacement=True,
        )
        return DataLoader(dataset, batch_size=cfg["batch_size"], sampler=sampler)
    return DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=False)

def predict_proba(model, loader):
    model.eval()
    probas = []
    with torch.no_grad():
        for batch in loader:
            output = model(batch)
            probas.append(output.cpu().numpy())
    return np.vstack(probas).flatten()

def pseudo_label(model, X_unlab_raw, scaler, cfg):
    X_unlab_scaled = scaler.transform(X_unlab_raw)
    loader = DataLoader(PulsarDataset(X_unlab_scaled), batch_size=cfg["batch_size"], shuffle=False)
    proba = predict_proba(model, loader)
    
    mask_pos = proba >= cfg["pseudo_pos_thresh"]
    mask_neg = proba <= cfg["pseudo_neg_thresh"]
    mask_unsure = ~mask_pos & ~mask_neg

    print(f"\n{'─'*50}\nPseudo-Labelling\n{'─'*50}")
    print(f"  Confident pulsars : {mask_pos.sum()}")
    print(f"  Confident noise   : {mask_neg.sum()}")
    print(f"  Discarded back    : {mask_unsure.sum()}")

    # Use forced labels for the entire test set to ensure no data is lost
    X_pseudo = X_unlab_raw 
    y_pseudo = (proba > 0.5).astype(np.float32)
    return X_pseudo, y_pseudo

class PulsarDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1) if y is not None else None
    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        return (self.X[idx], self.y[idx]) if self.y is not None else self.X[idx]

class PulsarMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, 16), nn.ReLU(),
            nn.Linear(16, 8), nn.ReLU(),
            nn.Linear(8, 1),  nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)

# --- 3. Training Pipeline ---

# Step 1: Initial Mixup Augmentation (Raw Space)
X_train_mix, y_train_mix = mixup_pulsars(X_train_raw, y_train_raw, cfg)

# Step 2: Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_mix)

# Step 3: Jitter Oversampling (Scaled Space)
X_train_jit, y_train_jit = jitter_oversample(X_train_scaled, y_train_mix, cfg, rng)

# Step 4: First Training Pass
train_loader = make_loader(X_train_jit, y_train_jit, cfg, weighted=True)
model = PulsarMLP()
optimizer = optim.Adam(model.parameters(), lr=cfg["lr"])
criterion = nn.BCELoss()

for _ in range(cfg["epochs"]):
    model.train()
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        criterion(model(X_batch), y_batch).backward()
        optimizer.step()

# Step 5: Pseudo-labeling (Test set)
X_ps_raw, y_ps = pseudo_label(model, X_test_raw, scaler, cfg)

# Step 6: Combine all data and re-scale
X_final_raw = np.vstack([X_train_mix, X_ps_raw])
y_final_raw = np.concatenate([y_train_mix, y_ps])
X_final_scaled = scaler.fit_transform(X_final_raw)

# Step 7: Final Jitter Oversampling
X_final_jit, y_final_jit = jitter_oversample(X_final_scaled, y_final_raw, cfg, rng)

# Step 8: Retrain
final_loader = make_loader(X_final_jit, y_final_jit, cfg, weighted=True)
for _ in range(20):
    for X_batch, y_batch in final_loader:
        optimizer.zero_grad()
        criterion(model(X_batch), y_batch).backward()
        optimizer.step()

# --- 4. Final Output ---
X_test_scaled = scaler.transform(X_test_raw)
final_probas = predict_proba(model, make_loader(X_test_scaled, None, cfg, weighted=False))
preds = (final_probas > 0.5).astype(int)

print(f"\n{'─'*50}\nFinal Predictions \n{'─'*50}")
print(f"  Predicted pulsars : {preds.sum()} / {len(preds)} ({(preds.sum()/len(preds))*100:.1f}%)")