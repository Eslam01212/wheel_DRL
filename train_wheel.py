#!/usr/bin/env python3
import os, glob
import numpy as np
import torch
import torch.nn as nn

DATA_DIR = "wheel_dataset"
PATTERN  = "wheel_part_*.npz"
OUT_CKPT = "wheel_predictor.pt"

def load_all_npz(files):
    Xs, Ys = [], []
    for f in files:
        d = np.load(f)
        X = d["X"].astype(np.float32)
        Y = d["Y"].astype(np.float32)
        m = np.isfinite(X).all(1) & np.isfinite(Y).all(1)
        Xs.append(X[m]); Ys.append(Y[m])
    X = np.concatenate(Xs, axis=0) if Xs else np.empty((0,8), np.float32)
    Y = np.concatenate(Ys, axis=0) if Ys else np.empty((0,2), np.float32)
    return X, Y

def mean_std(X):
    mu = X.mean(axis=0).astype(np.float32)
    sd = X.std(axis=0).astype(np.float32)
    sd = np.maximum(sd, 1e-6).astype(np.float32)
    return mu, sd

class WheelMLP(nn.Module):
    def __init__(self, in_dim=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 2),
        )
    def forward(self, x): return self.net(x)

def main():
    files = sorted(glob.glob(os.path.join(DATA_DIR, PATTERN)))
    assert files, f"No data found: {DATA_DIR}/{PATTERN}"

    X, Y = load_all_npz(files)
    assert len(X) > 0, "Empty dataset after filtering."

    mu, sd = mean_std(X)
    Xn = (X - mu[None, :]) / sd[None, :]

    # shuffle + split
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(Xn))
    Xn, Y = Xn[idx], Y[idx]
    n_tr = int(0.9 * len(Xn))
    Xtr, Ytr = Xn[:n_tr], Y[:n_tr]
    Xva, Yva = Xn[n_tr:], Y[n_tr:]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = WheelMLP().to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    lossf = nn.SmoothL1Loss(beta=0.05)  # robust, usually better than pure MSE

    Xtr_t = torch.from_numpy(Xtr).to(device)
    Ytr_t = torch.from_numpy(Ytr).to(device)
    Xva_t = torch.from_numpy(Xva).to(device)
    Yva_t = torch.from_numpy(Yva).to(device)

    BS = 1024
    EPOCHS = 20
    best = float("inf")
    bad = 0
    PATIENCE = 20

    for ep in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(len(Xtr_t), device=device)
        for i in range(0, len(Xtr_t), BS):
            b = perm[i:i+BS]
            pred = model(Xtr_t[b])
            loss = lossf(pred, Ytr_t[b])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            va = lossf(model(Xva_t), Yva_t).item()

        if va < best - 1e-6:
            best = va
            bad = 0
            torch.save({"state_dict": model.state_dict(), "x_mean": mu, "x_std": sd}, OUT_CKPT)
        else:
            bad += 1

        print(f"epoch {ep:03d}  val_loss={va:.6f}  best={best:.6f}")
        if bad >= PATIENCE:
            break

    print(f"saved best -> {OUT_CKPT}")

if __name__ == "__main__":
    main()
