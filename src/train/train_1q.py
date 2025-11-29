import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from ..data.one_qubit import make_1q_dataset
from ..models.mlp_1q import MLP1Q
from ..utils.seeding import set_seed
from ..eval.metrics import mse, bloch_angle_error

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def split_data(X, y, n_train, n_val):
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
    X_test, y_test = X[n_train+n_val:], y[n_train+n_val:]
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def standardize(train, val, test):
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True) + 1e-8
    return (train - mean)/std, (val - mean)/std, (test - mean)/std, (mean, std)

def train_epoch(model, loader, opt, device):
    model.train()
    total = 0.0
    n = 0
    crit = nn.MSELoss()
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        opt.zero_grad()
        pred = model(xb)
        loss = crit(pred, yb)
        loss.backward()
        opt.step()
        total += float(loss.item()) * xb.size(0)
        n += xb.size(0)
    return total / max(n,1)

@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    crit = nn.MSELoss()
    total = 0.0
    n = 0
    preds = []
    gts = []
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        pred = model(xb)
        loss = crit(pred, yb)
        total += float(loss.item()) * xb.size(0)
        n += xb.size(0)
        preds.append(pred.cpu().numpy())
        gts.append(yb.cpu().numpy())
    if n == 0:
        return 0.0, 0.0
    preds = np.concatenate(preds, axis=0)
    gts = np.concatenate(gts, axis=0)
    # Secondary metric: Bloch angle error
    ang = bloch_angle_error(gts, preds)
    return total / n, ang

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--shots", type=int, default=512)
    ap.add_argument("--train", type=int, default=20000)
    ap.add_argument("--val", type=int, default=5000)
    ap.add_argument("--test", type=int, default=5000)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--hidden", type=int, nargs="+", default=[64,64])
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device()

    n_total = args.train + args.val + args.test
    X, y = make_1q_dataset(n_total, args.shots, seed=args.seed)

    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = split_data(X, y, args.train, args.val)
    X_tr, X_va, X_te, norm = standardize(X_tr, X_va, X_te)

    train_ds = TensorDataset(torch.from_numpy(X_tr).float(), torch.from_numpy(y_tr).float())
    val_ds   = TensorDataset(torch.from_numpy(X_va).float(), torch.from_numpy(y_va).float())
    test_ds  = TensorDataset(torch.from_numpy(X_te).float(), torch.from_numpy(y_te).float())

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch)
    test_loader  = DataLoader(test_ds, batch_size=args.batch)

    model = MLP1Q(in_dim=3, hidden=tuple(args.hidden), out_dim=3).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = float("inf")
    best_state = None
    patience = 12
    bad = 0

    for epoch in range(1, args.epochs+1):
        tr_loss = train_epoch(model, train_loader, opt, device)
        va_loss, va_ang = eval_epoch(model, val_loader, device)
        print(f"epoch {epoch:03d} | train MSE {tr_loss:.6f} | val MSE {va_loss:.6f} | val angle {va_ang:.2f} deg")
        if va_loss < best_val - 1e-6:
            best_val = va_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print("Early stopping.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Evaluate on test
    te_loss, te_ang = eval_epoch(model, test_loader, device)
    print(f"TEST  | MSE {te_loss:.6f} | angle {te_ang:.2f} deg")

    # Save artifacts
    os.makedirs("reports", exist_ok=True)
    ckpt_path = os.path.join("reports", "mlp_1q_best.pt")
    torch.save(model.state_dict(), ckpt_path)
    with open(os.path.join("reports", "metrics_1q.txt"), "w") as f:
        f.write(f"shots={args.shots}\ntrain={args.train}\nval={args.val}\ntest={args.test}\n")
        f.write(f"test_mse={te_loss:.8f}\n")
        f.write(f"test_angle_deg={te_ang:.4f}\n")
    print(f"Saved checkpoint to {ckpt_path}")

    # Show a few predictions
    model.eval()
    xb, yb = next(iter(DataLoader(test_ds, batch_size=5, shuffle=True)))
    with torch.no_grad():
        pred = model(xb.to(device)).cpu().numpy()
    print("Sample predictions (pred  ->  true):")
    for i in range(len(pred)):
        print(np.round(pred[i], 3), "->", np.round(yb[i].numpy(), 3))

if __name__ == "__main__":
    main()
