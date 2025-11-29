
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from ..data.two_qubit import make_2q_dataset
from ..models.rho_net_2q import RhoNet2Q
from ..baselines.linear_inversion_2q import pauli_reconstruction_from_expectations, project_psd_trace1
from ..eval.metrics import frobenius_rho, fidelity
from ..utils.seeding import set_seed

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

def complex_to_np(t):
    return t.detach().cpu().numpy()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--shots", type=int, default=512)
    ap.add_argument("--train", type=int, default=30000)
    ap.add_argument("--val", type=int, default=5000)
    ap.add_argument("--test", type=int, default=5000)
    ap.add_argument("--features", choices=["counts","pauli"], default="counts")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--hidden", type=int, nargs="+", default=[256,256])
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device()

    n_total = args.train + args.val + args.test
    X, Y_rho = make_2q_dataset(n_total, args.shots, features=args.features, seed=args.seed)

    (X_tr, Y_tr), (X_va, Y_va), (X_te, Y_te) = split_data(X, Y_rho, args.train, args.val)
    X_tr, X_va, X_te, _ = standardize(X_tr, X_va, X_te)

    train_ds = TensorDataset(torch.from_numpy(X_tr).float(), torch.from_numpy(Y_tr).cfloat())
    val_ds   = TensorDataset(torch.from_numpy(X_va).float(), torch.from_numpy(Y_va).cfloat())
    test_ds  = TensorDataset(torch.from_numpy(X_te).float(), torch.from_numpy(Y_te).cfloat())

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch)
    test_loader  = DataLoader(test_ds, batch_size=args.batch)

    in_dim = X_tr.shape[1]
    model = RhoNet2Q(in_dim=in_dim, hidden=tuple(args.hidden)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    def frob_loss(pred, true):
        diff = pred - true
        return torch.mean(torch.real(torch.sum(diff.conj()*diff, dim=(-2,-1))))

    best_val = float("inf")
    best_state = None
    patience = 20
    bad = 0

    for epoch in range(1, args.epochs+1):
        model.train()
        tot = 0.0; n = 0
        for xb, yb in train_loader:
            xb = xb.to(device); yb = yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = frob_loss(pred, yb)
            loss.backward()
            opt.step()
            tot += float(loss.item()) * xb.size(0); n += xb.size(0)
        tr_loss = tot / max(1,n)

        model.eval()
        tot = 0.0; n = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device); yb = yb.to(device)
                pred = model(xb)
                loss = frob_loss(pred, yb)
                tot += float(loss.item()) * xb.size(0); n += xb.size(0)
        va_loss = tot / max(1,n)

        print(f"epoch {epoch:03d} | train Fro {tr_loss:.6f} | val Fro {va_loss:.6f}")
        if va_loss < best_val - 1e-6:
            best_val = va_loss
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print("Early stopping.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    frobs = []; fids = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device); yb = yb.to(device)
            pred = model(xb)
            P = complex_to_np(pred); T = complex_to_np(yb)
            for p, t in zip(P, T):
                frobs.append(frobenius_rho(p, t))
                fids.append(fidelity(p, t))
    print(f"TEST | Frobenius mean {np.mean(frobs):.6f} | Fidelity mean {np.mean(fids):.6f}")

    os.makedirs("reports", exist_ok=True)
    with open(os.path.join("reports","metrics_2q.txt"), "w") as f:
        f.write(f"features={args.features}\nshots={args.shots}\n")
        f.write(f"fro_mean={np.mean(frobs):.8f}\n")
        f.write(f"fid_mean={np.mean(fids):.8f}\n")

    if args.features == "pauli":
        print("Evaluating linear inversion baseline...")
        frobs_b = []; fids_b = []
        for i in range(X_te.shape[0]):
            feats = X_te[i]
            rho = pauli_reconstruction_from_expectations(feats)
            rho = project_psd_trace1(rho)
            frobs_b.append(frobenius_rho(rho, Y_te[i]))
            fids_b.append(fidelity(rho, Y_te[i]))
        print(f"Baseline | Frobenius mean {np.mean(frobs_b):.6f} | Fidelity mean {np.mean(fids_b):.6f}")
        with open(os.path.join("reports","metrics_2q_baseline.txt"), "w") as f:
            f.write(f"fro_mean={np.mean(frobs_b):.8f}\n")
            f.write(f"fid_mean={np.mean(fids_b):.8f}\n")

if __name__ == "__main__":
    main()
