# 🧠 Quantum State Tomography with Physics-Informed ML  

![Python](https://img.shields.io/badge/Python-3.10%2B-blue) 
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-success)

> **Author:** Sudip Sen Gupta Arka  

---

## ⚡ Overview
A practical demonstration of how **physics-informed neural networks (PINNs)** outperform classical linear inversion in quantum-state tomography.  
This project reconstructs **1- and 2-qubit density matrices** from simulated noisy measurement data while guaranteeing physical validity.

---

## 🚀 Highlights
✅ Physics-aware neural network enforcing ρ = LL† / Tr(LL†)  
✅ Produces valid density matrices (Hermitian, PSD, trace-1)  
✅ Simulated 1-qubit & 2-qubit datasets with finite-shot noise  
✅ Baseline vs ML comparison for reconstruction accuracy  
✅ Auto-saves metrics & plots in `reports/`

---

## 🧩 Repo Layout
```
quantum-tomography-ml-v2/
├─ src/
│  ├─ data/ ··· data generation (1q, 2q)
│  ├─ models/ ··· MLP & ρ-Net
│  ├─ train/  ··· training scripts
│  ├─ baselines/ ··· linear inversion
│  ├─ eval/ ··· metrics
│  └─ utils/ ··· seeding, Pauli ops
├─ reports/ ··· metrics + plots
├─ plot_results.py
├─ requirements.txt
└─ README.md
```

---

## 🧮 Key Results

| Model | Features | Shots | Frobenius ↓ | Fidelity ↑ |
|:------|:----------|------:|-------------:|------------:|
| **ρ-Net** | counts | 256 | 0.1005 | **0.9822** |
| **ρ-Net** | Pauli | 512 | 0.3719 | **0.8632** |
| **Linear Inversion** | Pauli | 512 | 0.5349 | 0.6951 |

> ρ-Net (counts) → near-perfect reconstruction (F ≈ 0.98).  
> ρ-Net (Pauli) → +0.17 fidelity gain over classical baseline.

---

## 🧠 How It Works
- **Data:** Random pure/mixed states using Ginibre ensembles + noisy Pauli measurements.  
- **Model:** PINN predicting lower-triangular \(L\) → \(ρ = LL^{†}/Tr(LL^{†})\).  
- **Loss:** Frobenius distance between predicted and true \(ρ\).  
- **Metrics:** Frobenius norm + quantum fidelity.  
- **Optimizer:** Adam (lr = 1e-3, weight decay = 1e-5).

---

## 🧪 Quickstart
```bash
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1-Qubit
python -m src.train.train_1q --shots 512 --epochs 60

# 2-Qubit (counts features)
python -m src.train.train_2q --features counts --shots 256 --epochs 80 --hidden 512 512

# 2-Qubit (Pauli features + baseline)
python -m src.train.train_2q --features pauli --shots 512 --epochs 80 --hidden 512 512

# Plot results
python plot_results.py
```

---

## 📊 Generated Artifacts
All results and figures are saved to the `reports/` folder:
- `metrics_2q.txt`  
- `metrics_2q_baseline.txt`  
- `fidelity_comparison.png`  
- `frobenius_comparison.png`

---

## 🧩 Insight
Classical linear inversion often produces **unphysical quantum states** (negative eigenvalues).  
By operating directly in the manifold of valid density matrices, **ρ-Net** ensures physically consistent, high-fidelity reconstructions even under noisy measurements.

---

## 📚 Citation
> Sudip Sen Gupta Arka, *Physics-Informed Neural Quantum State Tomography (2025)*  
> [github.com/sudip-95/quantum-tomography-ml-v2](https://github.com/sudip-arka/quantum-tomography-ml-v2)

---

## ✉️ Contact
**Email:** senguptasudip95@gmail.com
