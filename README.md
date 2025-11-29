# Quantum State Tomography with Physics-Informed Machine Learning

### Author: Sudip Sen Gupta Arka  

---

## 🧠 Overview

This repository implements **machine-learning-based quantum state tomography** for 1- and 2-qubit systems.  
The goal is to reconstruct an unknown quantum state (density matrix ρ) from simulated noisy measurement data.

The project demonstrates how **physics-informed neural networks (PINNs)** can outperform classical reconstruction (linear inversion) under realistic finite-shot noise.

---

## ⚙️ Key Features
- **1-qubit MLP baseline** — learns Bloch vector (X, Y, Z) reconstruction.  
- **2-qubit physics-informed network (ρ-Net)** — outputs valid quantum states via  
  \( \rho = LL^{\dagger}/\mathrm{Tr}(LL^{\dagger}) \).  
- **Synthetic data generator** for random density matrices with adjustable shot noise.  
- **Classical baseline** — linear inversion + PSD projection for comparison.  
- **Evaluation metrics** — mean Frobenius distance and quantum fidelity.  
- **Auto-saved metrics & plots** in `reports/`.

---

## 🧩 Project Structure

```
quantum-tomography-ml-v2/
│
├── README.md
├── requirements.txt
├── plot_results.py
│
├── reports/
│   ├── metrics_2q.txt
│   ├── metrics_2q_baseline.txt
│   ├── fidelity_comparison.png
│   └── frobenius_comparison.png
│
└── src/
    ├── data/           # one_qubit.py, two_qubit.py (data generators)
    ├── models/         # mlp_1q.py, rho_net_2q.py
    ├── train/          # train_1q.py, train_2q.py
    ├── baselines/      # linear_inversion_2q.py
    ├── eval/           # metrics.py
    └── utils/          # seeding.py, paulis.py
```

---

## 🚀 Quickstart

### 1️⃣ Install dependencies
```bash
  python -m venv .venv
  source .venv/bin/activate          # Windows: .venv\Scripts\activate
  pip install -r requirements.txt
```

### 2️⃣ Run the 1-qubit example
```bash
  python -m src.data.one_qubit
  python -m src.train.train_1q --shots 512 --train 20000 --val 5000 --test 5000 --epochs 60
```

### 3️⃣ Run 2-qubit training

**Counts features (36-dimensional):**
```bash
  python -m src.train.train_2q --features counts --shots 256        --train 15000 --val 3000 --test 3000 --epochs 80 --hidden 512 512
```

**Pauli features (15-dimensional) + Baseline comparison:**
```bash
  python -m src.train.train_2q --features pauli --shots 512        --train 15000 --val 3000 --test 3000 --epochs 80 --hidden 512 512
```

### 4️⃣ Plot results
```bash
  python plot_results.py
```

---

## 📊 Results

| Model | Features | Shots | Frobenius ↓ | Fidelity ↑ |
|--------|-----------|--------|-------------|-------------|
| ρ-Net | counts | 256 | 0.1005 | **0.9822** |
| ρ-Net | Pauli | 512 | 0.3719 | **0.8632** |
| Linear Inversion | Pauli | 512 | 0.5349 | 0.6951 |

### Discussion
- **ρ-Net (counts)** achieved near-perfect mean fidelity (~0.98), indicating highly accurate reconstruction even with limited (256) shots.  
- **ρ-Net (Pauli)** outperformed **linear inversion** by +0.17 fidelity, confirming that the physics-aware neural network robustly denoises finite-shot data.  
- The **ρ = LL† / Tr(LL†)** formulation enforces Hermiticity, positivity, and trace-1 automatically—producing physically valid quantum states.

Generated plots are available in the `reports/` folder:
- `fidelity_comparison.png`
- `frobenius_comparison.png`

---

## 🧮 Methods Summary

| Component | Description |
|------------|-------------|
| **Data Generation** | Random pure/mixed 2-qubit states using Ginibre ensembles and noisy Pauli measurements. |
| **Model** | Physics-informed neural network (PINN) predicting Cholesky factor \(L\). |
| **Loss Function** | Mean Frobenius distance between predicted and true density matrices. |
| **Optimizer** | Adam (lr=1e-3, weight_decay=1e-5). |
| **Metrics** | Frobenius norm and quantum fidelity. |
| **Baseline** | Linear inversion + PSD projection. |

---

## 🧠 Key Insight

Traditional linear inversion suffers from unphysical outputs under noise (negative eigenvalues).  
The physics-informed approach guarantees valid density matrices while learning a data-driven noise correction — a clear advantage for real-world quantum experiments.

---

## 🏁 Citation / Attribution

If referencing this work in academic writing or a portfolio:

> Sudip Sen Gupta Arka, *Physics-Informed Neural Quantum State Tomography (2025)*.  
> [github.com/sudip-95/quantum-tomography-ml-v2]

---

## 📬 Contact
For questions or collaboration:
- **Email:** senguptasudip95@gmail.com

---
