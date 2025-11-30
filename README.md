# Quantum State Tomography with Physics-Informed Machine Learning

[![CI](https://github.com/sudip-95/quantum-tomography-ml-v2/actions/workflows/ci.yml/badge.svg)](https://github.com/sudip-95/quantum-tomography-ml-v2/actions/workflows/ci.yml)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sudip-95/quantum-tomography-ml-v2/blob/main/colab_demo.ipynb)


**Author:** Sudip Sen Gupta Arka  
**Year:** 2025  
**Email:** senguptasudip95@gmail.com  

---

> **TL;DR:** I built a physics-informed neural network that reconstructs 1–2 qubit quantum states with ~0.98 fidelity, outperforming classical linear inversion.  
> The model enforces Hermiticity, positive semidefiniteness, and trace-1 automatically through a Cholesky parameterization.  
> Full reproducible code, pretrained models, and demo notebook included.

---

## Overview

This project implements a **physics-informed neural network (PINN)** for **quantum state tomography** of **1–2 qubit systems**.  
The model reconstructs the underlying density matrix of a quantum system from noisy measurement data while guaranteeing physical validity (Hermitian, positive semidefinite, trace = 1).

The approach achieves **~0.98 fidelity** on simulated 2-qubit states and **significantly outperforms classical linear inversion**.

This repository contains:
- Dataset generation for 1q and 2q systems  
- Physics-aware neural reconstruction model  
- Baseline classical algorithms  
- Evaluation metrics  
- Plots and quantitative results  
- Scripts for reproducing all experiments  

---

## Key Features

✔ Physics-Informed Architecture  
✔ Complete Tomography Pipeline  
✔ Reproducible Results  
✔ Research-Ready Structure  

---

## 📁 Repository Structure

```
quantum-tomography-ml-v2/
│
├── README.md
├── LICENSE
├── .gitignore
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
    ├── data/
    ├── models/
    ├── train/
    ├── baselines/
    ├── eval/
    └── utils/
```

---

##  Quickstart

### 1️⃣ Create a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
```

### 2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Run sanity checks
```bash
python -m src.data.one_qubit
python -m src.train.train_1q --shots 512 --epochs 60
```

---

##  Pretrained Models
- [2-qubit PINN checkpoint (`mlp_2q_best.pt`)](reports/mlp_2q_best.pt)

---

##  Evaluate Pretrained Model

To evaluate a pretrained 2-qubit model on a fresh synthetic test set:

```bash
python -m src.eval.eval_2q \
    --checkpoint reports/mlp_2q_best.pt \
    --features counts \
    --shots 256 \
    --test 2000 \
    --hidden 512,512
```
---

## Reproduce Main 2-Qubit Results (Fidelity ≈ 0.98)

### ML Model (counts features)
```bash
python -m src.train.train_2q --features counts --shots 256        --train 15000 --val 3000 --test 3000 --epochs 80 --hidden 512 512
```

### Pauli Features + Classical Baseline
```bash
python -m src.train.train_2q --features pauli --shots 512        --train 15000 --val 3000 --test 3000 --epochs 80 --hidden 512 512
```

### Generate Plots
```bash
python plot_results.py
```

---

## 📓 Demo Notebook

See `demo_quantum_tomography.ipynb` for an end-to-end walkthrough:
- 1-qubit and 2-qubit training
- Metric computation
- Plot generation and visualization

---

## 📊 Results

### Fidelity Comparison  
<img src="reports/fidelity_comparison.png" width="500"/>

### Frobenius Distance Comparison  
<img src="reports/frobenius_comparison.png" width="500"/>

---

##  Method Summary

- Data from Ginibre ensembles  
- PINN architecture ensuring valid density matrices  
- Loss: Frobenius distance  
- Metrics: fidelity + Frobenius norm  

---

For a more detailed physics discussion (density matrices, tomography, and the PINN parameterization), see:

➡️ [`PHYSICS_EXPLANATION.md`](PHYSICS_EXPLANATION.md)


For a structured description of the model, training data, metrics, and limitations, see:

➡️ [`MODEL_CARD_PINN.md`](MODEL_CARD_PINN.md)


For detailed ablation and sensitivity experiments (features, shots, hidden sizes, and train set size), see:

➡️ [`ABLATION_STUDIES.md`](ABLATION_STUDIES.md)

---

##  What I Learned
- How to enforce physical constraints (Hermitian, PSD, trace-1) in neural networks  
- How to generate synthetic quantum states via Ginibre ensembles  
- Importance of measurement noise and shot count in tomography  
- How Cholesky factorization guarantees valid density matrices  
- How to benchmark against classical linear inversion  
- Designing reproducible ML experiments and evaluation pipelines  

---

## 📚 References

**Quantum State Tomography**
- D. F. V. James, P. G. Kwiat, W. J. Munro, A. G. White (2001).  
  *"Measurement of qubits"* — The foundational paper on quantum state tomography.  
- M. Paris and J. Řeháček (2004).  
  *"Quantum State Estimation"* — Comprehensive textbook on tomographic reconstruction.

**Physics-Informed Neural Networks (PINNs)**
- M. Raissi, P. Perdikaris, G. E. Karniadakis (2019).  
  *"Physics-Informed Neural Networks: A Deep Learning Framework for Solving Forward and Inverse Problems Involving Nonlinear PDEs."*  
  Introduces the PINN framework used as inspiration for physics-aware constraints.

**Quantum Machine Learning**
- J. Biamonte et al. (2017).  
  *"Quantum Machine Learning."* Nature — Overview of ML techniques applied to quantum systems.
- Schuld & Killoran (2019).  
  *"Quantum Machine Learning in Feature Hilbert Spaces."*

**Cholesky / Positive Semidefinite Parameterization**
- J. Smolin, D. DiVincenzo (1996).  
  *"Five two-qubit quantum states are enough to characterize."*  
  (Introduces matrix reconstruction ideas used in tomography.)
- Cholesky PSD parameterization commonly used in density matrix reconstruction  
  (standard technique widely used across tomography and variational quantum algorithms).

**Classical Baselines**
- K. Banaszek, G. M. D’Ariano, M. Paris, M. Sacchi (1999).  
  *"Maximum-likelihood estimation of the density matrix."*  
- A. G. White et al. (1999).  
  *"Nonmaximally entangled states: Production, characterization, and tomography."*

**Neural Quantum State Reconstruction**
- Torlai et al. (2018).  
  *"Neural-network quantum state tomography."*  
  Demonstrates the power of neural networks for reconstructing quantum states from data.

**Quantum Software & Noise Models**
- Qiskit Development Team (2017–2024).  
  *Qiskit: An Open-source Framework for Quantum Computing.*  
  (Useful for comparing simulated measurement noise and experimental-style tomography.)

---

## 📘 Citation
```
Sudip Sen Gupta Arka, "Physics-Informed Neural Quantum State Tomography", 2025.
```

---

## ✉️ Contact  
📧 Email: senguptasudip95@gmail.com
