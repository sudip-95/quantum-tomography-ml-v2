# Physics-Informed Neural Networks for 1–2 Qubit Quantum State Tomography

**Author:** Sudip Sen Gupta Arka  
**Year:** 2025  
**Repository:** https://github.com/sudip-95/quantum-tomography-ml-v2  

---

## Abstract

Quantum state tomography is the task of reconstructing an unknown quantum state from many repeated measurements. For even a few qubits, this is a noisy and ill-posed inverse problem: naive reconstruction can easily produce matrices that are not valid physical states. In this work I implement a physics-informed neural network (PINN) that performs tomography for 1–2 qubit systems. The model reconstructs a density matrix directly from noisy measurement data while guaranteeing that the output is Hermitian, positive semidefinite, and trace-one via a Cholesky-like parameterization. On simulated 2-qubit data, the network achieves a mean fidelity of about 0.98 and clearly outperforms a classical linear inversion baseline. The project includes a fully reproducible pipeline with training scripts, evaluation tools, a Colab demo, and unit tests.

---

## 1 Introduction

At a very simple level, **tomography** means *reconstructing something you cannot see directly by looking at many different “shadows” of it*.  

- In a medical CT scan, you shine X-rays from many angles and reconstruct a 3D picture of the body.  
- In **quantum state tomography**, you prepare many copies of the same quantum state, measure them in different ways, and reconstruct the invisible quantum state from the measurement outcomes.

In this project I study **quantum state tomography** for 1–2 qubit systems using **physics-informed machine learning**. Instead of relying only on analytic formulas, I train a neural network that:

1. Takes noisy measurement statistics as input,  
2. Outputs a full density matrix for the 1–2 qubit state, and  
3. Is built so that its outputs are *always* valid quantum states (Hermitian, positive semidefinite, and trace-one).

The work is implemented in Python/PyTorch, organized as a reproducible GitHub repository with:

- Synthetic data generation,  
- Physics-informed network architectures for 1 and 2 qubits,  
- Classical baselines,  
- Evaluation metrics, plots, and tests,  
- A Colab notebook and a `reproduce.py` script for one-command experiments.

The goal is both scientific (showing that physics-informed neural networks can do high-fidelity tomography) and educational (demonstrating how ideas from quantum information and machine learning can be combined in a clean, end-to-end project).

---

## 2 Background

### 2.1 Quantum States and Density Matrices

A single qubit can be in a superposition
\[
|\psi\rangle = \alpha |0\rangle + \beta |1\rangle, \quad |\alpha|^2 + |\beta|^2 = 1.
\]

However, real systems are noisy and often better described as **mixed states**, which cannot be captured by a single state vector. Instead we use a **density matrix** \(\rho\):

- For \(n\) qubits, \(\rho\) is a \(2^n \times 2^n\) complex matrix.
- Physically valid density matrices satisfy:
  1. **Hermitian:** \(\rho = \rho^\dagger\)  
  2. **Positive semidefinite (PSD):** all eigenvalues \(\lambda_i \ge 0\)  
  3. **Unit trace:** \(\mathrm{Tr}(\rho) = 1\)

Pure states can be written as \(\rho = |\psi\rangle\langle\psi|\) with \(\mathrm{Tr}(\rho^2) = 1\); mixed states satisfy \(\mathrm{Tr}(\rho^2) < 1\). Any tomography procedure must produce matrices that obey these constraints to represent valid quantum states.

### 2.2 Quantum State Tomography

In practice, we never get to “look at” \(\rho\) directly. Instead, we perform many **measurements** on identically prepared copies of the system.

For 1–2 qubits, a natural set of measurements uses combinations of the **Pauli bases**:

- For 1 qubit: measure in X, Y, Z bases.  
- For 2 qubits: measure all 9 combinations from \(\{X, Y, Z\} \otimes \{X, Y, Z\}\).

For each setting, we perform a finite number of **shots** (repetitions). From the outcomes, we estimate either:

- **Counts:** frequencies of outcomes 00, 01, 10, 11 for each basis pair.  
- **Expectation values:** average values like \(\langle X \otimes Z \rangle\).

**Quantum state tomography** is the inverse problem of reconstructing \(\rho\) from these noisy statistics. Classical techniques include:

- **Linear inversion:** solve a linear system mapping measurement statistics to matrix elements of \(\rho\). Fast but can produce non-physical \(\rho\).  
- **Maximum likelihood estimation:** optimize over valid density matrices to best explain the observed data. More accurate but computationally heavier.

This project instead explores a **physics-informed neural approach**, where a network learns the mapping from measurements to \(\rho\) but is constrained to always output valid states.

---

## 3 Method

### 3.1 Data Generation

All data used in this project is **synthetic**, generated in Python:

1. **State sampling.** Random 1–2 qubit states are sampled (e.g., using Ginibre ensembles), giving ground-truth density matrices \(\rho_{\text{true}}\).  
2. **Measurement simulation.** For each state, we simulate projective measurements in Pauli bases (X, Y, Z on each qubit) with a finite number of shots (e.g., 256 or 512).  
3. **Feature construction.**  
   - For **counts features**, we collect normalized counts \(p(00), p(01), p(10), p(11)\) for each basis pair and concatenate them. In the 2-qubit case this yields a 36-dimensional feature vector (9 basis pairs × 4 outcomes).  
   - For **Pauli features**, we compute estimated expectation values of 2-qubit Pauli operators, giving a 15-dimensional feature vector.

The labels for supervised learning are the true density matrices \(\rho_{\text{true}}\) associated with each feature vector.

All dataset generation code lives in `src/data/`. The 1-qubit and 2-qubit generators are used by the training scripts in `src/train/`.

### 3.2 Physics-Informed Output Parameterization

A naive neural network regressor could try to predict all entries of \(\rho\) directly, but there is no guarantee that the resulting matrix will satisfy the physical constraints listed above. Instead, I use a **Cholesky-style parameterization**:

1. The neural network outputs parameters representing a **lower-triangular complex matrix** \(L\).  
2. I construct
   \[
   A = L L^\dagger .
   \]
   By construction, \(A\) is Hermitian and positive semidefinite.  
3. I normalize its trace:
   \[
   \rho = \frac{A}{\mathrm{Tr}(A)}.
   \]

This ensures that every prediction is:

- Hermitian: \(\rho = \rho^\dagger\)  
- PSD: \(\rho \succeq 0\)  
- Trace-one: \(\mathrm{Tr}(\rho) = 1\)

In other words, **the architecture itself enforces the physics**, and the optimizer does not need to learn these constraints from data.

### 3.3 Network Architecture

For both 1-qubit and 2-qubit tomography, the network has the same structure at a high level:

- **Input:** measurement feature vector (dimension depends on counts vs Pauli).  
- **Hidden layers:** a small multi-layer perceptron with two fully-connected layers (e.g., sizes [512, 512]) and nonlinear activations.  
- **Output:** real-valued parameters representing the entries of the lower-triangular matrix \(L\); these are reshaped and combined into a complex matrix.

The 2-qubit model implementing this architecture is named **RhoNet2Q** and lives in `src/models/rho_net_2q.py`. The 1-qubit model is an MLP operating on Bloch vector or Pauli expectation features (`src/models/mlp_1q.py`).

### 3.4 Loss Function and Optimization

The training objective is based on the **Frobenius distance** between the predicted and true density matrices:

\[
\mathcal{L}(\rho_{\text{pred}}, \rho_{\text{true}})
= \left\lVert \rho_{\text{pred}} - \rho_{\text{true}} \right\rVert_F
= \sqrt{\sum_{i,j} \lvert (\rho_{\text{pred}})_{ij}
- (\rho_{\text{true}})_{ij} \rvert^2 } .
\]

Optimization uses the Adam optimizer with early stopping based on validation performance. Training is run for tens of epochs (e.g., 40–80) depending on dataset size and configuration.

### 3.5 Classical Baseline: Linear Inversion

To provide a simple classical comparison, I implement a **linear inversion baseline** in `src/baselines/linear_inversion_2q.py`. This method:

- Uses Pauli expectation values to reconstruct \(\rho\) as a linear combination of Pauli operators.  
- Can be followed by a projection step to enforce basic physical constraints (e.g., truncating negative eigenvalues).

Linear inversion is fast and conceptually simple, but tends to produce noisier reconstructions than the physics-informed network.

---

## 4 Experimental Setup

### 4.1 2-Qubit Counts Configuration (Main Setting)

The main experiment evaluates RhoNet2Q using counts features:

- **Features:** counts (36-dimensional)  
- **Shots:** 256 per measurement setting  
- **Train / Val / Test sizes:** 15,000 / 3,000 / 3,000  
- **Model:** RhoNet2Q with hidden sizes [512, 512]  
- **Loss:** Frobenius distance  
- **Metrics:** mean Frobenius distance and mean fidelity over the test set

### 4.2 Pauli Features + Baseline

A second configuration uses Pauli expectation values as features:

- **Features:** Pauli expectations (15-dimensional)  
- **Shots:** 512 per setting (to estimate expectations more accurately)  
- **Train / Val / Test sizes:** 15,000 / 3,000 / 3,000  
- **Models:** RhoNet2Q and a classical linear inversion baseline

This configuration highlights the difference between a learned physics-informed model and a simple analytic reconstruction.

### 4.3 Reproducibility

The repository includes several entry points that make experiments easy to reproduce:

- `train_1q.py` and `train_2q.py` – train the 1-qubit and 2-qubit models.  
- `eval_2q.py` – load a pretrained checkpoint and evaluate on a fresh test set.  
- `plot_results.py` – generate plots from metrics in the `reports/` directory.  
- `reproduce.py` – run a complete training → plotting → evaluation demo with one command.  
- `colab_demo.ipynb` – run end-to-end experiments in Google Colab.

In addition, basic unit tests (`tests/test_data_and_metrics.py`) and a GitHub Actions CI workflow ensure that core functionality is stable.

---

## 5 Results

### 5.1 Quantitative Metrics

A representative 2-qubit experiment with counts features (256 shots, [512, 512] hidden layers, 15k/3k/3k split) yields:

- **RhoNet2Q (counts):**  
  - Test Frobenius mean \(\approx 0.1005\)  
  - Test fidelity mean \(\approx 0.9822\)

In the Pauli-feature configuration at 512 shots, the results are approximately:

- **RhoNet2Q (Pauli):**  
  - Test Frobenius mean \(\approx 0.3719\)  
  - Test fidelity mean \(\approx 0.8632\)
- **Linear inversion baseline (Pauli):**  
  - Test Frobenius mean \(\approx 0.5350\)  
  - Test fidelity mean \(\approx 0.6951\)

These numbers come from the final `TEST | ...` and `Baseline | ...` lines written to `reports/metrics_2q.txt` and `reports/metrics_2q_baseline.txt` in the repository.

Overall, the physics-informed network achieves substantially lower Frobenius error and higher fidelity than the classical baseline.

### 5.2 Plots

The script `plot_results.py` reads metrics from the `reports/` directory and generates comparison plots. Two key figures are:

**Figure 1 – Fidelity Comparison**  
![Fidelity comparison](reports/fidelity_comparison.png)

**Figure 2 – Frobenius Distance Comparison**  
![Frobenius distance comparison](reports/frobenius_comparison.png)

These plots visually summarize how often the PINN produces high-fidelity reconstructions compared to the classical linear inversion method.

---

## 6 Discussion

The experiments show that a relatively simple neural network, when combined with a physics-informed parameterization, can perform high-quality quantum state tomography for 2-qubit systems.

Key observations include:

1. **Physical constraints built into the architecture help.**  
   The \(LL^\dagger / \mathrm{Tr}(LL^\dagger)\) construction ensures that the network never outputs unphysical density matrices. This makes training more stable and prevents impossible predictions.

2. **Counts features are sufficient for strong performance.**  
   Even when trained directly on counts (which are closer to real hardware data), the network achieves \(\approx 0.98\) mean fidelity in the main configuration. This suggests that sophisticated feature engineering is not strictly necessary when the model is physics-aware.

3. **Improvement over classical linear inversion.**  
   Linear inversion provides a fast baseline but performs worse in both Frobenius error and fidelity. The PINN effectively acts as a learned regularizer, denoising the measurement statistics while respecting the structure of quantum states.

4. **Reproducibility and tooling matter.**  
   The combination of training scripts, a Colab demo, evaluation tools, tests, and CI gives the project the feel of a small research codebase rather than a one-off script. This is valuable for both learning and for sharing the work with others (professors, collaborators, reviewers).

---

## 7 Limitations and Future Work

This project has several limitations, many of which point to clear directions for future extensions:

- **System size.** The current implementation targets **1–2 qubit** systems only. Extending to 3–4 qubits would require more efficient architectures and possibly low-rank or tensor-network-inspired parameterizations to cope with the \(4^n\) scaling.

- **Synthetic data only.** All experiments are performed on simulated data. The next step would be to train and test on data from real quantum hardware (e.g., using Qiskit or other quantum SDKs), with realistic noise models and calibration procedures.

- **Noise models.** The current setup mostly models finite-shot noise; more complex effects such as crosstalk, coherent errors, and non-Markovian behavior are not included.

- **Limited baselines.** Only a basic linear inversion method is implemented as a classical baseline. Comparing against maximum-likelihood tomography or more advanced neural quantum state approaches would provide a richer picture.

Possible future directions include:

- Incorporating realistic device noise models and real backend data.  
- Running systematic ablation studies on shot count, hidden layer size, and training set size (templates are outlined in `ABLATION_STUDIES.md`).  
- Exploring architectures that exploit known symmetries or structure in quantum states.  
- Extending the framework to simple process (channel) tomography.

---

## 8 Conclusion

This project demonstrates that physics-informed neural networks can be effective tools for quantum state tomography in small systems. By combining:

- a principled density matrix parameterization,  
- simulated measurement data, and  
- a standard deep learning stack (PyTorch, training scripts, evaluation, and CI),  

the project reconstructs 1–2 qubit states with high fidelity and clear improvements over a classical linear inversion baseline.

Beyond the numerical results, the work serves as an example of how ideas from quantum information, linear algebra, and machine learning can be turned into a clean, reproducible codebase suitable for learning, teaching, and use in academic applications.

---

## 9 References

- D. F. V. James, P. G. Kwiat, W. J. Munro, A. G. White, *Measurement of qubits* (2001).  
- M. Paris and J. Řeháček (eds.), *Quantum State Estimation*, Lecture Notes in Physics, vol. 649, Springer (2004).  
- M. Raissi, P. Perdikaris, G. E. Karniadakis, *Physics-Informed Neural Networks: A Deep Learning Framework for Solving Forward and Inverse Problems Involving Nonlinear PDEs* (2019).  
- G. Torlai et al., *Neural-network quantum state tomography*, Nature Physics 14, 447–450 (2018).  
- K. Banaszek, G. M. D’Ariano, M. G. A. Paris, M. F. Sacchi, *Maximum-likelihood estimation of the density matrix*, Physical Review A 61, 010304 (1999).  
- A. G. White et al., *Nonmaximally entangled states: Production, characterization, and tomography*, Physical Review Letters 83, 3103 (1999).
