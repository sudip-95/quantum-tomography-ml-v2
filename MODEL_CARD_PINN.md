# Model Card – Physics-Informed 2‑Qubit Tomography Network (RhoNet2Q)

This document summarizes the main properties, assumptions, and limitations of the
physics‑informed neural network used for 2‑qubit quantum state tomography in this
repository.

---

## 1. Model Overview

**Name:** RhoNet2Q (physics‑informed 2‑qubit tomography network)  
**Task:** Reconstruct a 2‑qubit density matrix \(\rho\) from noisy measurement data.  
**Domain:** Quantum information / quantum state tomography (2‑qubit systems).  
**Author:** Sudip Sen Gupta Arka  

The model takes as input features derived from simulated measurement statistics
(e.g., counts over Pauli‑basis measurement settings) and outputs a **valid
2‑qubit density matrix** by construction:

- Hermitian: \(\rho = \rho^\dagger\)  
- Positive semidefinite: \(\rho \succeq 0\)  
- Trace‑1: \(\mathrm{Tr}(\rho) = 1\)

These constraints are enforced via a **Cholesky‑style parameterization**, where
the network predicts a lower‑triangular matrix \(L\) and constructs

\[
A = L L^\dagger, \qquad
\rho = \frac{A}{\mathrm{Tr}(A)}.
\]

This guarantees that all outputs are valid density matrices, even for states
that the model has not seen during training.

---

## 2. Intended Use

**Primary intended use:**

- Research / educational experiments in 1–2 qubit quantum state tomography.  
- Exploring physics‑informed neural networks for reconstructing quantum states
  from noisy measurements.  
- Providing a reproducible, end‑to‑end example for portfolios and applications
  (e.g., undergraduate physics / quantum information programs).

**Recommended scenarios:**

- Synthetic 2‑qubit states drawn from random ensembles (e.g., Ginibre‑based).  
- Simulated projective measurements in Pauli bases with finite shot noise.  
- Comparing a PINN‑style model to classical linear inversion baselines.

The model is **not** intended as a production‑ready tomography tool for real
hardware without further calibration and validation.

---

## 3. Out‑of‑Scope / Misuse

The current model and code are **not designed or validated for**:

- Systems with more than 2 qubits (Hilbert space grows exponentially; the
  architecture and training regime would need modification).  
- Direct, plug‑and‑play use on raw experimental data from real quantum devices
  without adapting noise models and measurement calibration.  
- Security‑critical or safety‑critical decision making.  
- Any claim of “certified” tomography; this is a research/educational pipeline,
  not a standards‑compliant tomography package.

Users should treat results as **research‑grade approximations**, not certified
reconstructions.

---

## 4. Data: How the Model is Trained

### 4.1 Data generation

Training and evaluation data are generated synthetically using Python code in
`src/data/` (not from real hardware):

1. **State generation**: random 2‑qubit states \(\rho\) sampled from a
   distribution such as the Ginibre ensemble.  
2. **Measurements**: simulated projective measurements in combinations of Pauli
   bases (e.g., \(X, Y, Z\) on each qubit) with a finite number of shots.  
3. **Features**:
   - **counts**: measurement outcome frequencies for each basis pair;  
   - **pauli**: estimated expectation values \(\langle P \rangle\) for
     2‑qubit Pauli operators (in some experiments).

### 4.2 Labels

- The target (label) for each example is the **true underlying density matrix**
  \(\rho\) from which the measurements were simulated.

### 4.3 Distribution shift warning

- The training data is generated from **idealized, simulated noise models**.  
- Real‑world hardware may exhibit calibration errors, crosstalk, drift, and
  non‑Markovian noise that are not captured by this dataset.  
- Accuracy on real devices may be different from reported synthetic results.

---

## 5. Model Architecture & Training

### 5.1 Architecture

RhoNet2Q is a fully connected feed‑forward network (multi‑layer perceptron):

- **Input dimension**:
  - counts features: 36 (9 basis pairs × 4 outcomes)  
  - pauli features: 15 (2‑qubit Pauli expectations), if used
- **Hidden layers**: typically 2 layers with sizes such as [512, 512]  
- **Output**: parameters of a lower‑triangular complex matrix \(L\) whose size
  corresponds to a 4×4 density matrix.

The network:
- Uses non‑linear activations between hidden layers.  
- Returns the real and imaginary parts of \(L\), then constructs \(\rho\)
  via \(\rho = LL^\dagger / \mathrm{Tr}(LL^\dagger)\).

### 5.2 Training setup (typical)

- Optimizer: Adam or similar first‑order optimizer.  
- Loss: based on Frobenius distance between predicted and true \(\rho\)
  (optionally combined with additional terms).  
- Batch size: chosen to balance GPU/CPU memory and speed.  
- Epochs: on the order of tens (e.g., 40–80) with early stopping.  
- Hyperparameters are configurable via CLI flags to `train_2q.py`.

Reference training commands are documented in the README and in the demo
notebooks, together with a `reproduce.py` script that runs a full demo pipeline.

---

## 6. Evaluation & Reported Performance

### 6.1 Metrics

The model is evaluated using:

- **Frobenius distance** between \(\rho_\text{pred}\) and \(\rho_\text{true}\):  

  \[
  \lVert \rho_{\text{pred}} - \rho_{\text{true}} \rVert_F
  = \sqrt{\sum_{i,j} \lvert (\rho_{\text{pred}})_{ij}
  - (\rho_{\text{true}})_{ij} \rvert^2 } .
  \]

- **Quantum fidelity** between the predicted and true density matrices:  

  \[
  F(\rho_{\text{pred}}, \rho_{\text{true}})
  = \bigl(\mathrm{Tr}\, \sqrt{\sqrt{\rho_{\text{true}}}\,
  \rho_{\text{pred}}\, \sqrt{\rho_{\text{true}}}}\bigr)^2 .
  \]

Both metrics are reported as **test set means** over many random states.

### 6.2 Example results (representative)

For a representative configuration (2‑qubit, counts features, 256 shots, a
moderately large training set, and a [512, 512] hidden architecture), the model
achieved roughly:

- **Frobenius mean**: ≈ 0.10  
- **Fidelity mean**: ≈ 0.98

In the same setting, a simple **linear inversion** baseline typically shows:

- Larger Frobenius error (≈ 0.5+)  
- Lower fidelity (≈ 0.68–0.70)

Exact numbers depend on random seeds, dataset size, and hyperparameters; detailed
metrics and plots are provided in `reports/` and summarized in the README.

---

## 7. Limitations

Important limitations of this model include:

1. **System size**: The current architecture is tailored for **2‑qubit** states.
   Extending to higher‑dimensional systems would require new architectures,
   regularization, and potentially different parameterizations.  
2. **Synthetic data**: All training and evaluation are performed on **simulated
   data**. Transfer to real quantum hardware is not evaluated here.  
3. **Noise modeling**: Only simple noise models (finite shots, basic simulated
   noise) are considered. Real devices may have more complex error processes.  
4. **Coverage of state space**: Training distributions (e.g., random Ginibre
   states) may not perfectly match the distributions encountered in specific
   experiments (e.g., particular entangled families or hardware‑restricted
   circuits).  
5. **Explainability**: While the physics‑informed parameterization ensures
   valid \(\rho\), the internal feature representations learned by the
   network remain somewhat opaque.

Users should keep these limitations in mind when interpreting results or
attempting to apply the model outside its original experimental setup.

---

## 8. Ethical & Responsible Use Considerations

- This model is intended for **research and education** in quantum information,
  not for safety‑critical or adversarial contexts.  
- Results should not be misrepresented as production‑grade or hardware‑verified
  tomography without further validation.  
- When using this work in publications or applications, users should **clearly
  state the synthetic nature of the data** and the assumptions of the model.

---

## 9. How to Cite / Acknowledge

If you build on this model or use it in an application, please consider citing
the repository and acknowledging:

> Sudip Sen Gupta Arka, *Physics‑Informed Neural Network for 1–2 Qubit Quantum
> State Tomography*, GitHub: `sudip-95/quantum-tomography-ml-v2`.

You may also provide a link to the GitHub repository in your materials so that
reviewers or collaborators can inspect the implementation and the experiments.
