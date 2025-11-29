# Physics Background: Quantum State Tomography with Density Matrices

This note explains the physics ideas behind the project in a way that a physics undergraduate (or a curious reviewer) can follow. It is meant to be read alongside the main code in the repository.

---

## 1. Qubits and Quantum States

A **qubit** is the quantum analogue of a classical bit.  
Instead of being only 0 or 1, a pure qubit state can be written as

\[
\lvert \psi \rangle = \alpha \lvert 0 \rangle + \beta \lvert 1 \rangle, \quad
\alpha, \beta \in \mathbb{C}, \quad \lvert \alpha \rvert^2 + \lvert \beta \rvert^2 = 1 .
\]

For a single qubit, every pure state can be represented by a point on the **Bloch sphere**. The coordinates of that point form a 3‑vector \(\mathbf{r} = (x,y,z)\) with \(\lVert \mathbf{r} \rVert = 1\). The state is then

\[
\rho = \frac{1}{2}\bigl(I + x X + y Y + z Z\bigr),
\]

where \(X, Y, Z\) are the Pauli matrices.

Real experiments are noisy and often produce **mixed states** rather than pure ones, which is why we work with **density matrices** instead of only state vectors.

---

## 2. Density Matrices and Physical Constraints

A **density matrix** \(\rho\) describes the state of a quantum system (pure or mixed).  
For an \(n\)-qubit system, \(\rho\) is a \(2^n \times 2^n\) complex matrix that satisfies three key properties:

1. **Hermitian**:  
   \(\rho = \rho^\dagger\).  
   This ensures that observable quantities (like measurement probabilities) are real.

2. **Positive semidefinite (PSD)**:  
   For any state \(\lvert \phi \rangle\),  
   \[ \langle \phi \rvert \rho \lvert \phi \rangle \ge 0. \]  
   This guarantees non‑negative probabilities.

3. **Unit trace**:  
   \(\mathrm{Tr}(\rho) = 1\).  
   This says that the total probability over all outcomes is 1.

For a **pure state**, \(\rho = \lvert \psi \rangle \langle \psi \rvert\) and \(\mathrm{Tr}(\rho^2) = 1\).  
For a **mixed state**, \(\rho\) is a convex combination of projectors, and \(\mathrm{Tr}(\rho^2) < 1\).

In this project, the neural network must output a matrix that always obeys these three constraints, otherwise the result would not represent a valid quantum state.

---

## 3. Measurements and Pauli Bases

To learn \(\rho\), we do not have direct access to it.  
Instead, we have outcomes from **measurements** on many copies of the state.

For one or two qubits, a convenient choice is to measure in combinations of the **Pauli bases**:

- \(X\) basis: eigenstates of the Pauli \(X\) matrix  
- \(Y\) basis: eigenstates of \(Y\)  
- \(Z\) basis: eigenstates of \(Z\)

For two qubits, we measure tensor products such as \(Z \otimes Z\), \(X \otimes Z\), etc.

From repeated measurements (with a finite number of **shots**), we estimate either:

- **Counts**: how many times each outcome (00, 01, 10, 11) occurred for each measurement setting; or  
- **Expectation values**: average values of observables like \(\langle X \otimes Z \rangle\).

Both contain enough information, in principle, to reconstruct \(\rho\). The project supports two feature types:
- `counts`: raw measurement frequencies (noisy but more realistic);
- `pauli`: estimated expectation values of Pauli operators.

---

## 4. Quantum State Tomography

**Quantum state tomography** is the process of reconstructing \(\rho\) from measurement statistics.

Classical approaches include:

- **Linear inversion**: solve a linear system that maps measurement statistics to matrix elements of \(\rho\). This is fast but can easily produce matrices that violate PSD or trace‑1 due to noise.
- **Maximum likelihood methods**: optimize over valid density matrices to best match observed statistics. These enforce physical constraints but can be computationally heavier.

In practice, tomography is challenging because:

- Measurements are noisy (finite shots).  
- The number of parameters grows quickly with qubit number (for \(n\) qubits, \(\rho\) has \(4^n - 1\) real parameters).  
- Simple methods can give *unphysical* states.

This project uses a **neural network** as a flexible estimator that is explicitly designed to output only valid density matrices.

---

## 5. Physics‑Informed Parameterization: \(\rho = LL^\dagger / \mathrm{Tr}(LL^\dagger)\)

A naive neural network that outputs arbitrary matrices would not automatically satisfy the Hermitian/PSD/trace constraints. To fix this, we use a **physics‑informed parameterization**:

1. The network outputs the entries of a lower‑triangular complex matrix \(L\) (the **Cholesky factor**).
2. We form
   \[
   A = L L^\dagger .
   \]
   By construction, \(A\) is Hermitian and PSD.
3. We normalize the trace:
   \[
   \rho = \frac{A}{\mathrm{Tr}(A)} .
   \]

This guarantees, for every network output:

- \(\rho = \rho^\dagger\) (Hermitian)  
- \(\rho \succeq 0\) (positive semidefinite)  
- \(\mathrm{Tr}(\rho) = 1\) (unit trace)

So even if the network is not perfectly trained, its predictions are **always valid quantum states**. This is the core “physics‑informed” idea in this project.

---

## 6. Learning from Noisy Data

The training data is generated by:

1. Sampling random 1–2 qubit states (e.g., via Ginibre ensembles).  
2. Simulating measurements in different Pauli bases.  
3. Using a finite number of **shots** to obtain noisy counts.  

The model sees input vectors derived from these counts or Pauli expectations and learns a mapping

\[
\text{(features from measurements)} \; \longrightarrow \; \text{density matrix } \rho .
\]

Because the model is trained on **noisy** data, it learns to be robust to measurement noise and finite sampling, similar to realistic experiments.

---

## 7. Loss Function and Evaluation Metrics

During training, the network parameters are optimized to minimize a loss that measures how close the predicted \(\rho_{\text{pred}}\) is to the true \(\rho_{\text{true}}\). Two main quantities are used:

### Frobenius Distance

The **Frobenius norm** between two matrices is

\[
\lVert \rho_{\text{pred}} - \rho_{\text{true}} \rVert_F
= \sqrt{\sum_{i,j} \lvert (\rho_{\text{pred}})_{ij} - (\rho_{\text{true}})_{ij} \rvert^2 } .
\]

Smaller values mean the matrices are closer entry‑wise.

### Fidelity

**Quantum state fidelity** measures how close two density matrices are as quantum states. For general mixed states:

\[
F(\rho_{\text{pred}}, \rho_{\text{true}})
= \bigl( \mathrm{Tr}\, \sqrt{\sqrt{\rho_{\text{true}}}\, \rho_{\text{pred}}\, \sqrt{\rho_{\text{true}}}} \bigr)^2.
\]

- \(F = 1\) means the states are identical.  
- Smaller \(F\) means less overlap.

In this project, we report **mean fidelity** over a test set of states. Values around \(0.98\) indicate very high reconstruction quality.

---

## 8. Why This Counts as Physics‑Informed Machine Learning

Several aspects of the model and training procedure encode physical knowledge:

1. **Valid state parameterization**: The \(LL^\dagger / \mathrm{Tr}(LL^\dagger)\) construction guarantees that outputs are always density matrices.
2. **Pauli measurements and counts**: Inputs are structured by the physics of qubit measurements, not arbitrary features.
3. **Noise modeling**: Finite shots and sampling noise in training data reflect realistic experimental conditions.
4. **Evaluation in physical metrics**: Performance is measured in fidelity and Frobenius norm, not just generic ML losses.

This combination of domain knowledge and neural networks is what makes the approach “physics‑informed” rather than just “black‑box” machine learning.

---

## 9. Limitations and Extensions

- The current implementation targets **1–2 qubit** systems. Extending to more qubits requires more efficient parameterizations and training strategies due to the exponential growth of the Hilbert space.
- Training and evaluation are performed on **simulated data**. Applying the method to real hardware (e.g., IBM Q devices) would require careful noise characterization and calibration.
- The project compares against a classical linear inversion baseline. Additional comparisons (e.g., maximum likelihood tomography, neural quantum states) would give an even richer picture.

Possible future directions include:

- Scaling to 3–4 qubits with factorized or low‑rank parameterizations.  
- Incorporating realistic device noise models from quantum hardware toolkits.  
- Performing ablation studies on shot count, feature types, and network architectures.  

---

This explanation is intended to bridge the gap between the code in the repository and the underlying quantum physics. It can be used as a standalone document for applications, teaching, or as a companion note for anyone inspecting the project.
