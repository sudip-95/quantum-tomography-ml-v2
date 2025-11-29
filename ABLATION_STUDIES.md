# Ablation & Sensitivity Studies

This document outlines suggested ablation and sensitivity experiments for the
*Quantum State Tomography with Physics-Informed ML* project. It is designed for
reviewers and for future extensions of the work. All experiments are based on
the 2-qubit PINN model (`train_2q.py`).

Each subsection provides:
- A **question** we are asking,
- The **commands** to run, and
- A **table template** to record results (Frobenius distance and fidelity).

You can fill in the tables after running the experiments.

---

## 1. Features: Counts vs Pauli

**Question.** How does performance change when we feed the network raw counts
versus Pauli expectation values?

**Baseline configuration (demo scale).**

- Shots: 256 for counts, 512 for Pauli
- Train/Val/Test sizes: 15k / 3k / 3k (or the maximum you can run)
- Hidden layers: 512, 512

### Commands

**Counts features**

```bash
python -m src.train.train_2q --features counts --shots 256        --train 15000 --val 3000 --test 3000 --epochs 80 --hidden 512 512
```

**Pauli features + baseline**

```bash
python -m src.train.train_2q --features pauli --shots 512        --train 15000 --val 3000 --test 3000 --epochs 80 --hidden 512 512
```

After each run, copy the final `TEST | ...` lines and (for Pauli) the
`Baseline | ...` line into the table below.

### Results (to be filled in)

| Features | Shots | Train / Val / Test | Model Frobenius ↓ | Model Fidelity ↑ | Baseline Frobenius ↓ | Baseline Fidelity ↑ |
|----------|-------|--------------------|-------------------|------------------|----------------------|---------------------|
| counts   | 256   | 15000 / 3000 / 3000| TBD               | TBD              | –                    | –                   |
| pauli    | 512   | 15000 / 3000 / 3000| TBD               | TBD              | TBD                  | TBD                 |

**Expected trend.** Counts are closer to realistic hardware readout and may
perform differently from Pauli expectations; this ablation highlights the
trade‑off between raw-data realism and engineered features.

---

## 2. Shot Count Sensitivity

**Question.** How does the reconstruction quality depend on the number of
measurement shots per setting?

Use **counts features** and keep other hyperparameters fixed.

### Suggested shot values

- 64, 128, 256, 512

### Command template

```bash
python -m src.train.train_2q --features counts --shots SHOTS        --train 8000 --val 2000 --test 2000 --epochs 40 --hidden 512 512
```

Replace `SHOTS` with 64, 128, 256, 512 and record the final test metrics.

### Results (to be filled in)

| Shots | Train / Val / Test | Frobenius ↓ | Fidelity ↑ |
|-------|--------------------|-------------|------------|
| 64    | 8000 / 2000 / 2000 | TBD         | TBD        |
| 128   | 8000 / 2000 / 2000 | TBD         | TBD        |
| 256   | 8000 / 2000 / 2000 | TBD         | TBD        |
| 512   | 8000 / 2000 / 2000 | TBD         | TBD        |

**Expected trend.** Higher shot counts reduce statistical noise, so fidelity
should generally increase and Frobenius distance should decrease, with
diminishing returns after some point.

---

## 3. Hidden Layer Size / Model Capacity

**Question.** How sensitive is performance to the size of the hidden layers in
the PINN (RhoNet2Q)?

Keep the data configuration fixed and vary the hidden sizes.

### Baseline configuration

- Features: counts
- Shots: 256
- Train/Val/Test: 8000 / 2000 / 2000
- Epochs: 40

### Command template

```bash
python -m src.train.train_2q --features counts --shots 256        --train 8000 --val 2000 --test 2000 --epochs 40 --hidden H1 H2
```

Suggested hidden sizes:

- 256 256
- 512 512
- 512 256
- 256 512

### Results (to be filled in)

| Hidden sizes | Params (approx) | Frobenius ↓ | Fidelity ↑ |
|--------------|-----------------|-------------|------------|
| 256, 256     | TBD             | TBD         | TBD        |
| 512, 512     | TBD             | TBD         | TBD        |
| 512, 256     | TBD             | TBD         | TBD        |
| 256, 512     | TBD             | TBD         | TBD        |

**Expected trend.** Larger models may fit the data better but can overfit or
be slower. A sweet spot in capacity often emerges where performance and
efficiency are balanced.

---

## 4. Training Set Size

**Question.** How does performance depend on the number of training examples,
holding test conditions fixed?

Use a fixed evaluation dataset and vary the training set size.

### Baseline config

- Features: counts
- Shots: 256
- Validation/Test: 3000 / 3000
- Epochs: 60

### Command template

```bash
python -m src.train.train_2q --features counts --shots 256        --train N_TRAIN --val 3000 --test 3000 --epochs 60 --hidden 512 512
```

Suggested values for `N_TRAIN`:

- 2000, 5000, 10000, 15000

### Results (to be filled in)

| Train size | Val / Test | Frobenius ↓ | Fidelity ↑ |
|-----------:|------------|-------------|------------|
| 2000       | 3000/3000  | TBD         | TBD        |
| 5000       | 3000/3000  | TBD         | TBD        |
| 10000      | 3000/3000  | TBD         | TBD        |
| 15000      | 3000/3000  | TBD         | TBD        |

**Expected trend.** Performance should improve as more data is used, then
saturate when the model capacity becomes the main bottleneck.

---

## 5. Suggested Writing for the Results Section

Once you have filled in the tables, you can summarize the ablations in a few
sentences in your report or README, for example:

- *“Using counts features at 256 shots achieves comparable fidelity to Pauli
features at 512 shots, while being closer to realistic hardware measurements.”*
- *“Fidelity increases rapidly between 64 and 256 shots, with only minor gains
beyond 256, suggesting diminishing returns in the high-shot regime.”*
- *“Hidden layer sizes of 512–512 offer the best trade‑off between accuracy and
training cost; smaller networks underfit, while larger ones give marginal
improvements.”*
- *“Increasing the training set from 2k to 10k examples significantly improves
reconstruction fidelity; beyond 10k, improvements are modest, indicating that
model capacity becomes the limiting factor.”*

These statements are placeholders; replace them with observations based on your
actual results.
