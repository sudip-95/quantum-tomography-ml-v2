
import numpy as np
from ..utils.seeding import set_seed
from ..utils.paulis import X, Y, Z, I, kron, projector

AXES = ["X","Y","Z"]

def _ginibre_density(rng: np.random.Generator) -> np.ndarray:
    G = rng.normal(size=(4,4)) + 1j*rng.normal(size=(4,4))
    A = G @ G.conj().T
    return A / np.trace(A)

def _noisy_pure_density(rng: np.random.Generator) -> np.ndarray:
    v = rng.normal(size=(4,)) + 1j * rng.normal(size=(4,))
    v = v / np.linalg.norm(v)
    rho_pure = np.outer(v, v.conj())
    eps = rng.uniform(0.0, 0.4)
    rho = (1-eps)*rho_pure + eps * np.eye(4)/4.0
    return rho / np.trace(rho)

def _sample_states(n: int, seed: int, mix: float = 0.5):
    rng = np.random.default_rng(seed)
    states = []
    for _ in range(n):
        if rng.random() < mix:
            states.append(_ginibre_density(rng))
        else:
            states.append(_noisy_pure_density(rng))
    return np.array(states)

def _pauli_expectations(rho: np.ndarray) -> np.ndarray:
    # 15 expectations: (X,Y,Z)⊗I, I⊗(X,Y,Z), and all pairwise (X,Y,Z)⊗(X,Y,Z)
    ops_single = [X, Y, Z]
    feats = []
    for a in ops_single:
        M = kron(a, I); feats.append(np.real(np.trace(rho @ M)))
    for b in ops_single:
        M = kron(I, b); feats.append(np.real(np.trace(rho @ M)))
    for a in ops_single:
        for b in ops_single:
            M = kron(a, b)
            feats.append(np.real(np.trace(rho @ M)))
    return np.array(feats, dtype=np.float64)  # shape (15,)

def _counts_from_rho(rho: np.ndarray, n_shots: int, rng: np.random.Generator) -> np.ndarray:
    # 9 settings (a,b) with a,b in {X,Y,Z}; 4 outcomes per setting => 36 features (freqs)
    counts = []
    for a in AXES:
        for b in AXES:
            # 4 projectors: (++,+-,-+,--)
            projs = [
                kron(projector(a,+1), projector(b,+1)),
                kron(projector(a,+1), projector(b,-1)),
                kron(projector(a,-1), projector(b,+1)),
                kron(projector(a,-1), projector(b,-1)),
            ]
            probs = np.array([np.real(np.trace(rho @ P)) for P in projs], dtype=np.float64)
            probs = np.clip(probs, 0.0, 1.0)
            probs = probs / probs.sum()
            c = rng.multinomial(n_shots, probs)
            freqs = c / float(n_shots)
            counts.extend(freqs.tolist())
    return np.array(counts, dtype=np.float64)  # (36,)

def make_2q_dataset(n_samples: int, n_shots: int, features: str = "counts", seed: int = 0):
    """features: "counts" -> (n,36); "pauli" -> (n,15)
    returns X, y where y is (n,4,4) complex density matrices
    """
    set_seed(seed)
    rng = np.random.default_rng(seed)
    rhos = _sample_states(n_samples, seed)
    if features == "counts":
        X = np.stack([_counts_from_rho(rho, n_shots, rng) for rho in rhos], axis=0)
    elif features == "pauli":
        # simulate counts first, then convert to expectations for realism
        counts = np.stack([_counts_from_rho(rho, n_shots, rng) for rho in rhos], axis=0)
        X = []
        for idx in range(n_samples):
            feats = []
            pair = {}
            k = 0
            for a in AXES:
                for b in AXES:
                    f = counts[idx, 4*k:4*k+4]
                    outcomes = [(+1,+1),(+1,-1),(-1,+1),(-1,-1)]
                    exp_ab = sum(s1*s2*freq for (s1,s2), freq in zip(outcomes, f))
                    pair[(a,b)] = exp_ab
                    k += 1
            # singles via averaging across partner axis
            singles_a = {a: np.mean([pair[(a,b)] for b in AXES]) for a in AXES}
            singles_b = {b: np.mean([pair[(a,b)] for a in AXES]) for b in AXES}
            for a in AXES:
                feats.append(singles_a[a])
            for b in AXES:
                feats.append(singles_b[b])
            for a in AXES:
                for b in AXES:
                    feats.append(pair[(a,b)])
            X.append(np.array(feats, dtype=np.float64))
        X = np.stack(X, axis=0)
    else:
        raise ValueError("features must be 'counts' or 'pauli'")
    return X, rhos
