
import numpy as np
from ..utils.paulis import I, X, Y, Z, kron

OPS = {"I": I, "X": X, "Y": Y, "Z": Z}
AXES = ["X","Y","Z"]

def pauli_reconstruction_from_expectations(feats: np.ndarray) -> np.ndarray:
    """feats: (15,) ordered as (X⊗I, Y⊗I, Z⊗I, I⊗X, I⊗Y, I⊗Z, then all a⊗b)"""
    c = {}
    c[("I","I")] = 1.0
    idx = 0
    for a in AXES:
        c[(a,"I")] = float(feats[idx]); idx += 1
    for b in AXES:
        c[("I",b)] = float(feats[idx]); idx += 1
    for a in AXES:
        for b in AXES:
            c[(a,b)] = float(feats[idx]); idx += 1
    rho = np.zeros((4,4), dtype=np.complex128)
    for a_label, A in OPS.items():
        for b_label, B in OPS.items():
            val = c.get((a_label,b_label), 0.0)
            rho += val * kron(A, B)
    rho = rho / 4.0
    return rho

def project_psd_trace1(rho: np.ndarray) -> np.ndarray:
    H = (rho + rho.conj().T)/2
    w, V = np.linalg.eigh(H)
    w = np.clip(np.real(w), 0.0, None)
    if w.sum() <= 0:
        w = np.ones_like(w)/len(w)
    else:
        w = w / w.sum()
    rho_psd = V @ np.diag(w) @ V.conj().T
    rho_psd = rho_psd / np.trace(rho_psd)
    return rho_psd
