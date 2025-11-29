
import numpy as np

def mse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a); b = np.asarray(b)
    return float(np.mean((a - b) ** 2))

def bloch_angle_error(r_true: np.ndarray, r_pred: np.ndarray, eps: float = 1e-8) -> float:
    r_true = np.asarray(r_true); r_pred = np.asarray(r_pred)
    n_true = np.linalg.norm(r_true, axis=-1) + eps
    n_pred = np.linalg.norm(r_pred, axis=-1) + eps
    dots = np.sum(r_true * r_pred, axis=-1) / (n_true * n_pred)
    dots = np.clip(dots, -1.0, 1.0)
    angles = np.degrees(np.arccos(dots))
    return float(np.mean(angles))

def frobenius_rho(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    return float(np.linalg.norm(diff, 'fro'))

def fidelity(rho: np.ndarray, sigma: np.ndarray) -> float:
    # F(rho,sigma) = (Tr sqrt( sqrt(rho) sigma sqrt(rho) ))^2
    H1 = (rho + rho.conj().T)/2
    w, V = np.linalg.eigh(H1)
    w = np.clip(np.real(w), 0.0, None)
    sqrt_rho = V @ np.diag(np.sqrt(w + 1e-12)) @ V.conj().T
    M = sqrt_rho @ sigma @ sqrt_rho
    H2 = (M + M.conj().T)/2
    w2, _ = np.linalg.eigh(H2)
    w2 = np.clip(np.real(w2), 0.0, None)
    return float(np.square(np.sum(np.sqrt(w2 + 1e-12))))
