import argparse
import numpy as np
from . import __init__  # noqa: F401
from ..utils.seeding import set_seed

def _sample_bloch_vectors(n: int, rng: np.random.Generator) -> np.ndarray:
    # Sample directions from normal, radii ~ U(0,1)^(1/3) for uniform in ball
    v = rng.normal(size=(n, 3))
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    v = v / norms
    radii = rng.random(n) ** (1.0 / 3.0)
    return v * radii[:, None]

def make_1q_dataset(n_samples: int, n_shots: int, seed: int = 0):
    """
    Returns:
      X: (n_samples, 3) noisy estimates of <X>, <Y>, <Z>
      y: (n_samples, 3) true Bloch vectors (rx, ry, rz)
    """
    set_seed(seed)
    rng = np.random.default_rng(seed)
    r = _sample_bloch_vectors(n_samples, rng)  # true Bloch vectors

    # True expectations are r itself
    exps_true = r.copy()

    # Simulate finite-shot measurements for X, Y, Z
    X_est = np.empty((n_samples, 3), dtype=np.float64)
    for i, axis in enumerate(range(3)):  # 0->X,1->Y,2->Z mapping
        p_plus = (1.0 + exps_true[:, i]) * 0.5  # p(+|axis)
        p_plus = np.clip(p_plus, 0.0, 1.0)
        n_plus = rng.binomial(n=n_shots, p=p_plus)
        # <A> = (n_plus - n_minus)/N = (2*n_plus - N)/N
        X_est[:, i] = (2.0 * n_plus - n_shots) / float(n_shots)

    return X_est, r

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--shots", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    X, y = make_1q_dataset(args.n, args.shots, args.seed)
    print("Noisy <X,Y,Z> estimates:\n", np.round(X, 3))
    print("True Bloch vectors:\n", np.round(y, 3))
