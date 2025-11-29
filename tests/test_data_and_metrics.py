import numpy as np

from src.data.two_qubit import make_2q_dataset
from src.eval.metrics import frobenius_rho, fidelity


def test_two_qubit_dataset_shapes_and_density_properties():
    """Basic sanity check on the 2-qubit synthetic dataset.

    We only generate a tiny dataset so this test runs quickly in CI.
    """
    X, Y = make_2q_dataset(
        n_samples=5,
        n_shots=64,
        features="counts",
        seed=0,
    )

    # Feature matrix: (N, D)
    assert X.shape[0] == 5
    assert X.ndim == 2

    # Density matrices: (N, 4, 4)
    assert Y.shape == (5, 4, 4)

    for rho in Y:
        # Hermitian: rho = rho^\dagger
        assert np.allclose(rho, rho.conj().T, atol=1e-6)

        # Trace ~= 1
        tr = np.trace(rho)
        assert np.allclose(tr, 1.0, atol=1e-5)

        # Positive semidefinite: eigenvalues >= 0 (up to numerical tolerance)
        eigvals = np.linalg.eigvalsh(rho)
        assert eigvals.min() >= -1e-6


def test_metrics_on_simple_states():
    """Check that Frobenius distance and fidelity behave correctly on trivial cases."""
    # |00><00|
    rho_00 = np.zeros((4, 4), dtype=np.complex64)
    rho_00[0, 0] = 1.0

    # |11><11|
    rho_11 = np.zeros((4, 4), dtype=np.complex64)
    rho_11[3, 3] = 1.0

    # Identical states: zero Frobenius distance, unit fidelity
    f_same = frobenius_rho(rho_00, rho_00)
    F_same = fidelity(rho_00, rho_00)
    assert np.isclose(f_same, 0.0, atol=1e-6)
    assert np.isclose(F_same, 1.0, atol=1e-6)

    # Orthogonal pure states: non-zero distance, near-zero fidelity
    f_diff = frobenius_rho(rho_00, rho_11)
    F_diff = fidelity(rho_00, rho_11)

    # Frobenius distance between diag(1,0,0,0) and diag(0,0,0,1) is sqrt(2)
    assert np.isclose(f_diff, np.sqrt(2.0), atol=1e-6)
    assert F_diff <= 1e-6
