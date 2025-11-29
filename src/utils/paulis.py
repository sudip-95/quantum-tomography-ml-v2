
import numpy as np

I = np.array([[1, 0],
              [0, 1]], dtype=np.complex128)
X = np.array([[0, 1],
              [1, 0]], dtype=np.complex128)
Y = np.array([[0, -1j],
              [1j, 0]], dtype=np.complex128)
Z = np.array([[1, 0],
              [0, -1]], dtype=np.complex128)

PAULI_DICT = {"I": I, "X": X, "Y": Y, "Z": Z}

def kron(a, b):
    return np.kron(a, b)

def projector(axis: str, outcome: int):
    """axis in {"X","Y","Z"}, outcome in {+1,-1}
    Returns the rank-1 projector onto the eigenstate with eigenvalue outcome.
    """
    if axis == "Z":
        if outcome == +1:
            v = np.array([1,0], dtype=np.complex128)
        else:
            v = np.array([0,1], dtype=np.complex128)
    elif axis == "X":
        if outcome == +1:
            v = (1/np.sqrt(2)) * np.array([1,1], dtype=np.complex128)
        else:
            v = (1/np.sqrt(2)) * np.array([1,-1], dtype=np.complex128)
    elif axis == "Y":
        if outcome == +1:
            v = (1/np.sqrt(2)) * np.array([1,1j], dtype=np.complex128)
        else:
            v = (1/np.sqrt(2)) * np.array([1,-1j], dtype=np.complex128)
    else:
        raise ValueError("axis must be X,Y,Z")
    v = v.reshape(-1,1)
    return v @ v.conj().T
