import argparse
import os
import torch
import numpy as np

# Project imports – adjust names here if they differ in your codebase
from src.data.two_qubit import make_2q_dataset
from src.models.rho_net_2q import RhoNet2Q
from src.eval.metrics import fidelity, frobenius_rho


def build_model(features: str, hidden: str):
    """Build a RhoNet2Q model matching the training configuration.

    Parameters
    ----------
    features : {"counts", "pauli"}
        Type of input features the model expects.
    hidden : str
        Comma-separated hidden layer sizes, e.g. "512,512".
    """
    # Match the input dimension to the feature type
    if features == "counts":
        in_dim = 36  # 9 bases × 4 outcomes
    elif features == "pauli":
        in_dim = 15  # 15 two-qubit Pauli expectations
    else:
        raise ValueError(f"Unknown features type: {features}")

    # Parse hidden sizes
    if hidden.strip() == "":
        hidden_sizes = [512, 512]
    else:
        hidden_sizes = [int(h) for h in hidden.split(",")]

    # IMPORTANT: your RhoNet2Q __init__ does NOT use 'hidden_sizes=' as a keyword,
    # so we pass the hidden sizes as a positional argument (second argument),
    # just like in train_2q.py.
    model = RhoNet2Q(in_dim, hidden_sizes)
    return model



def evaluate_model(model, x_test, y_test, device):
    """Evaluate a trained model on a test dataset.

    We loop over samples one by one because frobenius_rho/fidelity in
    metrics.py expect single 4x4 matrices, not a batch.
    """
    model.eval()
    frob_vals = []
    fid_vals = []

    import numpy as np
    import torch

    N = x_test.shape[0]

    with torch.no_grad():
        for i in range(N):
            # x: (in_dim,)
            x_i = torch.as_tensor(x_test[i], dtype=torch.float32, device=device).unsqueeze(0)  # (1, in_dim)

            # Forward pass: rho_pred_i: (4, 4)
            rho_pred_i = model(x_i)[0].cpu().numpy()

            # True rho: make sure it's a 4x4 numpy array
            y_i = y_test[i]
            if np.iscomplexobj(y_i):
                rho_true_i = np.array(y_i)
            else:
                rho_true_i = np.array(y_i)

            # metrics.frobenius_rho and fidelity expect single matrices (4x4)
            f_i = frobenius_rho(rho_pred_i, rho_true_i)
            F_i = fidelity(rho_pred_i, rho_true_i)

            frob_vals.append(f_i)
            fid_vals.append(F_i)

    frob_mean = float(np.mean(frob_vals))
    fid_mean = float(np.mean(fid_vals))
    return frob_mean, fid_mean



def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a pretrained 2‑qubit tomography model (RhoNet2Q)."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the pretrained model checkpoint (.pt)."
    )
    parser.add_argument(
        "--features",
        type=str,
        default="counts",
        choices=["counts", "pauli"],
        help="Feature type used during training (must match the checkpoint)."
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=256,
        help="Number of measurement shots used to generate the test set."
    )
    parser.add_argument(
        "--test",
        type=int,
        default=2000,
        help="Number of test samples to generate."
    )
    parser.add_argument(
        "--hidden",
        type=str,
        default="512,512",
        help="Comma‑separated hidden sizes, e.g. '512,512'. Must match training."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for test‑set generation."
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Build model
    model = build_model(args.features, args.hidden).to(device)

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    print(f"Loading weights from {args.checkpoint} ...")
    state = torch.load(args.checkpoint, map_location=device)
    # In many training scripts, model is saved as {'model_state_dict': ..., ...}
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        # Fallback: assume entire state_dict was saved directly
        model.load_state_dict(state)

    # Generate test dataset
    print(f"Generating test set: N={args.test}, shots={args.shots}, features={args.features} ...")
    x_test, y_test = make_2q_dataset(
        n_samples=args.test,
        n_shots=args.shots,
        features=args.features,
        seed=args.seed,
    )

    frob, fid = evaluate_model(model, x_test, y_test, device)

    print("\n=== Evaluation on synthetic test set ===")
    print(f"Frobenius mean: {frob:.6f}")
    print(f"Fidelity mean : {fid:.6f}")


if __name__ == "__main__":
    main()

