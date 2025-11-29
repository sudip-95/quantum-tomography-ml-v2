import subprocess
import sys
import os
from pathlib import Path


def run(cmd: list[str]) -> None:
    print("\n>>> Running:", " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"Command failed with exit code {result.returncode}: {' '.join(cmd)}")
        sys.exit(result.returncode)


def main():
    project_root = Path(__file__).resolve().parent
    os.chdir(project_root)
    print(f"Working directory set to: {project_root}")

    # 1) 2-qubit training (counts features, demo-scale)
    run([
        sys.executable, "-m", "src.train.train_2q",
        "--features", "counts",
        "--shots", "256",
        "--train", "4000",
        "--val", "1000",
        "--test", "1000",
        "--epochs", "30",
        "--hidden", "512", "512",
    ])

    # 2) Generate plots
    run([sys.executable, "plot_results.py"])

    # 3) Evaluate pretrained checkpoint (if present)
    ckpt_path = project_root / "reports" / "mlp_2q_best.pt"
    if ckpt_path.exists():
        run([
            sys.executable, "-m", "src.eval.eval_2q",
            "--checkpoint", str(ckpt_path),
            "--features", "counts",
            "--shots", "256",
            "--test", "2000",
            "--hidden", "512,512",
        ])
    else:
        print(f"Checkpoint not found at {ckpt_path}. Skipping evaluation step.")


if __name__ == "__main__":
    main()
