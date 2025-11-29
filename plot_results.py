import os
import matplotlib.pyplot as plt

# === Helper to read key=value pairs from metrics files ===
def read_metrics(path):
    data = {}
    with open(path) as f:
        for line in f:
            if "=" in line:
                k, v = line.strip().split("=", 1)
                try:
                    data[k] = float(v)
                except ValueError:
                    data[k] = v
    return data

# === Read your stored results ===
counts_file = "reports/metrics_2q.txt"  # counts features run
pauli_file = "reports/metrics_2q_baseline.txt"  # baseline (if exists)
pauli_net_file = "reports/metrics_2q.txt"  # may reuse depending on run

# Adjust if your files have different names
print("Looking in:", os.getcwd())

metrics_files = []
for root, _, files in os.walk("reports"):
    for f in files:
        if f.endswith(".txt"):
            metrics_files.append(os.path.join(root, f))

print("Found metric files:", metrics_files)

# Collect data
models = []
fid_values = []
fro_values = []

for path in metrics_files:
    m = read_metrics(path)
    label = os.path.basename(path).replace(".txt", "")
    models.append(label)
    fid_values.append(m.get("fid_mean", 0))
    fro_values.append(m.get("fro_mean", 0))

# === Plot Fidelity Comparison ===
plt.figure(figsize=(7,5))
plt.bar(models, fid_values, color=["#6BAED6", "#9ECAE1", "#C6DBEF"])
plt.ylabel("Mean Fidelity ↑")
plt.title("Quantum State Reconstruction Fidelity (2-Qubit)")
plt.ylim(0, 1.05)
for i,v in enumerate(fid_values):
    plt.text(i, v + 0.01, f"{v:.3f}", ha='center')
plt.tight_layout()
plt.savefig("reports/fidelity_comparison.png", dpi=200)
plt.show()

# === Plot Frobenius Distance Comparison ===
plt.figure(figsize=(7,5))
plt.bar(models, fro_values, color=["#FC9272", "#FCAE91", "#FEE0D2"])
plt.ylabel("Mean Frobenius Distance ↓")
plt.title("Reconstruction Error (Frobenius)")
plt.ylim(0, max(fro_values)*1.3)
for i,v in enumerate(fro_values):
    plt.text(i, v + 0.01, f"{v:.3f}", ha='center')
plt.tight_layout()
plt.savefig("reports/frobenius_comparison.png", dpi=200)
plt.show()

print("✅ Plots saved to 'reports/fidelity_comparison.png' and 'reports/frobenius_comparison.png'")
