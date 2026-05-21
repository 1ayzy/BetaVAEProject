import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import numpy as np

with open("beta_vae_experiment_results_2.json", "r") as f:
    data = json.load(f)

results = data["results"]
betas = [r["beta_norm"] for r in results]
acc = [r["disentanglement_accuracy"] for r in results]
mse = [r["mse"] for r in results]

fig, ax1 = plt.subplots(figsize=(9, 5.5))

color1 = 'tab:blue'
ax1.set_xlabel('β_norm')
ax1.set_ylabel('Disentanglement Accuracy', color=color1, fontsize=12)
ax1.plot(betas, acc, marker='o', linestyle='-', color=color1, linewidth=1.8, markersize=7, label='Accuracy')
ax1.tick_params(axis='y', labelcolor=color1)
ax1.set_xscale('log', base=2)
ax1.set_ylim(0.15, 0.50)
ax1.legend(loc='upper left', fontsize=10)

ax2 = ax1.twinx()
color2 = 'tab:red'
ax2.set_ylabel('MSE', color=color2, fontsize=12)
ax2.plot(betas, mse, marker='s', linestyle='--', color=color2, linewidth=1.8, markersize=7, label='MSE')
ax2.tick_params(axis='y', labelcolor=color2)
ax2.legend(loc='upper right', fontsize=10)

# annotate best accuracy
best_idx = np.argmax(acc)
ax1.annotate(f'Best Acc = {acc[best_idx]:.3f}\n(β_norm = {betas[best_idx]})',
             xy=(betas[best_idx], acc[best_idx]),
             xytext=(betas[best_idx] * 2.5, acc[best_idx] - 0.04),
             arrowprops=dict(arrowstyle='->', color='black', lw=1.2),
             fontsize=10, color='black')

plt.title("Experiment 1: β-VAE — Trade-off: Reconstruction vs. Disentanglement", fontsize=13)
fig.tight_layout()
plt.savefig("exp1_beta_tradeoff_2.png", dpi=200)
plt.close()
print("Plot saved to exp1_beta_tradeoff.png")