import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import numpy as np

with open("representation_comparison_results.json") as f:
    data = json.load(f)

results = data["results"]
# assume order: first beta=1.0, second beta=4.0
mi1 = np.array(results[0]["mutual_information"])
mi2 = np.array(results[1]["mutual_information"])

factor_names = ['Shape', 'PosX', 'PosY', 'Scale', 'Rotation']
latent_labels = [f'z{i}' for i in range(mi1.shape[0])]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
cmap = plt.cm.viridis

im1 = axes[0].imshow(mi1, cmap=cmap, aspect='auto', vmin=0, vmax=1)
axes[0].set_title("β = 1.0 (standard VAE)")
axes[0].set_xticks(range(len(factor_names)))
axes[0].set_xticklabels(factor_names, rotation=45)
axes[0].set_yticks(range(len(latent_labels)))
axes[0].set_yticklabels(latent_labels)
axes[0].set_ylabel("Latent dimension")

im2 = axes[1].imshow(mi2, cmap=cmap, aspect='auto', vmin=0, vmax=1)
axes[1].set_title("β = 4.0 (β‑VAE)")
axes[1].set_xticks(range(len(factor_names)))
axes[1].set_xticklabels(factor_names, rotation=45)
axes[1].set_yticks(range(len(latent_labels)))
axes[1].set_yticklabels(latent_labels)

# shared colorbar
cbar = fig.colorbar(im2, ax=axes.ravel().tolist(), shrink=0.85)
cbar.set_label('Mutual Information')

plt.suptitle("Experiment 3: Mutual Information – latent selectivity comparison")
plt.savefig("exp3_mi_heatmaps.png", dpi=200)
plt.close()