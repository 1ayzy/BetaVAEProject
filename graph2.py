import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json

with open("continuity_experiment_results.json") as f:
    data = json.load(f)

results = data["results"]
# rot_step values: 1,2,4,8 -> labels with actual number of rotation values
labels = [f"{r['rot_values']} val.\n(step={r['rot_step']})" for r in results]
acc = [r["disentanglement_accuracy"] for r in results]

plt.figure(figsize=(7, 5))
bars = plt.bar(labels, acc, color='steelblue', edgecolor='black', alpha=0.85)
# highlight the best (first) bar
bars[0].set_color('darkorange')

plt.ylabel("Disentanglement Accuracy")
plt.title("Experiment 2: Disentanglement vs. rotation continuity")
plt.ylim(0.2, 0.35)
for bar, val in zip(bars, acc):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f'{val:.3f}', ha='center', fontsize=10)

plt.tight_layout()
plt.savefig("exp2_continuity.png", dpi=200)
plt.close()