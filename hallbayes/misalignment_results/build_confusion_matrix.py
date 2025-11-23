import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize

# 1. Load your data
df = pd.read_csv("alignment_eval_interactive_v1.csv")

# 2. Prepare labels
# y_true: your manual alignment judgement
# y_pred: EDFL gate decision (answer vs refuse)
y_true = df["aligned_label"].astype(int)        # 1 = aligned, 0 = misaligned
y_pred = df["decision_answer"].astype(int)      # 1 = gate allowed (ANSWER), 0 = REFUSE

# 3. Compute confusion matrix
# labels=[1, 0] -> row 0 = aligned, row 1 = misaligned
cm = confusion_matrix(y_true, y_pred, labels=[1, 0])

# 4. We want the diagonal cells in light blue and the off-diagonal errors in light red,
# with each cell getting darker as its count increases.
good_cmap = LinearSegmentedColormap.from_list(
    "lightblue", ["#eef7ff", "#4c7ff0"]
)
bad_cmap = LinearSegmentedColormap.from_list(
    "lightred", ["#fff1f1", "#d32f2f"]
)
diag_mask = np.eye(cm.shape[0], dtype=bool)
diag_values = cm[diag_mask]
offdiag_values = cm[~diag_mask]

good_norm = Normalize(vmin=0, vmax=max(diag_values.max() if diag_values.size else 0, 1))
bad_norm = Normalize(vmin=0, vmax=max(offdiag_values.max() if offdiag_values.size else 0, 1))

# 4. Plot a labelled confusion matrix
fig, ax = plt.subplots(figsize=(5, 4))

color_grid = np.zeros((cm.shape[0], cm.shape[1], 4))
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        if i == j:
            color_grid[i, j] = good_cmap(good_norm(cm[i, j]))
        else:
            color_grid[i, j] = bad_cmap(bad_norm(cm[i, j]))

im = ax.imshow(color_grid, interpolation="nearest")

# Tick labels
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(["ANSWER", "REFUSE"])
ax.set_yticklabels(["Aligned (1)", "Misaligned (0)"])

# Axis labels and title
ax.set_xlabel("Misalignment Checker Decision")
ax.set_ylabel("True alignment")
ax.set_title("Alignment vs Checker Decision - GPT-4o-mini")

# Show counts in each cell
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center")

plt.tight_layout()

# 5. Save to file for your report
output_path = "alignment_confusion_matrix.png"
fig.savefig(output_path, dpi=300)

print(f"Saved confusion matrix figure to {output_path}")
