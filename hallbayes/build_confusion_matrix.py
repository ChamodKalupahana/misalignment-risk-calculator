import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# 1. Load your data
df = pd.read_csv("alignment_eval_interactive.csv")

# 2. Prepare labels
# y_true: your manual alignment judgement
# y_pred: EDFL gate decision (answer vs refuse)
y_true = df["aligned_label"].astype(int)        # 1 = aligned, 0 = misaligned
y_pred = df["decision_answer"].astype(int)      # 1 = gate allowed (ANSWER), 0 = REFUSE

# 3. Compute confusion matrix
# labels=[1, 0] -> row 0 = aligned, row 1 = misaligned
cm = confusion_matrix(y_true, y_pred, labels=[1, 0])

# 4. Plot a labelled confusion matrix
fig, ax = plt.subplots(figsize=(5, 4))

im = ax.imshow(cm)  # default colormap

# Tick labels
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(["ANSWER", "REFUSE"])
ax.set_yticklabels(["Aligned (1)", "Misaligned (0)"])

# Axis labels and title
ax.set_xlabel("Gate decision")
ax.set_ylabel("True alignment")
ax.set_title("Alignment vs Gate Decision – Confusion Matrix")

# Show counts in each cell
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center")

plt.tight_layout()

# 5. Save to file for your report
output_path = "alignment_confusion_matrix.png"
fig.savefig(output_path, dpi=300)

print(f"Saved confusion matrix figure to {output_path}")