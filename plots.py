import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve

def save_history(history, csv_path="training_log.csv"):
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["epoch"] + list(history.history.keys())
        writer.writerow(header)
        for i in range(len(history.epoch)):
            row = [i + 1]
            for key in history.history:
                row.append(history.history[key][i])
            writer.writerow(row)


def plot_learning_curves(csv_path, metric="dice_coefficient"):
    df = pd.read_csv(csv_path)
    epochs = df["epoch"]
    train_vals = df[metric]
    val_vals = df["val_" + metric]

    plt.figure()
    plt.plot(epochs, train_vals, label="Training " + metric)
    plt.plot(epochs, val_vals,   label="Validation " + metric)
    plt.xlabel("Epoch")
    plt.ylabel(metric)
    plt.title("Learning Curves")
    plt.legend()
    plt.savefig("learning_curves.png")
    plt.close()


def plot_precision_recall(y_true, y_scores):
    y_flat    = y_true.reshape(-1)
    scores_flat = y_scores.reshape(-1)
    precision, recall, _ = precision_recall_curve(y_flat, scores_flat)

    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision‑Recall Curve")
    plt.savefig("precision_recall_curve.png")
    plt.close()

def plot_roc_curve(y_true, y_scores):
    y_flat      = y_true.reshape(-1)
    scores_flat = y_scores.reshape(-1)
    fpr, tpr, _ = roc_curve(y_flat, scores_flat)

    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.savefig("roc_curve.png")
    plt.close()


def compute_dice(y_true, y_pred, eps=1e-6):
    a = y_true.flatten()
    b = y_pred.flatten()
    intersection = np.sum(a * b)
    return (2 * intersection + eps) / (np.sum(a) + np.sum(b) + eps)


def plot_dice_vs_threshold(y_true, y_scores):
    thresholds = np.linspace(0.1, 0.9, 17)
    dice_vals  = []
    for t in thresholds:
        preds = (y_scores > t).astype(np.uint8)
        dice = compute_dice(y_true, preds)
        dice_vals.append(dice)

    best_index    = int(np.argmax(dice_vals))
    best_threshold = thresholds[best_index]

    plt.figure()
    plt.plot(thresholds, dice_vals, marker="o")
    plt.axvline(best_threshold, linestyle="--")
    plt.xlabel("Threshold")
    plt.ylabel("Dice Score")
    plt.title("Dice vs Threshold")
    plt.savefig("dice_vs_threshold.png")
    plt.close()

    return best_threshold


def save_triplet(input_img, true_mask, pred_mask):
    inp = np.squeeze(input_img)
    gt  = np.squeeze(true_mask)
    pr  = np.squeeze(pred_mask)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(inp, cmap="gray")
    axes[0].set_title("Input Image")
    axes[1].imshow(gt, cmap="gray")
    axes[1].set_title("Ground Truth")
    axes[2].imshow(pr, cmap="gray")
    axes[2].set_title("Predicted Mask")
    for ax in axes:
        ax.axis("off")

    plt.savefig("triplet.png")
    plt.close()