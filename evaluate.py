"""Measure the saved model on test images, and print results and create the two charts."""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn

from config import MODEL_FILE, OUTPUT_DIR
from dataset import make_loader
from model import WasteCNN


def plot_history(history):
    """Plot the training measurements stored inside the saved model file."""
    epochs = [int(row["epoch"]) for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, metric in zip(axes, ["loss", "accuracy"]):
        for split in ["train", "val"]:
            axis.plot(epochs, [float(row[f"{split}_{metric}"]) for row in history], label=split)
        axis.set(xlabel="Epoch", ylabel=metric.capitalize(), title=metric.capitalize())
        axis.legend()
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "training_curves.png", dpi=150)
    plt.close(fig)


def main():
    """Load the best model, print metrics and predictions, and save two charts."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    checkpoint = torch.load(MODEL_FILE, map_location="cpu", weights_only=True)
    split = checkpoint["split"]
    loader, classes = make_loader("test", split, checkpoint["image_size"])
    if classes != checkpoint["classes"]:
        raise ValueError("Class order differs from the saved model.")
    model = WasteCNN(len(classes))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    criterion = nn.CrossEntropyLoss()
    matrix = torch.zeros(len(classes), len(classes), dtype=torch.int64)
    total_loss = 0.0
    predictions = []
    with torch.no_grad():
        for images, labels in loader:
            scores = model(images)
            total_loss += criterion(scores, labels).item() * labels.size(0)
            probabilities = scores.softmax(dim=1)
            confidence, predicted = probabilities.max(dim=1)
            for actual, guess, probability in zip(labels.tolist(), predicted.tolist(), confidence.tolist()):
                matrix[actual, guess] += 1
                sample = loader.dataset.samples[len(predictions)]
                predictions.append([sample["path"], classes[actual], classes[guess], probability])
    per_class = {}
    for index, name in enumerate(classes):
        tp = matrix[index, index].item()
        support = matrix[index].sum().item()
        predicted_count = matrix[:, index].sum().item()
        precision = tp / predicted_count if predicted_count else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[name] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
    train_counts = [sum(item["label"] == i for item in split["train"]) for i in range(len(classes))]
    majority_index = train_counts.index(max(train_counts))
    metrics = {"test_images": len(predictions), "test_loss": total_loss / len(predictions),
               "test_accuracy": matrix.diag().sum().item() / len(predictions),
               "macro_f1": sum(row["f1"] for row in per_class.values()) / len(classes),
               "majority_baseline_class": classes[majority_index],
               "majority_baseline_accuracy": matrix[majority_index].sum().item() / len(predictions),
               "selected_epoch": checkpoint["epoch"], "validation_accuracy": checkpoint["val_accuracy"],
               "per_class": per_class, "classes": classes, "confusion_matrix": matrix.tolist()}
    print("Test image predictions")
    for path, actual, guessed, score in predictions:
        print(f"{path}: actual={actual}, predicted={guessed}, score={score:.4f}")
    fig, axis = plt.subplots(figsize=(7, 6))
    display = axis.imshow(matrix.numpy(), cmap="Blues")
    axis.set(xticks=range(len(classes)), yticks=range(len(classes)),
             xticklabels=classes, yticklabels=classes, xlabel="Predicted class", ylabel="Actual class",
             title="Test confusion matrix")
    plt.setp(axis.get_xticklabels(), rotation=35, ha="right")
    for i in range(len(classes)):
        for j in range(len(classes)):
            axis.text(j, i, str(matrix[i, j].item()), ha="center", va="center",
                      color="white" if matrix[i, j] > matrix.max() / 2 else "black")
    fig.colorbar(display, ax=axis)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    plot_history(checkpoint["history"])
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
