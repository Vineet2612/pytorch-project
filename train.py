"""Train on the training split; select the best model using validation loss."""

import argparse
import random

import torch
from torch import nn

from config import EPOCHS, IMAGE_SIZE, LEARNING_RATE, MODEL_FILE, SEED
from dataset import make_loader
from model import WasteCNN
from prepare_data import create_splits


def run_epoch(model, loader, criterion, device, optimizer=None):
    """Train if an optimizer is supplied, otherwise measure loss and accuracy."""
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    total = 0
    # Validation does not need gradients or parameter updates.
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad()
            scores = model(images)
            loss = criterion(scores, labels)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * labels.size(0)
            correct += (scores.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total


def main():
    """Set up training, record each epoch, and save the best checkpoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split = create_splits(verbose=False)
    train_loader, classes = make_loader("train", split)
    val_loader, _ = make_loader("val", split)
    model = WasteCNN(len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_loss = float("inf")
    print(f"Training on {device}. Classes: {classes}", flush=True)
    history = []
    checkpoint = None
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "train_accuracy": train_acc,
                        "val_loss": val_loss, "val_accuracy": val_acc})
        if val_loss < best_loss:
            best_loss = val_loss
            # Copy the best weights so later updates cannot change this checkpoint.
            weights = {name: value.detach().cpu().clone()
                       for name, value in model.state_dict().items()}
            checkpoint = {"model_state_dict": weights, "classes": classes,
                          "image_size": IMAGE_SIZE, "epoch": epoch, "seed": SEED,
                          "val_loss": val_loss, "val_accuracy": val_acc, "split": split}
        # Keep all epoch measurements inside the same model file for the chart.
        checkpoint["history"] = history
        torch.save(checkpoint, MODEL_FILE)
        print(f"Epoch {epoch:02d}/{args.epochs}: train loss={train_loss:.4f}, "
              f"accuracy={train_acc:.2%}; val loss={val_loss:.4f}, accuracy={val_acc:.2%}", flush=True)
    print(f"Best validation model saved to {MODEL_FILE}")


if __name__ == "__main__":
    main()
