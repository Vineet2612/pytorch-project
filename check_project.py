"""Small correctness checks; these do not train or overwrite the saved model."""

import torch
from torch import nn

from config import IMAGE_SIZE
from prepare_data import create_splits
from dataset import make_loader
from model import WasteCNN


def main():
    """Check independent splits, image tensors, learning and evaluation behavior."""
    torch.manual_seed(123)
    torch.set_num_threads(4)
    split = create_splits(verbose=False)
    paths = {name: {item["path"] for item in split[name]} for name in ["train", "val", "test"]}
    for name in paths:
        assert len(paths[name]) == len(split[name]), "Repeated file in a split"
        assert {item["label"] for item in split[name]} == set(range(len(split["classes"])))
    assert not paths["train"] & paths["val"]
    assert not paths["train"] & paths["test"]
    assert not paths["val"] & paths["test"]
    loader, classes = make_loader("val", split)
    images, labels = next(iter(loader))
    assert images.shape[1:] == (3, IMAGE_SIZE, IMAGE_SIZE)
    assert images.dtype == torch.float32 and labels.dtype == torch.int64
    assert images.min() >= -1 and images.max() <= 1
    model = WasteCNN(len(classes))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    before = model.features[0].weight.detach().clone()
    scores = model(images)
    assert scores.shape == (len(labels), len(classes))
    loss = nn.CrossEntropyLoss()(scores, labels)
    optimizer.zero_grad()
    loss.backward()
    assert torch.isfinite(model.features[0].weight.grad).all()
    assert model.features[0].weight.grad.abs().sum() > 0
    optimizer.step()
    assert not torch.equal(before, model.features[0].weight), "Weights did not change"
    model.eval()
    with torch.no_grad():
        first = model(images)
        second = model(images)
    assert torch.equal(first, second)
    assert not first.requires_grad
    assert torch.allclose(first.softmax(1).sum(1), torch.ones(len(labels)))
    print("PASS: splits are disjoint and contain every class")
    print("PASS: RGB tensor shapes, types and normalization")
    print("PASS: output dimensions, finite gradients and parameter updates")
    print("PASS: repeatable evaluation without gradients; probabilities sum to one")
    print(f"Model parameters: {sum(parameter.numel() for parameter in model.parameters()):,}")


if __name__ == "__main__":
    main()
