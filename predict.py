"""Predict the waste category of one image using the saved model."""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from config import MODEL_FILE
from dataset import get_transform
from model import WasteCNN


def predict_image(image_path, model_path=MODEL_FILE):
    """Return the top class and softmax scores in descending order."""
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    classes = checkpoint["classes"]
    model = WasteCNN(len(classes))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    with Image.open(image_path) as image:
        tensor = get_transform(image_size=checkpoint["image_size"])(image.convert("RGB"))
    # One image still needs a batch dimension: [3, 64, 64] -> [1, 3, 64, 64].
    with torch.no_grad():
        probabilities = model(tensor.unsqueeze(0)).softmax(dim=1)[0]
    ordered = sorted(zip(classes, probabilities.tolist()), key=lambda item: item[1], reverse=True)
    return {"image": str(image_path), "predicted_class": ordered[0][0],
            "softmax_scores": dict(ordered),
            "note": "Scores are not calibrated certainty. The model always chooses a known class."}


def main():
    """Read a filename from the command line and print its prediction."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Path to a JPG or PNG image")
    args = parser.parse_args()
    if not args.image.is_file():
        parser.error(f"Image does not exist: {args.image}")
    if not MODEL_FILE.exists():
        parser.error("Saved model missing. Run python train.py first.")
    print(json.dumps(predict_image(args.image), indent=2))


if __name__ == "__main__":
    main()
