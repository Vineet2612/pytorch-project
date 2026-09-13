"""Settings shared by the small project scripts."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "dataset-resized"
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_FILE = BASE_DIR / "best_model.pth"

IMAGE_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001
SEED = 42
