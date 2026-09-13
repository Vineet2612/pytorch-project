"""Download TrashNet and build reproducible train/validation/test lists in memory."""

import hashlib
import json
import random
import urllib.request
import zipfile
from pathlib import Path

from PIL import Image

from config import DATA_DIR, SEED

DATA_URL = "https://raw.githubusercontent.com/garythung/trashnet/master/data/dataset-resized.zip"
CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]


def download_dataset():
    """Download and extract the original resized images if they are absent."""
    if all((DATA_DIR / name).is_dir() for name in CLASSES):
        print("Dataset folders already exist. Checking images next.")
        return
    DATA_DIR.parent.mkdir(parents=True, exist_ok=True)
    archive = DATA_DIR.parent / "dataset-resized.zip"
    if not archive.exists():
        print("Downloading TrashNet...")
        temporary = archive.with_suffix(".download")
        urllib.request.urlretrieve(DATA_URL, temporary)
        temporary.replace(archive)
    # Extract only class images, ignore macOS metadata in the archive.
    with zipfile.ZipFile(archive) as zipped:
        for member in zipped.infolist():
            parts = Path(member.filename).parts
            if (len(parts) == 3 and parts[0] == "dataset-resized"
                    and parts[1] in CLASSES
                    and Path(parts[2]).suffix.lower() in [".jpg", ".jpeg", ".png"]):
                destination = DATA_DIR / parts[1] / parts[2]
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(zipped.read(member))


def create_splits(verbose=True):
    """Check images, remove exact pixel duplicates, and split each class 70/15/15."""
    rng = random.Random(SEED)
    split = {"seed": SEED, "classes": CLASSES, "train": [], "val": [], "test": []}
    groups = {}
    duplicates = []
    counts = {}
    # Inspect all classes before splitting so conflicting labels can be excluded.
    for label, name in enumerate(CLASSES):
        for path in sorted((DATA_DIR / name).glob("*")):
            if path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue
            with Image.open(path) as image:
                image = image.convert("RGB")
                digest = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
            relative_path = path.relative_to(DATA_DIR).as_posix()
            groups.setdefault(digest, []).append({"path": relative_path, "label": label})
    clean_samples = []
    conflicts = []
    for group in groups.values():
        if len({sample["label"] for sample in group}) > 1:
            conflicts.append(group)
            continue
        clean_samples.append(group[0])
        for sample in group[1:]:
            duplicates.append({"kept": group[0]["path"], "skipped": sample["path"]})
    for label, name in enumerate(CLASSES):
        files = [sample for sample in clean_samples if sample["label"] == label]
        if len(files) < 10:
            raise ValueError(f"Need at least 10 valid, unique images in {DATA_DIR / name}")
        rng.shuffle(files)
        train_end = int(0.70 * len(files))
        val_end = train_end + int(0.15 * len(files))
        portions = {"train": files[:train_end], "val": files[train_end:val_end], "test": files[val_end:]}
        counts[name] = {key: len(items) for key, items in portions.items()}
        for key, items in portions.items():
            split[key].extend(items)
    summary = {"source": DATA_URL, "counts": counts, "pixel_duplicates_removed": duplicates,
               "conflicting_pixel_groups_excluded": conflicts}
    if verbose:
        print(json.dumps(summary, indent=2))
        print("Data splits are ready in memory. No JSON or log files were saved.")
    return split


if __name__ == "__main__":
    download_dataset()
    create_splits()
