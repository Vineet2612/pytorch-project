"""Turn image filenames into tensors, then group them into batches."""

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from config import BATCH_SIZE, DATA_DIR, IMAGE_SIZE, SEED


def get_transform(training=False, image_size=IMAGE_SIZE):
    """To resize and normalize images; randomly flip training images only."""
    steps = [transforms.Resize((image_size, image_size))]
    if training:
        steps.append(transforms.RandomHorizontalFlip())
    steps.extend([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    return transforms.Compose(steps)


class WasteDataset(Dataset):
    """A dataset returns one RGB image tensor and its integer class label."""

    def __init__(self, samples, training=False, image_size=IMAGE_SIZE):
        self.samples = samples
        self.transform = get_transform(training, image_size)
        self.image_size = image_size
        self.image_cache = {}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        if index not in self.image_cache:
            with Image.open(DATA_DIR / sample["path"]) as image:
                self.image_cache[index] = image.convert("RGB").resize(
                    (self.image_size, self.image_size), Image.Resampling.BILINEAR)
        # Cache before augmentation so a new random flip is possible each epoch.
        image = self.transform(self.image_cache[index])
        return image, sample["label"]


def make_loader(split_name, split, image_size=IMAGE_SIZE):
    """Build batches from an in-memory split dictionary and return class names."""
    training = split_name == "train"
    dataset = WasteDataset(split[split_name], training, image_size)
    generator = torch.Generator().manual_seed(SEED)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=training,
                        num_workers=0, generator=generator)
    return loader, split["classes"]
