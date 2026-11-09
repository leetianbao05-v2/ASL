import csv
import json
import os

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Where the Kaggle images live (same path as before)
DATASET_PATH = r"C:\Users\USER\.cache\kagglehub\datasets\grassknoted\asl-alphabet\versions\1"
TRAIN_DIR = os.path.join(DATASET_PATH, "asl_alphabet_train", "asl_alphabet_train")

IMG_SIZE = 64      # every image is resized to 64 x 64 pixels
BATCH_SIZE = 64    # the model learns from 64 images at a time


def get_class_names():
    """Class names in a fixed order: A, B, ..., Z, del, nothing, space."""
    return sorted(
        d for d in os.listdir(TRAIN_DIR)
        if os.path.isdir(os.path.join(TRAIN_DIR, d))
    )


class ASLDataset(Dataset):
    """Reads splits/<name>.csv and loads one image at a time when asked."""

    def __init__(self, split_name, class_names):
        self.class_to_index = {c: i for i, c in enumerate(class_names)}
        self.items = []   # list of (image_path, integer_label)
        with open(os.path.join("splits", f"{split_name}.csv")) as f:
            for row in csv.DictReader(f):
                path = os.path.join(TRAIN_DIR, row["class"], row["filename"])
                self.items.append((path, self.class_to_index[row["class"]]))

        # Resize, then convert to a tensor with values scaled from 0-255 to 0-1
        self.transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        path, label = self.items[index]
        image = Image.open(path).convert("RGB")
        return self.transform(image), label


def make_loader(split_name, class_names, shuffle=False):
    """Build a DataLoader that hands the model batches of images."""
    dataset = ASLDataset(split_name, class_names)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=0)


if __name__ == "__main__":
    # Quick self-test: run this file directly to check the pipeline
    import matplotlib.pyplot as plt

    class_names = get_class_names()
    print("Number of classes:", len(class_names))

    # Save the class order so the webcam app can use exactly the same one later
    with open(os.path.join("splits", "labels.json"), "w") as f:
        json.dump(class_names, f)

    train_loader = make_loader("train", class_names, shuffle=True)
    images, labels = next(iter(train_loader))
    print("Batch of images:", tuple(images.shape))
    print("Batch of labels:", tuple(labels.shape))
    print("Pixel range:", float(images.min()), "to", float(images.max()))

    fig, axes = plt.subplots(2, 4, figsize=(10, 5))
    for ax, img, lab in zip(axes.flatten(), images, labels):
        ax.imshow(img.permute(1, 2, 0))   # PyTorch stores channels first; matplotlib wants channels last
        ax.set_title(class_names[lab])
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("docs/pipeline_check.png")
    plt.show()