import os
import re
import csv

# Path to the folder that contains one sub-folder per class (A, B, ... del, nothing, space)
dataset_path = r"C:\Users\USER\.cache\kagglehub\datasets\grassknoted\asl-alphabet\versions\1"
train_dir = os.path.join(dataset_path, "asl_alphabet_train", "asl_alphabet_train")

# Block boundaries, by photo number within each class (1..3000)
TRAIN_END = 2100   # photos 1-2100    -> training   (70%)
VAL_END = 2550     # photos 2101-2550 -> validation (15%)
                   # photos 2551-3000 -> test       (15%)

def photo_number(filename):
    """'A1000.jpg' -> 1000. Used to sort numerically instead of alphabetically."""
    return int(re.search(r"(\d+)", filename).group(1))

splits = {"train": [], "val": [], "test": []}

for cls in sorted(os.listdir(train_dir)):
    folder = os.path.join(train_dir, cls)
    if not os.path.isdir(folder):
        continue
    for filename in os.listdir(folder):
        n = photo_number(filename)
        if n <= TRAIN_END:
            splits["train"].append((cls, filename))
        elif n <= VAL_END:
            splits["val"].append((cls, filename))
        else:
            splits["test"].append((cls, filename))

# Safety check: no photo may appear in more than one split
all_files = [item for rows in splits.values() for item in rows]
assert len(all_files) == len(set(all_files)), "A photo appears in two splits!"

# Save each split as a small CSV file (class, filename)
os.makedirs("splits", exist_ok=True)
for name, rows in splits.items():
    with open(os.path.join("splits", f"{name}.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "filename"])
        writer.writerows(rows)
    print(name, len(rows))