import os

# Raw string (r"...") so Windows backslashes are read correctly
dataset_path = r"C:\Users\USER\.cache\kagglehub\datasets\grassknoted\asl-alphabet\versions\1"

# The training images are usually nested: asl_alphabet_train/asl_alphabet_train
train_dir = os.path.join(dataset_path, "asl_alphabet_train", "asl_alphabet_train")

for cls in sorted(os.listdir(train_dir)):
    folder = os.path.join(train_dir, cls)
    if os.path.isdir(folder):
        print(cls, len(os.listdir(folder)))