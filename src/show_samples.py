import os
import random
import cv2
import matplotlib.pyplot as plt

dataset_path = r"C:\Users\USER\.cache\kagglehub\datasets\grassknoted\asl-alphabet\versions\1"
train_dir = os.path.join(dataset_path, "asl_alphabet_train", "asl_alphabet_train")

classes = sorted(os.listdir(train_dir))

# Make a grid with 5 rows and 6 columns (30 slots for 29 classes)
fig, axes = plt.subplots(5, 6, figsize=(14, 12))

for ax, cls in zip(axes.flatten(), classes):
    folder = os.path.join(train_dir, cls)
    filename = random.choice(os.listdir(folder))      # pick a random photo
    image = cv2.imread(os.path.join(folder, filename))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)    # OpenCV uses BGR, matplotlib uses RGB
    ax.imshow(image)
    ax.set_title(cls)
    ax.axis("off")

# Hide the one unused slot
for ax in axes.flatten()[len(classes):]:
    ax.axis("off")

print("Image shape (height, width, colour channels):", image.shape)

plt.tight_layout()
plt.savefig("docs/samples.png")
plt.show()