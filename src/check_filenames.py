import os

dataset_path = r"C:\Users\USER\.cache\kagglehub\datasets\grassknoted\asl-alphabet\versions\1"
folder = os.path.join(dataset_path, "asl_alphabet_train", "asl_alphabet_train", "A")

names = sorted(os.listdir(folder))
print(names[:10])
print(names[-5:])