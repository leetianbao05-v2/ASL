import kagglehub

# Download the dataset (files are saved in a cache folder on your computer)
path = kagglehub.dataset_download("grassknoted/asl-alphabet")

print("Path to dataset files:", path)