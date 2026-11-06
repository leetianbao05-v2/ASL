import sys
import numpy as np
import cv2
import tensorflow as tf
import sklearn

print("Python:", sys.version.split()[0])
print("NumPy:", np.__version__)
print("OpenCV:", cv2.__version__)
print("TensorFlow:", tf.__version__)
print("scikit-learn:", sklearn.__version__)
print("GPUs found:", tf.config.list_physical_devices("GPU"))

cap = cv2.VideoCapture(0)
ok, frame = cap.read()
cap.release()
print("Webcam works:", ok, "| frame shape:", frame.shape if ok else None)