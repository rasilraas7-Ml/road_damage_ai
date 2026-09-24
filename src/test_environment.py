import sys

import cv2
import flask
import matplotlib
import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf
import torch
from PIL import Image


print("=" * 60)
print("ROAD DAMAGE AI - ENVIRONMENT CHECK")
print("=" * 60)

print(f"Python       : {sys.version.split()[0]}")
print(f"NumPy        : {np.__version__}")
print(f"Pandas       : {pd.__version__}")
print(f"OpenCV       : {cv2.__version__}")
print(f"PyTorch      : {torch.__version__}")
print(f"TensorFlow   : {tf.__version__}")
print(f"Scikit-learn : {sklearn.__version__}")
print(f"Flask        : {flask.__version__}")
print(f"Matplotlib   : {matplotlib.__version__}")
print(f"Pillow       : {Image.__version__}")

print("-" * 60)

print(f"PyTorch CUDA : {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU          : {torch.cuda.get_device_name(0)}")
else:
    print("GPU          : Not available - using CPU")

print("-" * 60)

print("All core libraries imported successfully.")
print("Environment is ready.")

print("=" * 60)