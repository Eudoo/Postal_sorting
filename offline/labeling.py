import cv2
import numpy as np
import os
from utils import *

# Path to training images
IMAGES_DIR = os.path.join("..", "images", "digits")
OUTPUT_FILE = "knn_data.npz"
EXPECTED_DIGITS_PER_IMAGE = 5

# Initialize data lists
X_data = []
y_data = []

# Process each digit image (0.png to 9.png)
for digit in range(10):
    img_path = os.path.join(IMAGES_DIR, f"{digit}.png")
    img = cv2.imread(img_path)

    if img is None:
        print(f"[ERROR] Cannot load {img_path}")
        continue

    # Pretreatment
    binarized = binarize_img(img)

    # Segmentation
    contours = detect_contours(binarized)
    rectangles = filter_contours(contours)

    # Safety check
    if len(rectangles) != EXPECTED_DIGITS_PER_IMAGE:
        print(f"[WARNING] {digit}.png : {len(rectangles)} contours found (expected {EXPECTED_DIGITS_PER_IMAGE})")

    # Feature extraction
    for rect in rectangles:
        digit_img = extract_characters(binarized, rect)
        features = create_feature_vector(digit_img)

        X_data.append(features)
        y_data.append(digit)

# Convert to numpy arrays
X = np.array(X_data)
y = np.array(y_data)

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

# Save training data
save_data(X, y, OUTPUT_FILE)
print(f"Training data saved to {OUTPUT_FILE}")
