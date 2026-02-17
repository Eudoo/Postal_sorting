import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add offline/ to path so we can import its utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'offline'))

from utils import *
from knn_utils import predict_knn
from metrics import display_metrics

# ============== CONFIGURATION ==============
MODEL_PATH = os.path.join("..", "offline", "knn_data.npz")
IMAGES_DIR = os.path.join("..", "images", "postal_code")
K = 3

# ============== 1. Load training data ==============
X_train, y_train = load_data(MODEL_PATH)
print(f"Training data loaded: X={X_train.shape}, y={y_train.shape}")

# ============== 2. Load test image ==============
test_images = sorted(os.listdir(IMAGES_DIR))
print(f"Test images found: {test_images}")

# Collect predictions for metrics
all_true_codes = []
all_predicted_codes = []

for img_name in test_images:
    img_path = os.path.join(IMAGES_DIR, img_name)
    img = cv2.imread(img_path)

    if img is None:
        print(f"[ERROR] Cannot load {img_path}")
        continue

    # ============== 3. Pretreatment & Segmentation ==============
    binarized = binarize_img(img)
    contours = detect_contours(binarized)
    rectangles = filter_contours(contours)

    print(f"\n--- {img_name} ---")
    print(f"Digits found: {len(rectangles)}")

    # ============== 4. Prediction Loop ==============
    postal_code = []
    img_display = img.copy()

    for rect in rectangles:
        # Extract digit
        digit_img = extract_characters(binarized, rect)

        # Create feature vector
        features = create_feature_vector(digit_img)

        # Predict with KNN
        prediction = predict_knn(features, X_train, y_train, k=K)
        postal_code.append(str(int(prediction)))

        # Annotate image
        x, y, w, h = rect
        cv2.rectangle(img_display, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img_display, str(int(prediction)), (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # ============== 5. Display ==============
    result = "".join(postal_code)
    print(f"Predicted postal code: {result}")

    # Extract true code from filename (e.g. "59130 (1).png" → "59130")
    true_code = img_name.split(" ")[0]
    all_true_codes.append(true_code)
    all_predicted_codes.append(result)

    plt.figure(figsize=(10, 4))
    plt.imshow(cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB))
    plt.title(f"True: {true_code} | Predicted: {result}")
    plt.axis('off')
    plt.show()

# ============== 6. Metrics ==============
display_metrics(all_true_codes, all_predicted_codes)
