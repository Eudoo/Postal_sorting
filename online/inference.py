import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import argparse

# Add offline/ to path so we can import its utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'offline'))

from utils import *
from knn_utils import predict_features_knn, predict_centroid_knn
from metrics import display_metrics

# ============== CONFIGURATION ==============
MODEL_PATH = os.path.join("..", "offline", "knn_data.npz")
IMAGES_DIR = os.path.join("..", "images", "postal_code")
K = 3


def inference(mode="features"):
    """
    Run inference on postal code images.
    mode: 'features' (k-NN), 'centroid' (nearest mean), or 'both' (compare)
    """

    # ============== 1. Load training data ==============
    X_train, y_train, min_vals, max_vals, centroids = load_data(MODEL_PATH)
    print(f"Training data loaded: X={X_train.shape}, y={y_train.shape}, centroids={centroids.shape}")
    print(f"Mode: {mode}\n")

    # ============== 2. Load test images ==============
    test_images = sorted(os.listdir(IMAGES_DIR))
    print(f"Test images found: {test_images}")

    # Determine which methods to run
    methods = []
    if mode == "features":
        methods = ["features"]
    elif mode == "centroid":
        methods = ["centroid"]
    elif mode == "both":
        methods = ["features", "centroid"]

    # Collect predictions for metrics (per method)
    results = {m: {"true": [], "predicted": []} for m in methods}

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

        # Q9: Split touching digits
        rectangles = split_touching_digits(binarized, rectangles)

        true_code = img_name.split(" ")[0]
        print(f"\n--- {img_name} ---")
        print(f"Digits found: {len(rectangles)}")

        # ============== 4. Prediction Loop ==============
        predictions = {m: [] for m in methods}
        img_display = img.copy()

        for rect in rectangles:
            digit_img = extract_characters(binarized, rect)
            features = create_feature_vector(digit_img)
            features_norm = apply_normalization(features, min_vals, max_vals)

            # Predict with each method
            if "features" in methods:
                pred_f = predict_features_knn(features_norm, X_train, y_train, k=K)
                predictions["features"].append(str(int(pred_f)))

            if "centroid" in methods:
                pred_c = predict_centroid_knn(features_norm, centroids)
                predictions["centroid"].append(str(int(pred_c)))

            # Annotate image (use first method for display)
            display_pred = predictions[methods[1]][-1]
            x, y, w, h = rect
            cv2.rectangle(img_display, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img_display, display_pred, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # ============== 5. Display ==============
        for m in methods:
            code = "".join(predictions[m])
            results[m]["true"].append(true_code)
            results[m]["predicted"].append(code)
            print(f"  [{m:>8}] Predicted: {code}")

        plt.figure(figsize=(10, 4))
        plt.imshow(cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB))
        title = f"True: {true_code}"
        for m in methods:
            title += f" | {m}: {''.join(predictions[m])}"
        plt.title(title)
        plt.axis('off')
        plt.show()

    # ============== 6. Metrics ==============
    for m in methods:
        print(f"\n{'#'*50}")
        print(f"# METRICS — {m.upper()}")
        print(f"{'#'*50}")
        display_metrics(results[m]["true"], results[m]["predicted"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Postal code recognition")
    parser.add_argument("--mode", choices=["features", "centroid", "both"],
                        default="features", help="Prediction method")
    args = parser.parse_args()
    inference(mode=args.mode)
