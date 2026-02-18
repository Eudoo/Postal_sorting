import cv2
import numpy as np
import os

# Global Constants
NOISE_THRESHOLD = 20
LINE_Y_TOLERANCE = 20
DIGIT_SIZE = (64, 64)

# 1- Pretreatment (Cleaning) Functions

def binarize_img(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_img_thr = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    return gray_img_thr


# 2- Segmentation (Slicing) functions

def detect_contours(binary_img):
    contours, hierarchy = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def filter_contours(contours_list):
    rectangles = []
    for contour in contours_list:
        x, y, w, h = cv2.boundingRect(contour)
        if w * h > NOISE_THRESHOLD:
            rectangles.append((x, y, w, h))
    
    rectangles.sort(key=lambda rect: rect[1]) # Sort by y
    
    lines = []
    current_line = []
    last_y = None
    
    for x, y, w, h in rectangles:
        if last_y is None or abs(y - last_y) < LINE_Y_TOLERANCE:  # Same line
            current_line.append((x, y, w, h))
        else:  # New line
            lines.append(current_line)
            current_line = [(x, y, w, h)]
        last_y = y
    
    if current_line:
        lines.append(current_line)
    
    for line in lines:
        line.sort(key=lambda rect: rect[0]) # sort by x
    
    result = [rect for line in lines for rect in line]
    return result

def extract_characters(img, rectangle):
    roi = img[rectangle[1]:rectangle[1]+rectangle[3], rectangle[0]:rectangle[0]+rectangle[2]]
    roi = cv2.resize(roi, DIGIT_SIZE)
    _, roi = cv2.threshold(roi, 127, 255, cv2.THRESH_BINARY)
    return roi


# 3- Feature Extraction Functions

def calculate_cavities(digit_binary_img):
    rows, cols = digit_binary_img.shape
    img_area = rows * cols

    # Mask of digit pixels (white = 255) and background pixels
    digit_mask = digit_binary_img == 255
    bg_mask = ~digit_mask

    # --- Build 4 wall visibility masks ---

    # wall_north[r,c] = True if there is a digit pixel ABOVE row r in column c
    wall_north = np.zeros_like(digit_mask)
    for r in range(1, rows):
        wall_north[r, :] = wall_north[r - 1, :] | digit_mask[r - 1, :]

    # wall_south[r,c] = True if there is a digit pixel BELOW row r in column c
    wall_south = np.zeros_like(digit_mask)
    for r in range(rows - 2, -1, -1):
        wall_south[r, :] = wall_south[r + 1, :] | digit_mask[r + 1, :]

    # wall_west[r,c] = True if there is a digit pixel LEFT of column c in row r
    wall_west = np.zeros_like(digit_mask)
    for c in range(1, cols):
        wall_west[:, c] = wall_west[:, c - 1] | digit_mask[:, c - 1]

    # wall_east[r,c] = True if there is a digit pixel RIGHT of column c in row r
    wall_east = np.zeros_like(digit_mask)
    for c in range(cols - 2, -1, -1):
        wall_east[:, c] = wall_east[:, c + 1] | digit_mask[:, c + 1]

    # --- Combine masks with logical operations ---

    # Central cavity: background pixel blocked in ALL 4 directions (closed hole)
    central = bg_mask & wall_north & wall_south & wall_east & wall_west

    # North cavity: open to North, blocked S + E + W
    north = bg_mask & (~wall_north) & wall_south & wall_east & wall_west

    # South cavity: open to South, blocked N + E + W
    south = bg_mask & wall_north & (~wall_south) & wall_east & wall_west

    # East cavity: open to East, blocked N + S + W
    east = bg_mask & wall_north & wall_south & (~wall_east) & wall_west

    # West cavity: open to West, blocked N + S + E
    west = bg_mask & wall_north & wall_south & wall_east & (~wall_west)

    # --- Central: surface + connectivity (nb connected blocks) ---
    central_surface = np.sum(central) / img_area
    if np.sum(central) > 0:
        num_labels, _ = cv2.connectedComponents(central.astype(np.uint8) * 255)
        central_nb_blocks = (num_labels - 1) / 2  # 0->0, 1->0.5, 2->1.0
    else:
        central_nb_blocks = 0.0

    # --- Helper: surface + barycenter Y for a directional cavity ---
    def cavity_features(mask):
        surface = np.sum(mask) / img_area
        if np.sum(mask) > 0:
            ys = np.where(mask)[0]  # row indices of True pixels
            barycenter_y = np.mean(ys) / rows  # normalize by height [0, 1]
        else:
            barycenter_y = 0.5  # neutral default (center)
        return surface, barycenter_y

    north_surface, north_bary = cavity_features(north)
    south_surface, south_bary = cavity_features(south)
    east_surface, east_bary = cavity_features(east)
    west_surface, west_bary = cavity_features(west)

    return (
        central_surface, central_nb_blocks,
        north_surface, north_bary,
        south_surface, south_bary,
        east_surface, east_bary,
        west_surface, west_bary
    )

def calculate_solidity(digit_binary_img):
    """Solidity = contour area / convex hull area. Invariant to scale."""
    contours, _ = cv2.findContours(digit_binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    # Take largest contour
    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    if hull_area == 0:
        return 0.0
    return area / hull_area

def create_feature_vector(digit_img):
    cavities = list(calculate_cavities(digit_img))
    solidity = calculate_solidity(digit_img)
    return cavities + [solidity]


# 4- Normalization Functions

def normalize_features(X):
    """Min-Max normalization. Returns normalized X, min_vals, max_vals."""
    min_vals = np.min(X, axis=0)
    max_vals = np.max(X, axis=0)
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1  # avoid division by zero
    X_norm = (X - min_vals) / range_vals
    return X_norm, min_vals, max_vals

def apply_normalization(features, min_vals, max_vals):
    """Apply pre-computed Min-Max normalization to a single feature vector."""
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1
    return (np.array(features) - min_vals) / range_vals

def calculate_centroids(X, y, num_classes=10):
    """Calculate mean feature vector per class (centroid)."""
    centroids = np.zeros((num_classes, X.shape[1]))
    for c in range(num_classes):
        centroids[c] = np.mean(X[y == c], axis=0)
    return centroids


# 5- Data Management Functions

def save_data(features_list, labels_list, filename, min_vals=None, max_vals=None, centroids=None):
    np.savez(filename,
             features_list=features_list,
             labels_list=labels_list,
             min_vals=min_vals,
             max_vals=max_vals,
             centroids=centroids)

def load_data(filename):
    data = np.load(filename)
    features_list = data['features_list']
    labels_list = data['labels_list']
    min_vals = data['min_vals']
    max_vals = data['max_vals']
    centroids = data['centroids']
    return features_list, labels_list, min_vals, max_vals, centroids
