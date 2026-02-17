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

    # Normalize each surface by total image area
    return (
        np.sum(central) / img_area,
        np.sum(north) / img_area,
        np.sum(south) / img_area,
        np.sum(east) / img_area,
        np.sum(west) / img_area
    )

def create_feature_vector(digit_img):
    central, north, south, east, west = calculate_cavities(digit_img)
    return [central, north, south, east, west]


# 4- Data Management Functions

def save_data(features_list, labels_list, filename):
    np.savez(filename, features_list=features_list, labels_list=labels_list)

def load_data(filename):
    data = np.load(filename)
    features_list = data['features_list']
    labels_list = data['labels_list']
    return features_list, labels_list
