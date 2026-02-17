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

def calculate_cavities_number(digit_binary_img):
    contours, hierarchy = cv2.findContours(digit_binary_img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    if hierarchy is None:
        return 0
    
    cavities = 0
    for h in hierarchy[0]:
        if h[3] != -1:
            cavities += 1

    return cavities  # 0, 1 or 2

def calculate_cavities_area(digit_binary_img):
    contours, hierarchy = cv2.findContours(digit_binary_img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    if hierarchy is None:
        return 0
    
    cavities_area = 0
    for i, h in enumerate(hierarchy[0]):
        if h[3] != -1:  # If it's a cavity
            cavities_area += cv2.contourArea(contours[i])

    img_area = digit_binary_img.shape[0] * digit_binary_img.shape[1]
    ratio = cavities_area / img_area
    return ratio

def calculate_directional_concavities(digit_binary_img):
    rows, cols = digit_binary_img.shape
    img_area = rows * cols

    # West: for each row, distance from left edge to first white pixel
    west = 0
    for r in range(rows):
        for c in range(cols):
            if digit_binary_img[r, c] == 255:
                west += c
                break

    # East: for each row, distance from right edge to first white pixel
    east = 0
    for r in range(rows):
        for c in range(cols - 1, -1, -1):
            if digit_binary_img[r, c] == 255:
                east += (cols - 1 - c)
                break

    # North: for each column, distance from top edge to first white pixel
    north = 0
    for c in range(cols):
        for r in range(rows):
            if digit_binary_img[r, c] == 255:
                north += r
                break

    # South: for each column, distance from bottom edge to first white pixel
    south = 0
    for c in range(cols):
        for r in range(rows - 1, -1, -1):
            if digit_binary_img[r, c] == 255:
                south += (rows - 1 - r)
                break

    # Normalize by image area
    return north / img_area, south / img_area, east / img_area, west / img_area

def create_feature_vector(digit_img):
    nb = calculate_cavities_number(digit_img)
    area = calculate_cavities_area(digit_img)
    north, south, east, west = calculate_directional_concavities(digit_img)
    return [nb, area, north, south, east, west]


# 4- Data Management Functions

def save_data(features_list, labels_list, filename):
    np.savez(filename, features_list=features_list, labels_list=labels_list)

def load_data(filename):
    data = np.load(filename)
    features_list = data['features_list']
    labels_list = data['labels_list']
    return features_list, labels_list
