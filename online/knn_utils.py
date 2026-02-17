import numpy as np
from collections import Counter

# 1- Distance Calculation

def calculate_distance(vector1, vector2):
    return np.sqrt(np.sum((np.array(vector1) - np.array(vector2)) ** 2))

# 2- KNN Prediction 

def predict_knn(new_features, X_train, y_train, k=3):
    # Calculate distance between new_features and all training vectors
    distances = []
    for i in range(len(X_train)):
        dist = calculate_distance(new_features, X_train[i])
        distances.append((dist, y_train[i]))
    
    # Sort by distance (ascending)
    distances.sort(key=lambda x: x[0])
    
    # Take k nearest neighbors
    k_nearest = distances[:k]
    
    # Vote: most frequent label wins
    k_labels = [label for _, label in k_nearest]
    vote = Counter(k_labels)
    prediction = vote.most_common(1)[0][0]
    
    return prediction
