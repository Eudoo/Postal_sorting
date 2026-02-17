import numpy as np

# 1- Accuracy per digit
def digit_accuracy(true_digits, predicted_digits):
    correct = sum(1 for t, p in zip(true_digits, predicted_digits) if t == p)
    total = len(true_digits)
    accuracy = correct / total if total > 0 else 0
    return correct, total, accuracy

# 2- Accuracy per postal code
def postal_code_accuracy(true_codes, predicted_codes):
    correct = sum(1 for t, p in zip(true_codes, predicted_codes) if t == p)
    total = len(true_codes)
    accuracy = correct / total if total > 0 else 0
    return correct, total, accuracy

# 3- Confusion matrix (10x10)
def confusion_matrix(true_digits, predicted_digits, num_classes=10):
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(true_digits, predicted_digits):
        matrix[int(t)][int(p)] += 1
    return matrix

# 4- Precision per class
def precision_per_class(conf_matrix):
    num_classes = conf_matrix.shape[0]
    precisions = []
    for c in range(num_classes):
        tp = conf_matrix[c][c]
        fp = np.sum(conf_matrix[:, c]) - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        precisions.append(precision)
    return precisions

# 5- Recall per class
def recall_per_class(conf_matrix):
    num_classes = conf_matrix.shape[0]
    recalls = []
    for c in range(num_classes):
        tp = conf_matrix[c][c]
        fn = np.sum(conf_matrix[c, :]) - tp
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        recalls.append(recall)
    return recalls

# 6- Display all metrics
def display_metrics(true_codes, predicted_codes):
    # Extract individual digits
    true_digits = [int(d) for code in true_codes for d in str(code)]
    predicted_digits = [int(d) for code in predicted_codes for d in str(code)]

    # Digit accuracy
    d_correct, d_total, d_acc = digit_accuracy(true_digits, predicted_digits)
    print(f"\n{'='*50}")
    print(f"METRICS")
    print(f"{'='*50}")
    print(f"Digit accuracy:       {d_correct}/{d_total} = {d_acc*100:.1f}%")

    # Postal code accuracy
    pc_correct, pc_total, pc_acc = postal_code_accuracy(true_codes, predicted_codes)
    print(f"Postal code accuracy: {pc_correct}/{pc_total} = {pc_acc*100:.1f}%")

    # Confusion matrix
    conf = confusion_matrix(true_digits, predicted_digits)
    print(f"\nConfusion Matrix:")
    print(f"       {'  '.join([str(i) for i in range(10)])}")
    print(f"      {'---'*10}")
    for i in range(10):
        row = '  '.join([f"{conf[i][j]:2d}" for j in range(10)])
        print(f"  {i} |  {row}")

    # Precision per class
    precisions = precision_per_class(conf)
    print(f"\nPrecision per class:")
    for i, p in enumerate(precisions):
        print(f"  Digit {i}: {p*100:.1f}%")

    # Recall per class
    recalls = recall_per_class(conf)
    print(f"\nRecall per class:")
    for i, r in enumerate(recalls):
        print(f"  Digit {i}: {r*100:.1f}%")

    print(f"{'='*50}")
