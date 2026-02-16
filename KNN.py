"""
K-Nearest Neighbors (KNN) Classification
---------------------------------------
- Supports multiple attributes
- User-defined attribute names
- Categorical output (class labels)
- Pure Python implementation
"""

import math
from collections import Counter

def euclidean_distance(p1, p2):
    """Calculate Euclidean distance between two points"""
    return math.sqrt(sum((p1[i] - p2[i]) ** 2 for i in range(len(p1))))

def knn_classification(data, labels, test_point, k):
    """Predict class using KNN Classification"""
    distances = []

    for i in range(len(data)):
        dist = euclidean_distance(data[i], test_point)
        distances.append((dist, labels[i]))

    # Sort by distance
    distances.sort(key=lambda x: x[0])

    # Get k nearest labels
    k_nearest_labels = [label for _, label in distances[:k]]

    # Majority voting
    predicted_label = Counter(k_nearest_labels).most_common(1)[0][0]

    return predicted_label


def main():
    # Input attribute and output names
    attributes = input("Enter attribute names (space separated): ").split()
    target = input("Enter output column name: ")

    num_records = int(input("Enter number of records: "))

    data = []
    labels = []

    print("\nEnter dataset values:")
    for _ in range(num_records):
        values = list(map(float, input(f"Enter values for {attributes}: ").split()))
        label = input(f"Enter {target}: ")

        data.append(values)
        labels.append(label)

    k = int(input("\nEnter value of k: "))

    test_point = list(
        map(float, input(f"Enter values for {attributes} to predict {target}: ").split())
    )

    result = knn_classification(data, labels, test_point, k)

    print(f"\nPredicted {target}: {result}")


if __name__ == "__main__":
    main()
