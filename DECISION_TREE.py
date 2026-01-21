"""
Decision Tree (ID3) – Human-Readable Version
-------------------------------------------
- Works only with categorical data
- Uses Entropy & Information Gain
- Simple variable names
- Clear flow, easy to understand for exams
"""

import math
from collections import Counter, defaultdict


# Calculate entropy of a list of class labels
def entropy(labels):
    total = len(labels)
    counts = Counter(labels)
    ent = 0.0

    for count in counts.values():
        probability = count / total
        ent -= probability * math.log2(probability)

    return ent


# Calculate Information Gain for one attribute
def information_gain(data, labels, attr_index):
    base_entropy = entropy(labels)

    groups = defaultdict(list)
    for row, label in zip(data, labels):
        groups[row[attr_index]].append(label)

    weighted_entropy = 0.0
    total = len(labels)

    for group_labels in groups.values():
        weighted_entropy += (len(group_labels) / total) * entropy(group_labels)

    return base_entropy - weighted_entropy


# Find the attribute with maximum information gain
def best_attribute(data, labels, attributes):
    best_gain = -1
    best_index = -1

    for i in range(len(attributes)):
        gain = information_gain(data, labels, i)
        if gain > best_gain:
            best_gain = gain
            best_index = i

    return best_index


# Recursively build the decision tree
def build_tree(data, labels, attributes):

    # If all outputs are same → leaf node
    if len(set(labels)) == 1:
        return labels[0]

    # If no attributes left → majority vote
    if not attributes:
        return Counter(labels).most_common(1)[0][0]

    best_attr_index = best_attribute(data, labels, attributes)
    best_attr_name = attributes[best_attr_index]

    tree = {best_attr_name: {}}

    values = set(row[best_attr_index] for row in data)

    for val in values:
        new_data = []
        new_labels = []

        for row, label in zip(data, labels):
            if row[best_attr_index] == val:
                new_row = row[:best_attr_index] + row[best_attr_index + 1:]
                new_data.append(new_row)
                new_labels.append(label)

        new_attributes = attributes[:best_attr_index] + attributes[best_attr_index + 1:]
        tree[best_attr_name][val] = build_tree(
            new_data, new_labels, new_attributes
        )

    return tree


# Predict output using the built tree
def predict(tree, attributes, test_point):

    if not isinstance(tree, dict):
        return tree

    attribute = next(iter(tree))
    index = attributes.index(attribute)
    value = test_point[index]

    if value not in tree[attribute]:
        return None

    subtree = tree[attribute][value]

    new_test = test_point[:index] + test_point[index + 1:]
    new_attributes = attributes[:index] + attributes[index + 1:]

    return predict(subtree, new_attributes, new_test)


def main():
    attributes = input("Enter attribute names (space separated): ").split()
    target = input("Enter output column name: ")

    while True:
        try:
            n = int(input("Enter number of records: "))
            break
        except ValueError:
            print("❌ Enter a valid number.")

    data = []
    labels = []

    print("\nEnter dataset:")
    for _ in range(n):
        row = input(f"Values for {attributes}: ").split()
        label = input(f"{target}: ")
        data.append(row)
        labels.append(label)

    test_point = input(
        f"\nEnter values for {attributes} to predict {target}: "
    ).split()

    tree = build_tree(data, labels, attributes)
    result = predict(tree, attributes, test_point)

    print("\nDecision Tree:")
    print(tree)

    print(f"\nPredicted {target}: {result}")


if __name__ == "__main__":
    main()
