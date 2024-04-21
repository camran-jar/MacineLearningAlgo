import numpy as np
import pandas as pd
import sys
from collections import Counter

# https://www.geeksforgeeks.org/iterative-dichotomiser-3-id3-algorithm-from-scratch/
# https://machinelearningmastery.com/implement-decision-tree-algorithm-scratch-python/ used for some further info 
# Algorithim inspiration from geeks for geeks 

# Node class to reprsent a node in the decision tree
class Node:
    def __init__(self, feature=None, value=None, results=None):
        self.feature = feature
        self.value = value
        self.results = results
        self.information_gain = None
        self.entropy = None

# Class to calculate entropy 
def entropy(data):
    count = np.bincount(data)
    probabilities = count / len(data)
    entropy = -np.sum([pi * np.log(pi) if pi > 0 else 0 for pi in probabilities])
    return max(entropy, 0)

# Class to split data based on feature and value 
def split_data(x, y, feature, value):
    true_idx = np.where(x[:, feature] <= value)[0]
    false_idx = np.where(x[:, feature] > value)[0]
    return x[true_idx], y[true_idx], x[false_idx], y[false_idx]

# Class to build tree, using recursion, calculates information gain and entropy
def build_tree(x, y, min_information_gain=0.00001):
    if len(set(y)) == 1:
        return Node(results=Counter(y).most_common(1)[0][0])

    best_information_gain = 0
    best_sets = None
    current_entropy = entropy(y)

    for feature in range(x.shape[1]):
        for value in set(x[:, feature]):
            x_true, y_true, x_false, y_false = split_data(x, y, feature, value)
            if len(y_true) == 0 or len(y_false) == 0:
                continue
            true_entropy = entropy(y_true)
            false_entropy = entropy(y_false)
            p_true = len(y_true) / len(y)
            p_false = len(y_false) / len(y)
            information_gain = current_entropy - (p_true * true_entropy + p_false * false_entropy)
            if information_gain > best_information_gain:
                best_information_gain = information_gain
                best_sets = (x_true, y_true, x_false, y_false)
                best_feature = feature
                best_value = value

    if best_information_gain > min_information_gain:
        true_branch = build_tree(best_sets[0], best_sets[1], min_information_gain)
        false_branch = build_tree(best_sets[2], best_sets[3], min_information_gain)
        node = Node(feature=best_feature, value=best_value)
        node.information_gain = best_information_gain
        node.entropy = current_entropy
        node.true_branch = true_branch
        node.false_branch = false_branch
        return node
    else:
        return Node(results=Counter(y).most_common(1)[0][0])

# class to predict the class of a sample
def predict(node, sample):
    if node.results is not None:
        return node.results
    else:
        branch = node.false_branch
        if sample[node.feature] <= node.value:
            branch = node.true_branch
        return predict(branch, sample)

def print_tree(node, depth=0, file=None, feature_names=None):
    if file is None:
        file = sys.stdout

    if isinstance(node.results, np.int64):
        class_count = {int(node.results): np.sum(node.results)}
        class_counts_str = ", ".join(["{}: {}".format(label, count) for label, count in class_count.items()])
        print('{}leaf {{{}}} - Count: {}'.format(depth * '  ', class_counts_str, np.sum(node.results)), file=file)
    elif node.results is not None:
        class_counts_str = ", ".join(["{}: {}".format(label, count) for label, count in node.results.items()])
        print('{}leaf {{{}}} - Count: {}'.format(depth * '  ', class_counts_str, sum(node.results.values())), file=file)
    else:
        feature_name = feature_names[node.feature] if feature_names is not None else "Feature {}".format(node.feature)
        print('{}{} (IG: {:.4f}, Entropy: {:.4f})'.format(depth * '  ', feature_name, node.information_gain, node.entropy), file=file)

        print('{}-- {} == {} --'.format((depth + 1) * '  ', feature_name, 0), file=file)
        print_tree(node.true_branch, depth + 1, file=file, feature_names=feature_names)

        print('{}-- {} == {} --'.format((depth + 1) * '  ', feature_name, 1), file=file)
        print_tree(node.false_branch, depth + 1, file=file, feature_names=feature_names)

# Load data from CSV file 
def load_data(train_data):
    data = pd.read_csv(train_data)
    x = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    return x, y

# Calculate accuracy of the model
def calculate_accuracy(tree, x, y):
    correct_predictions = 0
    total_samples = len(y)

    for i in range(total_samples):
        predicted_label = predict(tree, x[i])
        if predicted_label == y[i]:
            correct_predictions += 1

    accuracy = (correct_predictions / total_samples) * 100
    return accuracy

# Main function to read data, build decision tree and write the tree to a file
# Pritns accuracy to terminal 
def main(train_data_file, output_file):
    # Load data
    x, y = load_data(train_data_file)

    # Get feature names
    feature_names = list(range(x.shape[1]))

    # Build decision tree
    decision_tree = build_tree(x, y)

    # Print decision tree to file
    with open(output_file, 'w') as f:
        print_tree(decision_tree, file=f, feature_names=feature_names)

    # Calculate and print accuracy
    accuracy = calculate_accuracy(decision_tree, x, y)
    print("Accuracy on training data: {:.2f} %".format(accuracy))

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python DT.py <train_data.csv> <output>")
        sys.exit(1)

    main(sys.argv[1], sys.argv[2])
