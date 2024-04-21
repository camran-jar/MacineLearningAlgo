import pandas as pd
import numpy as np
import sys

##KNN algorithm to classify wine data
# https://www.geeksforgeeks.org/k-nearest-neighbors-with-python-ml/
# KNN algorithim inpsiration from geeks for geeks 

#normalise each number in colum/row in the dataset
def normalise_data(train_data, test_data):
    #exclude final column
    features = train_data.columns[:-1]

    #calculate min and max for each column
    min_vals = train_data[features].min()
    max_vals = train_data[features].max()

    #apply normalisation to train and test data
    train_data[features] = (train_data[features] - min_vals) / (max_vals - min_vals)
    test_data[features] = (test_data[features] - min_vals) / (max_vals - min_vals)

    return train_data, test_data

#eudlidean distance
def euclidean_distance(i1, i2):
    return np.sqrt(np.sum((i1 - i2) ** 2))

#KNN algorithm
def k_nearest_neighbours(train_data, test_data, k):
    predictions = []

    for test_instance in test_data.values:
        distances = []
        for train_instance in train_data.values:
            dist = euclidean_distance(test_instance[:-1], train_instance[:-1])
            distances.append((train_instance[-1], dist))

        distances.sort(key=lambda x: x[1])
        neighbours = distances[:k]

        #predict class
        prediction = max(set([neighbour[0] for neighbour in neighbours]),
                         key=[neighbour[0] for neighbour in neighbours].count)
        predictions.append(prediction)

    return predictions


# main function to read data, normalise, run KNN algorithm and write predictions to file
# main function takes 4 command line arguments: train_csv, test_csv, output, k
def main(train_csv, test_csv, output, k):
    #read data
    train_data = pd.read_csv(train_csv)
    test_data = pd.read_csv(test_csv)

    #normalise data
    train_data, test_data = normalise_data(train_data, test_data)

    train_predictions = k_nearest_neighbours(train_data, train_data, k)

    # Calculate and print train accuracy
    train_labels = train_data.iloc[:, -1].tolist()
    train_accuracy = round(sum(1 for true, pred in zip(train_labels, train_predictions) if true == pred) / len(train_labels) * 100, 2)
    print("Train Accuracy (k =", k, "):", train_accuracy, "%")

    # Test KNN algorithm on test data
    test_predictions = k_nearest_neighbours(train_data, test_data, k)

    # Calculate and print test accuracy
    test_labels = test_data.iloc[:, -1].tolist()
    test_accuracy = round(sum(1 for true, pred in zip(test_labels, test_predictions) if true == pred) / len(test_labels) * 100, 2)
    print("Test Accuracy (k =", k, "):", test_accuracy, "%")

    # write predictions to file, taking the number of k nearest neighbours into account
    # outputting the distances between the test instance and the k nearest neighbours
    with open(output, 'w') as f:
        f.write("y,predicted")
        for i in range(k):
            f.write(f",distance={i+1}")
        f.write("\n")
        for true, pred, test_instance in zip(test_labels, test_predictions, test_data.values):
            f.write(f"{true},{pred}")
            distances = []
            for train_instance in train_data.values:
                dist = euclidean_distance(test_instance[:-1], train_instance[:-1])
                distances.append(dist)

            distances.sort()
            for i in range(k):
                f.write(f",{distances[i]}")
            f.write("\n")


if __name__ == '__main__':
    if len(sys.argv) != 5:
        print("Usage: python3 knn.py wine_train.csv wine_test.csv output k")
        sys.exit(1)

    train_file, test_file, output, k = sys.argv[1:]
    main(train_file, test_file, output, int(k))
