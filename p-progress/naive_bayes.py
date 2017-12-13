from pubchempy import *
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
import sys
import numpy as np
training_data = []
testing_data = []
testing_y = []

def read_train_file(filename):
    input_file = open(filename, 'r')
    for line in input_file:
        split_line = line.split('\t')
        value = int(split_line[1])
        fingerprint = split_line[0]
        fingerprint = [int(c) for c in fingerprint]
        fingerprint.append(value)
        training_data.append(fingerprint)
    input_file.close()
    return training_data

def read_test_file(filename):
    input_file = open(filename, 'r')
    for line in input_file:
        split_line = line.split('\t')
        value = int(split_line[1])
        fingerprint = split_line[0]
        fingerprint = [int(c) for c in fingerprint]
        testing_data.append(fingerprint)
        testing_y.append(value)
    input_file.close()
    return testing_data


def train(training_data, nb):
    train_x = []
    train_y = []
    for train_instance in training_data:
        train_y.append(train_instance.pop())
        train_x.append(train_instance)
    nb.fit(train_x, train_y)


def main():
    filename = sys.argv[1]
    test_filename = sys.argv[2]
    training = read_train_file(filename)
    testing = read_test_file(test_filename)
    nb = GaussianNB()
    train(training_data, nb)
    predictions = nb.predict(testing_data)
    print predictions
    true = np.array(testing_y)
    print confusion_matrix(true, predictions)

if __name__ == "__main__":
    main()