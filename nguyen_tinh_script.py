from sklearn import svm
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.naive_bayes import MultinomialNB
from queue import PriorityQueue
import csv
import numpy as np
import pandas as pd
import re

"""
    Rule 14 DS challenge
    Tinh Nguyen

    Basic Information:
    - 42 classes
    - Term frequency performs poorly but does not overfit!

    Model Eval:
    - SVM
        (performs poorly, n << d) ~ 15%
    - Naive Bayes
        (performs better) ~ 50% cross-val
"""


def main():
    X_train, y_train, X_test, vocabulary, y_filenames = load_dataset()
    # preprocessData(X_train, y_train, vocabulary)

    modelSVM(X_train, y_train, X_test)
    modelNB(X_train, y_train, X_test)
    # Train the best
    predictions = modelNB_final(X_train, y_train, X_test)

    X_train, y_train, X_test, vocabulary, _ = load_dataset(form='tf')
    modelSVM(X_train, y_train, X_test)
    modelNB(X_train, y_train, X_test)

    writeCSV(y_filenames, predictions)


def load_dataset(form='c'):
    """
        Loads data set into np.array (term frequency weighting or counts).
        Returns (X_train, y_Train, X_test)

        form : 'tf' or 'c' for term-freq/counts
    """
    data_X = load_files('data', categories='Fixed Judgements')
    data_y = pd.read_csv('data/Interview_Mapping.csv').values

    vectorizer = None
    if form == 'tf':
        vectorizer = TfidfVectorizer(decode_error='ignore')
        print('Term Frequency')
    else:
        vectorizer = CountVectorizer(decode_error='ignore')
        print('Counts')

    v_X = vectorizer.fit_transform(data_X.data)

    # X and y do not correspond to each other
    x_filenames = [processString(d) for d in data_X.filenames]
    y_filenames = data_y[:, 0].tolist()
    correct_index = [x_filenames.index(yName) for yName in y_filenames]
    X = v_X.toarray()[correct_index, :]

    train_ind = data_y[:, 1] != 'To be Tested'

    X_train, y_train = X[train_ind, :], data_y[train_ind, 1]
    X_test = X[np.invert(train_ind), :]
    return X_train, y_train, X_test, vectorizer.vocabulary_, y_filenames


def preprocessData(X, y, vocabulary):
    """
        Runs mutual information and returns new design matrix X'
    """
    featureList = mutualInformation(X, y, 10, vocabulary)
    feature_index = [vocabulary[f] for f in featureList]
    return X[:, feature_index], y


def mutualInformation(X, y, k, vocabulary):
    """
        Given training data X and labels y, computes the mutual information of
        terms and classes. Feature extraction step.

        Returns the union of the top k terms for each class

        ** NOTE **
        1. X must contain the counts!
        2. Takes very long, did not have time to test!
    """
    d = {}   # maps class -> priority queue(-I, term)
    classes = set(y.tolist())
    for term, ind in vocabulary.items():
        for c in classes:
            # calculate N{0, 0}
            n00 = sum(np.logical_and((X[:, ind] == 0), y != c))
            # calculate N{1, 0}
            n10 = sum(np.logical_and((X[:, ind] > 0), y != c))
            # calculate N{0, 1}
            n01 = sum(np.logical_and((X[:, ind] == 0), y == c))
            # calculate N{1, 1}
            n11 = sum(np.logical_and((X[:, ind] > 0), y == c))
            N = n00 + n10 + n01 + n11
            I = (n11/N) * np.log2((N * n11)/((n11 + n10) * (n00 + n01))) +\
                (n01/N) * np.log2((N * n01)/((n01 + n00) * (n11 + n01))) +\
                (n10/N) * np.log2((N * n10)/((n10 + n11) * (n00 + n10))) +\
                (n00/N) * np.log2((N * n00)/((n01 + n00) * (n10 + n00)))
            if c not in d.keys():
                d[c] = PriorityQueue()
            d[c].put((-I, term))

    rSet = set()
    for c, queue in d.items():
        rSet.union(set([queue.get()[1] for _ in range(k)]))
    return list(rSet)


def modelSVM(X_train, y_train, X_test):
    """
        Trains/tests SVM model on K folds and prints cross validation results.
        Returns predictions on X_test
    """
    X_local_train, X_local_test, y_local_train, y_local_test =\
        train_test_split(X_train, y_train, test_size=0.25)
    print("Training ... SVM")
    clf = svm.SVC()
    clf.fit(X_local_train, y_local_train)

    yHat_train = clf.predict(X_local_train)
    yHat_test = clf.predict(X_local_test)
    print("Train accuracy : ", accuracy_score(y_local_train, yHat_train))
    print("Test accuracy :  ", accuracy_score(y_local_test, yHat_test))
    return clf.predict(X_test)


def modelNB(X_train, y_train, X_test):
    X_local_train, X_local_test, y_local_train, y_local_test =\
        train_test_split(X_train, y_train, test_size=0.25)
    print("Training ... NB")
    clf = MultinomialNB()
    clf.fit(X_local_train, y_local_train)

    yHat_train = clf.predict(X_local_train)
    yHat_test = clf.predict(X_local_test)
    print("Train accuracy : ", accuracy_score(y_local_train, yHat_train))
    print("Test accuracy :  ", accuracy_score(y_local_test, yHat_test))
    return clf.predict(X_test)


def processString(s):
    prog = re.compile('LNIND_[0-9]*_[A-Z]*_[0-9]*')
    return re.search(prog, s).group()


def writeCSV(filenames, predictions):
    with open('predictions.csv', 'w') as csvfile:
        w = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        w.writerow(['Filename', 'Area of Law'])
        for i in range(0, len(predictions)):
            w.writerow([filenames[i], predictions[i]])


def modelNB_final(X_train, y_train, X_test):
    print("Training ... NB final")
    clf = MultinomialNB()
    clf.fit(X_train, y_train)

    yHat_train = clf.predict(X_train)
    yHat_test = clf.predict(X_test)
    print("Train accuracy : ", accuracy_score(y_train, yHat_train))
    return yHat_test

if __name__ == "__main__":
    main()
