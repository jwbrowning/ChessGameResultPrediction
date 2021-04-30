#
#   Johnathan Browning HW4
#   run as normal python file
#   I discussed this homework thoroughly with Alan Wu (.4232)
#

# import scipy.io as sio
import numpy as np
import math
import argparse
import random
import requests
from collections import Counter


class DataReader:
    def __init__(self, url):
        self.feature_set = []
        self.data = []
        self.set_of_lines = []
        for i in range(22):
            feature_value_counter = Counter()
            self.feature_set.append(feature_value_counter)
        r = requests.get( url, stream=True )
        for line in r.iter_lines():
            line = line.decode('utf-8')
            self.set_of_lines.append(line)

    def read(self):
        for line in self.set_of_lines:
            line_values = line.split(',')
            if len(line_values) != 23:
                continue
            y_label = [line_values[0]]
            x_atributes = line_values[1:]
            updated_data = y_label + x_atributes
            self.data.append(updated_data)


def sigmoid(a):
    return 1 / (1 + np.exp(-a))


def Logisitc_Regression(X, Y, learningRate=0.01, maxIter=100):
    print("Starting LR...")
    print()
    """
    Input:
        X: a (D+1)-by-N matrix (numpy array) of the input data; that is, we have concatenate "1" for you
        Y: a N-by-1 matrix (numpy array) of the label
    Output:
        w: the linear weight vector. Please represent it as a (D+1)-by-1 matrix (numpy array).
    Useful tool:
        1. np.matmul: for matrix-matrix multiplication
        2. the builtin "reshape" and "transpose()" functions of a numpy array
    """
    N = X.shape[1]
    D_plus_1 = X.shape[0]
    w = np.zeros((D_plus_1, 1))

    Y[Y == -1] = 0.0  # change label to be {0, 1}
    for t in range(maxIter):
        loss = 0
        for n in range(N):
            loss += (Y[n][0] - sigmoid(np.matmul(np.transpose(w), X[:, n]))) * X[:, n]
        loss *= - 1 / N
        loss = np.reshape(loss, (D_plus_1, 1))

        w = w - learningRate * loss
        # if t % (maxIter / 20) == 0 or t % (100) == 0:
        #     print(100 * (t / maxIter), "%")
        #     acc = Accuracy(testX, testY, w)
        #     print(w)
        #     print("Current Accuracy:", acc)

        Y[Y == 0] = -1  # change label to be {-1, 1}
        print(t, Accuracy(X, Y, w))
        Y[Y == -1] = 0.0  # change label to be {0, 1}

    print("done")
    print()

    Y[Y == 0] = -1  # change label to be {-1, 1}
    return w


# def Accuracy(X, YPos, YNeu, wPos, wNeu):
#     YHatPos = np.sign(np.matmul(X.transpose(), wPos))
#     YHatNeu = np.sign(np.matmul(X.transpose(), wNeu))
#
#     correct = 0
#     for i in range(len(YHatPos)):
#         if YHatPos[i] == YPos[i] and YHatNeu[i] == YNeu[i]:
#             correct += 1
#
#     return correct / len(X)


def Accuracy(X, Y, w):
    Y_hat = np.sign(np.matmul(X.transpose(), w))
    # print(Y_hat.shape)
    # print(Y.shape)
    correct = (Y_hat == Y)
    # print(correct)
    return float(sum(correct)) / len(correct)


def main():

    trainNeg = 'https://raw.githubusercontent.com/jeniyat/CSE-5521-SP21/master/HW/HW4/Data/train/Negative.txt'
    trainNeu = 'https://raw.githubusercontent.com/jeniyat/CSE-5521-SP21/master/HW/HW4/Data/train/Neutral.txt'
    trainPos = 'https://raw.githubusercontent.com/jeniyat/CSE-5521-SP21/master/HW/HW4/Data/train/Positive.txt'
    testNeg = 'https://raw.githubusercontent.com/jeniyat/CSE-5521-SP21/master/HW/HW4/Data/test/Negative.txt'
    testNeu = 'https://raw.githubusercontent.com/jeniyat/CSE-5521-SP21/master/HW/HW4/Data/test/Neutral.txt'
    testPos = 'https://raw.githubusercontent.com/jeniyat/CSE-5521-SP21/master/HW/HW4/Data/test/Positive.txt'
    drNeg = DataReader(trainNeg)
    drNeg.read()
    drNeu = DataReader(trainNeu)
    drNeu.read()
    drPos = DataReader(trainPos)
    drPos.read()
    drNegT = DataReader(testNeg)
    drNegT.read()
    drNeuT = DataReader(testNeu)
    drNeuT.read()
    drPosT = DataReader(testPos)
    drPosT.read()

    uniqueWords = []
    uniqueWordsSet = set()
    for line in drNeg.set_of_lines + drNeu.set_of_lines + drPos.set_of_lines:
        split = line.split()
        for word in split:
            if word not in uniqueWords:
                uniqueWordsSet.add(word)
                uniqueWords.append(word)
    X = []
    Y = []
    testX = []
    testY = []

    for tweet in drNeg.set_of_lines:
        split = tweet.split()
        words = [0] * len(uniqueWords)
        for word in split:
            words[uniqueWords.index(word)] += 1
        words.append(1)
        X.append(words)
        Y.append([-1])
    for tweet in drNeu.set_of_lines:
        split = tweet.split()
        words = [0] * len(uniqueWords)
        for word in split:
            words[uniqueWords.index(word)] += 1
        words.append(1)
        X.append(words)
        Y.append([0])
    for tweet in drPos.set_of_lines:
        split = tweet.split()
        words = [0] * len(uniqueWords)
        for word in split:
            words[uniqueWords.index(word)] += 1
        words.append(1)
        X.append(words)
        Y.append([1])

    for tweet in drNegT.set_of_lines:
        split = tweet.split()
        words = [0] * len(uniqueWords)
        for word in split:
            if word in uniqueWordsSet:
                words[uniqueWords.index(word)] += 1
        words.append(1)
        testX.append(words)
        testY.append([-1])
    for tweet in drNeuT.set_of_lines:
        split = tweet.split()
        words = [0] * len(uniqueWords)
        for word in split:
            if word in uniqueWordsSet:
                words[uniqueWords.index(word)] += 1
        words.append(1)
        testX.append(words)
        testY.append([0])
    for tweet in drPosT.set_of_lines:
        split = tweet.split()
        words = [0] * len(uniqueWords)
        for word in split:
            if word in uniqueWordsSet:
                words[uniqueWords.index(word)] += 1
        words.append(1)
        testX.append(words)
        testY.append([1])

    X1 = np.transpose(np.asarray(X))
    YPos = (np.asarray(Y))
    YPos[YPos == 0] = -1
    YNeu = (np.asarray(Y))
    YNeu[YNeu == 1] = -1
    YNeu[YNeu == 0] = 1
    YNeg = (np.asarray(Y))
    YNeg[YNeg == -1] = 2
    YNeg[YNeg == 0] = -1
    YNeg[YNeg == 1] = -1
    YNeg[YNeg == 2] = 1

    testX1 = np.transpose(np.asarray(testX))
    testYPos = (np.asarray(testY))
    testYPos[testYPos == 0] = -1
    testYNeu = (np.asarray(testY))
    testYNeu[testYNeu == 1] = -1
    testYNeu[testYNeu == 0] = 1
    testYNeg = (np.asarray(testY))
    testYNeg[testYNeg == -1] = 2
    testYNeg[testYNeg == 0] = -1
    testYNeg[testYNeg == 1] = -1
    testYNeg[testYNeg == 2] = 1

    print("number of training data instances: ", X1.shape)
    # print("number of test data instances: ", testX1.shape)
    print("number of training data labels: ", YPos.shape)
    print("number of training data labels: ", YNeu.shape)
    # print("number of test data labels: ", testY1.shape)

    wPos = Logisitc_Regression(X1, YPos,  maxIter=100, learningRate=0.1)
    wNeu = Logisitc_Regression(X1, YNeu,  maxIter=100, learningRate=0.1)
    wNeg = Logisitc_Regression(X1, YNeg,  maxIter=100, learningRate=0.1)

    trainAccPos = Accuracy(X1, YPos, wPos)
    trainAccNeu = Accuracy(X1, YNeu, wNeu)
    trainAccNeg = Accuracy(X1, YNeg, wNeg)
    print("Train Accuracy: pos, neu, neg")
    print(trainAccPos, trainAccNeu, trainAccNeg)
    trainAccPos = Accuracy(testX1, testYPos, wPos)
    trainAccNeu = Accuracy(testX1, testYNeu, wNeu)
    trainAccNeg = Accuracy(testX1, testYNeg, wNeg)
    print("Test Accuracy: pos, neu, neg")
    print(trainAccPos, trainAccNeu, trainAccNeg)


if __name__ == "__main__":
    main()
