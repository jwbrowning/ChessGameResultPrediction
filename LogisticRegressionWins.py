# import scipy.io as sio
import numpy as np
import math
import argparse
import random


def sigmoid(a):
    return 1 / (1 + np.exp(-a))


def Logisitc_Regression(X, Y, learningRate=0.01, maxIter=100, testX = np.asarray([]), testY = np.asarray([])):
    print("Starting LR...")
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
        # if t % (maxIter / 20) == 0 or t % (50) == 0:
        #     print(t, 100 * (t / maxIter), "%")
        #     acc = Accuracy(testX, testY, w)
        #     print(w)
        #     print("Current Accuracy:", acc)

        Y[Y == 0] = -1  # change label to be {-1, 1}
        print(t, Accuracy(X, Y, w))
        Y[Y == -1] = 0.0  # change label to be {0, 1}

    Y[Y == 0] = -1  # change label to be {-1, 1}
    return w


def Accuracy(X, Y, w):
    Y_hat = np.sign(np.matmul(X.transpose(), w))
    # print(Y_hat.shape)
    # print(Y.shape)
    correct = (Y_hat == Y)
    # print(correct)
    return float(sum(correct)) / len(correct)


def main():

    X = []
    Y = []
    results = []
    testPct = .15
    testX = []
    testY = []
    with open(r"C:\Users\Johnathan\Downloads\ChessAI2600\wins.txt") as inFile:
        for line in inFile:
            split = line.split()
            nums = []
            for i in range(len(split)):
                # if i == 0 or i == 1 or i == 3 or i == 4 or i == 5 or i == 6 or i == 7 or i == 8:
                #     continue
                if i == 3:
                    continue
                num = float(split[i])
                if i == 0 or i == 1:
                    num /= 2000.0
                # if i == 2:
                #     num /= 3.0
                # if i == 2 or i == 4:
                #     num = abs(num)
                if i == 15:
                    num /= 10.0
                if not i == 3:
                    nums.append(num)
            # nums.append(abs(float(split[2]) - float(split[4])))
            nums.append(1.0)
            if random.uniform(0.0, 1.0) < testPct:
                testX.append(nums)
                testY.append([1.0])
                results.append(1)
            else:
                X.append(nums)
                Y.append([1.0])
    with open(r"C:\Users\Johnathan\Downloads\ChessAI2600\draws.txt") as inFile:
        for line in inFile:
            split = line.split()
            nums = []
            for i in range(len(split)):
                # if i == 0 or i == 1 or i == 3 or i == 4 or i == 5 or i == 6 or i == 7 or i == 8:
                #     continue
                if i == 3:
                    continue
                num = float(split[i])
                if i == 0 or i == 1:
                    num /= 2000.0
                # if i == 2:
                #     num /= 3.0
                # if i == 2 or i == 4:
                #     num = abs(num)
                if i == 15:
                    num /= 10.0
                if not i == 3:
                    nums.append(num)
            # nums.append(abs(float(split[2]) - float(split[4])))
            nums.append(1.0)
            if random.uniform(0.0, 1.0) < testPct:
                testX.append(nums)
                testY.append([-1.0])
                results.append(2)
            else:
                X.append(nums)
                Y.append([-1.0])
    with open(r"C:\Users\Johnathan\Downloads\ChessAI2600\losses.txt") as inFile:
        for line in inFile:
            split = line.split()
            nums = []
            for i in range(len(split)):
                # if i == 0 or i == 1 or i == 3 or i == 4 or i == 5 or i == 6 or i == 7 or i == 8:
                #     continue
                if i == 3:
                    continue
                num = float(split[i])
                if i == 0 or i == 1:
                    num /= 2000.0
                # if i == 2:
                #     num /= 3.0
                if i == 15:
                    num /= 10.0
                # if i == 2 or i == 4:
                #     num = abs(num)
                if not i == 3:
                    nums.append(num)
            # nums.append(abs(float(split[2]) - float(split[4])))
            nums.append(1.0)
            if random.uniform(0.0, 1.0) < testPct:
                testX.append(nums)
                testY.append([-1.0])
                results.append(0)
            else:
                X.append(nums)
                Y.append([-1.0])

    X1 = np.transpose(np.asarray(X))
    Y1 = (np.asarray(Y))
    testX1 = np.transpose(np.asarray(testX))
    testY1 = np.asarray(testY)

    print("number of training data instances: ", X1.shape)
    print("number of test data instances: ", testX1.shape)
    print("number of training data labels: ", Y1.shape)
    print("number of test data labels: ", testY1.shape)

    wLR = Logisitc_Regression(X1, Y1,  maxIter=10000000, learningRate=0.01, testX=testX1, testY=testY1)

    # drawWeights = [[ 0.17128975],
    #                  [ 0.27238666],
    #                  [-0.88986044],
    #                  [-0.02481595],
    #                  [ 0.05649346],
    #                  [ 0.05933688],
    #                  [ 0.18339354]]
    #fadffdfffff
    # [[0.16758875]
    #  [0.28625655]
    #  [-0.85690441]
    #  [-0.04407547]
    #  [0.05230802]
    #  [0.07374992]
    #  [0.18263119]]
    # winWeights = [[-0.17179157],
    #              [-0.66364057],
    #              [ 0.87581088],
    #              [ 0.0036652 ],
    #              [-0.02809924],
    #              [ 0.02222476],
    #              [-0.35063793]]
    #gsdfgsd
    # [[-0.16080804]
    #  [-0.53553298]
    #  [0.34042304]
    #  [-0.09389055]
    #  [-0.01228825]
    #  [-0.05049037]
    #  [-0.27129147]]
    # testData = [2763.0 / 2000.0,
    #             2777.0 / 2000.0,
    #             -1.99,
    #             -1,
    #             52 / 10.0,
    #             0.99,
    #             1]
    # testPosition = np.transpose(np.asarray(testData))
    # testDataAbs = []
    # for d in testData:
    #     testDataAbs.append(abs(d))
    # testPositionAbs = np.transpose(np.asarray(testDataAbs))
    # testGuessDraw = np.matmul(testPositionAbs.transpose(), drawWeights)
    # pDraw = pow(math.e, testGuessDraw) / (1 + pow(math.e, testGuessDraw))
    # testGuessWin = np.matmul(testPosition.transpose(), winWeights)
    # pWin = pow(math.e, testGuessWin) / (1 + pow(math.e, testGuessWin))
    # print()
    # print("pWin  =", pWin)
    # print("pDraw =", pDraw)
    # print("pLoss =", 1 - pWin - pDraw)


if __name__ == "__main__":
    main()
