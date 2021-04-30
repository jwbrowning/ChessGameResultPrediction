# import scipy.io as sio
import numpy as np
import math
import argparse
import random

def data_loader(data_name):
    if data_name == "starplus":

        # starplus = sio.loadmat("data-starplus-04847-v7.mat")
        # metadata = starplus['meta'][0, 0]

        """
        #meta.study gives the name of the fMRI study
        #meta.subject gives the identifier for the human subject
        #meta.ntrials gives the number of trials in this dataset
        #meta.nsnapshots gives the total number of images in the dataset
        #meta.nvoxels gives the number of voxels (3D pixels) in each image
        #meta.dimx gives the maximum x coordinate in the brain image. The minimum x coordinate is x=1. meta.dimy and meta.dimz give the same information for the y and z coordinates.
        #meta.colToCoord(v,:) gives the geometric coordinate (x,y,z) of the voxel corresponding to column v in the data
        #meta.coordToCol(x,y,z) gives the column index (within the data) of the voxel whose coordinate is (x,y,z)
        #meta.rois is a struct array defining a few dozen anatomically defined Regions Of Interest (ROIs) in the brain. Each element of the struct array defines on of the ROIs, and has three fields: "name" which gives the ROI name (e.g., 'LIFG'), "coords" which gives the xyz coordinates of each voxel in that ROI, and "columns" which gives the column index of each voxel in that ROI.
        #meta.colToROI{v} gives the ROI of the voxel corresponding to column v in the data.

        """
        # study      = metadata['study']
        # subject    = metadata['subject']
        # ntrials    = metadata['ntrials'][0][0]
        # nsnapshots = metadata['nsnapshots'][0][0]
        # dimx       = metadata['dimx'][0][0]
        # colToCoord = metadata['colToCoord']
        # coordToCol = metadata['coordToCol']
        # rois       = metadata['rois']
        # colToROI   = metadata['colToROI']
        # info = starplus['info'][0]
        # info: This variable defines the experiment in terms of a sequence of 'trials'. 'info' is a 1x54 struct array, describing the 54 time intervals, or trials. Most of these time intervals correspond to trials during which the subject views a single picture and a single sentence, and presses a button to indicate whether the sentence correctly describes the picture. Other time intervals correspond to rest periods. The relevant fields of info are illustrated in the following example:
        # info(18) mint: 894 maxt: 948 cond: 2 firstStimulus: 'P' sentence: ''It is true that the star is below the plus.'' sentenceRel: 'below' sentenceSym1: 'star' sentenceSym2: 'plus' img: sap actionAnswer: 0 actionRT: 3613
        # info.mint gives the time of the first image in the interval (the minimum time)
        # info.maxt gives the time of the last image in the interval (the maximum time)
        # info.cond has possible values 0,1,2,3. Cond=0 indicates the data in this segment should be ignored. Cond=1 indicates the segment is a rest, or fixation interval. Cond=2 indicates the interval is a sentence/picture trial in which the sentence is not negated. Cond=3 indicates the interval is a sentence/picture trial in which the sentence is negated.
        # info.firstStimulus: is either 'P' or 'S' indicating whether this trail was obtained during the session is which Pictures were presented before sentences, or during the session in which Sentences were presented before pictures. The first 27 trials have firstStimulus='P', the remained have firstStimulus='S'. Note this value is present even for trials that are rest trials. You can pick out the trials for which sentences and pictures were presented by selecting just the trials trials with info.cond=2 or info.cond=3.
        # info.sentence gives the sentence presented during this trial. If none, the value is '' (the empty string). The fields info.sentenceSym1, info.sentenceSym2, and info.sentenceRel describe the two symbols mentioned in the sentence, and the relation between them.
        # info.img describes the image presented during this trial. For example, 'sap' means the image contained a 'star above plus'. Each image has two tokens, where one is above the other. The possible tokens are star (s), plus (p), and dollar (d).
        # info.actionAnswer: has values -1 or 0. A value of 0 indicates the subject is expected to press the answer button during this trial (either the 'yes' or 'no' button to indicate whether the sentence correctly describes the picture). A value of -1 indicates it is inappropriate for the subject to press the answer button during this trial (i.e., it is a rest, or fixation trial).
        # info.actionRT: gives the reaction time of the subject, measured as the time at which they pressed the answer button, minus the time at which the second stimulus was presented. Time is in milliseconds. If the subject did not press the button at all, the value is 0.
        # data = starplus['data']
        # data: This variable contains the raw observed data. The fMRI data is a sequence of images collected over time, one image each 500 msec. The data structure 'data' is a [54x1] cell array, with one cell per 'trial' in the experiment. Each element in this cell array is an NxV array of observed fMRI activations. The element data{x}(t,v) gives the fMRI observation at voxel v, at time t within trial x. Here t is the within-trial time, ranging from 1 to info(x).len. The full image at time t within trial x is given by data{x}(t,:).
        # Note the absolute time for the first image within trial x is given by info(x).mint.
        # MIN_LOSS_CHANGE=0.0001
        # maxFeatures = max([data[i][0].flatten().shape[0] for i in range(data.shape[0])])

        # Inputs
        # X = np.zeros((ntrials, maxFeatures + 1))
        # for i in range(data.shape[0]):
        #     f = data[i][0].flatten()
        #     X[i, :f.shape[0]] = f
        #     X[i, f.shape[0]] = 1  # Bias

        # # Outputs (+1 = Picture, -1 = Sentence)
        # Y = np.ones(ntrials)
        # Y[np.array([info[i]['firstStimulus'][0] != 'P' for i in range(ntrials)])] = -1

        # # Randomly permute the data
        # np.random.seed(1)  # Seed the random number generator to preserve the dev/test split
        # permutation = np.random.permutation(ntrials)
        # permutation = np.random.permutation(X.shape[0])
        # X = X[permutation,]
        # Y = Y[permutation,]
        # X_train = X[:40]
        # X_test = X[40:]
        # Y_train = Y[:40]
        # Y_test = Y[40:]
        # np.savez('Starplus.npz', X_train=X_train, X_test=X_test, Y_train=Y_train,Y_test=Y_test)

        data = np.load('Starplus.npz')
        X_train = data['X_train'] / 10.0
        Y_train = data['Y_train']
        X_test = data['X_test'] / 10.0
        Y_test = data['Y_test']
        return np.transpose(X_train), Y_train.reshape(-1, 1), np.transpose(X_test), Y_test.reshape(-1, 1)

    else:
        X_train = [[1.0, 1.0, 1.0], [2.0, 2.0, 1.0], [4.0, 4.0, 1.0], [5.0, 5.0, 1.0], [1.0, 3.0, 1.0], [2.0, 4.0, 1.0],
                   [4.0, 6.0, 1.0], [5.0, 7.0, 1.0]]
        X_test = [[3.0, 3.0, 1.0], [3.0, 5.0, 1.0]]
        Y_train = [[1.0], [1.0], [1.0], [1.0], [-1.0], [-1.0], [-1.0], [-1.0]]
        Y_test = [[1.0], [-1.0]]
        return np.transpose(np.asarray(X_train)), np.asarray(Y_train), np.transpose(np.asarray(X_test)), np.asarray(
            Y_test)


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
        if t % (maxIter / 20) == 0 or t % (100) == 0:
            print(100 * (t / maxIter), "%")
            acc = Accuracy(testX, testY, w)
            print(w)
            print("Current Accuracy:", acc)

    Y[Y == 0] = -1  # change label to be {-1, 1}
    return w


def Perceptron(X, Y, learningRate=0.01, maxIter=100, testX = np.asarray([]), testY = np.asarray([])):
    print("Starting P...")
    """
    Input:
        X: a (D+1)-by-N matrix (numpy array) of the input data; that is, we have concatenate "1" for you
        Y: a N-by-1 matrix (numpy array) of the label; labels are {-1, 1} and you have to turn them to {0, 1}
    Output:
        w: the linear weight vector. Please represent it as a (D+1)-by-1 matrix (numpy array).
    Useful tool:
        1. np.matmul: for matrix-matrix multiplication
        2. the builtin "reshape" and "transpose()" functions of a numpy array
        3. np.sign: for sign
    """
    N = X.shape[1]
    D_plus_1 = X.shape[0]
    w = np.zeros((D_plus_1, 1))
    np.random.seed(1)

    for t in range(maxIter):
        permutation = np.random.permutation(N)
        X = X[:, permutation]
        Y = Y[permutation, :]
        for n in range(N):
            yHat = np.sign(np.matmul(np.transpose(w), X[:, n]))
            if not (yHat == Y[n][0]).all():
                reshaped = np.reshape((Y[n][0] * X[:, n]), (D_plus_1, 1))
                w = w + learningRate * reshaped

        if t % (maxIter / 20) == 0 or t % (100) == 0:
            print(100 * (t / maxIter), "%")
            acc = Accuracy(testX, testY, w)
            print(w)
            print("Current Accuracy:", acc)
    return w


def Accuracy(X, Y, w):
    Y_hat = np.sign(np.matmul(X.transpose(), w))
    # print(Y_hat.shape)
    # print(Y.shape)
    correct = (Y_hat == Y)
    # print(correct)
    return float(sum(correct)) / len(correct)

def Accuracy1(X, results, wW, wD):
    yHatWin = np.sign(np.matmul(X.transpose(), wW))
    yHatDraw = np.sign(np.matmul(X.transpose(), wD))
    # print(yHatWin)
    # print(Y_hat.shape)
    # print(Y.shape)
    w = 0
    d = 0
    ll = 0
    total = 0
    for i in range(len(results)):
        total += 1
        if yHatWin[i][0] == 1 and results[i] == 1:
            w += 1
        elif yHatDraw[i][0] == 1 and results[i] == 2:
            d += 1
        elif yHatWin[i][0] != 1 and yHatDraw[i][0] != 1 and results[i] == 0:
            ll += 1
    # print(correct)
    # print("Win:", w / len(yHatWin == 1))
    # print("Draw:", d / len(yHatDraw == 1))
    # print("Loss:", ll / (len(yHatWin) - len(yHatWin == 1) - len(yHatDraw == 1)))
    return (w + d + ll) / total


def main():
    # X_train, Y_train, X_test, Y_test = data_loader("starplus")

    X = []
    Y = []
    testtest = []
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
                if i == 3 or (i > 4 and i != 15):
                    continue
                num = float(split[i])
                if i == 0 or i == 1:
                    num /= 2000.0
                # if i == 2:
                #     num /= 3.0
                if i == 2 or i == 4:
                    num = abs(num)
                if i == 15:
                    num /= 10.0
                if not i == 3:
                    nums.append(num)
            nums.append(abs(float(split[2]) - float(split[4])))
            nums.append(1.0)
            if random.uniform(0.0, 1.0) < testPct:
                testX.append(nums)
                testY.append([-1.0])
                results.append(1)
            else:
                X.append(nums)
                Y.append([-1.0])
    with open(r"C:\Users\Johnathan\Downloads\ChessAI2600\draws.txt") as inFile:
        for line in inFile:
            split = line.split()
            nums = []
            for i in range(len(split)):
                # if i == 0 or i == 1 or i == 3 or i == 4 or i == 5 or i == 6 or i == 7 or i == 8:
                #     continue
                if i == 3 or (i > 4 and i != 15):
                    continue
                num = float(split[i])
                if i == 0 or i == 1:
                    num /= 2000.0
                # if i == 2:
                #     num /= 3.0
                if i == 2 or i == 4:
                    num = abs(num)
                if i == 15:
                    num /= 10.0
                if not i == 3:
                    nums.append(num)
            nums.append(abs(float(split[2]) - float(split[4])))
            nums.append(1.0)
            if random.uniform(0.0, 1.0) < testPct:
                testX.append(nums)
                testY.append([1.0])
                results.append(2)
            else:
                X.append(nums)
                Y.append([1.0])
    with open(r"C:\Users\Johnathan\Downloads\ChessAI2600\losses.txt") as inFile:
        for line in inFile:
            split = line.split()
            nums = []
            for i in range(len(split)):
                # if i == 0 or i == 1 or i == 3 or i == 4 or i == 5 or i == 6 or i == 7 or i == 8:
                #     continue
                if i == 3 or (i > 4 and i != 15):
                    continue
                num = float(split[i])
                if i == 0 or i == 1:
                    num /= 2000.0
                # if i == 2:
                #     num /= 3.0
                if i == 15:
                    num /= 10.0
                if i == 2 or i == 4:
                    num = abs(num)
                if not i == 3:
                    nums.append(num)
            nums.append(abs(float(split[2]) - float(split[4])))
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

    # if args.algorithm == "logistic":
    # ----------------Logistic Loss-----------------------------------
    # ----------------Perceptron-----------------------------------
    # else:
    # wLR = Logisitc_Regression(X1, Y1,  maxIter=100000, learningRate=0.01, testX=testX1, testY=testY1)
    # w = Perceptron(X1, Y1, maxIter=100000, learningRate=0.001, testX=testX1, testY=testY1)
    #
    # # winWeights = [  [ 6.949397  ],
    # #                 [-7.6475295 ],
    # #                 [ 3.02613333],
    # #                 [ 0.101     ],
    # #                 [ 0.068     ],
    # #                 [ 0.043     ],
    # #                 [ 0.083     ],
    # #                 [ 0.0827    ],
    # #                 [-0.038     ],
    # #                 [-0.791     ]]
    # # wWin = np.asarray(winWeights)
    # # drawWeights = [  [-2.303420e+00],
    # #                  [ 3.223722e+00],
    # #                  [-3.693200e-01],
    # #                  [-8.800000e-02],
    # #                  [-1.170000e-01],
    # #                  [ 3.000000e-03],
    # #                  [-2.530000e-01],
    # #                  [ 4.000000e-01],
    # #                  [ 4.800000e-02],
    # #                  [-1.040000e-01]]
    # # wDraw = np.asarray(drawWeights)
    #
    # # acc = Accuracy1(testX1, results, wWin, wDraw)
    # # print(acc)
    #
    # training_accuracy = Accuracy(X1, Y1, w)
    # test_accuracy = Accuracy(testX1, testY1, w)
    # print("P Accuracy: training set: ", training_accuracy)
    # print("P Accuracy: test set: ", test_accuracy)
    # training_accuracy = Accuracy(X1, Y1, wLR)
    # test_accuracy = Accuracy(testX1, testY1, wLR)
    # print("LR Accuracy: training set: ", training_accuracy)
    # print("LR Accuracy: test set: ", test_accuracy)
    # #           wElo    bElo    eval    R   N   B   Q   P   plyNum  ignore
    # # testData = [2700,   2600,      0,   0,  0,  0,  0,  0,     40,  1]
    # print("P weights:", w)
    # print("LR weights:", wLR)
    # # testGame = np.transpose(np.asarray(testData))
    # # testGuess = np.matmul(testGame.transpose(), w)
    # # print("Guess:", testGuess)
    # # if testGuess > 0:
    # #     print("White Win")
    # # else:
    # #     print("Not a win")

    # weights = [[-0.5596185 ],
    #          [ 0.05837491],
    #          [-0.01925951],
    #          [ 0.62501181]]
    # testData = [0, 7, 7, 1]
    drawWeights = [[ 0.17128975],
                     [ 0.27238666],
                     [-0.88986044],
                     [-0.02481595],
                     [ 0.05649346],
                     [ 0.05933688],
                     [ 0.18339354]]
    # drawWeights = [[ 0.22420195],
    #                  [ 0.23651338],
    #                  [-0.70158886],
    #                  [ 0.07638486],
    #                  [ 0.0435165 ],
    #                  [-0.07694228],
    #                  [ 0.17261007]]
    winWeights = [[-0.17179157],
                 [-0.66364057],
                 [ 0.87581088],
                 [ 0.0036652 ],
                 [-0.02809924],
                 [ 0.02222476],
                 [-0.35063793]]
    # winWeights = [[-0.33040591],
    #                  [-0.36630892],
    #                  [ 0.31490686],
    #                  [-0.07300285],
    #                  [-0.01192767],
    #                  [-0.03149933],
    #                  [-0.26140316]]
    testData = [2777.0 / 2000.0,
                2758.0 / 2000.0,
                1.64,
                -1,
                56 / 10.0,
                2.17,
                1]
    testPosition = np.transpose(np.asarray(testData))
    testDataAbs = []
    for d in testData:
        testDataAbs.append(abs(d))
    testPositionAbs = np.transpose(np.asarray(testDataAbs))
    testGuessDraw = np.matmul(testPositionAbs.transpose(), drawWeights)
    pDraw = pow(math.e, testGuessDraw) / (1 + pow(math.e, testGuessDraw))
    testGuessWin = np.matmul(testPosition.transpose(), winWeights)
    pWin = pow(math.e, testGuessWin) / (1 + pow(math.e, testGuessWin))
    print()
    print("pWin  =", pWin)
    print("pDraw =", pDraw)
    print("pLoss =", 1 - pWin - pDraw)

    # i = 0
    # corr = 0
    # tot = 0
    # for test in testX:
    #     testPosition = np.transpose(np.asarray(test))
    #     testDataAbs = []
    #     for d in test:
    #         testDataAbs.append(abs(d))
    #     testPositionAbs = np.transpose(np.asarray(testDataAbs))
    #     testGuessDraw = np.matmul(testPositionAbs.transpose(), drawWeights)
    #     pDraw = pow(math.e, testGuessDraw) / (1 + pow(math.e, testGuessDraw))
    #     testGuessWin = np.matmul(testPosition.transpose(), winWeights)
    #     pWin = pow(math.e, testGuessWin) / (1 + pow(math.e, testGuessWin))
    #     pLoss = 1 - pWin - pDraw
    #     if pWin > pDraw and pWin > pLoss and results[i] == 1:
    #         corr += 1
    #     elif pLoss > pDraw and pLoss > pWin and results[i] == 0:
    #         corr += 1
    #     elif pDraw > pLoss and pDraw > pWin and results[i] == 2:
    #         corr += 1
    #     tot += 1
    #     i += 1

    # print("hey:")
    # print(corr, tot)
    # print(corr / tot)

    # accc = Accuracy1(testX1, results, winWeights, drawWeights)
    # print("acc", accc)



if __name__ == "__main__":
    main()
