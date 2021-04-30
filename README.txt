CSE 3521 Intro to AI Project: Predicting the outcome of a pro-chess game
by Johnathan Browning (.295) and Alan Wu (.4232)

Files submitted:
    NaiveBayes.py
        contains our implementation of Naive Bayes for our data set
        running this file will conduct naive bayes and output the accuracy on test data, which is about 68%
    LogisticRegressionWins.py
        contains our implementation of the Logistic Regression algorithm, training a weight vector for white wins
        running this will start running logistic regression iterations and
        will print out a weight vector along with its accuracy every k frames
    LogisticRegressionDraws.py
        contains our implementation of the Logistic Regression algorithm, training a weight vector for draws
        running this will start running logistic regression iterations and
        will print out a weight vector along with its accuracy every k frames
    LogisticRegressionLosses.py
        contains our implementation of the Logistic Regression algorithm, training a weight vector for black wins
        running this will start running logistic regression iterations and
        will print out a weight vector along with its accuracy every k frames
    LogisticRegressionAccuracy.py
        contains code that tests the accuracy of all 3 weight vectors on the entire dataset
        we randomly split out database so that 85% would be training data and 15% would be test data,
        this file tests accuracy on both the training and test data, which is about 68.1%, marginally higher than
        naive bayes
    Data
        the data folder contains the wins, draws, and losses files with all our data points
