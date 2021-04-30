import math
import random


def mean(data, column):
    sum = 0
    for row in data:
        sum += row[column]
    return sum / len(data)


def stdDev(data, column, mean):
    sum = 0
    for row in data:
        sum += math.pow(row[column] - mean, 2)
    sum /= len(data) - 1
    return math.sqrt(sum)


def normalDist(mean, stdDev, x):
    term1 = 1 / (stdDev * math.sqrt(2 * math.pi))
    term2 = math.pow(math.e, -.5 * math.pow((x - mean) / stdDev, 2))
    return term1 * term2


alpha = 1.0
negWordProbs = dict()
neuWordProbs = dict()
posWordProbs = dict()
vocabulary = dict()

columns = ["wElo", "bElo", "eval"]


def naiveBayes(negData, neuData, posData):
    global vocabulary
    global negWordProbs
    global neuWordProbs
    global posWordProbs

    for row in negData + neuData + posData:
        i = 0
        for column in row:
            if i == 3:
                continue
            word = str(column) + columns[i]
            if word in vocabulary:
                vocabulary[word] += 1
            else:
                vocabulary[word] = 1
            i += 1

    totalWords = 0
    for row in negData:
        i = 0
        for column in row:
            if i == 3:
                continue
            word = str(column) + columns[i]
            if word in negWordProbs:
                negWordProbs[word] += 1
            else:
                negWordProbs[word] = 1
            i += 1
    for key in negWordProbs:
        negWordProbs[key] += alpha
        negWordProbs[key] /= totalWords + alpha * len(vocabulary)

    totalWords = 0
    for row in neuData:
        i = 0
        for column in row:
            if i == 3:
                continue
            word = str(column) + columns[i]
            if word in neuWordProbs:
                neuWordProbs[word] += 1
            else:
                neuWordProbs[word] = 1
            i += 1
    for key in neuWordProbs:
        neuWordProbs[key] += alpha
        neuWordProbs[key] /= totalWords + alpha * len(vocabulary)

    totalWords = 0
    for row in posData:
        i = 0
        for column in row:
            if i == 3:
                continue
            word = str(column) + columns[i]
            if word in posWordProbs:
                posWordProbs[word] += 1
            else:
                posWordProbs[word] = 1
            i += 1
    for key in posWordProbs:
        posWordProbs[key] += alpha
        posWordProbs[key] /= totalWords + alpha * len(vocabulary)


winData = []
drawData = []
lossData = []
winDataT = []
drawDataT = []
lossDataT = []
dataT = []
testPct = .15

with open(r"C:\Users\Johnathan\Downloads\ChessAI2600\wins.txt") as inFile:
    for line in inFile:
        split = line.split()
        nums = []
        for s in split:
            if len(nums) < 4:
                nums.append(float(s))
        if random.uniform(0,1) < testPct:
            winDataT.append(nums)
        else:
            winData.append(nums)
with open(r"C:\Users\Johnathan\Downloads\ChessAI2600\draws.txt") as inFile:
    for line in inFile:
        split = line.split()
        nums = []
        for s in split:
            if len(nums) < 4:
                nums.append(float(s))
        if random.uniform(0,1) < testPct:
            drawDataT.append(nums)
        else:
            drawData.append(nums)
with open(r"C:\Users\Johnathan\Downloads\ChessAI2600\losses.txt") as inFile:
    for line in inFile:
        split = line.split()
        nums = []
        for s in split:
            if len(nums) < 4:
                nums.append(float(s))
        if random.uniform(0,1) < testPct:
            lossDataT.append(nums)
        else:
            lossData.append(nums)

total = len(winData) + len(drawData) + len(lossData)
pWin = len(winData) / total
pDraw = len(drawData) / total
pLoss = len(lossData) / total

winMeanAndStdDevs = []
drawMeanAndStdDevs = []
lossMeanAndStdDevs = []
counter = 0
for i in range(4):
    if i == 3:
        continue
    data = []
    data.append(mean(winData, i))
    data.append(stdDev(winData, i, data[0]))
    winMeanAndStdDevs.append(data)
    counter += 1
for i in range(4):
    if i == 3:
        continue
    data = []
    data.append(mean(drawData, i))
    data.append(stdDev(drawData, i, data[0]))
    drawMeanAndStdDevs.append(data)
for i in range(4):
    if i == 3:
        continue
    data = []
    data.append(mean(lossData, i))
    data.append(stdDev(lossData, i, data[0]))
    lossMeanAndStdDevs.append(data)

naiveBayes(lossData, drawData, winData)

testPosition = [2800, 2700, 0, 4]


# eloWinProb = 1 / (1 + math.pow(10, (testPosition[1] - testPosition[0]) / 400))


def wordProb(wordProbsDict, word):
    if word in wordProbsDict:
        return wordProbsDict[word]
    else:
        return alpha / (len(wordProbsDict) + alpha * len(vocabulary))

correct = 0
totall = 0
for line in winDataT:
    winProb = pWin
    i = 0
    for row in winMeanAndStdDevs:
        winProb *= normalDist(row[0], row[1], line[i])
        i += 1
    while i < len(winData[0]) - 1:
        winProb *= wordProb(posWordProbs, str(line[i]) + columns[i])
        i += 1
    drawProb = pDraw
    i = 0
    for row in drawMeanAndStdDevs:
        drawProb *= normalDist(row[0], row[1], line[i])
        i += 1
    while i < len(drawData[0]) - 1:
        drawProb *= wordProb(neuWordProbs, str(line[i]) + columns[i])
        i += 1
    lossProb = pLoss
    i = 0
    for row in lossMeanAndStdDevs:
        lossProb *= normalDist(row[0], row[1], line[i])
        i += 1
    while i < len(lossData[0]) - 1:
        lossProb *= wordProb(negWordProbs, str(line[i]) + columns[i])
        i += 1
    if winProb >= drawProb and winProb >= lossProb:
        correct += 1
    totall += 1
for line in drawDataT:
    winProb = pWin
    i = 0
    for row in winMeanAndStdDevs:
        winProb *= normalDist(row[0], row[1], line[i])
        i += 1
    while i < len(winData[0]) - 1:
        winProb *= wordProb(posWordProbs, str(line[i]) + columns[i])
        i += 1
    drawProb = pDraw
    i = 0
    for row in drawMeanAndStdDevs:
        drawProb *= normalDist(row[0], row[1], line[i])
        i += 1
    while i < len(drawData[0]) - 1:
        drawProb *= wordProb(neuWordProbs, str(line[i]) + columns[i])
        i += 1
    lossProb = pLoss
    i = 0
    for row in lossMeanAndStdDevs:
        lossProb *= normalDist(row[0], row[1], line[i])
        i += 1
    while i < len(lossData[0]) - 1:
        lossProb *= wordProb(negWordProbs, str(line[i]) + columns[i])
        i += 1
    if drawProb >= winProb and drawProb >= lossProb:
        correct += 1
    totall += 1
for line in lossDataT:
    winProb = pWin
    i = 0
    for row in winMeanAndStdDevs:
        winProb *= normalDist(row[0], row[1], line[i])
        i += 1
    while i < len(winData[0]) - 1:
        winProb *= wordProb(posWordProbs, str(line[i]) + columns[i])
        i += 1
    drawProb = pDraw
    i = 0
    for row in drawMeanAndStdDevs:
        drawProb *= normalDist(row[0], row[1], line[i])
        i += 1
    while i < len(drawData[0]) - 1:
        drawProb *= wordProb(neuWordProbs, str(line[i]) + columns[i])
        i += 1
    lossProb = pLoss
    i = 0
    for row in lossMeanAndStdDevs:
        lossProb *= normalDist(row[0], row[1], line[i])
        i += 1
    while i < len(lossData[0]) - 1:
        lossProb *= wordProb(negWordProbs, str(line[i]) + columns[i])
        i += 1
    if lossProb >= drawProb and winProb >= winProb:
        correct += 1
    totall += 1

acc = correct / totall
print(correct, totall)
print(acc)

# total = winProb + drawProb + lossProb
# winProbAdj = winProb / total
# drawProbAdj = drawProb / total
# lossProbAdj = lossProb / total
# # eloAdjWin = eloWinProb - drawProbAdj / 2
# # eloAdjLoss = (1 - eloWinProb) - drawProbAdj / 2
# print("Calculated Values:")
# print("Win:  ", winProbAdj)
# print("Draw: ", drawProbAdj)
# print("Loss: ", lossProbAdj)
