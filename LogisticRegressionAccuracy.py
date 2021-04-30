import numpy as np

whiteWinWeights =   [[-0.34931975],
                     [-0.36694976],
                     [ 0.69070251],
                     [ 0.03531081],
                     [-0.03838654],
                     [-0.26802819]]

drawWeights =       [[ 0.14781548],
                     [ 0.15137451],
                     [-0.47341494],
                     [-0.01660633],
                     [ 0.04413341],
                     [ 0.11175098]]

blackWinWeights =   [[-0.50168762],
                     [-0.44034826],
                     [-0.82528505],
                     [-0.02854132],
                     [-0.03371242],
                     [-0.35260704]]

testX = []
results = []
with open("Data/wins.txt") as inFile:
    for line in inFile:
        split = line.split()
        nums = []
        for i in range(len(split)):
            # if i == 0 or i == 1 or i == 3 or i == 4 or i == 5 or i == 6 or i == 7 or i == 8:
            #     continue
            if i == 3 or (4 < i < 15):
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
        testX.append(nums)
        results.append(1)
with open("Data/draws.txt") as inFile:
    for line in inFile:
        split = line.split()
        nums = []
        for i in range(len(split)):
            # if i == 0 or i == 1 or i == 3 or i == 4 or i == 5 or i == 6 or i == 7 or i == 8:
            #     continue
            if i == 3 or (4 < i < 15):
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
        testX.append(nums)
        results.append(2)
with open("Data/losses.txt") as inFile:
    for line in inFile:
        split = line.split()
        nums = []
        for i in range(len(split)):
            # if i == 0 or i == 1 or i == 3 or i == 4 or i == 5 or i == 6 or i == 7 or i == 8:
            #     continue
            if i == 3 or (4 < i < 15):
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
        testX.append(nums)
        results.append(0)

testX1 = np.transpose(np.asarray(testX))

r = 0
correct = 0
total = 0
for test in testX:
    win = 0
    draw = 0
    loss = 0
    for i in range(len(test)):
        win += test[i] * whiteWinWeights[i][0]
        loss += test[i] * blackWinWeights[i][0]
        if i == 2 or i == 3:
            draw += abs(test[i]) * drawWeights[i][0]
        else:
            draw += test[i] * drawWeights[i][0]
    result = results[r]
    if result == 2 and win <= draw >= loss:
        correct += 1
    elif result == 1 and draw <= win >= loss:
        correct += 1
    elif result == 0 and draw <= loss >= win:
        correct += 1
    total += 1
    r += 1

print("Total Accuracy:")
print(correct, "/", total)
print(correct / total)
