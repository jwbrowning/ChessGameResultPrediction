
wins = open(r"C:\Users\Johnathan\Downloads\ChessAI2600\wins.txt", "w")
draws = open(r"C:\Users\Johnathan\Downloads\ChessAI2600\draws.txt", "w")
losses = open(r"C:\Users\Johnathan\Downloads\ChessAI2600\losses.txt", "w")

skip = 1
with open(r"C:\Users\Johnathan\Downloads\ChessAI2600\outputNew.txt") as inFile:
    for line in inFile:
        if skip > 0:
            skip -= 1
            continue
        if len(line) < 2:
            continue
        data = line.split()
        result = int(data[3])
        if result == 1:
            wins.write(line)
        elif result == 2:
            draws.write(line)
        elif result == 0:
            losses.write(line)

wins.close()
draws.close()
losses.close()
