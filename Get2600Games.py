import chess
import chess.engine
import chess.pgn
import sys
import random

random.seed(23053333)
# outFile = open(r"C:\Users\Johnathan\Downloads\ChessAI2600\TopPlayerGames.pgn", "w", encoding='cp1252')
outFile = open(r"C:\Users\Johnathan\Downloads\ChessAI2600\TopPlayerGames2.pgn", "w", encoding='cp1252')

# with open(r"C:\Users\Johnathan\Downloads\mb-3.45\mb-3.45.pgn", encoding='cp1252', errors='replace') as inFile:
total = 0
accepted = 0
badEventCount = 0
with open(r"C:\Users\Johnathan\Downloads\ChessAI2600\TopPlayerGames.pgn", encoding='cp1252', errors='replace') as inFile:
    try:
        count = 0
        currentGame = ""
        gettingData = False
        whiteEloAcceptable = False
        blackEloAcceptable = False
        eventAcceptable = True
        siteAcceptable = True
        for line in inFile:
            if line.startswith("[") and not gettingData:
                gettingData = True
                if whiteEloAcceptable and blackEloAcceptable and eventAcceptable and siteAcceptable:
                    try:
                        outFile.write(currentGame)
                        accepted += 1
                    except UnicodeEncodeError:
                        print("Encode Error")
                    count += 1
                    if count % 1000 == 0:
                        print(count)
                whiteEloAcceptable = False
                blackEloAcceptable = False
                eventAcceptable = True
                siteAcceptable = True
                currentGame = ""
                total += 1
            if line.startswith("[WhiteElo"):
                elo = float(line.split()[1][1:-2])
                whiteEloAcceptable = 2600 <= elo < 2900
            elif line.startswith("[BlackElo"):
                elo = float(line.split()[1][1:-2])
                blackEloAcceptable = 2600 <= elo < 2900
            elif line.startswith("[Event"):
                if ("Rapid" in line) or ("Blitz" in line) or ("rapid" in line) or ("blitz" in line):
                    eventAcceptable = False
            elif line.startswith("[Site"):
                if (" INT" in line) or ("Rapid" in line) or ("Blitz" in line) or ("rapid" in line) or ("blitz" in line):
                    siteAcceptable = False
            elif not line.startswith("["):
                gettingData = False
            currentGame += line

    except UnicodeDecodeError:
        print("Decode Error")

print(count)
print(accepted, " / ", total)
print(badEventCount)
outFile.close()
