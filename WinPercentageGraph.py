from os import system, name
import chess
import chess.engine
import chess.pgn
import sys
import random
import time
import numpy as np
import math

def clear():
    if name == 'nt':
        x = system('cls')
    else:
        x = system('clear')

random.seed(87756892)

stockfishPath = r"C:\Users\Johnathan\Downloads\stockfish_12_win_x64_bmi2\stockfish_20090216_x64_bmi2"
engine = chess.engine.SimpleEngine.popen_uci(stockfishPath)

pgnfilename = r"C:\Users\Johnathan\Downloads\RandomPGNs\lichess_pgn_2021.03.20_SundanceKid1019_vs_RaisinBranCrunch.TA5C3C2q.pgn"
# pgnfilename = r"C:\Users\Johnathan\Downloads\caruana_vachier-lagrave_2021.pgn"
# pgnfilename = r"C:\Users\Johnathan\Downloads\DullSword_vs_Bonkiye_2021.02.26.pgn"
pgnfilename = r"C:\Users\Johnathan\Downloads\ChessAI2600\NakamuraAnandCandidates2016.pgn"
pgnfilename = r"C:\Users\Johnathan\Downloads\Nepomniachtchi, Ian_vs_Alekseenko, Kirill_2021.04.21.pgn"
# pgnfilename = r"C:\Users\Johnathan\Downloads\pgn_data.pgn"


#Read pgn file:
with open(pgnfilename) as f:
    game = chess.pgn.read_game(f)

#Go to the end of the game and create a chess.Board() from it:
#game = game.end()

drawWeights = [[ 0.17128975],
                 [ 0.27238666],
                 [-0.88986044],
                 [-0.02481595],
                 [ 0.05649346],
                 [ 0.05933688],
                 [ 0.18339354]]
winWeights = [[-0.17179157],
             [-0.66364057],
             [ 0.87581088],
             [ 0.0036652 ],
             [-0.02809924],
             [ 0.02222476],
             [-0.35063793]]

whiteName = game.headers["White"]
blackName = game.headers["Black"]
file = open(r"C:\Users\Johnathan\Downloads\ChessAI2600\GameGraphs\\" + whiteName + " vs " + blackName + ".txt", "w")

board = game.board()
for move in game.mainline_moves():
    board.push(move)
    # print()
    engine = chess.engine.SimpleEngine.popen_uci(stockfishPath)
    info = engine.analyse(board, chess.engine.Limit(depth=25))
    score = info["score"].white().score(mate_score=100000)
    eval = score / 100.0
    clear()
    boardStr = str(board)
    split = boardStr.split("\n")

    whiteRating = float(game.headers["WhiteElo"]) if "WhiteElo" in game.headers and game.headers["WhiteElo"] != "?" else 2600.0
    blackRating = float(game.headers["BlackElo"]) if "BlackElo" in game.headers and game.headers["BlackElo"] != "?" else 2600.0
    fen = ""
    material = 0
    try:
        fen = board.board_fen()
    except NameError:
        print("bad fen")
    for c in fen:
        if c == 'R':
            material += 5
        elif c == 'N':
            material += 3
        elif c == 'B':
            material += 3
        elif c == 'Q':
            material += 9
        elif c == 'P':
            material += 1
        elif c == 'r':
            material += -5
        elif c == 'n':
            material += -3
        elif c == 'b':
            material += -3
        elif c == 'q':
            material += -9
        elif c == 'p':
            material += -1
    diff = abs(eval - material)
    testData = [whiteRating / 2000.0,
                blackRating / 2000.0,
                eval,
                material,
                board.ply() / 10.0,
                diff,
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
    pLoss = 1 - pWin - pDraw
    file.write(str(int(board.ply()/2)) + " " + str(pWin[0]) + " " + str(pDraw[0]) + " " + str(pLoss[0]) + "\n")
    # time.sleep(1.0)
    # game = game.next()

file.close()