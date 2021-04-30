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

board = game.board()
for move in game.mainline_moves():
    board.push(move)
    # print()
    engine = chess.engine.SimpleEngine.popen_uci(stockfishPath)
    info = engine.analyse(board, chess.engine.Limit(depth=20))
    score = info["score"].white().score()
    print(score)
    eval = score / 100.0
    clear()
    print()
    print("  ", game.headers["Black"], game.headers["BlackElo"] if "BlackElo" in game.headers else "")
    print()
    boardStr = str(board)
    split = boardStr.split("\n")
    for line in split:
        print("\t", line)
    print()
    print("  ", game.headers["White"], game.headers["WhiteElo"] if "WhiteElo" in game.headers else "")
    print()
    print("\tEval:", eval)
    adj = max(eval, -4)
    adj = min(adj, 4)
    adj += 4
    adj *= 4
    evalBar = "----------------------------------\n["
    for i in range(32):
        if i < adj:
            evalBar += "#"
        else:
            evalBar += " "
    evalBar += "]\n----------------------------------"
    print(evalBar)

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
    # print(pLoss)

    print("Win:", str(int(pWin * 100))+"%", "   Draw:", str(int(pDraw * 100))+"%", "   Loss:", str(int(pLoss * 100))+"%")

    pWin *= 32
    pDraw *= 32
    pLoss *= 32

    predictionBar = "----------------------------------\n["
    for i in range(32):
        if i < pWin:
            predictionBar += "#"
        elif i < pWin + pDraw:
            predictionBar += "="
        else:
            predictionBar += " "
    predictionBar += "]\n----------------------------------"
    print(predictionBar)

    time.sleep(1.0)
    # game = game.next()
