import chess
import chess.engine
import chess.pgn
import sys
import random

random.seed(8215892)

size = 100000

stockfishPath = r"C:\Users\Johnathan\Downloads\stockfish_12_win_x64_bmi2\stockfish_20090216_x64_bmi2"
engine = chess.engine.SimpleEngine.popen_uci(stockfishPath)


lines = []
count = 0
count2 = 0
#   cp1252  utf-8
with open(r"C:\Users\Johnathan\Downloads\ChessAI2600\TopPlayerGames.pgn", encoding='cp1252', errors='replace') as inFile:
    try:
        for line in inFile:
            lines.append(line)
            count += 1
            if count % 10000 == 0:
                count = 0
                count2 += 1
                print(count2)
    except UnicodeDecodeError:
        print("oof ", count2)

    print("LINES: ", len(lines))

fileName = r"C:\Users\Johnathan\Downloads\ChessAI2600\stupidUtilityFile.txt"
errorGames = r"C:\Users\Johnathan\Downloads\ChessAI2600\errorGames.txt"
outFile = open(r"C:\Users\Johnathan\Downloads\ChessAI2600\output.txt", "w")
# outFile.write("wElo\tbElo\teval\tresult\tR\tK\tB\tQ\tP\tplyNum\n")
outFile.close()
i = 0
while i < size:
    randLine = int(len(lines) * random.random())
    startLine = endLine = 0
    plyCount = 0
    whiteRating = 0
    blackRating = 0
    result = -1
    while randLine > 0 and not (len(lines[randLine]) > 1 and lines[randLine][0] == '['):
        randLine -= 1
    # while randLine < len(lines) - 1 and len(lines[randLine]) > 1:
    #     randLine += 1
    while randLine > 0 and len(lines[randLine]) > 1:
        randLine -= 1
    randLine += 1
    startLine = randLine
    while randLine < len(lines) - 1 and lines[randLine][0] != "1":
        if lines[randLine].startswith("[PlyCount \""):
            plyCount = int(lines[randLine].split("\"")[1])
        if lines[randLine].startswith("[WhiteElo \""):
            whiteRating = int(lines[randLine].split("\"")[1])
        if lines[randLine].startswith("[BlackElo \""):
            blackRating = int(lines[randLine].split("\"")[1])
        if lines[randLine].startswith("[Result \""):
            resultStr = lines[randLine].split("\"")[1]
            if resultStr.startswith("1/2"):
                result = 2
            elif resultStr.startswith("0"):
                result = 0
            elif resultStr.startswith("1"):
                result = 1

        randLine += 1
    while randLine < len(lines) - 1 and len(lines[randLine]) > 1:
        randLine += 1
        endLine = randLine
    if endLine <= startLine:
        continue
    if result == -1:
        continue
    if plyCount == 0:
        continue
    if whiteRating == 0 or blackRating == 0:
        continue
    fw = open(fileName, "w")
    fw.writelines(lines[startLine:-(len(lines)-endLine)])
    fw.close()

    with open(fileName) as f:
        game = chess.pgn.read_game(f)

    if game is None:
        continue

    ply = int(random.random() * plyCount)
    plyNum = ply
    while not ply <= 0 and not game.is_end():
        node = game.variations[0]
        board = game.board()  # print the board if you want, to make sure
        game = node
        ply -= 1

    if game is None or game.board is None:
        continue
    try:
        fen = board.board_fen()
    except NameError:
        continue
    rookCountW = 0
    knightCountW = 0
    bishopCountW = 0
    queenCountW = 0
    pawnCountW = 0
    rookCountB = 0
    knightCountB = 0
    bishopCountB = 0
    queenCountB = 0
    pawnCountB = 0
    for c in fen:
        if c == 'R':
            rookCountW += 1
        elif c == 'N':
            knightCountW += 1
        elif c == 'B':
            bishopCountW += 1
        elif c == 'Q':
            queenCountW += 1
        elif c == 'P':
            pawnCountW += 1
        elif c == 'r':
            rookCountB += 1
        elif c == 'n':
            knightCountB += 1
        elif c == 'b':
            bishopCountB += 1
        elif c == 'q':
            queenCountB += 1
        elif c == 'p':
            pawnCountB += 1
    materialDiff = queenCountW * 9 - queenCountB * 9
    materialDiff += rookCountW * 5 - rookCountB * 5
    materialDiff += bishopCountW * 3 - bishopCountB * 3
    materialDiff += knightCountW * 3 - knightCountB * 3
    materialDiff += pawnCountW - pawnCountB

    try:
        info = engine.analyse(board, chess.engine.Limit(depth=20))
        evaluation = info["score"].white().score() / 100
    except NameError:
        continue
    except TypeError:
        continue

    outFile = open(r"C:\Users\Johnathan\Downloads\mb-3.45\output.txt", "a")
    outFile.write(str(whiteRating)+"\t"+str(blackRating)+"\t"+str(evaluation)+"\t"+str(result)+"\t"+str(materialDiff)+"\t"
                  +str(rookCountW)+"\t"+str(knightCountW)+"\t"+str(bishopCountW)+"\t"+str(queenCountW)+"\t"+str(pawnCountW)+"\t"
                  +str(rookCountB)+"\t"+str(knightCountB)+"\t"+str(bishopCountB)+"\t"+str(queenCountB)+"\t"+str(pawnCountB)+"\t"+str(plyNum)+"\n")
    outFile.close()
    #print("Eval: ", info["score"].white().score() / 100)

    # i = startLine
    # print("-----")
    # while i <= endLine:
    #     print(lines[i])
    #     i += 1
    # print("-----")

    i += 1
    if i % 100 == 0:
        print(i)



# pgnfilename = r"C:\Users\Johnathan\Downloads\RandomPGNs\lichess_pgn_2021.03.20_SundanceKid1019_vs_RaisinBranCrunch.TA5C3C2q.pgn"
#
# #Read pgn file:
# with open(pgnfilename) as f:
#     game = chess.pgn.read_game(f)
#
# #Go to the end of the game and create a chess.Board() from it:
# #game = game.end()
# board = game.board()
#
# moveNum = 19
# whiteToMove = True
#
# ply = moveNum * 2 - (2 if whiteToMove else 1)
# while not ply <= 0 and not game.is_end():
#     node = game.variations[0]
#     board = game.board() #print the board if you want, to make sure
#     game = node
#     ply -= 1
#
# engine = chess.engine.SimpleEngine.popen_uci(stockfishPath)
#
# info = engine.analyse(board, chess.engine.Limit(depth=20))
# print("Eval: ", info["score"].white().score() / 100)
