from collections import deque
import math
import time
import random
import copy
def printChessBoard(size,sol) :
    for i in range(size) :
        for j in range(size) :
            temp = [i,j]
            if temp in sol :
                print("Q", end=" ")
            else :
                print("*", end =" ")
        print()
    print("\n")

def printBoard(n,board) :
    for i in range(n) :
        for b in board[i] :
            print(b, end=" ")
        print()
    print("\n")

def columnCrashTest(temp,currentPath) :
    i = temp[0]
    j = temp[1]
    for p in currentPath :
        if p[0] != i and p[1] == j :
            return True
    return False

def diagonalCrashTest(temp,currentPath,n) :
    i = temp[0] - 1
    left_j = temp[1] - 1
    right_j = temp[1] + 1
    while i >= 0 and (left_j >= 0 or right_j < n):
        if left_j >= 0 and [i,left_j] in currentPath :
            return True
        if right_j < n and [i,right_j] in currentPath :
            return True
        i -= 1
        right_j += 1
        left_j -= 1
    return False

def findSuccessor(currentNode,n) :
    current_i = currentNode[0]
    currentPath = currentNode[2]
    route = []
    for j in range(n) :
        temp = [current_i+1,j]
        if columnCrashTest(temp, currentPath) :
            continue
        if diagonalCrashTest(temp,currentPath,n) :
            continue
        route.append([current_i+1,j,currentPath+[temp]])
    return route

def makeRandomQueen(n) :
    currentBoard = [['*' for j in range(n)] for i in range(n)]
    currentQueen = [-1 for j in range(n)]
    randomLocation = random.sample(range(0,n),n)
    
    for i in range(n) :
        currentBoard[i][randomLocation[i]] = 'Q'
        currentQueen[i] = randomLocation[i]
    
    return currentBoard, currentQueen

def calHvalue(n,board,queen) :
    h = 0 
    for i in range(n) :
        r = queen[i] + 1
        l = queen[i] - 1
        for j in range(i+1,n) :
            # check if queen crashes in the same column
            if board[i] == board[j] :
                h+=1
            # check if queen crashes in right diagonal
            if r < n :
                right = ['*' for k in range(n)]
                right[r] = 'Q'
                if right == board[j] :
                    h+=1
                r += 1
            # check if queen crashes in left diagonal
            if l >= 0 :
                left = ['*' for k in range(n)]
                left[l] = 'Q'
                if left == board[j] :
                    h+=1
                l -= 1
    return h

def findSuccessorHC(n,currentBoard, currentQueen, current_hvalue) :
    sucBoard = copy.deepcopy(currentBoard)
    sucQueen = copy.deepcopy(currentQueen)
    suc_hvalue = current_hvalue
    
    for i in range(n) :
        tempBoard = copy.deepcopy(sucBoard)
        tempQueen = copy.deepcopy(sucQueen)
        temp_hvalue = suc_hvalue
        moved = False
        for j in range(n) :
            # check if [i,j] is same as current location of queen
            if j == currentQueen[i] :
                continue
            else :
                temp = ['*' for j in range(n)]
                temp[j] = 'Q'
                tempBoard[i] = copy.deepcopy(temp)
                tempQueen[i] = j
                temp_hvalue = calHvalue(n,tempBoard,tempQueen)

                if temp_hvalue == 0 : # if solution is found
                    print("i = ",i)
                    print("i = ",i, " h-value = ",temp_hvalue)
                    print("\n")
                    return True, tempBoard, tempQueen
                    
                elif temp_hvalue <= suc_hvalue :
                    sucBoard = copy.deepcopy(tempBoard)
                    sucQueen = copy.deepcopy(tempQueen)
                    suc_hvalue = temp_hvalue
                    moved = True
                    
                new = ['*' for j in range(n)]
                new[currentQueen[i]] = 'Q'
                tempBoard[i] = copy.deepcopy(new)
                tempQueen[i] = currentQueen[i]
        if moved :
            print("i = ",i, " h-value = ",suc_hvalue)
            printBoard(n,sucBoard)
    return False, sucBoard, sucQueen
            
def BFS(n) :
    sol = 0
    for i in range(n) :
        isSuccess = False
        frontier = deque([[0,i,[[0,i]]]]) #index i, first row = 0, location of queen [x,y]
        while True :
            if len(frontier) == 0 :
                break
            currentNode = frontier.popleft()
            if currentNode[0] == n-1 :
                break
            route = findSuccessor(currentNode,n)
            
            for r in route :
                if r[0] == n-1 :
                    if n < 8 :
                        printChessBoard(n,r[2])
                    isSuccess = True
                frontier.append(r)
            if isSuccess :
                isSuccess = False
                sol += 1
                
    return sol

def Hill_Climbing(n) :
    num_try = 0
    found = False
    while True :
        currentBoard, currentQueen = makeRandomQueen(n)
        print("initial board")
        printBoard(n,currentBoard)
        print()
        current_hvalue = calHvalue(n,currentBoard,currentQueen)
        
        if current_hvalue == 0 :
            found = True
        
        num_try += 1
        print("initial h value : ", current_hvalue)
        found, finalBoard, finalQueen = findSuccessorHC(n,currentBoard,currentQueen,current_hvalue)
        
        if found :
            print("SOLUTION FOUND")
            print("Number of tries : ", num_try)
            printBoard(n,finalBoard)
            break
        else :
            printBoard(n,finalBoard)
            print("make another random board")
    return

print("*****BFS*****")
n = int(input("N : "))
sol = 0
start = time.time()
if n == 1 :
    print("Number of Solution : 1")
    print("*")
else :
    sol = BFS(n)
    print("-----------------------\n")
    print("Number of Solution : ", sol)
math.factorial(100000)
end = time.time()
print(f"{end - start:.5f} sec")

if sol > 0 :
    print("\n*****Hill-Climbing*****")
    start = time.time()
    math.factorial(100000)
    Hill_Climbing(n)
    end = time.time()
    print(f"{end - start:.5f} sec")