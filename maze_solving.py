from collections import deque
from queue import PriorityQueue
import copy

def printMaze(maze,m,n) :
    for i in range(m) :
        for j in range(n) :
            print(maze[i][j], end=" ")
        print()

def makePath(maze,i,j,path) :
    temp = copy.deepcopy(maze)
    for p in range(len(path)-1) :
        if path[p] == 'D' :
            temp[i+1][j] = 'P'
            i += 1
        elif path[p] == 'U' :
            temp[i-1][j] = 'P'
            i -= 1
        elif path[p] == 'L' :
            temp[i][j-1] = 'P'
            j -= 1
        elif path[p] == 'R' :
            temp[i][j+1] = 'P'
            j += 1
        p += 1
    return temp
        
def possibleRouteBFS(maze,i,j,lv,cp) :
    route = []
    if i > 0 and maze[i-1][j] != 'W' :
        route.append([i-1,j,lv,cp+'U'])
    if i < len(maze) - 1 and maze[i+1][j] != 'W' :
        route.append([i+1,j,lv,cp+'D'])
    if j > 0 and maze[i][j-1] != 'W' :
        route.append([i,j-1,lv,cp+'L'])
    if j < len(maze[0]) - 1 and maze[i][j+1] != 'W' :
        route.append([i,j+1,lv,cp+'R'])
    if len(route) > 0 :
        return route
    return None

def possibleRouteAstar(maze,i,j,lv,goal,cp) :
    route = []
    if i > 0 and maze[i-1][j] != 'W' :
        h = (goal[0] - i + 1) + (goal[1] - j)
        route.append([i-1,j,lv,h,cp+'U'])
    if i < len(maze) - 1 and maze[i+1][j] != 'W' :
        h = (goal[0] - i - 1) + (goal[1] - j)
        route.append([i+1,j,lv,h,cp+'D'])
    if j > 0 and maze[i][j-1] != 'W' :
        h = (goal[0] - i) + (goal[1] - j + 1)
        route.append([i,j-1,lv,h,cp+'L'])
    if j < len(maze[0]) - 1 and maze[i][j+1] != 'W' :
        h = (goal[0] - i) + (goal[1] - j - 1)
        route.append([i,j+1,lv,h,cp+'R'])
    if len(route) > 0 :
        return route
    return None

def goalTest(maze,i,j) :
    return maze[i][j] == 'G'

def checkContainsBFS(r,arr) :
    for ex in arr :
        if r[0] == ex[0] and r[1] == ex[1] :
            return True
    return False

def checkContainsAstar(r,arr) :
    for ex in arr :
        if r[0] == ex[1][0] and r[1] == ex[1][1] and r[3] == ex[0] :
            return True
    return False

def BFS(maze, i, j) :
    explored = []
    frontier = deque([[i,j,0,'']]) # index i, index j, level, direction string
    
    while True :
        if len(frontier) == 0 :
            print("no solution")
            return [-1,-1]#failure
        currentNode = frontier.popleft()
        current_i = currentNode[0]
        current_j = currentNode[1]
        current_lv = currentNode[2]
        current_path = currentNode[3]
        explored.append(currentNode)
        route = possibleRouteBFS(maze,current_i,current_j,current_lv+1, current_path)
        
        if route is None :
            print("no solution")
            return [-1,-1]
        else :
            for r in route :
                if not checkContainsBFS(r,explored) and not checkContainsBFS(r,frontier) :
                    if goalTest(maze,r[0],r[1]) :
                        print("Success")
                        print("explored = ", len(explored))
                        print("path cost = ", r[2])
                        print("path length = ", r[2])
                        return [r[0],r[1],r[3]] # return goal's location and final path
                    else :
                        frontier.append(r)
    return [-1,-1]

def aStar(maze, i, j, goal) :
    explored = []
    h = (goal[0] - i) + (goal[1] - j) # minimum path length from current location to goal
    frontier = PriorityQueue() # index i, j, level, h-value and path
    frontier.put((h,[i,j,0,0,'']))
    
    while True :
        if frontier.qsize() == 0 :
            print("no solution")
            return -1 #failure
        currentNode = frontier.get()
        current_i = currentNode[1][0]
        current_j = currentNode[1][1]
        current_lv = currentNode[1][2]
        current_path = currentNode[1][4]
        #goal test
        if current_i == goal[0] and current_j == goal[1] :
            print("Success")
            print("explored = ", len(explored))
            print("path cost = ", current_lv)
            print("path length = ", current_lv)
            return [current_i,current_j,current_path]
        
        explored.append(currentNode)
        route = possibleRouteAstar(maze,current_i,current_j,current_lv+1,goal,current_path)
        
        if route is None :
            print("no solution")
            return -1
        else :
            for r in route :
                if not checkContainsAstar(r,explored) and not checkContainsAstar(r,frontier.queue) :
                    frontier.put((r[3],r[0:5]))
                elif checkContainsAstar(r,frontier.queue) :
                    for ex in frontier.queue :
                        if r[3] < ex[0] :
                            print(ex[0], r[3])
                            ex[0] = r[3]
    return -1

# read file
with open('input2.txt') as file:
    inputlist = file.read().splitlines() 
    
m = int(inputlist[0])
n = int(inputlist[1])
maze = []
aStarPath = []

start_i = 0
start_j = 0

#copy the board to an array 'maze'
#find the starting point
for i in range(2,2+m) :
    maze.append(inputlist[i].split())
    if 'S' in maze[i-2] :
        start_i = i-2
        start_j = maze[i-2].index('S')
        
print("---------Original Maze---------")
printMaze(maze,m,n)
# BFS
goal=[]
print("=========BFS=========")
goal=BFS(maze,start_i,start_j)

if goal[0] != -1 and goal[1] != -1 :
    path = goal[2]
    foundPath = makePath(maze, start_i, start_j,path)
    print("---------path---------")
    print(path[0], end="")
    for p in path[1:] :
        print(" -",p, end="")
    
    print("\n---------Solved Maze---------")
    printMaze(foundPath,m,n)
    
    
    print("=========A*=========")
    aStarSol = aStar(maze,start_i,start_j,goal)

    if aStarSol != -1 :
        aStarPath = aStarSol[2]
    print("---------path---------")
    print(aStarPath[0], end="")
    for p in aStarPath[1:] :
        print(" -",p, end="")

    foundPath = makePath(maze, start_i, start_j,aStarPath)

    print("\n---------Solved Maze---------")
    printMaze(foundPath,m,n)