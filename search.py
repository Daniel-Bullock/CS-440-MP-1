# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Michael Abir (abir2@illinois.edu) on 08/28/2018

"""
This is the main entry point for MP1. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
# Search should return the path.
# The path should be a list of tuples in the form (row, col) that correspond
# to the positions of the path taken by your search algorithm.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,astar,astar_multi,fast)

import heapq
import math
import queue
from copy import deepcopy
import copy
from collections import deque


def search(maze, searchMethod):
    return {
        "bfs": bfs,
        "astar": astar,
        "astar_corner": astar_corner,
        "astar_multi": astar_multi,
        "fast": fast,
    }.get(searchMethod)(maze)


def bfs(maze):
    """
    Runs BFS for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here

    q = []
    visited = {}
    keys = {}
    selected = None
    q.append(maze.getStart())

    while len(q) > 0:
        curr = q.pop(0)
        if maze.isObjective(curr[0], curr[1]):
            selected = curr
            break

        neighbors = maze.getNeighbors(curr[0], curr[1])

        for n in neighbors:
            if n not in visited:
                visited[n] = True
                q.append(n)
                keys[n] = curr

    curr = selected
    path = []
    while curr != maze.getStart():
        path.append(curr)
        curr = keys[curr]

    path.append(curr)
    path.reverse()  # backtrace
    print(path)
    return path


def astar(maze):
    """
    Runs A star for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here\

    start = maze.getStart()
    end = maze.getObjectives()[0]  # 0 needed so it's not the list it's the end spot

    pq = []  # priority queue  - filled with tuple of f, x&y, g(path distance from start)
    heapq.heappush(pq, (manhattan_distance(start, end), start, 0))

    visited = set()
    map_ = {}
    solvable = True
    at_collectible = None

    while len(pq) > 0:

        curr = heapq.heappop(pq)
        curr_pos = curr[1]

        if curr_pos == end:
            at_collectible = curr
            break

        neighbors = maze.getNeighbors(curr_pos[0], curr_pos[1])

        for n in neighbors:

            new_curr = (manhattan_distance(n, end) + curr[2] + 1, (n[0], n[1]), curr[2] + 1)

            if n not in visited and maze.isValidMove(n[0], n[1]):
                map_[new_curr] = curr
                heapq.heappush(pq, new_curr)
                visited.add(n)

    curr = at_collectible
    path = []
    while curr[1] != start:
        path.append(curr[1])
        curr = map_[curr]
    path.append(curr[1])
    path.reverse()

    return path


def manhattan_distance(start, end):
    distance = abs(start[0] - end[0]) + abs(start[1] - end[1])
    return distance


def backtrace(parent_map, start, end_):
    # print("end_")
    print(end_)
    path = [end_]
    while path[-1] != start:  # while last element doesn't equal start
        # print("path")
        # print(path)
        # print(path[-1])
        # print("parent_map")
        # print(parent_map)
        path.append(parent_map[path[-1]])
    path.reverse()
    return path


lengthMin = {}


def new_pq_copy(maze, goals, start):
    pq = []
    for goal in goals:
        f = min_manhattan(goals, start)
        heapq.heappush(pq, (f, goal))
    return pq


def min_manhattan(goals, start):
    #print(goals)
    #print(goals[0])
    minD = manhattan_distance(start, goals[0])
    min_goal = goals[0]
    for i in range(len(goals)):
        if manhattan_distance(start, goals[i]) < minD:
            minD = manhattan_distance(start, goals[i])
            min_goal = goals[i]
    goals_copy = goals.copy()
    goals_copy.remove(min_goal)
    paths = 0
    other = 0
    if tuple(goals_copy) in lengthMin.keys():
        other = lengthMin[tuple(goals_copy)]
    else:
        goals_copy2 = goals_copy.copy()
        curr_goal = min_goal
        while paths < len(goals) - 1:
            min_dist = manhattan_distance(curr_goal, goals_copy[0])
            flag = False
            for g in goals_copy:
                if manhattan_distance(curr_goal, g) < min_dist:
                    min_dist = manhattan_distance(curr_goal, g)
                    curr_goal = g
                    flag = True
            if flag is False:
                goals_copy.remove(goals_copy[0])
            else:
                goals_copy.remove(curr_goal)
            other += min_dist
            paths += 1
        # print(goals_copy2)
        # print(tuple(goals_copy2))
        lengthMin[tuple(goals_copy2)] = other
    return minD + other / 3


def mst_heur(start, goals, graph = None):
    minD = manhattan_distance(start, goals[0])
    goalMin = goals[0]

    for i in range(len(goals)):
        if manhattan_distance(start, goals[i]) < minD:
            minD = manhattan_distance(start, goals[i])
            goalMin = goals[i]

    goals_copy = goals.copy()
    goals_copy.remove(goalMin)

    return minD + graph.mst(goals_copy, goalMin)


def min_distance(goals, curr):
    dist = math.inf  # infinity
    next_goal = (0, 0)
    for goal in goals:
        h = manhattan_distance(goal, curr)
        if h < dist:
            dist = h
            next_goal = goal
    return dist


def nearest(maze, start, end):
    visited = set()
    q = [[start]]
    while len(q) > 0:

        curr_path = q.pop(0)
        curr = curr_path[-1]

        if curr in visited:
            continue

        visited.add(curr)

        if curr == end:
            return len(curr_path)

        for n in maze.getNeighbors(curr[0], curr[1]):
            if n not in visited:
                q.append(curr_path + [n])
    print("return 0")
    return 0


def new_pq(maze, goals, start):
    pq = []
    for goal in goals:
        f = closest(maze, start, goal)
        heapq.heappush(pq, (f, goal))
    return pq


def fast(maze):
    """
    Runs suboptimal search algorithm for part 4.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    pq = []
    visited = {}

    goals = maze.getObjectives()
    goals_pq = new_pq(maze, goals, maze.getStart())

    f, curr_goal = heapq.heappop(goals_pq)
    heapq.heappush(pq, (f, [maze.getStart()]))

    while len(pq) > 0:
        curr_path = heapq.heappop(pq)[1]
        curr = curr_path[-1]

        if curr in visited:
            continue
        heuristic = closest(maze, curr, curr_goal)

        f = heuristic + len(curr_path) - 1
        visited[curr] = f
        if curr in goals:
            goals.remove(curr)
            if len(goals) == 0:
                return curr_path
            else:
                # print("before")
                # print(curr_goal)
                goals_pq = new_pq(maze, goals, curr)
                f, curr_goal = heapq.heappop(goals_pq)
                # print("after")
                # print(curr_goal)
                pq = []
                heapq.heappush(pq, (f, curr_path))
                visited.clear()
                continue
        for item in maze.getNeighbors(curr[0], curr[1]):
            heuristic = closest(maze, item, curr_goal)
            new_f = heuristic + len(curr_path) - 1
            if item not in visited:
                heapq.heappush(pq, (new_f, curr_path + [item]))
            else:  # checks if overlap has smaller f
                if new_f < visited[item]:
                    visited[item] = new_f
                    heapq.heappush(pq, (new_f, curr_path + [item]))
    return []


def closest(maze, start, end):
    distance = abs(start[0] - end[0]) + abs(start[1] - end[1])
    return distance


def astar_corner(maze):
    """
    Runs A star for part 2 of the assignment in the case where there are four corner objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
        """
    # TODO: Write your code here
    """
    Plan:
    Do normal a* but then .clear visited after each new goal is found
    new  h = Manhattan distance to the nearest goal and then the manhattan distance to the other goals starting from this nearest goal. 
    new priority queue --  tuple (f, x&y, goals_left, 
    """
    pq = []
    visited = {}

    goals = maze.getObjectives()
    start = maze.getStart()

    tie = 1
    #
    # tuple = (f,g,h,x&y,tiebreaker, goals left, currpath, visited)
    f = min_manhattan(goals, start)
    curr = (f, 0, f, start, goals, 0, [])
    heapq.heappush(pq, curr)

    food = None
    while len(pq) > 0:
        curr = heapq.heappop(pq)
        #print("curr:", curr)
        if curr[3] in curr[4]:
            curr[4].remove(curr[3])
        if len(curr[4]) == 0:
            #print("DONE")
            #print(food)
            food = curr
            break
        neighbors = maze.getNeighbors(curr[3][0], curr[3][1])
        for n in neighbors:
            curr_goals_left = curr[4].copy()
            curr_visited = curr[6].copy()
            tie += 1
            #print("curr[6]: ", curr[6])
            #print("n: ", n)
            #print("curr[4]: ", curr[4])
            h2 = min_manhattan(curr[4], n)
            f2 = h2 + curr[1]
            g2 = curr[1] + 1

            node_new = (f2, g2, h2, n, curr_goals_left, tie, curr_visited)
            
            if node_new[3] not in visited or node_new[4] not in visited[node_new[3]][1]:
                if node_new[3] not in visited:
                    visited[node_new[3]] = (node_new[3], [])
                visited[node_new[3]][1].append(node_new[4])
                node_new[6].append(curr[3])
                heapq.heappush(pq, node_new)

    if food is None:
        return []

    food[6].append(food[3])

    return food[6]


def astar_multi(maze):
    """
    Implements A* search on a maze with multiple goals
    Currently using the naive strategy of "next dot = closest dot to current point"
    Arguments:
        states {set of tuples} -- represents the "empty" states in the maze
        start {tuple} -- the starting point
        goals {list of tuples} -- list of dots that need to be reached
    Returns:
        list of tuples, int -- returns the path between the dots and the number of nodes expanded
    """
    graph_ = Graph(maze.getObjectives())

    pq = []
    visited = {}

    goals = maze.getObjectives()
    start = maze.getStart()

    tie = 1
    #
    # tuple = (f,g,h,x&y,tiebreaker, goals left, currpath, visited)
    # h = min_manhattan(goals, start)
    h = mst_heur(start, goals, graph_)

    curr = (h, 0, h, start, goals, 0, [])
    heapq.heappush(pq, curr)

    food = None
    while len(pq) > 0:
        curr = heapq.heappop(pq)
        # print("curr:", curr)
        if curr[3] in curr[4]:
            curr[4].remove(curr[3])
        if len(curr[4]) == 0:
            # print("DONE")
            # print(food)
            food = curr
            break
        neighbors = maze.getNeighbors(curr[3][0], curr[3][1])
        for n in neighbors:
            curr_goals_left = curr[4].copy()
            curr_visited = curr[6].copy()
            tie += 1

            # print("curr[6]: ", curr[6])
            # print("n: ", n)
            # print

            # h2 = min_manhattan(curr[4], n)
            h2 = mst_heur(n, curr[4], graph_)
            f2 = h2 + curr[1]
            g2 = curr[1] + 1

            node_new = (f2, g2, h2, n, curr_goals_left, tie, curr_visited)

            if node_new[3] not in visited or node_new[4] not in visited[node_new[3]][1]:
                if node_new[3] not in visited:
                    visited[node_new[3]] = (node_new[3], [])
                visited[node_new[3]][1].append(node_new[4])
                node_new[6].append(curr[3])
                heapq.heappush(pq, node_new)

    if food is None:
        return []

    food[6].append(food[3])

    return food[6]


class Cell:
    edge_list = []
    value = 0
    coord = (0,0)

    def __init__(self, in_value, in_coord, in_edge_list):
        self.value = in_value
        self.edge_list = in_edge_list
        self.coord = in_coord


class Edge:

    start = None
    end = None
    length = 0

    def __init__(self, start_, end_, length_):
        self.start = start_
        self.end = end_
        self.length = length_

    def __gt__(self, other):
        return self.length > other.length

    def __lt__(self, other):
        return self.length < other.length


class Graph:

    cell_list = {}

    mst_ = {}

    def __init__(self, goals):

        for x in goals:
            edges = []
            for g in goals:
                if g != x:
                    e = Edge(x, g, manhattan_distance(x, g))
                    edges.append(e)

            n = Cell(math.inf, x, edges)
            self.cell_list[x] = n

        #print("cell_list: ", self.cell_list)

    def mst(self, goals_left, start):
        #print("goals_left: ", goals_left)
        #print("start: ", start)
        if tuple(goals_left) in self.mst_.keys():
            return self.mst_[tuple(goals_left)]
        else:
            goals_left_copy = goals_left.copy()

            eQ = []

            e_Sum = 0

            currgoal = start

            while len(goals_left) > 0:
                #print(self.cell_list)
                #print(cell_list)
                #print(currgoal)
                curr = self.cell_list[currgoal]

                edges = curr.edge_list

                for e in edges:

                    if e.start in goals_left or e.end in goals_left:
                        heapq.heappush(eQ, e)

                if currgoal in goals_left:
                    goals_left.remove(currgoal)

                b = False

                while b is False and len(eQ) > 0:
                    low = heapq.heappop(eQ)

                    if low.start in goals_left:
                        currgoal = low.start
                        e_Sum += low.length
                        b = True

                    elif low.end in goals_left:
                        currgoal = low.end
                        e_Sum += low.length

                    b = True

                if b is False:
                    break

            self.mst_[tuple(goals_left_copy)] = e_Sum

            return 0


