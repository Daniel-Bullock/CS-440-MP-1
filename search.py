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
    print("end_")
    print(end_)
    path = [end_]
    while path[-1] != start:  # while last element doesn't equal start
        print("path")
        print(path)
        # print(path[-1])
        print("parent_map")
        print(parent_map)
        path.append(parent_map[path[-1]])
    path.reverse()
    return path


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
    goals_pq = new_pq(maze, goals, maze.getStart())

    f, curr_goal = heapq.heappop(goals_pq)
    heapq.heappush(pq, (f, [maze.getStart()]))

    while len(pq) > 0:
        curr_path = heapq.heappop(pq)[1]
        curr = curr_path[-1]

        if curr in visited:
            continue
        heuristic = min_distance(goals, curr)

        f = heuristic + len(curr_path) - 1
        visited[curr] = f
        if curr in goals:
            goals.remove(curr)
            if len(goals) == 0:
                return curr_path
            else:
                goals_pq = new_pq(maze, goals, curr)
                f, curr_goal = heapq.heappop(goals_pq)
                pq = []
                heapq.heappush(pq, (f, curr_path))
                visited.clear()
                continue
        for item in maze.getNeighbors(curr[0], curr[1]):
            heuristic = min_distance(goals, item)
            new_f = heuristic + len(curr_path) - 1
            if item not in visited:
                heapq.heappush(pq, (new_f, curr_path + [item]))
            else:  # checks if overlap has smaller f
                if new_f < visited[item]:
                    visited[item] = new_f
                    heapq.heappush(pq, (new_f, curr_path + [item]))

    return []


def min_distance(goals, curr):
    dist = math.inf  # infinity
    next_goal = (0, 0)
    for goal in goals:
        h = manhattan_distance(goal, curr)
        if h < dist:
            dist = h
            next_goal = goal
    return dist


def astar_multi(maze):
    """
    Runs A star for part 3 of the assignment in the case where there are
    multiple objectives.

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
        heuristic = min_distance(goals, curr)

        f = heuristic + len(curr_path) - 1
        visited[curr] = f
        if curr in goals:
            goals.remove(curr)
            if len(goals) == 0:
                return curr_path
            else:
                goals_pq = new_pq(maze, goals, curr)
                f, curr_goal = heapq.heappop(goals_pq)
                pq = []
                heapq.heappush(pq, (f, curr_path))
                visited.clear()
                continue
        for item in maze.getNeighbors(curr[0], curr[1]):
            heuristic = min_distance(goals, item)
            new_f = heuristic + len(curr_path) - 1
            if item not in visited:
                heapq.heappush(pq, (new_f, curr_path + [item]))
            else:  # checks if overlap has smaller f
                if new_f < visited[item]:
                    visited[item] = new_f
                    heapq.heappush(pq, (new_f, curr_path + [item]))

    return []


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
