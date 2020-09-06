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
    path.reverse()   # backtrace

    return path


def astar(maze):
    """
    Runs A star for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here\

    start = maze.getStart()
    #print(start)
    end = maze.getObjectives()[0]  # 0 needed so it's not the list it's the end spot
    #print(end)
    pq = []     # priority queue  - filled with tuple of f, x&y, g(path distance from start)
    heapq.heappush(pq, (manhattan_distance(start, end), start, 0))
    visited = set()
    map = {}
    solvable = True
    at_collectible = None

    while len(pq) > 0:
        #print("in")

        curr = heapq.heappop(pq)
        curr_pos = curr[1]
        #print("curr_pos ")
        #print(curr_pos)
        #if curr_pos in visited:
        #    continue
        #visited.add(curr_pos) ADD TO VISITED IN THE NEIGHBOR LOOP

        if curr_pos == end:
            at_collectible = curr
            break


        neighbors = maze.getNeighbors(curr_pos[0], curr_pos[1])

        for n in neighbors:

            new_curr = (manhattan_distance(n, end)+curr[2]+1, (n[0], n[1]), curr[2]+1)
            if n not in visited and maze.isValidMove(n[0], n[1]):
                map[new_curr] = curr
                #f = manhattan_distance(curr_pos, end) + curr[2] # f(x) = h(x) = g(x)
                heapq.heappush(pq, new_curr)
                visited.add(n)
        #print("queue")
        #print(pq)
    curr = at_collectible
    path = []
    while curr[1] != start:
        path.append(curr[1])
        curr = map[curr]
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
    return []


def astar_multi(maze):
    """
    Runs A star for part 3 of the assignment in the case where there are
    multiple objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    return []


def fast(maze):
    """
    Runs suboptimal search algorithm for part 4.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    return []
