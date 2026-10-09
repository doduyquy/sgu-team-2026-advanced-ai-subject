"""Self-contained search implementations used by the maze example notebooks.

This module replaces the instructor-only ``tree_search_solution.py`` referenced
by the original notebooks.  It intentionally keeps the same small public API.
"""

from collections import deque
import heapq
import itertools
import random as _random

import numpy as np

import maze_helper as mh


_order = ("N", "E", "S", "W")
_random_order = False
_moves = {
    "N": (-1, 0),
    "E": (0, 1),
    "S": (1, 0),
    "W": (0, -1),
}


def set_order(order=None, random=False):
    """Set the successor order or request a new random order per expansion."""
    global _order, _random_order
    _random_order = bool(random)
    if order is not None:
        normalized = tuple(order)
        if len(normalized) != 4 or set(normalized) != set(_moves):
            raise ValueError("order must contain N, E, S and W exactly once")
        _order = normalized


def manhattan(state, goal):
    return abs(state[0] - goal[0]) + abs(state[1] - goal[1])


heuristic = manhattan


class _Node:
    __slots__ = ("state", "parent", "action", "cost", "depth")

    def __init__(self, state, parent=None, action=None, cost=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.cost = cost
        self.depth = 0 if parent is None else parent.depth + 1


def _directions():
    directions = list(_order)
    if _random_order:
        _random.shuffle(directions)
    return directions


def _successors(maze, state):
    rows, cols = maze.shape
    row, col = state
    for action in _directions():
        dr, dc = _moves[action]
        child = (row + dr, col + dc)
        if (0 <= child[0] < rows and 0 <= child[1] < cols
                and maze[child] != "X"):
            yield action, child


def _unpack(node):
    nodes = []
    while node is not None:
        nodes.append(node)
        node = node.parent
    nodes.reverse()
    return [node.state for node in nodes], [node.action for node in nodes[1:]]


def _result(goal_node, reached, expanded, generated, max_frontier, maze_anim=None):
    if goal_node is None:
        path, actions = None, None
    else:
        path, actions = _unpack(goal_node)
    return {
        "path": path,
        "actions": actions,
        "reached": reached,
        "expanded": expanded,
        "generated": generated,
        "max_frontier": max_frontier,
        "path_cost": None if actions is None else len(actions),
        "maze_anim": maze_anim or [],
    }


def best_first_search(maze, strategy="BFS", debug=False, vis=False, W=1):
    """Run BFS, GBFS or (weighted) A* graph search."""
    strategy = strategy.upper()
    if strategy == "DFS":
        return DFS(maze, vis=vis)
    if strategy not in {"BFS", "GBFS", "A*"}:
        raise ValueError("strategy must be BFS, DFS, GBFS or A*")

    start = mh.find_pos(maze, "S")
    goal = mh.find_pos(maze, "G")
    root = _Node(start)
    expanded = 0
    generated = 1

    if strategy == "BFS":
        frontier = deque([root])
        reached = {start: root}
        max_frontier = 1
        while frontier:
            node = frontier.popleft()
            if node.state == goal:
                return _result(node, reached, expanded, generated, max_frontier)
            expanded += 1
            for action, state in _successors(maze, node.state):
                if state not in reached:
                    child = _Node(state, node, action, node.cost + 1)
                    reached[state] = child
                    frontier.append(child)
                    generated += 1
            max_frontier = max(max_frontier, len(frontier))
        return _result(None, reached, expanded, generated, max_frontier)

    counter = itertools.count()
    frontier = []
    root_priority = heuristic(start, goal) * (W if strategy == "A*" else 1)
    heapq.heappush(frontier, (root_priority, -next(counter), root))
    best_cost = {start: 0}
    reached = {start: root}
    max_frontier = 1

    while frontier:
        _, _, node = heapq.heappop(frontier)
        if node.cost != best_cost.get(node.state):
            continue
        if node.state == goal:
            return _result(node, reached, expanded, generated, max_frontier)
        expanded += 1
        for action, state in _successors(maze, node.state):
            new_cost = node.cost + 1
            if strategy == "GBFS":
                should_add = state not in reached
                priority = heuristic(state, goal)
            else:
                should_add = new_cost < best_cost.get(state, float("inf"))
                priority = new_cost + W * heuristic(state, goal)
            if should_add:
                child = _Node(state, node, action, new_cost)
                best_cost[state] = new_cost
                reached[state] = child
                heapq.heappush(frontier, (priority, -next(counter), child))
                generated += 1
        max_frontier = max(max_frontier, len(frontier))
    return _result(None, reached, expanded, generated, max_frontier)


def DFS(maze, vis=False, max_tries=100000, debug_reached=False,
        check_cycle=True, limit=None, frontier_option=2):
    """Depth-first tree search with optional depth limit and ancestor checking."""
    del debug_reached
    start = mh.find_pos(maze, "S")
    goal = mh.find_pos(maze, "G")
    root = _Node(start)
    stack = [root]
    reached = {start: root}
    expanded = 0
    generated = 1
    max_frontier = 1
    # For depth-limited search, remembering the shallowest occurrence of a
    # state is safe: arriving deeper can never leave more search budget.
    use_depth_reached = limit is not None and frontier_option == 2
    best_depth = {start: 0}

    while stack and expanded < max_tries:
        node = stack.pop()
        if node.state == goal:
            return _result(node, reached, expanded, generated, max_frontier)
        if limit is not None and node.depth >= limit:
            continue
        expanded += 1
        ancestors = set()
        if check_cycle:
            cursor = node
            while cursor is not None:
                ancestors.add(cursor.state)
                cursor = cursor.parent
        children = []
        for action, state in _successors(maze, node.state):
            if check_cycle and state in ancestors:
                continue
            child = _Node(state, node, action, node.cost + 1)
            if (use_depth_reached
                    and child.depth >= best_depth.get(state, float("inf"))):
                continue
            if use_depth_reached:
                best_depth[state] = child.depth
            children.append(child)
            reached[state] = child
            generated += 1
        # A stack reverses insertion order; reverse here so the configured first
        # direction is explored first.
        stack.extend(reversed(children))
        max_frontier = max(max_frontier, len(stack))
    return _result(None, reached, expanded, generated, max_frontier)


def IDS(maze, frontier_option=2, max_tries=100000, vis=False):
    """Iterative deepening search starting at depth zero."""
    total_expanded = 0
    max_depth = int(np.count_nonzero(maze != "X")) - 1
    for limit in range(max_depth + 1):
        result = DFS(
            maze,
            vis=vis,
            max_tries=max_tries,
            check_cycle=True,
            limit=limit,
            frontier_option=frontier_option,
        )
        total_expanded += result["expanded"]
        if result["path"] is not None:
            result["limit"] = limit
            result["iterations"] = limit + 1
            result["total_expanded"] = total_expanded
            return result
    result["limit"] = max_depth
    result["iterations"] = max_depth + 1
    result["total_expanded"] = total_expanded
    return result


def show_path(maze, result):
    """Print a short summary and draw explored cells plus the final path."""
    path = result.get("path")
    if path is None:
        print("No solution found!")
        return None

    shown = np.copy(maze)
    start = mh.find_pos(maze, "S")
    goal = mh.find_pos(maze, "G")
    reached = result.get("reached", {})
    states = reached.keys() if hasattr(reached, "keys") else reached
    for state in states:
        if state not in {start, goal} and shown[state] == " ":
            shown[state] = "."
    for state in path:
        if state not in {start, goal}:
            shown[state] = "P"

    # A path without cycle checking can revisit the start.  Restore the two
    # semantic markers explicitly so maze_helper can always render the maze.
    shown[start] = "S"
    shown[goal] = "G"

    print(f"Path length: {len(path) - 1}")
    print(f"Reached squares: {len(reached)}")
    print(f"Action sequence: {result.get('actions')}")
    return mh.show_maze(shown)


show_maze = mh.show_maze


def min_index(values):
    return min(range(len(values)), key=values.__getitem__)
