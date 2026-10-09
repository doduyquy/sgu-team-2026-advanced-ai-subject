"""Self-contained search implementations used by the maze example notebooks.

This module replaces the instructor-only ``tree_search_solution.py`` referenced
by the original notebooks.  It intentionally keeps the same small public API.
"""

from collections import deque
from collections import Counter
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
        check_cycle=True, limit=None, frontier_option=2, animation=False):
    """Depth-first search with the three frontier policies used in Lab 02.

    ``frontier_option=3`` checks only the current root-to-node path.
    ``frontier_option=2`` also refuses duplicate states already in the stack.
    ``frontier_option=1`` moves an existing duplicate to the stack top.
    """
    del debug_reached, vis
    if frontier_option not in {1, 2, 3}:
        raise ValueError("frontier_option must be 1, 2 or 3")
    start = mh.find_pos(maze, "S")
    goal = mh.find_pos(maze, "G")
    root = _Node(start)
    stack = [root]
    frontier_counts = Counter({start: 1})
    frontier_nodes = {start: root}
    reached = {start: root}
    expanded = 0
    generated = 1
    max_frontier = 1
    iterations = 0
    repeated_pops = 0
    duplicate_encounters = 0
    cycle_skips = 0
    popped_counts = Counter()
    maze_anim = []
    # For depth-limited search, remembering the shallowest occurrence of a
    # state is safe: arriving deeper can never leave more search budget.
    use_depth_reached = limit is not None and frontier_option == 2
    best_depth = {start: 0}

    def record_frame(current=None):
        if not animation:
            return
        frame = np.copy(maze)
        for state in reached:
            if frame[state] == " ":
                frame[state] = "."
        for state in frontier_counts:
            if frontier_counts[state] and frame[state] == " ":
                frame[state] = "F"
        if current is not None:
            cursor = current
            while cursor is not None:
                if frame[cursor.state] not in {"S", "G"}:
                    frame[cursor.state] = "P"
                cursor = cursor.parent
        maze_anim.append(frame)

    record_frame(root)
    while stack and iterations < max_tries:
        node = stack.pop()
        iterations += 1
        frontier_counts[node.state] -= 1
        if frontier_counts[node.state] <= 0:
            del frontier_counts[node.state]
        if frontier_nodes.get(node.state) is node:
            frontier_nodes.pop(node.state, None)

        if popped_counts[node.state]:
            repeated_pops += 1
        popped_counts[node.state] += 1

        # A state already on its own ancestor path closes a cycle.
        ancestor_states = set()
        cursor = node.parent
        while cursor is not None:
            ancestor_states.add(cursor.state)
            cursor = cursor.parent
        if check_cycle and node.state in ancestor_states:
            cycle_skips += 1
            record_frame(node.parent)
            continue

        if node.state == goal:
            answer = _result(node, reached, expanded, generated, max_frontier, maze_anim)
            answer.update({
                "status": "FOUND", "iterations": iterations,
                "unique_popped": len(popped_counts),
                "repeated_pops": repeated_pops,
                "duplicate_encounters": duplicate_encounters,
                "cycle_skips": cycle_skips,
                "final_frontier": len(stack),
            })
            return answer
        if limit is not None and node.depth >= limit:
            record_frame(node)
            continue
        expanded += 1
        children = []
        reprioritized = []
        for action, state in _successors(maze, node.state):
            child = _Node(state, node, action, node.cost + 1)
            if (use_depth_reached
                    and child.depth >= best_depth.get(state, float("inf"))):
                continue
            if use_depth_reached:
                best_depth[state] = child.depth
            if frontier_counts.get(state, 0):
                duplicate_encounters += 1
                if frontier_option == 2:
                    continue
                if frontier_option == 1:
                    existing = frontier_nodes[state]
                    stack.remove(existing)
                    reprioritized.append(existing)
                    continue
            children.append(child)
            reached[state] = child
            generated += 1
        # A stack reverses insertion order; reverse here so the configured first
        # direction is explored first.
        for child in reversed(children):
            stack.append(child)
            frontier_counts[child.state] += 1
            frontier_nodes[child.state] = child
        # These states were already waiting in the stack.  Place them last so
        # they are genuinely the next entries popped by LIFO DFS.
        stack.extend(reprioritized)
        max_frontier = max(max_frontier, len(stack))
        record_frame(node)

    answer = _result(None, reached, expanded, generated, max_frontier, maze_anim)
    answer.update({
        "status": "LIMIT" if iterations >= max_tries else "FAILURE",
        "iterations": iterations,
        "unique_popped": len(popped_counts),
        "repeated_pops": repeated_pops,
        "duplicate_encounters": duplicate_encounters,
        "cycle_skips": cycle_skips,
        "final_frontier": len(stack),
    })
    return answer


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
animate_maze = mh.animate_maze


def min_index(values):
    return min(range(len(values)), key=values.__getitem__)
