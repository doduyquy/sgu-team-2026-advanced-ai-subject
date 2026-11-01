"""Breadth-first and uniform-cost graph search on the shared public route lattice."""

from collections import deque
import heapq
import math

from src.agents.planning_graph import PlanningGraph, SearchResult


def _validate(graph, max_expansions):
    if type(max_expansions) is not int or max_expansions < 0:
        raise ValueError("max_expansions must be a nonnegative integer")
    size = len(graph.points)
    if len(graph.edges) != size or not 0 <= graph.goal < size:
        raise ValueError("Invalid graph size or goal")
    for edges in graph.edges:
        for neighbor, cost in edges:
            if not 0 <= neighbor < size or not math.isfinite(cost) or cost < 0:
                raise ValueError("Graph edges need valid nodes and finite nonnegative costs")


def _found(node, parents, cost, expansions):
    path = [node]
    while node != 0:
        node = parents[node]
        path.append(node)
    return SearchResult(tuple(reversed(path)), float(cost), expansions, True)


def breadth_first_search(graph: PlanningGraph, max_expansions=256) -> SearchResult:
    """Minimize edge count. Report the actual weighted cost of the selected path."""
    _validate(graph, max_expansions)
    frontier = deque([0])
    parents, costs = {}, {0: 0.0}
    expansions = 0
    while frontier and expansions < max_expansions:
        node = frontier.popleft()
        expansions += 1
        if node == graph.goal:
            return _found(node, parents, costs[node], expansions)
        for neighbor, cost in graph.edges[node]:
            if neighbor not in costs:
                parents[neighbor] = node
                costs[neighbor] = costs[node] + cost
                frontier.append(neighbor)
    return SearchResult((), float("inf"), expansions, False)


def dijkstra_search(graph: PlanningGraph, max_expansions=256) -> SearchResult:
    """Expand the lowest accumulated cost, without a heuristic."""
    _validate(graph, max_expansions)
    frontier = [(0.0, 0)]
    parents, costs = {}, {0: 0.0}
    expansions = 0
    while frontier and expansions < max_expansions:
        cost, node = heapq.heappop(frontier)
        if cost > costs[node] + 1e-10:
            continue
        expansions += 1
        if node == graph.goal:
            return _found(node, parents, cost, expansions)
        for neighbor, edge_cost in graph.edges[node]:
            candidate = cost + edge_cost
            if candidate + 1e-10 < costs.get(neighbor, float("inf")):
                costs[neighbor] = candidate
                parents[neighbor] = node
                heapq.heappush(frontier, (candidate, neighbor))
    return SearchResult((), float("inf"), expansions, False)
