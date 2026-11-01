"""Stage 2 graph-search alternatives with identical observations and path following."""

from src.agents.astar_agent import AStarAgent
from src.agents.graph_search import breadth_first_search, dijkstra_search


class BFSAgent(AStarAgent):
    agent_id = "stage2_bfs"
    heuristic_name = "none_breadth_first"
    search_method = "bfs"
    objective = "minimum_edge_count"

    def search_graph(self, graph):
        return breadth_first_search(graph, self.config["max_expansions"])


class DijkstraAgent(AStarAgent):
    agent_id = "stage2_dijkstra"
    heuristic_name = "none_uniform_cost"
    search_method = "dijkstra"

    def search_graph(self, graph):
        return dijkstra_search(graph, self.config["max_expansions"])
