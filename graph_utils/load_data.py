"""Load graphs from different sources."""
from pathlib import Path
import networkx as nx
import numpy as np

UNDIRECTED = 1
DIRECTED = 2
MULTI_UNDIRECTED = 3
MULTI_DIRECTED = 4


np.random.seed(42)


def type_graph_to_int(type_graph: str):
    if type_graph == "undirected":
        return UNDIRECTED
    elif type_graph == "directed":
        return DIRECTED
    elif type_graph == "multi-undirected":
        return MULTI_UNDIRECTED
    elif type_graph == "multi-directed":
        return MULTI_DIRECTED


def load_graph(filepath: Path, delimiter: str, type_graph: int):
    with open(filepath, "r", encoding="utf-8") as file:
        lines = file.read().split("\n")
    if type_graph == UNDIRECTED:
        graph = nx.parse_edgelist(
            lines, delimiter=delimiter, nodetype=int, create_using=nx.Graph
        )
    elif type_graph == DIRECTED:
        graph = nx.parse_edgelist(
            lines, delimiter=delimiter, nodetype=int, create_using=nx.DiGraph
        )
    elif type_graph == MULTI_UNDIRECTED:
        graph = nx.parse_edgelist(
            lines, delimiter=delimiter, nodetype=int, create_using=nx.MultiGraph
        )
    elif type_graph == MULTI_DIRECTED:
        graph = nx.parse_edgelist(
            lines, delimiter=delimiter, nodetype=int, create_using=nx.MultiDiGraph
        )
    return graph


def extract_subgraph(graph: nx.Graph, nb_nodes: int):
    source = np.random.choice(np.array(graph.nodes()))
    nodes = []
    while len(nodes) < nb_nodes:
        new_nodes = list(nx.bfs_successors(graph, source=source, depth_limit=1))[0][1]
        for new in new_nodes:
            if not new in nodes:
                nodes.append(new)
        source = np.random.choice(new_nodes)
    return nx.subgraph(graph, nodes)


def save_subgraph(
    graph: nx.Graph, output_path: Path = Path("./subgraph_edgeslist.csv")
):
    with open(output_path, "wb") as file:
        nx.write_edgelist(graph, file, delimiter=",")
