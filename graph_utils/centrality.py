"""Calculate centrality measures on graphs."""
import logging
from pathlib import Path
import networkx as nx
import pandas as pd

BETWEENNESS = 1
DEGREE = 2
PAGERANK = 3
EIGENVECTOR = 4
CLOSENESS = 5


def calculate_degrees(graph: nx.Graph):
    logging.info("Calculating degrees ...")
    nodes = graph.nodes()
    return {node: len(list(graph.neighbors(node))) for node in nodes}


def calculate_centrality(graph: nx.Graph, centrality: int):
    if centrality == BETWEENNESS:
        logging.info("Calculating betweenness centrality ...")
        return nx.betweenness_centrality(graph)
    elif centrality == DEGREE:
        logging.info("Calculating degree centrality ...")
        return nx.degree_centrality(graph)
    elif centrality == PAGERANK:
        logging.info("Calculating pagerank centrality ...")
        return nx.pagerank(graph)
    elif centrality == EIGENVECTOR:
        logging.info("Calculating eigenvector centrality ...")
        return nx.eigenvector_centrality_numpy(graph)
    elif centrality == CLOSENESS:
        logging.info("Calculating closeness centraliyty ...")
        return nx.closeness_centrality(graph)


def calculate_centralities(graph: nx.Graph) -> dict:
    eigenvector_centrality = calculate_centrality(graph=graph, centrality=EIGENVECTOR)
    degree_centrality = calculate_centrality(graph=graph, centrality=DEGREE)
    pagerank_centrality = calculate_centrality(graph=graph, centrality=PAGERANK)
    closeness_centrality = calculate_centrality(graph=graph, centrality=CLOSENESS)
    betweenness_centrality = calculate_centrality(graph, centrality=BETWEENNESS)
    degrees_nodes = calculate_degrees(graph=graph)
    nodes = list(graph.nodes())
    return {
        "nodes": nodes,
        "degrees_nodes": [degrees_nodes[node] for node in nodes],
        "degree_centrality": [degree_centrality[node] for node in nodes],
        "pagerank_centrality": [pagerank_centrality[node] for node in nodes],
        "eigenvector_centrality": [eigenvector_centrality[node] for node in nodes],
        "closeness_centrality": [closeness_centrality[node] for node in nodes],
        "betweenness_centrality": [betweenness_centrality[node] for node in nodes],
    }


def save_centralities_to_csv(centralities: dict, output_path: Path) -> None:
    """Save the analytics to a csv file.

    Args:
        analytics (dict): statistics calculated on the graph
        output_path (Path): path to the output csv file
    """
    logging.info("Saving the centralities to: %s", str(output_path))
    centralities_df = pd.DataFrame(centralities)
    centralities_df.to_csv(output_path, index=False)
