"""Calculate general statistics on graphs."""
import logging
from pathlib import Path
import networkx as nx
import pandas as pd

logging.basicConfig(level=logging.INFO)


def compute_general_statistics(graph: nx.Graph):
    stats_dict = {}
    stats_dict["nb_nodes"] = len(list(graph.nodes()))
    stats_dict["nb_edges"] = len(list(graph.edges()))
    logging.info("[CALCULATE_STATS] Calculating density ...")
    stats_dict["density"] = nx.density(graph)
    logging.info("[CALCULATE_STATS] Calculating transitivity ...")
    stats_dict["transitivity"] = nx.transitivity(graph)
    logging.info("[CALCULATE_STATS] Calculating average clustering ...")
    stats_dict["average_clustering_coeff"] = nx.average_clustering(graph)
    logging.info("[CALCULATE_STATS] Calculating diameter ...")
    stats_dict["diameter"] = nx.diameter(graph)
    logging.info("[CALCULATE_STATS] Calculating radius ...")
    stats_dict["radius"] = nx.radius(graph)
    return stats_dict


def save_stats_to_csv(stats_dict: dict, output_path: Path):
    logging.info("Saving the statistics to : %s", str(output_path))
    stats_series = pd.Series(list(stats_dict.values()), index=list(stats_dict.keys()))
    stats_series.to_csv(output_path)
