from pathlib import Path
import networkx as nx
import numpy as np
from graph_utils import load_data as ld
from graph_utils.diffusion import (
    independent_cascade_model,
    linear_threshold_model,
    maximize_influence,
)

TEST_GRAPH_PATH = Path("./data/test_data/toy_graph.csv")


def test_icm_single_seed():
    graph = ld.load_graph(TEST_GRAPH_PATH, delimiter=",", type_graph=ld.UNDIRECTED)
    adj_matrix = nx.to_numpy_array(graph)
    informed_nodes = independent_cascade_model(adj_matrix, seeds=np.array([0]))
    assert len(informed_nodes) > 0


def test_icm_random_seeds():
    graph = ld.load_graph(TEST_GRAPH_PATH, delimiter=",", type_graph=ld.UNDIRECTED)
    adj_matrix = nx.to_numpy_array(graph)
    random_seeds = np.random.choice(np.arange(adj_matrix.shape[0]), 2, replace=False)
    informed_nodes = independent_cascade_model(adj_matrix, seeds=random_seeds)
    assert len(informed_nodes) > 0


def test_ltm_single_seed():
    graph = ld.load_graph(TEST_GRAPH_PATH, delimiter=",", type_graph=ld.UNDIRECTED)
    adj_matrix = nx.to_numpy_array(graph)
    informed_nodes = linear_threshold_model(adj_matrix, seeds=np.array([0]))
    assert len(informed_nodes) > 0


def test_ltm_random_seeds():
    graph = ld.load_graph(TEST_GRAPH_PATH, delimiter=",", type_graph=ld.UNDIRECTED)
    adj_matrix = nx.to_numpy_array(graph)
    random_seeds = np.random.choice(np.arange(adj_matrix.shape[0]), 2, replace=False)
    informed_nodes = linear_threshold_model(adj_matrix, seeds=random_seeds)
    assert len(informed_nodes) > 0


def test_icm_influence_maximization():
    graph = ld.load_graph(TEST_GRAPH_PATH, delimiter=",", type_graph=ld.UNDIRECTED)
    adj_matrix = nx.to_numpy_array(graph)
    best_nodes, _, _ = maximize_influence(
        model="icm", adj_matrix=adj_matrix, max_nb_nodes=2, nb_runs=5
    )
    assert len(best_nodes) == 2
