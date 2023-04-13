import networkx as nx
from pathlib import Path
import graph_utils.load_data as ld
import graph_utils.centrality as cent

TEST_GRAPH_PATH = Path("./data/test_data/toy_graph.csv")


def test_calculate_degrees():
    graph = ld.load_graph(TEST_GRAPH_PATH, delimiter=",", type_graph=ld.UNDIRECTED)
    degrees = cent.calculate_degrees(graph)
    assert list(degrees.values()) == [2, 2, 2, 3, 1]


def test_calculate_degree_centrality():
    graph = ld.load_graph(TEST_GRAPH_PATH, delimiter=",", type_graph=ld.UNDIRECTED)
    degree_centrality = cent.calculate_centrality(graph=graph, centrality=cent.DEGREE)
    assert list(degree_centrality.values()) == [1 / 2, 1 / 2, 1 / 2, 3 / 4, 1 / 4]
