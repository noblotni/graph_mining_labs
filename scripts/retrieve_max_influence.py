import json
from pathlib import Path
import networkx as nx
from graph_utils import diffusion
from graph_utils import load_data as ld
import pandas as pd
import numpy as np

with open(
    "./data/lastfm_asia/lastfm_asia_ltm_max_nodes.json", "r", encoding="utf-8"
) as f:
    best_nodes = np.array(json.load(f)["best_nodes"])
graph = ld.load_graph(
    Path("./data/lastfm_asia/lastfm_asia_edges.csv"),
    delimiter=",",
    type_graph=ld.UNDIRECTED,
)
nb_seeds = [i + 1 for i in range(len(best_nodes))]
nb_activateds_nodes = []
std_activated_nodes = []

adj_matrix = nx.to_numpy_array(graph)
for i in range(len(best_nodes)):
    mean, std = diffusion.run_n_diffusions(
        adj_matrix, model="ltm", seeds=best_nodes[: i + 1], nb_runs=20
    )
    nb_activateds_nodes.append(mean)
    std_activated_nodes.append(std)

diffusion_df = pd.DataFrame(
    {
        "nb_seeds": nb_seeds,
        "nb_activated_nodes": nb_activateds_nodes,
        "std_activated_nodes": std_activated_nodes,
        "nb_runs": [20 for _ in range(len(best_nodes))],
    }
)
diffusion_df.to_csv("./diffusion_lastfm_ltm_influence.csv", index=False)
