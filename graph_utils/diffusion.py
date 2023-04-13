"""Diffusion on graphs."""
from typing import Optional
import logging
import numba
import numpy as np
import pandas as pd
from graph_utils import numba_utils

# Constants
RANDOM_SELECTION = 1
PAGERANK_SELECTION = 2
EIGENVECTOR_SELECTION = 3
CLOSENESS_SELECTION = 4
BETWEENNESS_SELECTION = 5
DEGREE_SELECTION = 6

logging.basicConfig(level=logging.INFO)


def seeds_selection_to_int(seeds_selection: str):
    if seeds_selection == "random":
        return RANDOM_SELECTION
    elif seeds_selection == "pagerank":
        return PAGERANK_SELECTION
    elif seeds_selection == "degree":
        return DEGREE_SELECTION
    elif seeds_selection == "betweenness":
        return BETWEENNESS_SELECTION
    elif seeds_selection == "closeness":
        return DEGREE_SELECTION
    elif seeds_selection == "eigenvector":
        return EIGENVECTOR_SELECTION
    else:
        raise ValueError("Unknown seeds selection mode.")


def random_seeds_selection(adj_matrix: np.ndarray, nb_nodes: int):
    return np.random.choice(np.arange(adj_matrix.shape[0]), nb_nodes)


def centrality_seeds_selection(centrality: np.ndarray, nb_nodes: int):
    nodes_sorted = np.argsort(centrality)
    return nodes_sorted[centrality.shape[0] - nb_nodes : centrality.shape[0]]


@numba.njit()
def independent_cascade_model(adj_matrix: np.ndarray, seeds: np.ndarray):
    active_nodes = seeds.copy()
    new_active = seeds.copy()
    while new_active.size > 0:
        activated_nodes = np.array([-1])
        for node in new_active:
            neighbors = np.where(adj_matrix[node, :] > 0)[0]
            activation_prob = 1 / np.array(
                [np.sum(adj_matrix[ind, :]) for ind in neighbors]
            )
            random_choice = np.random.uniform(0, 1, neighbors.size)
            success = np.array(
                [random_choice[i] < activation_prob[i] for i in range(neighbors.size)]
            )
            activated_nodes = np.append(activated_nodes, np.extract(success, neighbors))
        activated_nodes = np.delete(activated_nodes, 0)
        new_active = numba_utils.set_diff1d(activated_nodes, active_nodes)
        active_nodes = np.append(active_nodes, new_active)
    return active_nodes


@numba.njit()
def get_infectedprop_neighbors(neighbors: np.ndarray, active_nodes: np.ndarray):
    count = np.sum(np.array([node in active_nodes for node in neighbors]))
    return count / neighbors.size


@numba.njit()
def linear_threshold_model(adj_matrix: np.ndarray, seeds: np.ndarray):
    acceptances = np.random.random(adj_matrix.shape[0])
    active_nodes = seeds.copy()
    new_active = seeds.copy()
    while new_active.size > 0:
        activated_nodes = np.array([-1])
        for node_a in new_active:
            neighbors_a = np.where(adj_matrix[node_a, :] > 0)[0]
            for node_u in neighbors_a:
                neighbors_u = np.where(adj_matrix[node_u, :] > 0)[0]
                infected_prop = get_infectedprop_neighbors(
                    neighbors=neighbors_u, active_nodes=active_nodes
                )
                if infected_prop > acceptances[node_u]:
                    activated_nodes = np.append(activated_nodes, np.array([node_u]))
        activated_nodes = np.delete(activated_nodes, 0)
        new_active = numba_utils.set_diff1d(activated_nodes, active_nodes)
        active_nodes = np.append(active_nodes, new_active)
    return active_nodes


def select_seeds(
    mode: int,
    adj_matrix: np.ndarray,
    nb_nodes: int = 5,
    centrality_df: Optional[pd.DataFrame] = None,
):
    if mode == RANDOM_SELECTION:
        seeds = random_seeds_selection(adj_matrix, nb_nodes)
    elif mode == PAGERANK_SELECTION:
        centrality = centrality_df["pagerank_centrality"].to_numpy()
        seeds = centrality_seeds_selection(centrality=centrality, nb_nodes=nb_nodes)
    elif mode == CLOSENESS_SELECTION:
        centrality = centrality_df["closeness_centrality"].to_numpy()
        seeds = centrality_seeds_selection(centrality=centrality, nb_nodes=nb_nodes)
    elif mode == BETWEENNESS_SELECTION:
        centrality = centrality_df["betweenness_centrality"].to_numpy()
        seeds = centrality_seeds_selection(centrality=centrality, nb_nodes=nb_nodes)
    elif mode == EIGENVECTOR_SELECTION:
        centrality = centrality_df["eigenvector_centrality"].to_numpy()
        seeds = centrality_seeds_selection(centrality=centrality, nb_nodes=nb_nodes)
    elif mode == DEGREE_SELECTION:
        centrality = centrality_df["degree_centrality"].to_numpy()
        seeds = centrality_seeds_selection(centrality=centrality, nb_nodes=nb_nodes)
    return seeds


def maximize_influence(
    model: str, adj_matrix: np.ndarray, max_nb_nodes: int, nb_runs: int
):
    best_nodes = np.empty(0, dtype=int)
    counter_best_nodes = 0
    scores_means = []
    scores_stds = []
    while counter_best_nodes < max_nb_nodes:
        best_score = 0
        best_current_node = -1
        for node in range(adj_matrix.shape[0]):
            best_nodes_copy = best_nodes.copy()
            if node not in best_nodes:
                best_nodes_copy = np.append(best_nodes_copy, node)
                logging.info(
                    f"[Diffusion]Number of best nodes: {counter_best_nodes}. Best nodes: {best_nodes}. Running {nb_runs} diffusions."
                )
                score_mean, score_std = run_n_diffusions(
                    adj_matrix=adj_matrix,
                    model=model,
                    seeds=best_nodes_copy,
                    nb_runs=nb_runs,
                )
                if score_mean > best_score:
                    best_current_node = node
                    best_score = score_mean
                    scores_means.append(score_mean)
                    scores_stds.append(score_std)
        best_nodes = np.append(best_nodes, best_current_node)
        logging.info("[Diffusion] Best nodes: {best_nodes}")
        counter_best_nodes += 1
    return best_nodes, scores_means, scores_stds


def run_n_diffusions(
    adj_matrix: np.ndarray, model: str, seeds: np.ndarray, nb_runs: int = 5
):
    nb_informed_nodes = []
    for i in range(nb_runs):
        logging.info("[Diffusion] Run %d", i)
        if model == "icm":
            informed_nodes = independent_cascade_model(
                adj_matrix=adj_matrix,
                seeds=seeds,
            )
        elif model == "ltm":
            informed_nodes = linear_threshold_model(adj_matrix, seeds=seeds)
        nb_informed_nodes.append(len(informed_nodes))
    return np.mean(nb_informed_nodes), np.std(nb_informed_nodes)


def simulate_diffusion(
    adj_matrix: np.ndarray,
    model: str,
    nb_runs: int = 5,
    nb_nodes: int = 5,
    seeds_selection: int = RANDOM_SELECTION,
    centrality_df: Optional[pd.DataFrame] = None,
):
    seeds = select_seeds(
        mode=seeds_selection,
        adj_matrix=adj_matrix,
        nb_nodes=nb_nodes,
        centrality_df=centrality_df,
    )
    logging.info(
        f"[Diffusion] Running {nb_runs} diffusions with {model.upper()} model."
    )
    mean_nb_informed, std_nb_informed = run_n_diffusions(
        adj_matrix=adj_matrix, model=model, seeds=seeds, nb_runs=nb_runs
    )
    return mean_nb_informed, std_nb_informed
