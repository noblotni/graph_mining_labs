"""Shortest path-related functions."""
import numpy as np
import numba


@numba.njit()
def init_dist_matrix(adj_matrix: np.ndarray):
    dist_matrix = np.full_like(adj_matrix, np.inf)
    np.fill_diagonal(dist_matrix, 0)
    return dist_matrix


@numba.njit()
def single_source_shortest_path(
    source: int, adj_matrix: np.ndarray, dist_matrix: np.ndarray
):
    """Calculate the single source shortest path for an undirected unweighted graph."""
    visited = np.array([False for _ in range(adj_matrix.shape[0])])
    next_to_visit = np.array([source])
    dist = np.zeros(adj_matrix.shape[0])
    while next_to_visit.size > 0:
        next_node = next_to_visit[0]
        next_to_visit = np.delete(next_to_visit, 0)
        visited[next_node] = True
        neighbors = np.where(adj_matrix[source, :] > 0)[0]
        for neighbor in neighbors:
            if not visited[neighbor]:
                next_to_visit = np.append(next_to_visit, neighbor)
                dist[neighbor] = dist[next_node] + 1
                dist_matrix[source, neighbor] = dist[neighbor]
    return dist_matrix


@numba.njit()
def all_pairs_shortest_paths(adj_matrix: np.ndarray):
    """Compute all pairs shortest paths for an undirected graph."""
    dist_matrix = init_dist_matrix(adj_matrix)
    for i in range(adj_matrix.shape[0]):
        dist_matrix = single_source_shortest_path(i, adj_matrix, dist_matrix)
    return dist_matrix
