"""Entry point of the package."""
import click
import logging
from typing import Optional
from pathlib import Path
import networkx as nx
import pandas as pd
import numpy as np
import graph_utils.load_data as ld
import graph_utils.centrality as cent
import graph_utils.diffusion as diff
import graph_utils.calculate_stats as cs
from graph_utils import graph_dash

logging.basicConfig(level=logging.INFO)


@click.group()
def main():
    pass


@main.command()
@click.option(
    "--graph_path",
    type=Path,
    help="Path to list of the edges of the graph.",
    required=True,
)
@click.option(
    "--type-graph",
    type=click.Choice(["undirected", "directed", "multi-undirected", "multi-directed"]),
    default="undirected",
    help="Type of the graph.",
    show_default=True,
)
@click.option(
    "--output-path",
    "-o",
    type=Path,
    default=Path("./centrality_measures.csv"),
    help="Path to the file where to save the centrality values.",
    show_default=True,
)
def centrality(graph_path: Path, type_graph: str, output_path: Path):
    """Calculate centrality measures of a graph."""
    type_graph = ld.type_graph_to_int(type_graph)
    graph = ld.load_graph(graph_path, delimiter=",", type_graph=type_graph)
    centralities_dict = cent.calculate_centralities(graph)
    cent.save_centralities_to_csv(centralities_dict, output_path=output_path)


@main.command()
@click.option(
    "--graph_path",
    type=Path,
    help="Path to list of the edges of the graph.",
    required=True,
)
@click.option(
    "--type-graph",
    type=click.Choice(["undirected", "directed", "multi-undirected", "multi-directed"]),
    default="undirected",
    help="Type of the graph.",
    show_default=True,
)
@click.option(
    "--centrality-path",
    type=Path,
    default=None,
    help="Path to the file containing the centrality measures.",
    show_default=True,
)
@click.option(
    "--diffusion-model",
    type=click.Choice(["ltm", "icm"]),
    help="Diffusion model.",
    default="icm",
    show_default=True,
)
@click.option(
    "--maximize-influence",
    is_flag=True,
    help="Use the influence maximization heuristic only.",
    default=False,
    show_default=True,
)
@click.option(
    "--seeds-selection",
    type=str,
    help="Selection mode of the seeds.",
    default="random",
    show_default=True,
)
@click.option(
    "--nb-runs", type=int, default=5, help="Number of runs.", show_default=True
)
@click.option(
    "--output-path",
    "-o",
    type=Path,
    default=Path("./diffusion.csv"),
    help="Path to the file where to save the diffusion results.",
)
def diffusion(
    graph_path: Path,
    type_graph: str,
    output_path: Path,
    diffusion_model: str,
    seeds_selection: str,
    centrality_path: Path = None,
    nb_runs: int = 5,
    maximize_influence: bool = False,
):
    """Apply a diffusion model on the graph."""
    type_graph = ld.type_graph_to_int(type_graph)
    graph = ld.load_graph(graph_path, delimiter=",", type_graph=type_graph)
    adj_matrix = nx.to_numpy_array(graph)
    if not maximize_influence:
        nb_activated_nodes = []
        std_activated_nodes = []
        nb_seeds = np.linspace(1, 30, 20, dtype=int)
        centrality_df = pd.read_csv(centrality_path) if centrality_path else None
        logging.info("Seeds selection mode: %s", seeds_selection)
        seeds_selection = diff.seeds_selection_to_int(seeds_selection)
        for i in nb_seeds:
            logging.info("Running diffusion with %d seeds.", i)
            mean_nb_activated, std_nb_activated = diff.simulate_diffusion(
                adj_matrix,
                nb_nodes=i,
                model=diffusion_model,
                seeds_selection=seeds_selection,
                centrality_df=centrality_df,
                nb_runs=nb_runs,
            )
            nb_activated_nodes.append(mean_nb_activated)
            std_activated_nodes.append(std_nb_activated)
        diffusion_df = pd.DataFrame(
            {
                "nb_seeds": nb_seeds,
                "nb_activated_nodes": nb_activated_nodes,
                "std_activated_nodes": std_activated_nodes,
                "nb_runs": [nb_runs for _ in range(20)],
            }
        )
    else:
        _, nb_activated_nodes, std_activated_nodes = diff.maximize_influence(
            model=diffusion_model,
            adj_matrix=adj_matrix,
            max_nb_nodes=30,
            nb_runs=nb_runs,
        )
        diffusion_df = pd.DataFrame(
            {
                "nb_seeds": [i + 1 for i in range(30)],
                "nb_activated_nodes": nb_activated_nodes,
                "std_activated_nodes": std_activated_nodes,
                "nb_runs": [nb_runs for _ in range(30)],
            }
        )
    logging.info("Saving the results to %s", str(output_path))
    diffusion_df.to_csv(output_path, index=False)


@main.command()
@click.option(
    "--graph_path",
    type=Path,
    help="Path to list of the edges of the graph.",
    required=True,
)
@click.option(
    "--type-graph",
    type=click.Choice(["undirected", "directed", "multi-undirected", "multi-directed"]),
    default="undirected",
    help="Type of the graph.",
)
@click.option(
    "--output-path",
    "-o",
    type=Path,
    default=Path("./general_stats.csv"),
    help="Path to the file where to save the general statistics.",
)
def general_graph_stats(graph_path: Path, type_graph: str, output_path: Path):
    """Compute general statistics on a graph."""
    type_graph = ld.type_graph_to_int(type_graph)
    graph = ld.load_graph(graph_path, delimiter=",", type_graph=type_graph)
    stats_dict = cs.compute_general_statistics(graph)
    cs.save_stats_to_csv(stats_dict, output_path)


@main.command()
@click.option(
    "--graph_path",
    type=Path,
    help="Path to list of the edges of the graph.",
    required=True,
)
@click.option(
    "--type-graph",
    type=click.Choice(["undirected", "directed", "multi-undirected", "multi-directed"]),
    default="undirected",
    help="Type of the graph.",
)
@click.option("--centrality-path", type=Path, help="Path to the centrality measures.")
def dash(graph_path: Path, type_graph: str, centrality_path: Path):
    """Create a dashboard with graph analytics."""
    type_graph = ld.type_graph_to_int(type_graph)
    graph_dash.make_dash(
        graph_path=graph_path, type_graph=type_graph, analytics_path=centrality_path
    )


if __name__ == "__main__":
    main()
