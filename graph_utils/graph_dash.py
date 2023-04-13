"""Make a dashboard about a graph."""
import argparse
from pathlib import Path
import webbrowser
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go
from dash import html, dcc, Dash
import pandas as pd
from graph_utils.load_data import load_graph, type_graph_to_int

HTTPS_SERVER = "http://127.0.0.1:8050/"


def plot_measure_histogram(analytics_df: pd.DataFrame, measure_name: str):
    fig = px.histogram(analytics_df, x=measure_name, nbins=50)
    return dcc.Graph(id=measure_name + "_hist", figure=fig)


def plot_network_graph(graph: nx.Graph):
    # Nodes positions
    nodes_pos = nx.spring_layout(graph)
    edges_x = []
    edges_y = []
    for edge in graph.edges():
        node1_x, node1_y = nodes_pos[edge[0]]
        edges_x.append(node1_x)
        edges_y.append(node1_y)
        node2_x, node2_y = nodes_pos[edge[1]]
        edges_x.append(node2_x)
        edges_y.append(node2_y)

    edges_trace = go.Scatter(
        x=edges_x,
        y=edges_y,
        line=dict(width=0.5, color="#888"),
        hoverinfo="none",
        mode="lines",
    )

    nodes_x = []
    nodes_y = []
    for node in graph.nodes():
        nodes_x.append(nodes_pos[node][0])
        nodes_y.append(nodes_pos[node][1])

    nodes_trace = go.Scatter(x=nodes_x, y=nodes_y, fillcolor="red", mode="markers")

    figure = go.Figure(
        [edges_trace, nodes_trace],
        layout=go.Layout(
            title="Representation of the graph",
            titlefont_size=16,
            showlegend=False,
            hovermode="closest",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        ),
    )
    return dcc.Graph(id="graph-representation", figure=figure)


def make_dash(graph_path: Path, type_graph: int, analytics_path: Path):
    """Make a dash webpage to display analytics about a graph."""
    graph = load_graph(graph_path, delimiter=",", type_graph=type_graph)
    analytics_df = pd.read_csv(analytics_path, delimiter=",")

    # Init app
    app = Dash(__name__)
    graph_plot = [plot_network_graph(graph)]
    centrality_df = analytics_df.drop(["nodes"], axis=1)
    measure_histograms = [
        plot_measure_histogram(analytics_df, name)
        for name in list(centrality_df.columns)
    ]
    app.layout = html.Div(
        children=[html.H1(children="Graph Dashboard")] + graph_plot + measure_histograms
    )
    webbrowser.get().open(url=HTTPS_SERVER)
    app.run(debug=True)
