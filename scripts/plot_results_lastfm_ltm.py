import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import pandas as pd

paths_dict = {
    "pagerank": Path("./data/lastfm_asia/diffusion_ltm_lastfm_pagerank.csv"),
    "degree": Path("./data/lastfm_asia/diffusion_ltm_lastfm_degree.csv"),
    "closeness": Path("./data/lastfm_asia/diffusion_ltm_lastfm_closeness.csv"),
    "betweenness": Path("./data/lastfm_asia/diffusion_ltm_lastfm_betweenness.csv"),
    "eigenvector": Path("./data/lastfm_asia/diffusion_ltm_lastfm_eigenvector.csv"),
    "heuristic_greedy": Path("./data/lastfm_asia/diffusion_ltm_lastfm_influence.csv"),
    "random": Path("./data/lastfm_asia/diffusion_ltm_lastfm_random.csv"),
}
colors_dict = {
    "pagerank": {
        "color": "rgb(136, 100, 200)",
        "fillcolor": "rgba(136, 100, 200, 0.3)",
    },
    "degree": {"color": "rgb(68, 212, 126)", "fillcolor": "rgba(68,212,126,0.3)"},
    "closeness": {
        "color": "rgb(140, 41, 200)",
        "fillcolor": "rgba(140,41,200,0.3)",
    },
    "betweenness": {
        "color": "rgb(222,110, 104)",
        "fillcolor": "rgba(222,110,104,0.3)",
    },
    "eigenvector": {
        "color": "rgb(254, 159, 109)",
        "fillcolor": "rgba(254,159,109, 0.3)",
    },
    "heuristic_greedy": {
        "color": "rgb(222, 255, 104)",
        "fillcolor": "rgba(222,255,104,0.3)",
    },
    "random": {"color": "rgb(80,150,60)", "fillcolor": "rgba(80, 150, 60, 0.3)"},
}
fig = go.Figure()
for key, path in paths_dict.items():
    df = pd.read_csv(path)

    traces = [
        go.Scatter(
            name=key,
            x=df["nb_seeds"],
            y=df["nb_activated_nodes"],
            mode="lines",
            line=dict(color=colors_dict[key]["color"]),
        ),
        go.Scatter(
            x=df["nb_seeds"],
            y=df["nb_activated_nodes"] + df["std_activated_nodes"],
            showlegend=False,
            mode="lines",
            line=dict(width=0),
        ),
        go.Scatter(
            x=df["nb_seeds"],
            y=df["nb_activated_nodes"] - df["std_activated_nodes"],
            fill="tonexty",
            mode="lines",
            showlegend=False,
            line=dict(width=0),
            fillcolor=colors_dict[key]["fillcolor"],
        ),
    ]
    fig.add_traces(traces)
fig.update_layout(
    title=dict(text="Diffusion with Linear Threshold Model", font=dict(size=20)),
    xaxis_title=dict(text="Number of seeds", font=dict(size=20)),
    yaxis_title=dict(text="Number of infected nodes", font=dict(size=20)),
    legend=dict(font=dict(size=20)),
    xaxis=dict(tickfont=dict(size=20)),
    yaxis=dict(tickfont=dict(size=20)),
)
fig.show()
