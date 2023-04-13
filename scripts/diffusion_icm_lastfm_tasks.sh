python -m graph_utils diffusion --graph_path ./data/lastfm_asia/lastfm_asia_edges.csv --diffusion-model icm --seeds-selection random --nb-runs 100 -o ./diffusion_icm_lastfm_random.csv
python -m graph_utils diffusion --graph_path ./data/lastfm_asia/lastfm_asia_edges.csv --centrality-path ./data/lastfm_asia/centrality_measures.csv --diffusion-model icm --seeds-selection pagerank --nb-runs 100 -o ./diffusion_icm_lastfm_pagerank.csv
python -m graph_utils diffusion --graph_path ./data/lastfm_asia/lastfm_asia_edges.csv --centrality-path ./data/lastfm_asia/centrality_measures.csv --diffusion-model icm --seeds-selection eigenvector --nb-runs 100 -o ./diffusion_icm_lastfm_eigenvector.csv
python -m graph_utils diffusion --graph_path ./data/lastfm_asia/lastfm_asia_edges.csv --centrality-path ./data/lastfm_asia/centrality_measures.csv --diffusion-model icm --seeds-selection closeness --nb-runs 100 -o ./diffusion_icm_lastfm_closeness.csv
python -m graph_utils diffusion --graph_path ./data/lastfm_asia/lastfm_asia_edges.csv --centrality-path ./data/lastfm_asia/centrality_measures.csv --diffusion-model icm --seeds-selection betweenness --nb-runs 100 -o ./diffusion_icm_lastfm_betweenness.csv
python -m graph_utils diffusion --graph_path ./data/lastfm_asia/lastfm_asia_edges.csv --centrality-path ./data/lastfm_asia/centrality_measures.csv --diffusion-model icm --seeds-selection degree --nb-runs 100 -o ./diffusion_icm_lastfm_degree.csv