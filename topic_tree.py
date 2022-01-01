import hdbscan
import numpy as np

data = np.load('clusterable_data.npy')

clusterer = hdbscan.HDBSCAN(min_cluster_size=5, gen_min_span_tree=True)
clusterer.fit(data)

clusterer.minimum_spanning_tree_.plot(edge_cmap='viridis',
                                      edge_alpha=0.6,
                                      node_size=8,
                                      edge_linewidth=2)