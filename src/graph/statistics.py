"""
Graph statistics utilities.
"""

import networkx as nx
import numpy as np


class GraphStatistics:

    @staticmethod
    def summarize(G):

        degrees = [degree for _, degree in G.degree()]

        return {

            "timestep": G.graph["timestep"],

            "nodes": G.number_of_nodes(),

            "edges": G.number_of_edges(),

            "average_degree": float(np.mean(degrees)),

            "density": nx.density(G),

            "weakly_connected_components":
                nx.number_weakly_connected_components(G),

            "isolated_nodes":
                len(list(nx.isolates(G)))
        }