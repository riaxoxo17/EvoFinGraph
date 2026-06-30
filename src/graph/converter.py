"""
Convert NetworkX graphs to PyTorch Geometric Data objects.
"""

import torch
from torch_geometric.data import Data


class GraphConverter:

    @staticmethod
    def to_pyg(graph):

        nodes = list(graph.nodes())

        node_mapping = {
            node: idx
            for idx, node in enumerate(nodes)
        }

        edge_index = []

        for source, target in graph.edges():

            edge_index.append([
                node_mapping[source],
                node_mapping[target]
            ])

        edge_index = (
            torch.tensor(edge_index)
            .t()
            .contiguous()
        )

        x = torch.tensor([
            graph.nodes[node]["features"]
            for node in nodes
        ], dtype=torch.float)

        y = torch.tensor([
            graph.nodes[node]["label"]
            for node in nodes
        ])

        return Data(
            x=x,
            edge_index=edge_index,
            y=y
        )