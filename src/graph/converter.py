"""
Graph Converter.

Converts NetworkX temporal transaction graphs into
PyTorch Geometric Data objects.
"""

import torch
from torch_geometric.data import Data


class GraphConverter:
    """
    Converts NetworkX graphs to PyTorch Geometric.
    """

    @staticmethod
    def to_pyg(
        graph,
        bidirectional=True,
    ):
        """
        Convert a NetworkX graph into a PyTorch Geometric
        Data object.

        Parameters
        ----------
        graph : networkx.Graph
            Input graph.

        bidirectional : bool
            Whether to add reverse edges.

        Returns
        -------
        Data
            PyTorch Geometric graph.
        """

    

        # --------------------------------------------------
        # Node Mapping
        # --------------------------------------------------

        nodes = list(graph.nodes())

        node_mapping = {

            node: index

            for index, node in enumerate(nodes)

        }

        # --------------------------------------------------
        # Edge Index
        # --------------------------------------------------

        edges = []

        for source, target in graph.edges():

            source_index = node_mapping[source]

            target_index = node_mapping[target]

            edges.append([

                source_index,

                target_index,

            ])

            if bidirectional:

                edges.append([

                    target_index,

                    source_index,

                ])

        if len(edges):

            edge_index = (

                torch.tensor(
                    edges,
                    dtype=torch.long,
                )

                .t()

                .contiguous()

            )

        else:

            edge_index = torch.empty(

                (2, 0),

                dtype=torch.long,

            )

        # --------------------------------------------------
        # Node Features
        # --------------------------------------------------

        x = torch.tensor(

            [

                graph.nodes[node]["features"]

                for node in nodes

            ],

            dtype=torch.float,

        )

        # --------------------------------------------------
        # Labels
        # --------------------------------------------------

        y = torch.tensor(

            [

                graph.nodes[node]["label"]

                for node in nodes

            ],

            dtype=torch.long,

        )

        # --------------------------------------------------
        # Community IDs
        # --------------------------------------------------

        community = torch.tensor(

            [

                graph.nodes[node]["community"]

                for node in nodes

            ],

            dtype=torch.long,

        )

        # --------------------------------------------------
        # Build Data Object
        # --------------------------------------------------

        data = Data(

            x=x,

            edge_index=edge_index,

            y=y,

        )

        # --------------------------------------------------
        # Extra Metadata
        # --------------------------------------------------

        data.community = community

        data.node_mapping = node_mapping

        data.reverse_mapping = {

            value: key

            for key, value in node_mapping.items()

        }

        data.timestep = graph.graph.get(

            "timestep",

            None,

        )

        # --------------------------------------------------
        # Community Statistics
        # --------------------------------------------------

        data.num_communities = len(

            torch.unique(
                community
            )

        )

        return data
    # --------------------------------------------------
    # Convert Multiple Graphs
    # --------------------------------------------------

    @staticmethod
    def convert_all(
        graphs,
        bidirectional=True,
    ):
        """
        Convert all NetworkX graph snapshots into
        PyTorch Geometric Data objects.

        Parameters
        ----------
        graphs : dict
            Dictionary mapping timestep -> NetworkX graph.

        bidirectional : bool
            Whether to add reverse edges.

        Returns
        -------
        dict
            Dictionary mapping timestep -> PyG Data object.
        """

        converted = {}

        for timestep in sorted(graphs.keys()):

            converted[timestep] = GraphConverter.to_pyg(

                graphs[timestep],

                bidirectional=bidirectional,

            )

        return converted
    
    # --------------------------------------------------
    # Convert to Ordered List
    # --------------------------------------------------

    @staticmethod
    def to_list(
        graphs,
        bidirectional=True,
    ):
        """
        Convert graph snapshots into an ordered list
        of PyTorch Geometric Data objects.
        """

        converted = GraphConverter.convert_all(
            graphs,
            bidirectional=bidirectional,
        )

        return [

            converted[timestep]

            for timestep in sorted(converted.keys())

        ]
