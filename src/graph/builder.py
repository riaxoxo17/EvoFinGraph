"""
builder.py

Builds temporal NetworkX graphs from the processed Elliptic Bitcoin dataset.

Author: Ria Parikh
Project: EvoFinGraph
"""

from typing import Optional

import networkx as nx
import pandas as pd

from src.data.preprocessing import DataPreprocessor
from src.logger import logger


class TemporalGraphBuilder:
    """
    Builds directed transaction graphs for individual time steps.

    Each node represents a Bitcoin transaction.

    Each directed edge represents the flow of Bitcoin
    from one transaction to another.
    """

    def __init__(self):

        logger.info("Loading processed dataset...")

        processor = DataPreprocessor()

        self.df = processor.run()

        self.edges = processor.edges

        self.graph: Optional[nx.DiGraph] = None

        logger.info("Dataset loaded successfully.")

    def build_snapshot(self, timestep: int) -> nx.DiGraph:
        """
        Build a directed NetworkX graph for a single timestep.

        Parameters
        ----------
        timestep : int
            Temporal snapshot number.

        Returns
        -------
        nx.DiGraph
            Graph corresponding to the selected timestep.
        """

        logger.info(f"Building graph for timestep {timestep}")

        snapshot = self.df[
            self.df["time_step"] == timestep
        ]

        if snapshot.empty:
            raise ValueError(
                f"No transactions found for timestep {timestep}"
            )

        feature_columns = [
            column
            for column in snapshot.columns
            if column.startswith("feature_")
        ]

        G = nx.DiGraph()

        G.graph["timestep"] = timestep

        # ---------------------------------------------------
        # Add Nodes
        # ---------------------------------------------------

        for _, row in snapshot.iterrows():

            G.add_node(
                row["txId"],
                label=int(row["class"]),
                timestep=int(row["time_step"]),
                features=row[feature_columns]
                .to_numpy(dtype=float),
            )

        # ---------------------------------------------------
        # Add Edges
        # ---------------------------------------------------

        valid_nodes = set(snapshot["txId"])

        snapshot_edges = self.edges[
            self.edges["txId1"].isin(valid_nodes)
            &
            self.edges["txId2"].isin(valid_nodes)
        ]

        for _, edge in snapshot_edges.iterrows():

            G.add_edge(
                edge["txId1"],
                edge["txId2"],
            )

        self.graph = G

        logger.info(
            f"Graph built successfully | "
            f"Nodes: {G.number_of_nodes()} | "
            f"Edges: {G.number_of_edges()}"
        )

        return G

    def get_graph(self) -> nx.DiGraph:
        """
        Returns the most recently built graph.
        """

        if self.graph is None:
            raise RuntimeError(
                "Graph has not been built yet."
            )

        return self.graph