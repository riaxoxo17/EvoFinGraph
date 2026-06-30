"""
Community detection using Louvain.
"""

import community as community_louvain
import networkx as nx

from src.logger import logger


class CommunityDetector:

    @staticmethod
    def detect(graph):

        undirected = graph.to_undirected()

        partition = community_louvain.best_partition(
            undirected,
            random_state=42
        )

        nx.set_node_attributes(
            graph,
            partition,
            "community"
        )

        logger.info(
            f"Detected {len(set(partition.values()))} communities."
        )

        return partition