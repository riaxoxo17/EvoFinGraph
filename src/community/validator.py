"""
Graph validation utilities.
"""

from src.logger import logger


class GraphValidator:

    @staticmethod
    def validate(graph):

        assert graph.number_of_nodes() > 0

        assert graph.number_of_edges() >= 0

        node = next(iter(graph.nodes()))

        attrs = graph.nodes[node]

        assert "features" in attrs

        assert "label" in attrs

        assert "timestep" in attrs

        logger.info(
            f"Snapshot {graph.graph['timestep']} validated."
        )

        return True