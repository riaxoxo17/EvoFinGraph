"""
Community Matching Engine.

Matches communities between consecutive temporal graph snapshots
using the Community Continuity Score (CCS).
"""

from typing import Dict, Set

import networkx as nx

from src.config import (
    COMMUNITY_MATCHING_WEIGHTS,
    COMMUNITY_MATCH_THRESHOLD,
)


class CommunityMatcher:
    """
    Matches communities across consecutive graph snapshots.
    """

    @staticmethod
    def get_communities(graph: nx.Graph) -> Dict[int, Set]:

        communities = {}

        for node, attrs in graph.nodes(data=True):

            cid = attrs["community"]

            communities.setdefault(
                cid,
                set()
            ).add(node)

        return communities

    @staticmethod
    def compute_member_overlap(
        community_a: Set,
        community_b: Set,
    ) -> float:
        """
        Compute Jaccard similarity between two communities.
        """

        intersection = len(community_a & community_b)
        union = len(community_a | community_b)

        if union == 0:
            return 0.0

        return intersection / union

    @staticmethod
    def compute_fraud_ratio_similarity(
        graph_a,
        nodes_a,
        graph_b,
        nodes_b,
    ):
        """
        Compare fraud ratios of two communities.
        """

        fraud_a = sum(
            graph_a.nodes[n]["label"]
            for n in nodes_a
        ) / len(nodes_a)

        fraud_b = sum(
            graph_b.nodes[n]["label"]
            for n in nodes_b
        ) / len(nodes_b)

        return 1 - abs(
            fraud_a - fraud_b
        )

    @staticmethod
    def compute_density_similarity(
        graph_a,
        nodes_a,
        graph_b,
        nodes_b,
    ):
        """
        Compare graph densities of two communities.
        """

        density_a = nx.density(
            graph_a.subgraph(nodes_a)
        )

        density_b = nx.density(
            graph_b.subgraph(nodes_b)
        )

        return 1 - abs(
            density_a - density_b
        )

    @staticmethod
    def compute_average_degree_similarity(
        graph_a,
        nodes_a,
        graph_b,
        nodes_b,
    ):
        """
        Compare average node degrees.
        """

        degree_a = (
            sum(
                graph_a.degree(n)
                for n in nodes_a
            )
            / len(nodes_a)
        )

        degree_b = (
            sum(
                graph_b.degree(n)
                for n in nodes_b
            )
            / len(nodes_b)
        )

        maximum = max(
            degree_a,
            degree_b,
            1
        )

        return 1 - (
            abs(degree_a - degree_b)
            / maximum
        )

    @staticmethod
    def community_similarity(
        graph_a,
        nodes_a,
        graph_b,
        nodes_b,
    ):
        """
        Compute the Community Continuity Score (CCS).
        """

        overlap = CommunityMatcher.compute_member_overlap(
            nodes_a,
            nodes_b,
        )

        fraud = CommunityMatcher.compute_fraud_ratio_similarity(
            graph_a,
            nodes_a,
            graph_b,
            nodes_b,
        )

        density = CommunityMatcher.compute_density_similarity(
            graph_a,
            nodes_a,
            graph_b,
            nodes_b,
        )

        degree = CommunityMatcher.compute_average_degree_similarity(
            graph_a,
            nodes_a,
            graph_b,
            nodes_b,
        )

        score = (
            COMMUNITY_MATCHING_WEIGHTS["member_overlap"] * overlap
            + COMMUNITY_MATCHING_WEIGHTS["fraud_ratio"] * fraud
            + COMMUNITY_MATCHING_WEIGHTS["density"] * density
            + COMMUNITY_MATCHING_WEIGHTS["average_degree"] * degree
        )

        return score

    @staticmethod
    def match(
        graph_a,
        graph_b,
    ):
        """
        Match communities between two consecutive snapshots.
        """

        communities_a = CommunityMatcher.get_communities(
            graph_a
        )

        communities_b = CommunityMatcher.get_communities(
            graph_b
        )

        matches = {}

        for cid_a, nodes_a in communities_a.items():

            best_score = 0.0
            best_match = None

            for cid_b, nodes_b in communities_b.items():

                score = CommunityMatcher.community_similarity(
                    graph_a,
                    nodes_a,
                    graph_b,
                    nodes_b,
                )

                if score > best_score:

                    best_score = score
                    best_match = cid_b

            if (
                best_match is not None
                and best_score >= COMMUNITY_MATCH_THRESHOLD
            ):

                matches[cid_a] = {

                    "community": best_match,

                    "score": round(
                        best_score,
                        4,
                    ),

                }

        return matches