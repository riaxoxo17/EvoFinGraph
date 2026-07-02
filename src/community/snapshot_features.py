"""
Snapshot Feature Extraction.

Extracts structural and behavioral features for a single
community within a single temporal graph snapshot.
"""

import numpy as np
import networkx as nx
import pandas as pd


class SnapshotFeatureExtractor:
    """
    Extracts snapshot-level features from communities.

    Parameters
    ----------
    graph : networkx.Graph
        Graph corresponding to one timestep.
    """

    def __init__(self, graph):

        self.graph = graph

    # --------------------------------------------------
    # Utility
    # --------------------------------------------------

    def get_nodes(
        self,
        community_id,
    ):
        """
        Return nodes belonging to a community.
        """

        return [

            node

            for node, attrs in self.graph.nodes(data=True)

            if attrs["community"] == community_id

        ]

    # --------------------------------------------------
    # Structural Features
    # --------------------------------------------------

    def compute_size(
        self,
        community_id,
    ):
        """
        Community size.
        """

        return len(
            self.get_nodes(
                community_id
            )
        )

    def compute_density(
        self,
        community_id,
    ):
        """
        Community density.
        """

        nodes = self.get_nodes(
            community_id
        )

        if len(nodes) < 2:

            return 0.0

        return nx.density(
            self.graph.subgraph(nodes)
        )

    def compute_average_degree(
        self,
        community_id,
    ):
        """
        Average node degree.
        """

        nodes = self.get_nodes(
            community_id
        )

        if not nodes:

            return 0.0

        degrees = [

            self.graph.degree(node)

            for node in nodes

        ]

        return float(
            np.mean(degrees)
        )

    def compute_average_clustering(
        self,
        community_id,
    ):
        """
        Average clustering coefficient.
        """

        nodes = self.get_nodes(
            community_id
        )

        if len(nodes) < 2:

            return 0.0

        return nx.average_clustering(
            self.graph.subgraph(nodes)
        )

    # --------------------------------------------------
    # Fraud Features
    # --------------------------------------------------

    def compute_fraud_ratio(
        self,
        community_id,
    ):
        """
        Fraction of fraudulent transactions.
        """

        nodes = self.get_nodes(
            community_id
        )

        if not nodes:

            return 0.0

        labels = [

            self.graph.nodes[node]["label"]

            for node in nodes

        ]

        fraud = sum(

            label == 1

            for label in labels

        )

        return fraud / len(nodes)

    # --------------------------------------------------
    # Feature Space
    # --------------------------------------------------

    def compute_feature_matrix(
        self,
        community_id,
    ):
        """
        Return feature matrix for a community.
        """

        nodes = self.get_nodes(
            community_id
        )

        if not nodes:

            return np.empty((0, 0))

        return np.array(

            [

                self.graph.nodes[node]["features"]

                for node in nodes

            ]

        )

    def compute_feature_centroid(
        self,
        community_id,
    ):
        """
        Mean feature vector.
        """

        matrix = self.compute_feature_matrix(
            community_id
        )

        if matrix.size == 0:

            return None

        return np.mean(
            matrix,
            axis=0,
        )

    def compute_feature_variance(
        self,
        community_id,
    ):
        """
        Variance of feature vectors.
        """

        matrix = self.compute_feature_matrix(
            community_id
        )

        if matrix.size == 0:

            return None

        return np.var(
            matrix,
            axis=0,
        )

    # --------------------------------------------------
    # Summary Statistics
    # --------------------------------------------------

    def compute_centroid_norm(
        self,
        community_id,
    ):
        """
        L2 norm of centroid.
        """

        centroid = self.compute_feature_centroid(
            community_id
        )

        if centroid is None:

            return 0.0

        return float(
            np.linalg.norm(
                centroid
            )
        )

    def compute_variance_mean(
        self,
        community_id,
    ):
        """
        Mean feature variance.
        """

        variance = self.compute_feature_variance(
            community_id
        )

        if variance is None:

            return 0.0

        return float(
            np.mean(
                variance
            )
        )

    def compute_variance_std(
        self,
        community_id,
    ):
        """
        Standard deviation of feature variance.
        """

        variance = self.compute_feature_variance(
            community_id
        )

        if variance is None:

            return 0.0

        return float(
            np.std(
                variance
            )
        )

    # --------------------------------------------------
    # Main Extraction
    # --------------------------------------------------

    def extract(
        self,
        community_id,
    ):
        """
        Extract all snapshot features.
        """

        return {

            "community":

                community_id,

            "timestep":

                self.graph.graph["timestep"],

            "size":

                self.compute_size(
                    community_id
                ),

            "density":

                self.compute_density(
                    community_id
                ),

            "average_degree":

                self.compute_average_degree(
                    community_id
                ),

            "average_clustering":

                self.compute_average_clustering(
                    community_id
                ),

            "fraud_ratio":

                self.compute_fraud_ratio(
                    community_id
                ),

            "feature_centroid":

                self.compute_feature_centroid(
                    community_id
                ),

            "feature_variance":

                self.compute_feature_variance(
                    community_id
                ),

            "centroid_norm":

                self.compute_centroid_norm(
                    community_id
                ),

            "variance_mean":

                self.compute_variance_mean(
                    community_id
                ),

            "variance_std":

                self.compute_variance_std(
                    community_id
                ),

        }
        # --------------------------------------------------
    # Extract All Communities
    # --------------------------------------------------

    def extract_all(self):
        """
        Extract snapshot features for every community
        in the current graph snapshot.

        Returns
        -------
        pandas.DataFrame
            One row per community.
        """

        from src.community.matcher import CommunityMatcher

        communities = CommunityMatcher.get_communities(
            self.graph
        )

        records = []

        for community_id in sorted(communities.keys()):

            features = self.extract(
                community_id
            )

            records.append(
                features
            )

        return pd.DataFrame(
            records
        )