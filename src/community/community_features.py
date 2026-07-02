"""
Community Feature Extraction.

Computes temporal evolution features for persistent communities.
"""

import numpy as np
import pandas as pd

from src.community.snapshot_features import SnapshotFeatureExtractor


class CommunityFeatureExtractor:
    """
    Extracts temporal features from persistent communities.

    Parameters
    ----------
    graphs : dict
        Dictionary of timestep -> graph.

    tracker : CommunityTracker

    timeline : CommunityTimeline
    """

    def __init__(
        self,
        graphs,
        tracker,
        timeline,
    ):

        self.graphs = graphs

        self.tracker = tracker

        self.timeline = timeline

        # ------------------------------------------
        # Cache snapshot-level features
        # ------------------------------------------

        self.snapshot_cache = {}

        self.build_snapshot_cache()

    # -------------------------------------------------
    # Snapshot Cache
    # -------------------------------------------------

    def build_snapshot_cache(
        self,
    ):
        """
        Cache snapshot features for every community
        in every graph.

        Cache Key

        (timestep, community_id)
        """

        for timestep, graph in self.graphs.items():

            extractor = SnapshotFeatureExtractor(
                graph
            )

            df = extractor.extract_all()

            for _, row in df.iterrows():

                self.snapshot_cache[
                    (
                        timestep,
                        row["community"],
                    )
                ] = row.to_dict()

    # -------------------------------------------------
    # Snapshot Lookup
    # -------------------------------------------------

    def get_snapshot(
        self,
        timestep,
        community,
    ):
        """
        Return cached snapshot features.
        """

        return self.snapshot_cache.get(

            (
                timestep,
                community,
            ),

            None,

        )

    # -------------------------------------------------
    # Utility
    # -------------------------------------------------

    @staticmethod
    def relative_change(
        previous,
        current,
    ):
        """
        Relative change.

        (current - previous) / previous
        """

        if previous == 0:

            return 0.0

        return (

            current - previous

        ) / previous
    # -------------------------------------------------
    # Growth Rate
    # -------------------------------------------------

    def compute_growth_rate(
        self,
        previous,
        current,
    ):
        """
        Relative community growth.
        """

        return self.relative_change(

            previous["size"],

            current["size"],

        )

    # -------------------------------------------------
    # Density Change
    # -------------------------------------------------

    def compute_density_change(
        self,
        previous,
        current,
    ):
        """
        Relative density change.
        """

        return self.relative_change(

            previous["density"],

            current["density"],

        )

    # -------------------------------------------------
    # Degree Change
    # -------------------------------------------------

    def compute_degree_change(
        self,
        previous,
        current,
    ):
        """
        Relative average-degree change.
        """

        return self.relative_change(

            previous["average_degree"],

            current["average_degree"],

        )

    # -------------------------------------------------
    # Fraud Growth
    # -------------------------------------------------

    def compute_fraud_growth(
        self,
        previous,
        current,
    ):
        """
        Relative fraud-ratio change.
        """

        return self.relative_change(

            previous["fraud_ratio"],

            current["fraud_ratio"],

        )

    # -------------------------------------------------
    # Clustering Change
    # -------------------------------------------------

    def compute_clustering_change(
        self,
        previous,
        current,
    ):
        """
        Relative clustering change.
        """

        return self.relative_change(

            previous["average_clustering"],

            current["average_clustering"],

        )
    # -------------------------------------------------
    # Feature Drift
    # -------------------------------------------------

    def compute_feature_drift(
        self,
        previous,
        current,
    ):
        """
        Euclidean distance between feature centroids.
        """

        return float(

            np.linalg.norm(

                current["feature_centroid"]

                -

                previous["feature_centroid"]

            )

        )
    # -------------------------------------------------
    # Variance Drift
    # -------------------------------------------------

    def compute_variance_drift(
        self,
        previous,
        current,
    ):
        """
        Relative change in behavioral variance.
        """

        return self.relative_change(

            previous["variance_mean"],

            current["variance_mean"],

        )
    # -------------------------------------------------
    # Temporal Feature Extraction
    # -------------------------------------------------

    def compute_temporal_features(
        self,
        previous,
        current,
    ):
        """
        Compute temporal evolution features.
        """

        return {

            "growth_rate":

                self.compute_growth_rate(

                    previous,

                    current,

                ),

            "density_change":

                self.compute_density_change(

                    previous,

                    current,

                ),

            "degree_change":

                self.compute_degree_change(

                    previous,

                    current,

                ),

            "fraud_growth":

                self.compute_fraud_growth(

                    previous,

                    current,

                ),

            "clustering_change":

                self.compute_clustering_change(

                    previous,

                    current,

                ),

            "feature_drift":

                self.compute_feature_drift(

                    previous,

                    current,

                ),

            "variance_drift":

                self.compute_variance_drift(

                    previous,

                    current,

                ),

        }
    # -------------------------------------------------
    # Lifetime
    # -------------------------------------------------

    def compute_lifetime(
        self,
        pcid,
    ):
        """
        Number of snapshots that a persistent
        community exists.
        """

        records = self.timeline.get_timeline(
            pcid
        )

        return len(records)
    # -------------------------------------------------
    # Persistence
    # -------------------------------------------------

    def compute_persistence(
        self,
        pcid,
    ):
        """
        Fraction of snapshots in which the
        community survives.
        """

        lifetime = self.compute_lifetime(
            pcid
        )

        total = len(
            self.graphs
        )

        if total == 0:

            return 0.0

        return lifetime / total
    # -------------------------------------------------
    # Average Matching Score
    # -------------------------------------------------

    def compute_average_match_score(
        self,
        pcid,
    ):
        """
        Average Community Continuity Score (CCS)
        across the lifetime of a community.
        """

        records = self.timeline.get_timeline(
            pcid
        )

        scores = []

        for record in records:

            metadata = self.tracker.get_metadata(

                record["timestep"],

                record["community"],

            )

            scores.append(

                metadata["score"]

            )

        if not scores:

            return 0.0

        return float(

            np.mean(scores)

        )
    # -------------------------------------------------
    # Node Churn
    # -------------------------------------------------

    def compute_node_churn(
        self,
        previous,
        current,
    ):
        """
        Fraction of nodes that joined or left
        between two consecutive snapshots.
        """

        previous_graph = self.graphs[
            previous["timestep"]
        ]

        current_graph = self.graphs[
            current["timestep"]
        ]

        previous_nodes = set(

            SnapshotFeatureExtractor(
                previous_graph
            ).get_nodes(
                previous["community"]
            )

        )

        current_nodes = set(

            SnapshotFeatureExtractor(
                current_graph
            ).get_nodes(
                current["community"]
            )

        )

        if len(previous_nodes) == 0:

            return 0.0

        joined = len(

            current_nodes -
            previous_nodes

        )

        left = len(

            previous_nodes -
            current_nodes

        )

        return (

            joined + left

        ) / len(previous_nodes)
    
    # -------------------------------------------------
    # Structural Stability
    # -------------------------------------------------

    def compute_structural_stability(
        self,
        temporal_features,
    ):
        """
        Stability score.

        Lower variation
        →

        Higher stability.
        """

        values = [

            abs(

                temporal_features["density_change"]

            ),

            abs(

                temporal_features["degree_change"]

            ),

            abs(

                temporal_features["clustering_change"]

            ),

        ]

        return 1 / (

            1 +

            np.mean(values)

        )
    # -------------------------------------------------
    # Stability Features
    # -------------------------------------------------

    def compute_stability_features(
        self,
        pcid,
        previous,
        current,
    ):
        """
        Compute stability metrics.
        """

        temporal = self.compute_temporal_features(

            previous,

            current,

        )

        return {

            "lifetime":

                self.compute_lifetime(
                    pcid
                ),

            "persistence":

                self.compute_persistence(
                    pcid
                ),

            "average_match_score":

                self.compute_average_match_score(
                    pcid
                ),

            "node_churn":

                self.compute_node_churn(

                    previous,

                    current,

                ),

            "structural_stability":

                self.compute_structural_stability(

                    temporal

                ),

        }
    # -------------------------------------------------
    # Extract Temporal Feature Matrix
    # -------------------------------------------------

    def extract_all(
        self,
    ):
        """
        Build the complete temporal feature matrix.

        Returns
        -------
        pandas.DataFrame
            One row per (PCID, timestep).
        """

        records = []

        for pcid in sorted(

            self.timeline.timelines.keys()

        ):

            timeline = self.timeline.get_timeline(
                pcid
            )

            if len(timeline) < 2:

                continue

            for i in range(

                1,

                len(timeline),

            ):

                previous_record = timeline[
                    i - 1
                ]

                current_record = timeline[
                    i
                ]

                previous = self.get_snapshot(

                    previous_record["timestep"],

                    previous_record["community"],

                )

                current = self.get_snapshot(

                    current_record["timestep"],

                    current_record["community"],

                )

                temporal = self.compute_temporal_features(

                    previous,

                    current,

                )

                stability = self.compute_stability_features(

                    pcid,

                    previous,

                    current,

                )

                row = {

                    "pcid": pcid,

                    "timestep": current_record[
                        "timestep"
                    ],

                    "community": current_record[
                        "community"
                    ],

                }

                row.update(
                    temporal
                )

                row.update(
                    stability
                )
                row["is_fraud_community"] = (

                    self.compute_community_label(
                        current
                    )

                )

                records.append(
                    row
                )

        return pd.DataFrame(
            records
        )
    # -------------------------------------------------
    # Community Label
    # -------------------------------------------------

    def compute_community_label(
        self,
        snapshot,
    ):
        """
        Compute the ground-truth label for a community.

        A community is considered fraudulent if it
        contains at least one fraudulent transaction.
        """

        return int(

            snapshot["fraud_ratio"] > 0

        )