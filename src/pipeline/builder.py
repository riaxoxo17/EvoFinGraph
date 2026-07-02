"""
EvoFinGraph Pipeline Builder.

Builds the complete EvoFinGraph pipeline from the
Elliptic Bitcoin Dataset.

Pipeline
--------
Dataset
    ↓
Temporal Graphs
    ↓
Community Tracking
    ↓
Community Timelines
    ↓
Temporal Community Features
"""

from src.config import DEV_SNAPSHOTS

from src.graph.manager import TemporalGraphManager

from src.community.tracker import CommunityTracker
from src.community.timeline import CommunityTimeline
from src.community.community_features import CommunityFeatureExtractor


class EvoFinGraphPipeline:
    """
    Builds the complete EvoFinGraph pipeline.
    """

    def __init__(
        self,
        snapshots=DEV_SNAPSHOTS,
    ):
        self.snapshots = snapshots

        self.manager = None

        self.tracker = None

        self.timeline = None

        self.extractor = None

        self.feature_matrix = None

    # --------------------------------------------------
    # Graph Construction
    # --------------------------------------------------

    def build_graphs(
        self,
    ):
        """
        Build all temporal graph snapshots.
        """

        self.manager = TemporalGraphManager()

        self.manager.build_all(
            self.snapshots
        )

    # --------------------------------------------------
    # Community Tracking
    # --------------------------------------------------

    def build_tracker(
        self,
    ):
        """
        Track communities across snapshots.
        """

        self.tracker = CommunityTracker()

        self.tracker.track(
            self.manager.graphs
        )

    # --------------------------------------------------
    # Community Timeline
    # --------------------------------------------------

    def build_timeline(
        self,
    ):
        """
        Build Persistent Community timelines.
        """

        self.timeline = CommunityTimeline()

        self.timeline.build(

            self.tracker,

            self.manager.graphs,

        )

    # --------------------------------------------------
    # Community Features
    # --------------------------------------------------

    def build_features(
        self,
    ):
        """
        Extract temporal community features.
        """

        self.extractor = CommunityFeatureExtractor(

            self.manager.graphs,

            self.tracker,

            self.timeline,

        )

        self.feature_matrix = (

            self.extractor.extract_all()

        )

    # --------------------------------------------------
    # Build Complete Pipeline
    # --------------------------------------------------

    def build(
        self,
    ):
        """
        Execute the complete EvoFinGraph pipeline.
        """

        self.build_graphs()

        self.build_tracker()

        self.build_timeline()

        self.build_features()

        return self

    # --------------------------------------------------
    # Helpers
    # --------------------------------------------------

    @property
    def graphs(self):
        return self.manager.graphs

    @property
    def communities(self):
        return self.tracker.community_map

    @property
    def timelines(self):
        return self.timeline.timelines

    def summary(
        self,
    ):
        """
        Print pipeline summary.
        """

        print("=" * 60)
        print("EVOFINGRAPH PIPELINE")
        print("=" * 60)

        print(
            f"Snapshots              : {len(self.graphs)}"
        )

        print(
            f"Tracked Communities    : {len(self.timeline.timelines)}"
        )

        print(
            f"Feature Matrix Shape   : {self.feature_matrix.shape}"
        )

        print("=" * 60)
    # --------------------------------------------------
    # Convenience Methods
    # --------------------------------------------------

    def get_graph(
        self,
        timestep,
    ):
        """
        Return a graph snapshot for a timestep.
        """

        return self.manager.get_snapshot(
            timestep
        )

    def get_timeline(
        self,
        pcid,
    ):
        """
        Return the timeline for a Persistent Community.
        """

        return self.timeline.get_timeline(
            pcid
        )

    def get_feature_matrix(
        self,
    ):
        """
        Return the temporal feature matrix.
        """

        return self.feature_matrix

    def get_tracker(
        self,
    ):
        """
        Return the community tracker.
        """

        return self.tracker

    def get_manager(
        self,
    ):
        """
        Return the graph manager.
        """

        return self.manager