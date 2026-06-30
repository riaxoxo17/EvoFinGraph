"""
Tracks communities across temporal graph snapshots.

Assigns Persistent Community IDs (PCIDs) to communities
across consecutive snapshots.
"""

from src.community.matcher import CommunityMatcher


class CommunityTracker:

    def __init__(self):

        # Next available Persistent Community ID
        self.next_pcid = 0

        # (timestep, community_id) -> metadata
        self.community_map = {}

    def initialize(self, graph):
        """
        Initialize tracking using the first graph snapshot.

        Every detected community is assigned a new
        Persistent Community ID.
        """

        timestep = graph.graph["timestep"]

        communities = CommunityMatcher.get_communities(graph)

        for community_id in communities:

            self.community_map[
                (timestep, community_id)
            ] = {

                "pcid": self.next_pcid,

                "previous_community": None,

                "current_community": community_id,

                "score": 1.0

            }

            self.next_pcid += 1

    def update(
        self,
        previous_graph,
        current_graph,
    ):
        """
        Track communities from one snapshot
        to the next.
        """

        previous_time = previous_graph.graph["timestep"]

        current_time = current_graph.graph["timestep"]

        matches = CommunityMatcher.match(
            previous_graph,
            current_graph,
        )

        assigned = set()

        # --------------------------------------------------
        # Assign existing PCIDs to matched communities
        # --------------------------------------------------

        for previous_comm, result in matches.items():

            current_comm = result["community"]

            score = result["score"]

            pcid = self.community_map[
                (previous_time, previous_comm)
            ]["pcid"]

            self.community_map[
                (current_time, current_comm)
            ] = {

                "pcid": pcid,

                "previous_community": previous_comm,

                "current_community": current_comm,

                "score": score

            }

            assigned.add(current_comm)

        # --------------------------------------------------
        # Assign new PCIDs to unmatched communities
        # --------------------------------------------------

        current_communities = CommunityMatcher.get_communities(
            current_graph
        )

        for community_id in current_communities:

            if community_id not in assigned:

                self.community_map[
                    (current_time, community_id)
                ] = {

                    "pcid": self.next_pcid,
                    "previous_community": None,
                    "current_community": community_id,
                    "score": 1.0

                }

                self.next_pcid += 1

    def track(self, graphs):
        """
        Track communities across all graph snapshots.
        """

        timesteps = sorted(graphs.keys())

        if not timesteps:
            return

        self.initialize(
            graphs[timesteps[0]]
        )

        for i in range(1, len(timesteps)):

            self.update(
                graphs[timesteps[i - 1]],
                graphs[timesteps[i]],
            )

    def get_pcid(
        self,
        timestep,
        community_id,
    ):
        """
        Return the Persistent Community ID.
        """

        return self.community_map[
            (timestep, community_id)
        ]["pcid"]

    def get_score(
        self,
        timestep,
        community_id,
    ):
        """
        Return the matching confidence score.
        """

        return self.community_map[
            (timestep, community_id)
        ]["score"]

    def get_metadata(
        self,
        timestep,
        community_id,
    ):
        """
        Return the complete tracking metadata.
        """

        return self.community_map[
            (timestep, community_id)
        ]