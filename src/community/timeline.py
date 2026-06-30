"""
Community Timeline Generator.

Builds temporal histories for Persistent Community IDs (PCIDs).
"""

from src.logger import logger


class CommunityTimeline:

    def __init__(self):

        self.timelines = {}

    def build(
        self,
        tracker,
        graphs,
    ):
        """
        Build timelines for all Persistent Community IDs.
        """

        self.timelines = {}

        for (timestep, community), metadata in tracker.community_map.items():

            pcid = metadata["pcid"]

            self.timelines.setdefault(
                pcid,
                []
            ).append({

                "timestep": timestep,

                "graph": graphs[timestep],

                "community": metadata["current_community"],

                "previous_community": metadata["previous_community"],

                "score": metadata["score"]

            })

        for pcid in self.timelines:

            self.timelines[pcid] = sorted(

                self.timelines[pcid],

                key=lambda record: record["timestep"]

            )

        logger.info(
            f"Built {len(self.timelines)} community timelines."
        )

    def get_timeline(
        self,
        pcid,
    ):
        """
        Return the complete timeline for a Persistent Community.
        """

        return self.timelines.get(
            pcid,
            []
        )

    def get_snapshot(
        self,
        pcid,
        timestep,
    ):
        """
        Return the timeline record for a specific timestep.
        """

        for record in self.get_timeline(pcid):

            if record["timestep"] == timestep:

                return record

        return None

    def birth(
        self,
        pcid,
    ):
        """
        Return the birth timestep.
        """

        timeline = self.get_timeline(pcid)

        if not timeline:
            return None

        return timeline[0]["timestep"]

    def death(
        self,
        pcid,
    ):
        """
        Return the final timestep.
        """

        timeline = self.get_timeline(pcid)

        if not timeline:
            return None

        return timeline[-1]["timestep"]

    def persistence(
        self,
        pcid,
    ):
        """
        Number of snapshots in which the community exists.
        """

        return len(
            self.get_timeline(pcid)
        )

    def lifetime(
        self,
        pcid,
    ):
        """
        Lifetime from birth to death (inclusive).
        """

        birth = self.birth(pcid)
        death = self.death(pcid)

        if birth is None:

            return 0

        return death - birth + 1

    def get_all_pcids(self):
        """
        Return all Persistent Community IDs.
        """

        return sorted(
            self.timelines.keys()
        )

    def summarize(self):
        """
        Return summary statistics for every Persistent Community.
        """

        rows = []

        for pcid in self.get_all_pcids():

            rows.append({

                "pcid": pcid,

                "birth": self.birth(pcid),

                "death": self.death(pcid),

                "lifetime": self.lifetime(pcid),

                "persistence": self.persistence(pcid),

                "snapshots": len(
                    self.get_timeline(pcid)
                )

            })

        return rows