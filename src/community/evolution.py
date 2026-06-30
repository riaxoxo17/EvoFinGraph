"""
Community evolution utilities.
"""

import pandas as pd


class CommunityEvolution:

    @staticmethod
    def summarize(graph):

        rows = []

        communities = {}

        for node, attrs in graph.nodes(data=True):

            cid = attrs["community"]

            communities.setdefault(
                cid,
                []
            ).append(node)

        for cid, members in communities.items():

            rows.append({

                "timestep":

                    graph.graph["timestep"],

                "community":

                    cid,

                "size":

                    len(members)

            })

        return pd.DataFrame(rows)