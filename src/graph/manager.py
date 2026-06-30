"""
Manage temporal graph snapshots.
"""

from pathlib import Path
import pickle

import pandas as pd

from src.config import SNAPSHOT_DIR
from src.graph.builder import TemporalGraphBuilder
from src.graph.statistics import GraphStatistics
from src.logger import logger


class TemporalGraphManager:

    def __init__(self):

        self.builder = TemporalGraphBuilder()

        self.graphs = {}

        self.statistics = []

    def build_all(self, timesteps):

        logger.info("Building temporal graphs...")

        for timestep in timesteps:

            graph = self.builder.build_snapshot(timestep)

            self.graphs[timestep] = graph

            self.statistics.append(
                GraphStatistics.summarize(graph)
            )

        logger.info(
            f"Built {len(self.graphs)} graphs."
        )

    def get_snapshot(self, timestep):

        return self.graphs[timestep]

    def save_graphs(self):

        SNAPSHOT_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        for timestep, graph in self.graphs.items():

            with open(
                SNAPSHOT_DIR /
                f"snapshot_{timestep}.gpickle",
                "wb"
            ) as file:

                pickle.dump(graph, file)

    def save_statistics(self):

        df = pd.DataFrame(self.statistics)

        df.to_csv(
            SNAPSHOT_DIR /
            "graph_statistics.csv",
            index=False
        )