"""
Preprocessing pipeline.
"""

import pandas as pd

from src.data.loader import load_dataset
from src.config import PROCESSED_DATA_DIR


class DataPreprocessor:

    def __init__(self):

        self.features = None
        self.classes = None
        self.edges = None

    def load(self):

        self.features, self.classes, self.edges = load_dataset()

    def merge(self):

        self.features = self.features.merge(
            self.classes,
            on="txId"
        )

    def remove_unknown(self):

        self.features = self.features[
            self.features["class"] != "unknown"
        ]

    def encode_labels(self):

        self.features["class"] = (
            self.features["class"]
            .astype(int)
            .replace({
                1: 1,
                2: 0
            })
        )

    def save(self):

        PROCESSED_DATA_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        self.features.to_csv(
            PROCESSED_DATA_DIR /
            "processed_transactions.csv",
            index=False
        )

    def run(self):

        self.load()

        self.merge()

        self.remove_unknown()

        self.encode_labels()

        self.save()

        return self.features