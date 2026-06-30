"""
Data loading utilities for the Elliptic Bitcoin Dataset.
"""

from pathlib import Path
import pandas as pd

from src.config import FEATURE_FILE, CLASS_FILE, EDGE_FILE


def load_features() -> pd.DataFrame:
    """
    Load transaction feature data.
    """
    df = pd.read_csv(FEATURE_FILE, header=None)

    df.columns = (
        ["txId", "time_step"]
        + [f"feature_{i}" for i in range(165)]
    )

    return df


def load_classes() -> pd.DataFrame:
    """
    Load transaction labels.
    """
    return pd.read_csv(CLASS_FILE)


def load_edges() -> pd.DataFrame:
    """
    Load transaction graph edges.
    """
    return pd.read_csv(EDGE_FILE)


def load_dataset():
    """
    Load the complete dataset.
    """

    return (
        load_features(),
        load_classes(),
        load_edges()
    )