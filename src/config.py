from pathlib import Path

# -----------------------------------------------------------------------------
# Project Paths
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SNAPSHOT_DIR = DATA_DIR / "snapshots"
EXPORT_DIR = DATA_DIR / "exports"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
MODEL_DIR = OUTPUT_DIR / "models"
METRIC_DIR = OUTPUT_DIR / "metrics"

NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"
DOCS_DIR = PROJECT_ROOT / "docs"

# -----------------------------------------------------------------------------
# Dataset Files
# -----------------------------------------------------------------------------

FEATURE_FILE = RAW_DATA_DIR / "elliptic_txs_features.csv"
CLASS_FILE = RAW_DATA_DIR / "elliptic_txs_classes.csv"
EDGE_FILE = RAW_DATA_DIR / "elliptic_txs_edgelist.csv"

# -----------------------------------------------------------------------------
# Development Settings
# -----------------------------------------------------------------------------

DEV_SNAPSHOTS = [20, 21, 22, 23, 24]

ALL_SNAPSHOTS = list(range(1, 50))

RANDOM_STATE = 42

# ---------------------------------------------------------
# Community Matching Configuration
# ---------------------------------------------------------

COMMUNITY_MATCHING_WEIGHTS = {
    "member_overlap": 0.50,
    "fraud_ratio": 0.25,
    "density": 0.15,
    "average_degree": 0.10,
}

COMMUNITY_MATCH_THRESHOLD = 0.60