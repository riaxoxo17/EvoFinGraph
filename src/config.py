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

    "feature_similarity": 0.40,

    "fraud_ratio": 0.20,

    "density": 0.15,

    "average_degree": 0.15,

    "community_size": 0.10,

}

COMMUNITY_MATCH_THRESHOLD = 0.45
MIN_COMMUNITY_SIZE = 3

"""Then, during community extraction, we ignore communities smaller than that threshold.

This has several benefits:

removes noisy singleton communities,
makes community tracking more stable,
improves AFEI by focusing on meaningful groups,
aligns better with the idea of fraud rings, which are inherently multi-node structures."""

AFEI_WEIGHTS = {

    "growth_rate": 0.20,

    "density_change": 0.10,

    "degree_change": 0.10,

    "fraud_growth": 0.20,

    "clustering_change": 0.05,

    "feature_drift": 0.15,

    "variance_drift": 0.05,

    "node_churn": 0.10,

    "structural_stability": 0.05,

}

# -----------------------------------------------------------------------------
# AFEI Features
# -----------------------------------------------------------------------------

AFEI_FEATURE_COLUMNS = [

    "growth_rate",

    "density_change",

    "degree_change",

    "fraud_growth",

    "clustering_change",

    "feature_drift",

    "variance_drift",

    "node_churn",

    "structural_stability",

]