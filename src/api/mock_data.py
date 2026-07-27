"""
Mock data for Day 3 FastAPI development.

This mimics the shape of data that will eventually come from real
Neo4j Cypher queries (once CSE's Community/PCID pipeline exists).
Keeping the *shape* consistent now means swapping mock -> real later
only touches the query logic, not the API contract.
"""

MOCK_COMMUNITIES = {
    "PCID_001": {
        "pcid": "PCID_001",
        "timeStep": 21,
        "size": 42,
        "density": 0.31,
        "fraudRatio": 0.62,
        "growthRate": 0.15,
        "afei": 0.78,
        "riskScore": 0.81,
    },
    "PCID_002": {
        "pcid": "PCID_002",
        "timeStep": 22,
        "size": 17,
        "density": 0.12,
        "fraudRatio": 0.05,
        "growthRate": -0.02,
        "afei": 0.10,
        "riskScore": 0.08,
    },
}

MOCK_TIMELINES = {
    "PCID_001": [
        {"timeStep": 20, "size": 30, "afei": 0.55, "riskScore": 0.60},
        {"timeStep": 21, "size": 42, "afei": 0.78, "riskScore": 0.81},
        {"timeStep": 22, "size": 55, "afei": 0.85, "riskScore": 0.90},
    ],
    "PCID_002": [
        {"timeStep": 21, "size": 20, "afei": 0.12, "riskScore": 0.10},
        {"timeStep": 22, "size": 17, "afei": 0.10, "riskScore": 0.08},
    ],
}

MOCK_ALERTS = [
    {
        "pcid": "PCID_001",
        "timeStep": 22,
        "riskScore": 0.90,
        "reason": "Rapid growth combined with high fraud ratio",
    },
    {
        "pcid": "PCID_007",
        "timeStep": 24,
        "riskScore": 0.87,
        "reason": "Sudden density increase after new member additions",
    },
]

MOCK_GRAPHS = {
    "PCID_001": {
        "pcid": "PCID_001",
        "nodes": [
            {"txId": 232438397, "label": "illicit"},
            {"txId": 232022460, "label": "illicit"},
            {"txId": 232344069, "label": "unknown"},
        ],
        "edges": [
            {"source": 232022460, "target": 232438397},
            {"source": 232438397, "target": 232344069},
        ],
    }
}