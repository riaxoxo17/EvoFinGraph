"""
Adaptive Fraud Evolution Index (AFEI).

Computes rule-based fraud evolution scores
from temporal community features.
"""

import numpy as np
import pandas as pd
from src.config import (
    AFEI_WEIGHTS,
    AFEI_FEATURE_COLUMNS,
)

from dataclasses import dataclass

class AFEIScorer:
    """
    Computes the Adaptive Fraud Evolution Index
    from temporal community features.
    """
    def __init__(
        self,
        dataframe,
    ):
        """
        Parameters
        ----------
        dataframe : pandas.DataFrame
            Output of CommunityFeatureExtractor.
        """

        self.df = dataframe.copy()
    FEATURE_COLUMNS = [

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
    def validate_features(
        self,
    ):
        """
        Ensure all required features exist.
        """

        missing = [

            feature

            for feature in self.FEATURE_COLUMNS

            if feature not in self.df.columns

        ]

        if missing:

            raise ValueError(

                f"Missing features: {missing}"

            )
    def normalize_features(
        self,
    ):
        """
        Normalize all temporal features
        using Min-Max scaling.
        """

        self.validate_features()

        normalized = self.df.copy()

        for feature in self.FEATURE_COLUMNS:

            minimum = normalized[
                feature
            ].min()

            maximum = normalized[
                feature
            ].max()

            if maximum == minimum:

                normalized[
                    feature
                ] = 0.0

            else:

                normalized[
                    feature
                ] = (

                    normalized[feature]

                    -

                    minimum

                ) / (

                    maximum

                    -

                    minimum

                )

        return normalized
    @staticmethod
    def clip_score(
        score,
    ):
        """
        Keep AFEI within [0,1].
        """

        return max(

            0.0,

            min(

                1.0,

                score,

            ),

        )
    
    def normalized_dataframe(
        self,
    ):
        """
        Return normalized features.
        """

        return self.normalize_features()
    
    # -------------------------------------------------
    # Feature Contributions
    # -------------------------------------------------

    def compute_contributions(
        self,
        row,
    ):
        """
        Compute weighted contribution of every
        temporal feature.

        Structural stability is inverted since
        lower stability indicates higher fraud risk.
        """

        contributions = {}

        for feature in AFEI_FEATURE_COLUMNS:

            value = row[feature]

            if feature == "structural_stability":

                value = 1 - value

            contributions[feature] = (

                value

                * AFEI_WEIGHTS[feature]

            )

        return contributions
    # -------------------------------------------------
    # Rule-Based AFEI
    # -------------------------------------------------

    def compute_rule_based(
        self,
        row,
    ):
        """
        Compute the rule-based
        Adaptive Fraud Evolution Index.
        """

        contributions = self.compute_contributions(
            row
        )

        score = sum(

            contributions.values()

        )

        return {

            "afei": self.clip_score(
                score
            ),

            "contributions": contributions,

        }
    # -------------------------------------------------
    # Score One Community
    # -------------------------------------------------

    def score(
        self,
        row,
    ):
        """
        Score one community.
        """

        return self.compute_rule_based(
            row
        )
    # -------------------------------------------------
    # Score Entire Dataset
    # -------------------------------------------------

    def score_all(
        self,
    ):
        """
        Score every community.
        """

        normalized = self.normalize_features()

        scores = []

        for _, row in normalized.iterrows():

            result = self.compute_rule_based(
                row
            )

            record = row.to_dict()

            record["afei"] = result[
                "afei"
            ]

            record[
                "feature_contributions"
            ] = result[
                "contributions"
            ]

            scores.append(
                record
            )

        return pd.DataFrame(
            scores
        )
    # -------------------------------------------------
    # Ranking
    # -------------------------------------------------

    def rank(
        self,
        ascending=False,
    ):
        """
        Rank communities by AFEI.
        """

        scored = self.score_all()

        return scored.sort_values(

            "afei",

            ascending=ascending,

        )
    # -------------------------------------------------
    # Summary
    # -------------------------------------------------

    def summary(
        self,
    ):
        """
        Print summary statistics for AFEI.
        """

        scored = self.score_all()

        print("=" * 50)
        print("AFEI SUMMARY")
        print("=" * 50)

        print(
            f"Communities scored : {len(scored)}"
        )

        print(
            f"Mean AFEI          : {scored['afei'].mean():.3f}"
        )

        print(
            f"Max AFEI           : {scored['afei'].max():.3f}"
        )

        print(
            f"Min AFEI           : {scored['afei'].min():.3f}"
        )

        if "is_fraud_community" in scored.columns:

            fraud = scored[
                scored["is_fraud_community"] == 1
            ]

            clean = scored[
                scored["is_fraud_community"] == 0
            ]

            print()

            print(
                f"Fraud Communities     : {len(fraud)}"
            )

            print(
                f"Non-Fraud Communities : {len(clean)}"
            )

            if len(fraud):

                print(
                    f"Mean Fraud AFEI       : {fraud['afei'].mean():.3f}"
                )

            if len(clean):

                print(
                    f"Mean Clean AFEI       : {clean['afei'].mean():.3f}"
                )