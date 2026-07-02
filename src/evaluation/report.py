"""
Experiment comparison utilities.

Compares multiple EvoFinGraph models
using a common evaluation format.
"""

import pandas as pd


class EvaluationReport:
    """
    Collects experiment results.
    """

    def __init__(self):

        self.results = []

    # --------------------------------------------------
    # Add Experiment
    # --------------------------------------------------

    def add(
        self,
        model_name,
        metrics,
    ):
        """
        Add one experiment.
        """

        record = {

            "model": model_name,

        }

        record.update(
            metrics
        )

        self.results.append(
            record
        )

    # --------------------------------------------------
    # DataFrame
    # --------------------------------------------------

    def dataframe(
        self,
    ):
        """
        Return experiment table.
        """

        return pd.DataFrame(
            self.results
        )

    # --------------------------------------------------
    # Ranking
    # --------------------------------------------------

    def rank(
        self,
        metric="f1",
    ):
        """
        Rank models by metric.
        """

        return self.dataframe().sort_values(

            metric,

            ascending=False,

        )

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    def summary(
        self,
    ):
        """
        Print experiment summary.
        """

        df = self.rank()

        print("=" * 60)
        print("MODEL COMPARISON")
        print("=" * 60)

        print(df)

        return df