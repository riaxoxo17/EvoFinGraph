"""
Evaluation metrics for EvoFinGraph.

Provides standard machine learning metrics and
research-oriented metrics for fraud community detection.
"""

import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


class EvaluationMetrics:
    """
    Computes evaluation metrics.
    """

    # --------------------------------------------------
    # Standard Metrics
    # --------------------------------------------------

    @staticmethod
    def compute(
        y_true,
        predictions,
        probabilities=None,
    ):
        """
        Compute evaluation metrics.
        """

        results = {

            "accuracy":

                accuracy_score(
                    y_true,
                    predictions,
                ),

            "precision":

                precision_score(
                    y_true,
                    predictions,
                    zero_division=0,
                ),

            "recall":

                recall_score(
                    y_true,
                    predictions,
                    zero_division=0,
                ),

            "f1":

                f1_score(
                    y_true,
                    predictions,
                    zero_division=0,
                ),

        }

        if probabilities is not None:

            results["roc_auc"] = roc_auc_score(

                y_true,

                probabilities,

            )

        return results

    # --------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------

    @staticmethod
    def confusion(
        y_true,
        predictions,
    ):
        """
        Return confusion matrix.
        """

        return confusion_matrix(

            y_true,

            predictions,

        )

    # --------------------------------------------------
    # Top-K Precision
    # --------------------------------------------------

    @staticmethod
    def top_k_precision(
        y_true,
        probabilities,
        k=10,
    ):
        """
        Precision among the Top-K
        highest-risk communities.
        """

        order = np.argsort(
            probabilities
        )[::-1]

        top = order[:k]

        return np.mean(

            np.array(y_true)[top]

        )

    # --------------------------------------------------
    # Detection Lag
    # --------------------------------------------------

    @staticmethod
    def detection_lag(
        true_timesteps,
        predicted_timesteps,
    ):
        """
        Mean delay between
        true fraud appearance
        and model detection.
        """

        return np.mean(

            np.array(predicted_timesteps)

            -

            np.array(true_timesteps)

        )

    # --------------------------------------------------
    # Early Detection Gain
    # --------------------------------------------------

    @staticmethod
    def early_detection_gain(
        baseline_lag,
        model_lag,
    ):
        """
        Improvement over baseline.
        """

        return baseline_lag - model_lag