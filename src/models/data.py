"""
Utilities for preparing temporal
community features for machine learning.
"""

from sklearn.model_selection import train_test_split

from src.config import AFEI_FEATURE_COLUMNS


class ModelData:

    """
    Prepares feature matrices
    for prediction models.
    """

    @staticmethod
    def prepare(
        dataframe,
        test_size=0.2,
        random_state=42,
    ):
        """
        Split dataframe into
        train/test sets.
        """

        X = dataframe[
            AFEI_FEATURE_COLUMNS
        ]

        y = dataframe[
            "is_fraud_community"
        ]

        return train_test_split(

            X,

            y,

            test_size=test_size,

            random_state=random_state,

            stratify=y,

        )