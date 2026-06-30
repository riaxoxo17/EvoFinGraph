"""
Dataset validation utilities.
"""

import pandas as pd


class DataValidator:

    @staticmethod
    def check_duplicates(df: pd.DataFrame, column: str):

        duplicates = df[column].duplicated().sum()

        print(f"Duplicate {column}: {duplicates}")

        return duplicates

    @staticmethod
    def check_missing(df: pd.DataFrame):

        missing = df.isnull().sum().sum()

        print(f"Missing Values: {missing}")

        return missing

    @staticmethod
    def check_unique_labels(classes):

        print(classes["class"].value_counts())

    @staticmethod
    def check_time_steps(features):

        print(
            f"Time Steps: {features.time_step.nunique()}"
        )