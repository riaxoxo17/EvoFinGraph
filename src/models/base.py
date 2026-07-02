"""
Base model interface for EvoFinGraph.

Defines a common API for all machine learning
models used in fraud community prediction.
"""

from abc import ABC, abstractmethod
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


class BaseModel(ABC):
    """
    Abstract base class for all prediction models.
    """

    def __init__(self):

        self.model = None

    # --------------------------------------------------
    # Core API
    # --------------------------------------------------

    @abstractmethod
    def fit(
        self,
        X_train,
        y_train,
    ):
        """
        Train the model.
        """
        pass

    @abstractmethod
    def predict(
        self,
        X_test,
    ):
        """
        Predict class labels.
        """
        pass

    @abstractmethod
    def predict_proba(
        self,
        X_test,
    ):
        """
        Predict class probabilities.
        """
        pass

    # --------------------------------------------------
    # Persistence
    # --------------------------------------------------

    def save(
        self,
        path,
    ):
        """
        Save trained model.
        """

        joblib.dump(
            self.model,
            path,
        )

    def load(
        self,
        path,
    ):
        """
        Load trained model.
        """

        self.model = joblib.load(
            path
        )