"""
Logistic Regression baseline.

Provides an interpretable baseline model
for community-level fraud prediction.
"""

from sklearn.linear_model import LogisticRegression

from src.models.base import BaseModel


class LogisticModel(BaseModel):
    """
    Logistic Regression baseline.
    """

    def __init__(
        self,
        random_state=42,
    ):

        super().__init__()

        self.model = LogisticRegression(

            max_iter=1000,

            random_state=random_state,

            class_weight="balanced",

        )

    # --------------------------------------------------
    # Training
    # --------------------------------------------------

    def fit(
        self,
        X_train,
        y_train,
    ):
        """
        Train Logistic Regression.
        """

        self.model.fit(

            X_train,

            y_train,

        )

    # --------------------------------------------------
    # Prediction
    # --------------------------------------------------

    def predict(
        self,
        X_test,
    ):
        """
        Predict fraud labels.
        """

        return self.model.predict(
            X_test
        )

    def predict_proba(
        self,
        X_test,
    ):
        """
        Predict fraud probabilities.
        """

        return self.model.predict_proba(

            X_test

        )[:, 1]

    # --------------------------------------------------
    # Feature Importance
    # --------------------------------------------------

    def feature_importance(
        self,
        feature_names,
    ):
        """
        Return Logistic Regression coefficients.
        """

        importance = {

            feature: coefficient

            for feature, coefficient in zip(

                feature_names,

                self.model.coef_[0],

            )

        }

        return dict(

            sorted(

                importance.items(),

                key=lambda x: abs(x[1]),

                reverse=True,

            )

        )