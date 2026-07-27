"""
GraphSAGE model for EvoFinGraph.

Learns node representations from temporal
transaction graphs using neighborhood aggregation.
"""

import torch
import torch.nn.functional as F

from torch import nn
from torch_geometric.nn import SAGEConv

from src.models.base import BaseModel


class GraphSAGEModel(BaseModel):
    """
    Two-layer GraphSAGE classifier.
    """

    class GraphSAGE(nn.Module):
        """
        Internal GraphSAGE network.
        """

        def __init__(
            self,
            input_dim,
            hidden_dim,
            output_dim,
        ):
            super().__init__()

            self.conv1 = SAGEConv(
                input_dim,
                hidden_dim,
            )

            self.conv2 = SAGEConv(
                hidden_dim,
                output_dim,
            )

        def forward(
            self,
            x,
            edge_index,
        ):

            x = self.conv1(
                x,
                edge_index,
            )

            x = F.relu(x)

            x = F.dropout(
                x,
                p=0.3,
                training=self.training,
            )

            x = self.conv2(
                x,
                edge_index,
            )

            return x

    def __init__(
        self,
        input_dim,
        hidden_dim=128,
        output_dim=2,
        learning_rate=0.001,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate

        self.model = self.GraphSAGE(
            input_dim,
            hidden_dim,
            output_dim,
        )

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
        )

        self.loss_fn = nn.CrossEntropyLoss()

    # --------------------------------------------------
    # Forward
    # --------------------------------------------------

    def forward(
        self,
        data,
    ):

        return self.model(
            data.x,
            data.edge_index,
        )
    
    # --------------------------------------------------
    # Training
    # --------------------------------------------------

    def fit(
        self,
        graphs,
        epochs=100,
        verbose=True,
    ):
        """
        Train GraphSAGE on one or more PyG graphs.
        """

        if not isinstance(graphs, (list, tuple)):
            graphs = [graphs]

        self.model.train()

        history = []

        for epoch in range(epochs):

            epoch_loss = 0.0

            for data in graphs:

                self.optimizer.zero_grad()

                logits = self.forward(
                    data
                )

                loss = self.loss_fn(
                    logits,
                    data.y,
                )

                loss.backward()

                self.optimizer.step()

                epoch_loss += loss.item()

            epoch_loss /= len(graphs)

            history.append(epoch_loss)

            if verbose:

                print(
                    f"Epoch {epoch + 1:03d} | "
                    f"Loss: {epoch_loss:.4f}"
                )

        return {

            "loss_history": history,

            "epochs": epochs,

            "final_loss": history[-1],

        }

    # --------------------------------------------------
    # Training Mode
    # --------------------------------------------------

    def train_mode(
        self,
    ):

        self.model.train()

    # --------------------------------------------------
    # Evaluation Mode
    # --------------------------------------------------

    def eval_mode(
        self,
    ):

        self.model.eval()

    # --------------------------------------------------
    # Loss
    # --------------------------------------------------

    def compute_loss(
        self,
        data,
    ):
        """
        Compute loss without gradient update.
        """

        self.model.eval()

        with torch.no_grad():

            logits = self.forward(
                data
            )

            loss = self.loss_fn(
                logits,
                data.y,
            )

        return loss.item()
    # --------------------------------------------------
    # Prediction
    # --------------------------------------------------

    def predict(
        self,
        data,
    ):
        """
        Predict node labels.
        """

        self.model.eval()

        with torch.no_grad():

            logits = self.forward(
                data,
            )

            predictions = torch.argmax(

                logits,

                dim=1,

            )

        return predictions.cpu().numpy()

    # --------------------------------------------------
    # Prediction Probabilities
    # --------------------------------------------------

    def predict_proba(
        self,
        data,
    ):
        """
        Predict node probabilities.
        """

        self.model.eval()

        with torch.no_grad():

            logits = self.forward(
                data,
            )

            probabilities = torch.softmax(

                logits,

                dim=1,

            )

        return probabilities.cpu().numpy()

    # --------------------------------------------------
    # Node Embeddings
    # --------------------------------------------------

    def embeddings(
        self,
        data,
    ):
        """
        Return hidden GraphSAGE embeddings.
        """

        self.model.eval()

        with torch.no_grad():

            x = self.model.conv1(

                data.x,

                data.edge_index,

            )

            x = F.relu(x)

        return x.cpu().numpy()

    # --------------------------------------------------
    # Save Model
    # --------------------------------------------------

    def save(
        self,
        path,
    ):
        """
        Save trained model.
        """

        torch.save(

            self.model.state_dict(),

            path,

        )

    # --------------------------------------------------
    # Load Model
    # --------------------------------------------------

    def load(
        self,
        path,
    ):
        """
        Load trained model.
        """

        self.model.load_state_dict(

            torch.load(

                path,

                map_location="cpu",

            )

        )

        self.model.eval()

    # --------------------------------------------------
    # Feature Importance
    # --------------------------------------------------

    def feature_importance(
        self,
        feature_names=None,
    ):
        """
        GraphSAGE does not expose intrinsic
        feature importance.

        Future versions may integrate
        GNNExplainer or Captum.
        """

        raise NotImplementedError(

            "GraphSAGE does not provide native feature importance."

        )