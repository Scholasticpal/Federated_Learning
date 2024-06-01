"""Define our models, and training and eval functions.

If your model is 100% off-the-shelf (e.g. directly from torchvision without requiring
modifications) you might be better off instantiating your  model directly from the Hydra
config. In this way, swapping your model for  another one can be done without changing
the python code at all
"""

from collections import OrderedDict
from typing import Union

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict
from flwr.common import NDArray
from hydra.utils import instantiate
from omegaconf import DictConfig
from xgboost import XGBClassifier, XGBRegressor

def fit_gru(
    config: DictConfig,
    task_type: str,
    x_train: NDArray,
    y_train: NDArray,
    # hidden_size: int, num_layers: int, input_size: int
) -> nn.Module:
    """Train a fixed GRU model for regression."""
    # Define the GRU model
    class GRUModel(nn.Module):
        def __init__(self):
            super(GRUModel, self).__init__()
            self.input_size = 12
            self.hidden_size = 64
            self.num_layers = 1
            self.batch_first = True
            self.gru = nn.GRU(self.input_size, self.hidden_size, self.num_layers, batch_first=self.batch_first)
            self.fc = nn.Linear(self.hidden_size, 1)

        def forward(self, x, h0=None):
            # Ensure input data is 3-Dimensional (batch_size, seq_length, input_size)
            if len(x.shape) == 2:  # Add a batch dimension if it's missing
                x = x.unsqueeze(0)
            
            if h0 is None:
                h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            else:
                # Ensure initial hidden state is 3-Dimensional
                if len(h0.shape) == 2:  # Add a batch dimension if it's missing
                    h0 = h0.unsqueeze(0)

            out, _ = self.gru(x, h0)
            out = self.fc(out[:, -1, :])  # Use only the output of the last time step
            return out

    model = GRUModel()
    return model


def fit_xgboost(
    config: DictConfig,
    task_type: str,
    x_train: NDArray,
    y_train: NDArray,
    n_estimators: int,
) -> Union[XGBClassifier, XGBRegressor]:
    """Fit XGBoost model to training data.

    Parameters
    ----------
        config (DictConfig): Hydra configuration.
        task_type (str): Type of task, "REG" for regression and "BINARY"
        for binary classification.
        x_train (NDArray): Input features for training.
        y_train (NDArray): Labels for training.
        n_estimators (int): Number of trees to build.

    Returns
    -------
        Union[XGBClassifier, XGBRegressor]: Fitted XGBoost model.
    """
    if config.dataset.dataset_name == "all":
        if task_type.upper() == "REG":
            tree = instantiate(config.XGBoost.regressor, n_estimators=n_estimators)
        elif task_type.upper() == "BINARY":
            tree = instantiate(config.XGBoost.classifier, n_estimators=n_estimators)
    else:
        tree = instantiate(config.XGBoost)
    tree.fit(x_train, y_train)
    return tree


class CNN(nn.Module):
    """CNN model."""

    def __init__(self, cfg: DictConfig, n_channel: int = 64) -> None:
        super().__init__()
        n_out = 1
        self.task_type = cfg.dataset.task.task_type
        n_estimators_client = cfg.n_estimators_client
        client_num = cfg.client_num

        self.conv1d = nn.Conv1d(
            in_channels=1,
            out_channels=n_channel,
            kernel_size=n_estimators_client,
            stride=n_estimators_client,
            padding=0,
        )

        self.layer_direct = nn.Linear(n_channel * client_num, n_out)

        self.relu = nn.ReLU()

        if self.task_type == "BINARY":
            self.final_layer = nn.Sigmoid()
        elif self.task_type == "REG":
            self.final_layer = nn.Identity()

        # Add weight initialization
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(
                    layer.weight, mode="fan_in", nonlinearity="relu"
                )

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass.

        Parameters
        ----------
            input_features (torch.Tensor): Input features to the network.

        Returns
        -------
            output (torch.Tensor): Output of the network after the forward pass.
        """
        output = self.conv1d(input_features)
        output = output.flatten(start_dim=1)
        output = self.relu(output)
        output = self.layer_direct(output)
        output = self.final_layer(output)
        return output

    def get_weights(self) -> fl.common.NDArrays:
        """Get model weights.

        Parameters
        ----------
            a list of NumPy arrays.
        """
        return [
            np.array(val.cpu().numpy(), copy=True)
            for _, val in self.state_dict().items()
        ]

    def set_weights(self, weights: fl.common.NDArrays) -> None:
        """Set the CNN model weights.

        Parameters
        ----------
            weights:a list of NumPy arrays
        """
        layer_dict = {}
        for key, value in zip(self.state_dict().keys(), weights):
            if value.ndim != 0:
                layer_dict[key] = torch.Tensor(np.array(value, copy=True))
        state_dict = OrderedDict(layer_dict)
        self.load_state_dict(state_dict, strict=True)


# Define the GRU model using Keras
# def create_gru_model(input_shape, units=64, task_type='REG'):
#     model = nn.Sequential(
#             nn.GRU(
#                 batch_first = True
#             )
#         )
#     if task_type.upper() == 'BINARY':
#         model.add(Dense(1, activation='sigmoid'))
#         model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
#     elif task_type.upper() == 'REG':
#         model.add(Dense(1, activation='linear'))
#         model.compile(optimizer=Adam(), loss='mean_squared_error')
#     return model

# def fit_gru(
#     config: DictConfig,
#     task_type: str,
#     x_train: NDArray,
#     y_train: NDArray,
#     epochs: int = 50,
#     batch_size: int = 32
# ) -> nn.GRU:
#     """Fit GRU model to training data.

#     Parameters
#     ----------
#         config (DictConfig): Hydra configuration.
#         task_type (str): Type of task, "REG" for regression and "BINARY" for binary classification.
#         x_train (NDArray): Input features for training.
#         y_train (NDArray): Labels for training.
#         epochs (int): Number of epochs to train.
#         batch_size (int): Batch size for training.

#     Returns
#     -------
#     """
#     # Determine input shape based on training data
#         # Ensure x_train is a 3D array
#     if len(x_train.shape) == 2:  # (samples, features)
#         x_train = np.expand_dims(x_train, axis=1)  # add a timestep dimension
    
#     # If x_train is 1D or some unexpected shape, handle appropriately
#     if len(x_train.shape) != 3:
#     raise ValueError(f"Unexpected shape for x_train: {x_train.shape}")
    
#     input_shape = (x_train.shape[1], x_train.shape[2])

#     # Instantiate and compile the model based on task type
#     model = create_gru_model(input_shape, task_type=task_type)

#     # Fit the model to the training data
#     model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=2)
#     return model

