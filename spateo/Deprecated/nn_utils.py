"""

"""
import collections
import sys
from typing import Callable, Iterable, List, Optional, Union

import torch
from torch import nn

# For now, add Spateo working directory to sys path so compiler doesn't look in the installed packages:
sys.path.insert(0, "/mnt/c/Users/danie/Desktop/Github/Github/spateo-release-main")

from spateo.logging import logger_manager as lm


# ---------------------------------------------------------------------------------------------------
# Fully connected layers
# ---------------------------------------------------------------------------------------------------
class MultiLayer(nn.Module):
    """Base class to build customizable fully-connected layers through PyTorch.

    Args:
        n_in: The dimensionality of the input
        n_out: The dimensionality of the output
        n_layers: The number of fully-connected hidden layers. If 'n_layers' is 1, will include no hidden layers-
            only input connected to output.
        n_hidden: The number of output nodes in all hidden layers (if given as int) or the number of output nodes in
            each hidden layer (if given as sequence)
        dropout_rate: Dropout rate to apply to each of the hidden layers
        use_batch_norm: Set True to have `BatchNorm` layers after each hidden layer
        momentum: Higher momentum means updates are "slowed" by the batch norm layer placing increased weightage to
            the previous batches
        use_layer_norm: Set True to have `LayerNorm` layers after each hidden layer
        use_activation: Set True to apply activation function at each layer.
        bias: Set True to include bias terms in hidden layers
        activation_fn: Object of class :func implementing an activation function, e.g. torch.F.relu(). Only applied
            to each layer if 'use_activation' is True.
    """

    def __init__(
        self,
        n_in: int,
        n_out: int,
        n_layers: int = 1,
        n_hidden: Union[List[int], int] = 128,
        dropout_rate: float = 0.1,
        use_batch_norm: bool = True,
        momentum: Optional[float] = 0.01,
        use_layer_norm: bool = False,
        use_activation: bool = True,
        bias: bool = True,
        activation_fn: Callable = nn.ReLU,
    ):
        super().__init__()
        self.logger = lm.get_main_logger()

        if not isinstance(n_hidden, Iterable):
            try:
                n_hidden = [n_hidden]
            except:
                raise TypeError(
                    "Invalid input given to 'n_hidden' of :class `~MultiLayer`. Must be a list, sequence, "
                    "or integer."
                )
        # Net either has no hidden layers or number of hidden layers equal to n_layers - 1:
        if len(n_hidden) != 0 and len(n_hidden) != n_layers - 1:
            raise ValueError(
                f"Length of hidden layer list {len(n_hidden)} is not not compatible with number of "
                f"layers specified to network: {n_layers}, which will result in {n_layers-1} hidden "
                f"layers that must each have an assigned number of outputs."
            )

        if len(n_hidden) == 1:
            layers_dim = [n_in] + (n_layers - 1) * n_hidden + [n_out]
        else:
            layers_dim = [n_in] + n_hidden + [n_out]
        total_n_layers = len(layers_dim) - 2

        self.layers = nn.Sequential(
            collections.OrderedDict(
                [
                    (
                        f"Layer {i}",
                        nn.Sequential(
                            nn.Linear(
                                n_in,
                                n_out,
                                bias=bias,
                            ),
                            nn.BatchNorm1d(n_out, momentum=momentum, eps=1e-4) if use_batch_norm else None,
                            nn.LayerNorm(n_out, elementwise_affine=False) if use_layer_norm else None,
                            activation_fn() if (use_activation and i != total_n_layers) else nn.Linear(n_out, n_out),
                            nn.Dropout(p=dropout_rate) if dropout_rate > 0 else None,
                        ),
                    )
                    for i, (n_in, n_out) in enumerate(zip(layers_dim[:-1], layers_dim[1:]))
                ]
            )
        )

    def forward(
        self,
        x: torch.Tensor,
        cat_list: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        cont_tensor: Optional[torch.Tensor] = None,
    ):
        """Process of feeding tensor 'x' through the network.

        Args:
            x: tensor of values with shape [, n_in]
            cat_list: Optional list of tensors for categorical covariates. One tensor array for each categorical
                variable.
            cont_tensor: Tensor array for continuous covariates.

        Returns:
            x: tensor of values from the final layer of the network, with shape [n_out,]
        """

        for i, layers in enumerate(self.layers):
            for layer in layers:
                if layer is not None:
                    if isinstance(layer, nn.BatchNorm1d):
                        if x.dim() == 3:
                            x = torch.cat([(layer(slice_x)).unsqueeze(0) for slice_x in x], dim=0)
                        else:
                            x = layer(x)

        return x
