"""
Helper functions.
"""

from typing import Any, Union, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.nn
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def two_dim_transform(
        data: Union[pd.DataFrame, np.ndarray],
        method: str = 'pca',
        exclude_col: str = 'target',
        args: Union[None, Dict[str, Any]] = {'random_state': 42},
        plot: bool = True
) -> np.ndarray:
    """
    Transforms high-dimensional data into a 2D representation using PCA or t-SNE.

    Args:
        data (np.ndarray): The high-dimensional data.
        method (str, optional): The transformation method, either 'pca' or 'tsne'. Default is 'pca'.
        exclude_col (str, default="target"): The column to exclude from computation.
        args (Union[None, Dict[str, Any]], optional): Additional arguments for the transformation method. Default is {'random_state': 42}.
        plot (bool, optional): Whether to display a scatter plot of the transformed data. Default is True.

    Returns:
        np.ndarray: 2D transformed data.
    """
    if method == 'pca':
        model = PCA(n_components=2, **args)
    elif method == 'tsne':
        model = TSNE(n_components=2, **args)
    else:
        raise ValueError(f"Unknown method {method}")

    data = data.copy()
    if isinstance(data, pd.DataFrame):
        if exclude_col in data.columns:
            data = data.drop([exclude_col], axis=1)

    x_transformed = model.fit_transform(data)
    if plot:
        plt.figure()
        plt.scatter(x_transformed[:, 0], x_transformed[:, 1], c='b')
        plt.title(f'{method.upper()} visualization')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.show()
        plt.close()

    return x_transformed


def get_optimizer(
        model: torch.nn.Module,
        optim: str = "adam",
        lr: float = 1e-3,
        args: Dict[str, Any] = None
):
    """
    Retrieves the specified optimizer for the provided model.

    Parameters:
        model (torch.nn.Module): The model for which the optimizer is needed.
        optim (str, optional): The type of optimizer, either 'adam' or 'sgd'. Default is 'adam'.
        lr (float, optional): The learning rate. Default is 1e-3.
        args (Dict[str, Any], optional): Additional arguments for the optimizer like 'weight_decay' and 'momentum'. Default is None.

    Returns:
        torch.optim.Optimizer: The specified optimizer initialized with given arguments.
    """
    if args is None:
        weight_decay = 0
        momentum = 0
    else:
        try:
            weight_decay = args['weight_decay']
        except KeyError:
            weight_decay = 0
        try:
            momentum = args['momentum']
        except KeyError:
            momentum = 0

    if optim == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
    elif optim == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum
        )
    else:
        raise NotImplementedError
    return optimizer
