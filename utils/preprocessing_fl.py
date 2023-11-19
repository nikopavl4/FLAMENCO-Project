"""
Pre-processing methods specifically designed for federated learning simulation.
"""

from typing import Dict, Tuple, Union

import pandas as pd
from torch.utils.data import DataLoader

from utils.preprocessing import to_torch_data, get_X_y


def get_X_y_client(
        clients: Dict[str, Dict[str, pd.DataFrame]],
        data_all: pd.DataFrame,
        target_col: str = "target"
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Dict[str, pd.DataFrame]]]:
    x_y_client_data = dict()
    X_all, y_all = None, None
    for i, client in enumerate(clients):
        train_data = clients[client]["train"]
        test_data = clients[client]["test"]
        if i == 0:
            X_train, X_test, y_train, y_test, X_all, y_all = get_X_y(
                data_all=data_all, data_train=train_data, data_test=test_data, target_col=target_col
            )
        else:
            X_train, X_test, y_train, y_test, _, _ = get_X_y(
                data_all=None, data_train=train_data, data_test=test_data, target_col=target_col
            )

        x_y_client_data[client] = {"X_train": X_train, "y_train": y_train, "X_test": X_test, "y_test": y_test}
    assert X_all is not None and y_all is not None
    return X_all, y_all, x_y_client_data


def to_torch_client_data(
        clients: Dict[str, Dict[str, pd.DataFrame]],
        X_all: Union[pd.DataFrame, None],
        y_all: Union[pd.DataFrame, None],
        X_train_batch_size: int = 32,
        X_test_batch_size: int = 1,
        X_all_batch_size: int = 1,
) -> Tuple[Dict[str, Dict[str, DataLoader]], Union[DataLoader, None]]:
    data_loader = None
    client_loaders = dict()

    for i, client in enumerate(clients):
        X_train, X_test = clients[client]["X_train"], clients[client]["X_test"]
        y_train, y_test = clients[client]["y_train"], clients[client]["y_test"]

        if i == 0:
            if X_all is not None:
                train_loader, test_loader, data_loader = to_torch_data(
                    X_train=X_train,
                    X_test=X_test,
                    y_train=y_train,
                    y_test=y_test,
                    X_all=X_all,
                    y_all=y_all,
                    X_train_batch_size=X_train_batch_size,
                    X_test_batch_size=X_test_batch_size,
                    X_all_batch_size=X_all_batch_size
                )
            else:
                train_loader, test_loader = to_torch_data(
                    X_train=X_train,
                    X_test=X_test,
                    y_train=y_train,
                    y_test=y_test,
                    X_train_batch_size=X_train_batch_size,
                    X_test_batch_size=X_test_batch_size
                )
        else:
            train_loader, test_loader = to_torch_data(
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                X_train_batch_size=X_train_batch_size,
                X_test_batch_size=X_test_batch_size
            )
        client_loaders[client] = {"train": train_loader, "test": test_loader}
    return client_loaders, data_loader
