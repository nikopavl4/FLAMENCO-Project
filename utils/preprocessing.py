"""
Pre-processing function for model training.
"""

from typing import Tuple, List, Union, Optional, Dict

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader


def generate_dataset(
        df: pd.DataFrame,
        split_column: str = "scores",
        target_column: str = "target",
        test_size: float = 0.2,
        random_state=42,
        log: bool = True
) -> Dict[str, Dict[str, pd.DataFrame]]:
    df_copy = df.copy()
    # calculate the mean
    mean_score = df_copy[split_column].mean()

    # assign samples that have score >= mean as normal and target != 1 (if present). Also, get the rest of the samples.
    normal_data, rest_data = split_normal_abnormal(
        df, mean_score, target_column=target_column
    )

    # split the normal data into training and testing
    train_data, test_normal_data = normal_data_train_test_split(
        normal_data, test_size=test_size, random_state=random_state
    )

    # merge the testing normal data with the rest data
    test_data = merge_normal_abnormal(
        test_normal_data, rest_data
    )

    if log:
        print(f"Mean={mean_score:.4f}\t#Normal={len(normal_data)}\t\t#Rest={len(rest_data)}\n"
              f"#Train={len(train_data)}, #Normal_Test={len(test_normal_data)}\n"
              f"#Rest={len(test_data)}")
        if target_column in train_data.columns:
            train_counts = dict(train_data[target_column].value_counts())
            print(f"Counts in train: {train_counts}")
        if target_column in test_data.columns:
            test_counts = dict(test_data[target_column].value_counts())
            print(f"Counts in test: {test_counts}")
    full_data = pd.concat([train_data, test_data], ignore_index=True)

    dataset = {"Data": {"full_data": full_data, "train": train_data, "test": test_data}}
    return dataset


def split_normal_abnormal(
        df: pd.DataFrame,
        split_value: float,
        split_column: str = "scores",
        target_column: str = "target"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits a DataFrame into two based on a threshold value in a specified column.

    Args:
        df (pd.DataFrame): The input DataFrame to be split.
        split_value (float): The threshold value to decide the split. Rows with values greater than or equal to this
        will be classified as 'normal'.
        split_column (str, default='scores'): The name of the column in df to check against split_value.
        target_column (str, default='target'): The name of the column in df which acts as the target.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple where the first DataFrame contains rows considered 'normal'
        (i.e., rows where the value in split_column is >= split_value) and the second DataFrame contains the rest data.
    """
    normal_data = df[df[split_column] >= split_value]
    if target_column in normal_data.columns:
        normal_data = normal_data[normal_data[target_column] != 1]
    rest_data = df.loc[~df.index.isin(normal_data.index)]
    if target_column in normal_data.columns:
        rest_data.loc[(rest_data[split_column] < split_value) & (rest_data[target_column] == 0), target_column] = -1

    return normal_data, rest_data


def normal_data_train_test_split(
        df: pd.DataFrame,
        test_size: float = 0.2,
        random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits the input dataframe into training and testing datasets.

    Args:
        df (pd.DataFrame): Input data to be split.
        test_size (float, default=0.2): Proportion of the dataset to include in the test split.
        random_state (int, default=42): Seed used by the random number generator.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the training and testing datasets.
    """
    train_data, test_normal_data = train_test_split(
        df, test_size=test_size, random_state=random_state
    )
    return train_data, test_normal_data


def merge_normal_abnormal(
        normal_data: pd.DataFrame,
        abnormal_data: pd.DataFrame,
        random_state: int = 42) -> pd.DataFrame:
    """
    Merge normal and abnormal data and shuffle the combined data.

    Args:
        normal_data (pd.DataFrame): DataFrame containing normal data.
        abnormal_data (pd.DataFrame): DataFrame containing abnormal data.
        random_state (int, default=42): Seed for the random number generator.

    Returns:
        pd.DataFrame: A shuffled DataFrame containing a combination of normal and abnormal data.
    """
    test_data = pd.concat([normal_data, abnormal_data], axis=0)
    test_data = test_data.sample(
        frac=1, random_state=random_state
    ).reset_index(drop=True)
    return test_data


def scale_data(
        df: pd.DataFrame,
        exclude_columns: List[str] = ['child_id']
) -> pd.DataFrame:
    """
    Scale data in a DataFrame to a range between 0 and 1, excluding specified columns.

    Args:
        df (pd.DataFrame): Input DataFrame containing data to be scaled.
        exclude_columns (List[str], default=['child_id']): List of column names to exclude from scaling.

    Returns:
        pd.DataFrame: A new DataFrame with scaled values, excluding columns specified in exclude_columns.

    Note:
    The scaling is done by clipping the data to a range of [0,100] and then dividing by 100.
    Any values below 0 are set to 0, and values above 100 are set to 100 before scaling.
    """
    scaled_df = df.copy()
    columns_to_scale = [col for col in scaled_df.columns if col not in exclude_columns]

    scaled_df[columns_to_scale] = scaled_df[columns_to_scale].clip(lower=0, upper=100)
    scaled_df[columns_to_scale] = scaled_df[columns_to_scale] / 100

    return scaled_df


def get_X_y(
        data_train: pd.DataFrame,
        data_test: pd.DataFrame,
        data_all: Union[pd.DataFrame, None] = None,
        target_col: str = "target"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if target_col in data_train.columns:
        X_train = data_train.drop(columns=[target_col])
        y_train = data_train[target_col]
    else:
        X_train = data_train.copy()
        y_train = data_train.copy()

    if target_col in data_test.columns:
        X_test = data_test.drop(columns=[target_col])
        y_test = data_test[target_col]
    else:
        X_test = data_test.copy()
        y_test = data_test.copy()

    X_all, y_all = None, None
    if data_all is not None:
        if target_col in data_all.columns:
            X_all = data_all.drop(columns=[target_col])
            y_all = data_all[target_col]
        else:
            X_all = data_all.copy()
            y_all = data_all.copy()

    return X_train, X_test, y_train, y_test, X_all, y_all


def to_torch_data(
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: Union[pd.DataFrame, pd.Series],
        y_test: Union[pd.DataFrame, pd.Series],
        X_all: Optional[Union[pd.DataFrame, None]] = None,
        y_all: Union[pd.DataFrame, pd.Series] = None,
        X_train_batch_size: int = 32,
        X_test_batch_size: int = 1,
        X_all_batch_size: int = 1,
) -> Union[Tuple[DataLoader, DataLoader, DataLoader], Tuple[DataLoader, DataLoader]]:
    """
    Convert pandas DataFrames into PyTorch DataLoader objects.

    This function takes in training, testing, and optionally, an 'all' dataset in the form of pandas DataFrames.
    It then creates PyTorch TensorDatasets and DataLoaders for each provided dataset.

    Args:
        X_train (pd.DataFrame): Training dataset.
        X_test (pd.DataFrame): Testing dataset.
        X_all (Optional[Union[pd.DataFrame, None]], default=None): An optional dataset that represents the entire data.
        X_train_batch_size (int, default=32): Batch size for the training DataLoader.
        X_test_batch_size (int, default=1): Batch size for the testing DataLoader.
        X_all_batch_size (int, default=1): Batch size for the 'all' DataLoader.

    Returns:
        Union[Tuple[DataLoader, DataLoader, DataLoader], Tuple[DataLoader, DataLoader]]:
        A tuple containing the training, testing, and (if provided) 'all' DataLoaders.
        If the 'all' dataset is not provided, the tuple will only contain the training and testing DataLoaders.

    Note:
        This function currently creates TensorDatasets using the same tensor for both features and targets
        (i.e., an autoencoder setting). You might want to modify this behavior if this is not the intended use case.
    """
    X_train_tensor = torch.Tensor(X_train.values)
    X_test_tensor = torch.Tensor(X_test.values)

    y_train_tensor = torch.Tensor(y_train.values)
    y_test_tensor = torch.Tensor(y_test.values)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=X_train_batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=X_test_batch_size, shuffle=False)

    if X_all is not None:
        X_all_tensor = torch.Tensor(X_all.values)
        y_all_tensor = torch.clone(X_all_tensor)
        if y_all is not None:
            y_all_tensor = torch.Tensor(y_all.values)

        all_dataset = TensorDataset(X_all_tensor, y_all_tensor)
        all_loader = DataLoader(all_dataset, batch_size=X_all_batch_size, shuffle=False)
        return train_loader, test_loader, all_loader
    return train_loader, test_loader
