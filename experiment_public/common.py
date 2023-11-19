import random
import sys
from typing import Union, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from models.AutoEncoder import AutoEncoder, VariationalAutoEncoder


def read_data(file_path: str = "../dataset/categories_per_child.csv") -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError as e:
        if file_path.startswith("../"):
            file_path = file_path[3:]
            try:
                df = pd.read_csv(file_path)
                return df
            except FileNotFoundError as e:
                print(f"Error reading data file: {e}")
                sys.exit(1)
        else:
            print(f"Error reading data file: {e}")
            sys.exit(1)


def load_dataset(
        case: str = "autism",
        identifier: Union[str, None] = "case_id"
):
    if case in ["asd", "autism", "autismos"]:
        train_data_path = "../dataset_public/autism_train.csv"
        test_data_path = "../dataset_public/autism_test.csv"
    elif case in ["id", "intellectual", "nohtikh"]:
        train_data_path = "../dataset_public/intellectual_train.csv"
        test_data_path = "../dataset_public/intellectual_test.csv"
    else:
        raise ValueError

    df_train, df_test = read_data(train_data_path), read_data(test_data_path)
    print(f"The number of subjects in train is: {len(df_train)} and in test: {len(df_test)}")

    print("Training Data:")
    print(df_train.head())
    print("Test Data:")
    print(df_test.head())

    columns_train, columns_test = df_train.columns, df_test.columns
    assert list(columns_train) == list(columns_test)
    assert identifier in list(columns_train)

    return df_train, df_test


def plot_dist(df: pd.DataFrame, mean: float, cid=None) -> None:
    if mean != -1:
        tmp = df[df['scores'] < mean]
    else:
        tmp = df.copy()
    sns.set(style="whitegrid")
    sns.displot(tmp['scores'].values, kde=True)
    plt.xlabel("Scores")
    plt.ylabel("Frequency")
    if cid is None:
        plt.title("Distribution of Child Mean Scores")
    else:
        plt.title(f"[{cid}]Distribution of Child Mean Scores")
    plt.tight_layout()
    plt.show()
    plt.close()


def get_model(model_name, input_dim, hidden_dim) -> torch.nn.Module:
    if model_name == "autoencoder":
        return AutoEncoder(input_dim, hidden_dim)
    elif model_name == "vae":
        return VariationalAutoEncoder(input_dim, hidden_dim)
    else:
        raise ValueError(f"The provided model: {model_name} is not implemented")


def seed_all(seed):
    if seed is None:
        return
    # ensure reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
