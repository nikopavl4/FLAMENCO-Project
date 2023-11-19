import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import Tuple, Dict, List, Union, Any

import numpy as np
import pandas as pd
import seaborn as sns
import torch.cuda
from torch.nn.modules.loss import _Loss
from torch.utils.data import ConcatDataset, DataLoader

from experiment_public.common import load_dataset, get_model, seed_all
from utils.fl_sampler import Sampler, RandomSampler, QuantitySampler, StdSampler
from utils.fl_server import Server
from utils.helpers import two_dim_transform
from utils.preprocessing_fl import (get_X_y_client,
                                    to_torch_client_data)
from utils.train_utils import predict


def get_selection_metric(
        global_history: Dict[Any, Any],
        key: str,
        aim: str,
        k: int = 5
) -> [float, int, float]:
    val_list = global_history[key]
    if key == 'p_scores':
        nval_list = [d[k] for d in val_list]
        val_list = nval_list
    if aim == 'min':
        aim_val = min(val_list)
    elif aim == 'max':
        aim_val = max(val_list)

    val_round = val_list.index(aim_val) + 1

    return [aim_val, val_round, aim_val / val_round, k]


def get_clients(
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        cid: str = "client_id",
        log: bool = True
) -> Dict[str, Union[Dict[pd.DataFrame, np.ndarray], pd.DataFrame]]:
    clients_train = list(df_train[cid].unique())
    clients_test = list(df_test[cid].unique())
    print("Are clients in train and test the same?", clients_train == clients_test)
    client_data = dict()

    for client in clients_train:
        tmp_df_train: pd.DataFrame = df_train.loc[df_train[cid] == client]
        client_data[client] = dict()
        client_data[client]["train"] = tmp_df_train.drop(columns=cid)

    for client in clients_test:
        tmp_df_test: pd.DataFrame = df_test.loc[df_test[cid] == client]
        if client not in client_data.keys():
            client_data[client] = dict()
        client_data[client]["test"] = tmp_df_test.drop(columns=cid)

    all_data = pd.concat([df_train, df_test], ignore_index=True)
    client_data["full_data"] = all_data.drop(columns=cid)

    if log:
        for client in client_data:
            if client == "full_data":
                print(f"[All data] #Samples: {len(client_data[client])}")
            else:
                print(
                    f"[Client: {client}] #Train: {len(client_data[client]['train'])}, #Test: {len(client_data[client]['test'])}")

    return client_data


def get_sampler(sampler_name: str, client_names: List[str]) -> Sampler:
    if sampler_name == "random":
        return RandomSampler(clients=client_names)
    elif sampler_name == "quantity":
        return QuantitySampler(clients=client_names)
    elif sampler_name == "std":
        return StdSampler(clients=client_names)
    else:
        raise ValueError(f"Cannot find Sampler: {sampler_name} implementation.")


def transform_data(
        full_data, client_data, exclude_col: str = "target", plot=True,
):
    x_transformed_all = two_dim_transform(full_data, plot=plot, exclude_col=exclude_col)
    x_client_transformed = dict()
    for client in client_data:
        train_data = client_data[client]["train"]
        test_data = client_data[client]["test"]
        x_transformed_train = two_dim_transform(train_data, plot=plot)
        x_transformed_test = two_dim_transform(test_data, plot=plot)
        x_client_transformed[client] = {
            "train": x_transformed_train, "test": x_transformed_test
        }
    return x_transformed_all, x_client_transformed


def train(
        client_data: Dict[str, Union[Dict[str, pd.DataFrame], pd.DataFrame]],
        identifier: str = "case_id",
        y_column: Union[str, None] = "target",
        two_dim_transform_: bool = True,
        plot_transformed: bool = False,
        scale_data: bool = True,
        train_batch_size: int = 32,
        model_name: str = "autoencoder",
        hidden_dim: int = 32,
        fl_rounds: int = 50,
        local_epochs: int = 5,
        optim_name: str = "adam",
        optim_args: Union[Dict, Any] = None,
        lr: float = 1e-3,
        criterion: _Loss = None,
        sampler_name: str = "random",
        aggregation_alg: str = "fedavg",
        aggregation_params: Union[None, Dict[str, Any]] = None,
        selection_fraction: float = 0.3,
        anomaly_percentile: int = 90,
        kappas: List[int] = [1, 2, 3, 4, 5, 10, 15, 20],
        plot_history: bool = False,
        plot_losses: bool = False,
        device: str = 'cuda',
        fhe=False
):
    full_data: pd.DataFrame = client_data["full_data"]
    del client_data["full_data"]

    if scale_data:
        scaled_client_data = dict()
        columns_to_scale = [col for col in full_data.columns if col not in [identifier, y_column]]

        for i, client in enumerate(client_data):
            client_train_data = client_data[client]["train"]
            client_test_data = client_data[client]["test"]
            scaled_client_train_data, scaled_client_test_data = (client_train_data.drop(columns=[identifier]),
                                                                 client_test_data.drop(columns=[identifier]))
            scaled_client_train_data[columns_to_scale] = scaled_client_train_data[columns_to_scale].clip(lower=0,
                                                                                                         upper=100)
            scaled_client_train_data[columns_to_scale] = scaled_client_train_data[columns_to_scale] / 100
            scaled_client_test_data[columns_to_scale] = scaled_client_test_data[columns_to_scale].clip(lower=0,
                                                                                                       upper=100)
            scaled_client_test_data[columns_to_scale] = scaled_client_test_data[columns_to_scale] / 100
            scaled_client_data[client] = {"train": scaled_client_train_data, "test": scaled_client_test_data}

        full_data_scaled = full_data.drop(columns=[identifier])

        full_data_scaled[columns_to_scale] = full_data_scaled[columns_to_scale].clip(lower=0, upper=100)
        full_data_scaled[columns_to_scale] = full_data_scaled[columns_to_scale] / 100
    else:
        full_data_scaled = full_data
        scaled_client_data = client_data

    # get 2d transformed
    x_transformed_all, x_transformed_client = None, None
    if two_dim_transform_:
        x_transformed_all, x_transformed_client = transform_data(
            full_data_scaled, scaled_client_data, plot=plot_transformed
        )

    # try to get X and y. If y cannot be found, then only features X are returned
    X_all, y_all, x_y_client_data = get_X_y_client(
        clients=scaled_client_data, data_all=full_data_scaled, target_col="target"
    )

    client_loaders, all_loader = to_torch_client_data(
        clients=x_y_client_data, X_all=X_all, y_all=y_all,
        X_train_batch_size=train_batch_size, X_test_batch_size=1, X_all_batch_size=1
    )

    model = get_model(
        model_name=model_name,
        input_dim=X_all.shape[1],
        hidden_dim=hidden_dim,
    ).to(device)
    print(model)
    print("Number of parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    training_params = dict()
    for client in client_loaders:
        training_params[client] = {
            "optimizer": optim_name,
            "optim_args": optim_args,
            "lr": lr,
            "local_epochs": local_epochs,
            "criterion": criterion,
            "percentile": anomaly_percentile,
            "normalize": True,
            "kappas": kappas,
            "log_interval": None,
            "plot_interval": None,
            "device": device
        }

    sampler = get_sampler(sampler_name=sampler_name, client_names=list(client_loaders.keys()))

    server = Server(
        model=model,
        clients=client_loaders,
        clients_transformed=x_transformed_client,
        aggregation_algo=aggregation_alg,
        aggregation_params=aggregation_params,
        sampler=sampler,
        training_params=training_params,
        fhe=fhe
    )

    global_model, local_models, global_history = server.fit(fl_rounds, selection_fraction,
                                                            return_last_model=True)

    find_all = False
    has_labels = False
    if y_column is not None:
        has_labels = True
        find_all = True

    client_predictions = dict()
    for client, local_model in local_models.items():
        print(f"\n\n[Client {client}]")

        _, all_sireos, all_normalized_p, all_auc_roc, all_ap, all_anomalies, all_scores = predict(
            model=local_model,
            data_loader=all_loader,
            data_id="test",
            criterion=criterion,
            percentile=90,
            epoch=0,
            x_transformed=x_transformed_all,
            has_labels=has_labels,
            kappas=kappas,
            plot_interval=None,
            normalize_scores=True,
            find_all=find_all,
            device=device
        )
        print("[PREDICTION USING ALL AVAILABLE DATA]")
        print("SIREOS", all_sireos)
        print("P@K", all_normalized_p)
        print("AUC", all_auc_roc)
        print("AP", all_ap)

        test_datasets = [d["test"].dataset for _, d in client_loaders.items()]
        merged_test_datasets = ConcatDataset(test_datasets)
        merged_test_loader = DataLoader(merged_test_datasets, batch_size=1, shuffle=False)

        x_test_transformed = [d["test"] for _, d in x_transformed_client.items()]
        merges_x_test_transformed = np.concatenate(x_test_transformed)
        _, all_sireos, all_normalized_p, all_auc_roc, all_ap, all_anomalies, all_scores = predict(
            model=local_model,
            data_loader=merged_test_loader,
            data_id="test",
            criterion=criterion,
            percentile=90,
            epoch=0,
            x_transformed=merges_x_test_transformed,
            has_labels=has_labels,
            kappas=kappas,
            plot_interval=None,
            normalize_scores=True,
            find_all=find_all,
            device=device
        )
        print("[PREDICTION USING TEST DATA ONLY]")
        print("SIREOS", all_sireos)
        print("P@K", all_normalized_p)
        print("AUC", all_auc_roc)
        print("AP", all_ap)

        p_scores = all_normalized_p

        client_predictions[client] = {"SIREOS": all_sireos, "AUC-ROC": all_auc_roc, "AP": all_ap}
        for k in p_scores:
            client_predictions[client][str(k)] = p_scores[k]
    try:
        file_path = f"../results/local/metrics/model_{model_name}_metrics_seed_{global_seed}_case_{global_case}.csv"
    except OSError:
        file_path = f"./results/local/metrics/model_{model_name}_metrics_seed_{global_seed}_case_{global_case}.csv"
    metrics_df = pd.DataFrame.from_dict(client_predictions, orient='index').reset_index()
    metrics_df = metrics_df.rename(columns={'index': 'client_id'})
    metrics_df = metrics_df.sort_values(by='client_id').reset_index(drop=True)
    metrics_df.to_csv(file_path, index=False)


def run_experiment(seed=0,
                   target_col="target",
                   identifier="case_id",
                   case="autism",
                   cid="client_id",
                   plot_datasets=False,
                   model="autoencoder",
                   hidden_dim=32,
                   fl_rounds=100,
                   local_epochs=3,
                   sampler_name="random",
                   aggregation_alg="fedavg",
                   selection_fraction=1.,
                   plot_history=True,
                   plot_losses=True,
                   kappas=[5, 10, 15, 20, 25, 30],
                   fhe=False
                   ):
    seed_all(seed)

    sns.set(style="whitegrid")
    df_train, df_test = load_dataset(
        case=case,
        identifier=identifier,
    )

    if target_col not in df_train.columns:
        target_col = None

    client_data = get_clients(
        df_train=df_train, df_test=df_test, cid=cid
    )

    # training params
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    criterion = torch.nn.MSELoss()

    train(
        client_data=client_data,
        identifier=identifier,
        y_column=target_col,
        two_dim_transform_=True,
        plot_transformed=False,
        scale_data=True,
        train_batch_size=32,
        model_name=model,
        hidden_dim=hidden_dim,
        fl_rounds=fl_rounds,
        local_epochs=local_epochs,
        optim_name="adam",
        optim_args=None,
        lr=1e-3,
        criterion=criterion,
        sampler_name=sampler_name,
        aggregation_alg=aggregation_alg,
        aggregation_params=None,
        selection_fraction=selection_fraction,
        device=device,
        plot_history=plot_history,
        plot_losses=plot_losses,
        kappas=kappas,
        fhe=fhe,
    )


global_case, global_seed = None, None


def main():
    #seeds = [0, 42, 2023, 453749, 353576, 455367, 453321, 200200, 795326, 999999]
    #models = ["autoencoder", "vae"]
    #cases = ["autism", "id"]
    seeds, models, cases, kappas = [0], ['autoencoder'], ['autism'], [5, 10, 15, 20]

    global global_case, global_seed
    for case in cases:
        global_case = case
        if case == "id":
            kappas = [1, 2, 3, 4]
        else:
            kappas = [5, 10, 15, 20]
        for model in models:
            for seed in seeds:
                global_seed = seed
                print(f"Running using seed={seed}, Model={model}, Case={case}")
                run_experiment(
                    seed=seed,
                    case=case,
                    model=model,
                    kappas=kappas,
                    plot_datasets=False,
                    plot_history=False,
                    plot_losses=False
                )


if __name__ == "__main__":
    main()
