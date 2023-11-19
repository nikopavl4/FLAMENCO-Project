import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import Dict, List, Union, Any

import pandas as pd
import seaborn as sns
import torch.cuda
from matplotlib import pyplot as plt
from torch.nn.modules.loss import _Loss

from experiment_public.common import plot_dist, get_model, seed_all, load_dataset
from utils.helpers import two_dim_transform
from utils.preprocessing import (get_X_y, to_torch_data, generate_dataset)
from utils.train_utils import fit, predict

global_seed = -1
global_case = "none"


def get_dataset(
        df: pd.DataFrame,
        split_column: str = "scores",
        target_column: str = "target",
        test_size: float = 0.2,
        random_state: int = 42,
        log: bool = True,
        plot: bool = True
) -> Dict[str, Dict[str, pd.DataFrame]]:
    dataset = generate_dataset(
        df, split_column=split_column, target_column=target_column,
        test_size=test_size, random_state=random_state, log=log
    )
    means = [100]
    if plot:
        for mean in means:
            plot_dist(df, mean)
    return dataset


def transform_data(
        full_data,
        train_data,
        test_data,
        exclude_col: str = "target",
        plot=True,
):
    x_transformed_train = two_dim_transform(train_data, plot=plot, exclude_col=exclude_col)
    x_transformed_test = two_dim_transform(test_data, plot=plot, exclude_col=exclude_col)
    x_transformed_all = two_dim_transform(full_data, plot=plot, exclude_col=exclude_col)
    return x_transformed_train, x_transformed_test, x_transformed_all


def train(
        dataset: Dict[str, pd.DataFrame],
        identifier: str = "case_id",
        y_column: Union[str, None] = "target",
        two_dim_transform_: bool = True,
        plot_transformed: bool = False,
        scale_data: bool = True,
        train_batch_size: int = 32,
        model_name: str = "autoencoder",
        hidden_dim: int = 32,
        epochs: int = 200,
        optim_name: str = "adam",
        optim_args: Union[Dict, Any] = None,
        lr: float = 1e-3,
        criterion: _Loss = None,
        anomaly_percentile: int = 90,
        kappas: List[int] = [1, 2, 3, 4, 5, 10, 15, 20],
        log_interval: int = 1,
        plot_interval: Union[None, int] = 10,
        plot_history: bool = True,
        plot_losses: bool = True,
        device: str = 'cuda',
):
    train_data, test_data = dataset["train"], dataset["test"]

    columns_to_scale = [col for col in train_data.columns if col not in [identifier, y_column]]

    if scale_data:
        train_scaled = train_data.drop(columns=[identifier])
        train_scaled[columns_to_scale] = train_scaled[columns_to_scale].clip(lower=0, upper=100)
        train_scaled[columns_to_scale] = train_scaled[columns_to_scale] / 100

        test_scaled = test_data.drop(columns=[identifier])
        test_scaled[columns_to_scale] = test_scaled[columns_to_scale].clip(lower=0, upper=100)
        test_scaled[columns_to_scale] = test_scaled[columns_to_scale] / 100
    else:
        train_scaled = train_data
        test_scaled = test_data

    full_data_scaled = pd.concat([train_scaled, test_scaled])

    # get 2d transformed data
    x_transformed_train, x_transformed_test, x_transformed_all = None, None, None
    if two_dim_transform_:
        x_transformed_train, x_transformed_test, x_transformed_all = transform_data(
            train_data=train_scaled, test_data=test_scaled, full_data=full_data_scaled, plot=plot_transformed,
        )

    # try to get X and y. If y cannot be found, then only features X are returned.
    X_train, X_test, y_train, y_test, X_all, y_all = get_X_y(
        train_scaled, test_scaled, full_data_scaled
    )

    # to torch dataloader
    train_loader, test_loader, all_loader = to_torch_data(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        X_all=X_all,
        y_all=y_all,
        X_train_batch_size=train_batch_size,
        X_test_batch_size=1,
        X_all_batch_size=1
    )

    # init pytorch model
    model = get_model(
        model_name=model_name,
        input_dim=X_train.shape[1],
        hidden_dim=hidden_dim
    ).to(device)

    model, history, anomalies, scores = fit(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        data_loader=all_loader,
        criterion=criterion,
        optim=optim_name,
        optim_args=optim_args,
        lr=lr,
        epochs=epochs,
        x_transformed_test=x_transformed_test,
        x_transformed_data=x_transformed_all,
        percentile=anomaly_percentile,
        normalize_scores=True,
        kappas=kappas,
        log_interval=log_interval,
        plot_interval=plot_interval,
        device=device,
        plot_history=plot_history,
        fl_note=None
    )

    if plot_losses:
        train_losses, test_losses = history["train_losses"], history["test_losses"]
        test_normal_losses, test_abnormal_losses = history["test_losses_normal"], history["test_losses_abnormal"]
        test_unknown_losses = history["test_losses_unknown"]
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(3.5, 3.5))

        axs.plot(train_losses, label="Train Loss")
        axs.plot(test_losses, label="Test Loss")
        axs.plot(test_normal_losses, label="Test Normal Loss")
        axs.plot(test_abnormal_losses, label="Test Abnormal Loss")
        axs.plot(test_unknown_losses, label="Test Unknown Loss")
        axs.set_ylabel("Loss")
        axs.legend()
        plt.show()
        plt.close()

    find_all = False
    has_labels = False
    if y_column is not None:
        has_labels = True
        find_all = True

    _, all_sireos, all_normalized_p, all_auc_roc, all_ap, all_anomalies, all_scores = predict(
        model=model,
        data_loader=all_loader,
        data_id="test",
        criterion=criterion,
        percentile=90,
        epoch=0,
        x_transformed=x_transformed_all,
        has_labels=has_labels,
        kappas=kappas,
        plot_interval=None,  # do not plot
        normalize_scores=True,
        find_all=find_all,
        device=device
    )
    print("SIREOS", all_sireos)
    print("P@K", all_normalized_p)
    print("AUC", all_auc_roc)
    print("AP", all_ap)

    # plt.hist(all_scores, bins=10)
    # plt.show()
    # plt.close()

    _, test_sireos, test_normalized_p, test_auc_roc, test_ap, test_anomalies, test_scores = predict(
        model=model,
        data_loader=test_loader,
        data_id="test",
        criterion=criterion,
        percentile=90,
        epoch=0,
        x_transformed=x_transformed_test,
        has_labels=has_labels,
        kappas=kappas,
        plot_interval=None,  # do not plot
        normalize_scores=True,
        find_all=find_all,
        device=device
    )
    print("SIREOS", test_sireos)
    print("P@K", test_normalized_p)
    print("AUC", test_auc_roc)
    print("AP", test_ap)
    # plt.hist(test_scores, bins=10)
    # plt.show()
    # plt.close()

    # write this to file
    convergence_cols = [
        "Train Loss", "Test Loss", "Test Normal Loss", "Test Abnormal Loss",
        "Test Unknown Loss", "SIREOS"
    ]

    p_scores_keys = list(next(iter(history["p_scores"])).keys())

    res_cols = [
        "SIREOS", "AUC-ROC", "AP"
    ]
    res_cols.extend(p_scores_keys)

    convergence_df = pd.DataFrame(columns=convergence_cols)
    convergence_df["Train Loss"] = history['train_losses']
    convergence_df["Test Loss"] = history['test_losses']
    convergence_df['Test Normal Loss'] = history['test_losses_normal']
    convergence_df['Test Abnormal Loss'] = history['test_losses_abnormal']
    convergence_df['Test Unknown Loss'] = history['test_losses_unknown']
    convergence_df['SIREOS'] = history['sireos']

    try:
        convergence_file_path = f"../results/centralized/convergence/model_{model_name}_history_seed_{global_seed}_case_{global_case}.csv"
        convergence_df.to_csv(convergence_file_path, index=False)
    except OSError:
        convergence_file_path = f"./results/centralized/convergence/model_{model_name}_history_seed_{global_seed}_case_{global_case}.csv"
        convergence_df.to_csv(convergence_file_path, index=False)

    metrics_df = pd.DataFrame(columns=res_cols)
    metrics_df["SIREOS"] = [test_sireos]
    metrics_df["AUC-ROC"] = [test_auc_roc]
    metrics_df["AP"] = [test_ap]
    p_scores = test_normalized_p
    for k in p_scores_keys:
        metrics_df[k] = [p_scores[k]]

    try:
        final_res_file_path = f"../results/centralized/metrics/model_{model_name}_metrics_seed_{global_seed}_case_{global_case}.csv"
        metrics_df.to_csv(final_res_file_path, index=False)
    except OSError:
        final_res_file_path = f"./results/centralized/metrics/model_{model_name}_metrics_seed_{global_seed}_case_{global_case}.csv"
        metrics_df.to_csv(final_res_file_path, index=False)

    print(history)


def run_experiment(
        seed=0,
        target_col="target",
        identifier="case_id",
        case="autism",
        cid="client_id",
        model="autoencoder",
        hidden_dim=32,
        epochs=100,
        plot_history=True,
        plot_losses=True,
        kappas=[5, 10, 15, 20, 25, 30]
):
    seed_all(seed)
    df_train, df_test = load_dataset(
        case=case,
        identifier=identifier
    )
    if cid in df_train.columns:
        df_train = df_train.drop(columns=[cid])
    if cid in df_test.columns:
        df_test = df_test.drop(columns=[cid])

    if target_col not in df_train.columns:
        target_col = None

    full_dataset = {"train": df_train, "test": df_test}

    # training params
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    criterion = torch.nn.MSELoss()

    train(
        dataset=full_dataset,
        y_column=target_col,
        plot_transformed=False,
        model_name=model,
        hidden_dim=hidden_dim,
        criterion=criterion,
        device=device,
        plot_interval=None,
        epochs=epochs,
        plot_history=plot_history,
        kappas=kappas,
        plot_losses=plot_losses
    )


def main():
    sns.set(style="whitegrid")
    #seeds = [0, 42, 2023, 453749, 353576, 455367, 453321, 200200, 795326, 999999]
    #models = ["autoencoder", "vae"]
    #cases = ["autism", "id"]
    #kappas = [5, 10, 15, 20]'''
    seeds, models, cases, kappas = [0], ['autoencoder'], ['autism'], [5, 10, 15, 20]
    global global_case, global_seed
    for case in cases:
        global_case = case
        if case == "id":
            kappas = [1, 2, 3, 4]
        for model in models:
            for seed in seeds:
                global_seed = seed
                print(f"Running using seed={seed}, Model={model}, Case={case}")
                run_experiment(
                    seed=seed,
                    case=case,
                    model=model,
                    plot_history=False,
                    plot_losses=False,
                    kappas=kappas
                )


if __name__ == '__main__':
    main()
