"""
Training and evaluation functions.
"""

from typing import Tuple, Union, Dict, Any, List

import matplotlib.pyplot as plt
import numpy as np
import torch.nn
from torch.utils.data import DataLoader

from torch.nn.modules.loss import _Loss

from utils.helpers import get_optimizer
from utils.metrics import (sireos,
                           precision_at_k,
                           average_precision,
                           roc_auc)
from utils.scoring import normalize


def fit(
        model: torch.nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        data_loader: Union[None, DataLoader],
        criterion: _Loss,
        optim: str = "adam",
        optim_args: Union[Dict, Any] = None,
        lr: float = 1e-3,
        epochs: int = 100,
        x_transformed_test: Union[None, np.ndarray] = None,
        x_transformed_data: Union[None, np.ndarray] = None,
        percentile: int = 90,
        normalize_scores: bool = True,
        kappas: List[int] = [1, 2, 3, 4, 5, 10, 15, 20],
        log_interval: int = 1,
        plot_interval: Union[None, int] = 10,
        device: str = 'cuda',
        plot_history: bool = True,
        fl_note: str = None,
) -> Tuple[torch.nn.Module, Dict[str, List[float]], np.ndarray, np.ndarray]:
    """"""
    optimizer = get_optimizer(model, optim, lr, args=optim_args)
    losses, test_losses, test_normal_losses, test_abnormal_losses, test_unknown_losses = [], [], [], [], []
    anomalies, scores = None, None
    sireos_scores, p_at_k_scores, normalized_p_scores, auc_roc_scores, ap_scores = [], [], [], [], []

    convergence_checker = None

    tmp_batch_x, tmp_batch_y = next(iter(train_loader))
    if tmp_batch_x.shape == tmp_batch_y.shape:
        has_labels = False
    else:
        has_labels = True

    for epoch in range(epochs):
        model.train()
        running_loss = []
        for idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            encoded, recon = model(data)
            loss = criterion(recon, data)
            loss.backward()
            optimizer.step()
            running_loss.append(loss.item())

        epoch_loss = sum(running_loss) / len(running_loss)
        if fl_note is None:
            print(f"[Epoch {epoch + 1}] Train Loss: {epoch_loss:.4f}")
        elif fl_note is not None and epoch + 1 == epochs:
            print(f"[{fl_note}][Epoch {epoch + 1}] Train Loss: {epoch_loss:.4f}")
        losses.append(epoch_loss)

        if x_transformed_test is not None and plot_interval is not None and (
                epoch == 0 or (epoch + 1) % plot_interval == 0):
            predict(
                model, train_loader, "train",
                criterion, percentile, epoch + 1, x_transformed_test,
                has_labels=has_labels,
                kappas=kappas, plot_interval=plot_interval,
                normalize_scores=normalize_scores,
                device=device, fl_note=fl_note
            )
        if x_transformed_data is not None and plot_interval is not None and (
                epoch == 0 or (epoch + 1) % plot_interval == 0):
            predict(
                model, data_loader, "all",
                criterion, percentile, epoch + 1, x_transformed_data,
                has_labels=has_labels,
                kappas=kappas, plot_interval=plot_interval,
                normalize_scores=normalize_scores, device=device, fl_note=fl_note
            )

        test_loss, sireos_score, normalized_p_at_k, auc_roc, ap, anomalies, scores = predict(
            model, test_loader, "test",
            criterion, percentile, epoch + 1, x_transformed_test,
            has_labels=has_labels,
            kappas=kappas, plot_interval=plot_interval,
            normalize_scores=normalize_scores, device=device, fl_note=fl_note
        )

        test_loss, test_normal_loss, test_abnormal_loss, test_unknown_loss = test_loss[0], test_loss[1], test_loss[2], \
        test_loss[3]
        test_losses.append(test_loss)
        test_normal_losses.append(test_normal_loss)
        test_abnormal_losses.append(test_abnormal_loss)
        test_unknown_losses.append(test_unknown_loss)

        sireos_scores.append(sireos_score)
        # p_at_k_scores.append(p_at_k)
        normalized_p_scores.append(normalized_p_at_k)
        auc_roc_scores.append(auc_roc)
        ap_scores.append(ap)

        if log_interval is not None and (epoch == 0 or (epoch + 1) % log_interval == 0):
            if has_labels:
                print(
                    f"[Epoch {epoch + 1}]"
                    f"\tSIREOS Score: {sireos_score:.4f}\n"
                    # f"\t\t\tP@k: {p_at_k}\n"
                    f"\t\t\tNP@k: {normalized_p_at_k}\n"
                    f"\t\t\tAUC-ROC: {auc_roc:.4f}\n"
                    f"\t\t\tAP: {ap}\n"
                )
            else:
                print(
                    f"[Epoch {epoch + 1} SIREOS Score: {sireos_score:.4f}\n"
                )

        if convergence_checker is not None:
            if convergence_checker.check_convergence(epoch_loss):
                print("Training loss converged. Stopping training")
                break

    if fl_note is None:
        top_sireos = min(sireos_scores)
        sireos_idx = sireos_scores.index(top_sireos)
        if has_labels:
            top_auc = max(auc_roc_scores)
            auc_idx = auc_roc_scores.index(top_auc)
            top_ap = max(ap_scores)
            ap_idx = ap_scores.index(top_ap)
            print(f"[Top Scores]\n\t[SIREOS] {top_sireos} (epoch={sireos_idx + 1})\n"
                  f"\t[AUC-ROC] {top_auc} (epoch={auc_idx})\n"
                  f"\t[AP] {top_ap} (epoch={ap_idx})")
        else:
            print(f"[Top Scores]\n\t[SIREOS] {top_sireos} (epoch={sireos_idx + 1})\n")

    if plot_history and has_labels:
        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))

        # auc-roc
        axs[0, 0].plot(auc_roc_scores)
        axs[0, 0].set_ylabel("AUC-ROC Score")
        # AP
        axs[0, 1].plot(ap_scores, "tab:orange")
        axs[0, 1].set_ylabel("AP Score")
        # sireos
        axs[1, 0].plot(sireos_scores, "tab:green")
        axs[1, 0].set_ylabel("Sireos Score")
        # losses
        axs[1, 1].plot(losses, "tab:red")
        axs[1, 1].set_ylabel("Train Loss")

        for ax in axs.flat:
            ax.set(xlabel="Epoch")

        plt.show()
        plt.close()
    elif plot_history and not has_labels:
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(5, 5))

        # sireos
        axs[0].plot(sireos_scores, "tab:green")
        axs[0].set_ylabel("Sireos Score")
        # losses
        axs[1].plot(losses, "tab:red")
        axs[1].set_ylabel("Train Loss")

        for ax in axs.flat:
            ax.set(xlabel="Epoch")

        plt.show()
        plt.close()

    history = {
        # losses
        "train_losses": losses,
        "test_losses": test_losses,
        "test_losses_normal": test_normal_losses,
        "test_losses_abnormal": test_abnormal_losses,
        "test_losses_unknown": test_unknown_losses,
        # metrics
        "sireos": sireos_scores,
        "p_scores": normalized_p_scores,
        "auc_roc": auc_roc_scores,
        "ap_scores": ap_scores
    }

    return model, history, anomalies, scores


def predict(
        model: torch.nn.Module,
        data_loader: DataLoader,
        data_id: str,
        criterion: _Loss,
        percentile: int,
        epoch: int,
        x_transformed: np.ndarray = None,
        has_labels: bool = True,
        kappas: List[int] = [1, 2, 3, 4, 5, 10, 15, 20],
        plot_interval: Union[None, int] = 10,
        normalize_scores: bool = True,
        find_all: bool = False,
        device: str = 'cuda',
        fl_note: Union[None, str] = None
) -> Union[
    None,
    Tuple[List[float], float, Union[None, Dict[int, float]], Union[None, float], Union[
        None, float], np.ndarray, np.ndarray]
]:
    """"""
    model.eval()
    model.to(device)
    scores = []
    running_normal_loss, running_abnormal_loss, running_unknown_loss = [], [], []
    data_list = []
    target_list = []
    data_normal_list = []
    scores_normal = []
    with torch.no_grad():
        for idx, (data, target) in enumerate(data_loader):
            data = data.to(device)
            for sample, t in zip(data, target):
                _, recon = model(sample)
                loss = criterion(recon, sample)
                scores.append(loss.item())
                if t == 0:
                    running_normal_loss.append(loss.item())
                    data_normal_list.append(sample.cpu().numpy())
                    scores_normal.append(loss.item())
                elif t == 1:
                    running_abnormal_loss.append(loss.item())
                else:
                    running_unknown_loss.append(loss.item())

            data_list.append(data.cpu().numpy())
            target_list.append(target.cpu().numpy())

    running_loss = sum(scores) / len(scores)
    running_normal_loss = sum(running_normal_loss) / len(running_normal_loss)
    if len(running_abnormal_loss) == 0:
        running_abnormal_loss = None
    else:
        running_abnormal_loss = sum(running_abnormal_loss) / len(running_abnormal_loss)
    running_unknown_loss = sum(running_unknown_loss) / len(running_unknown_loss)
    losses = [running_loss, running_normal_loss, running_abnormal_loss, running_unknown_loss]

    if data_id == "train":
        plt.figure()
        plt.hist(scores, bins=50, density=True, stacked=True,
                 alpha=0.7, color='blue', label='Train Loss Distribution')
        plt.xlabel("Loss")
        plt.ylabel("Density")
        plt.title(f"Train Loss Distribution (Epoch {epoch})")
        plt.show()
        plt.close()
        return None

    data_list = np.concatenate(data_list)
    data_normal_list = np.concatenate(data_normal_list)
    target_list = np.concatenate(target_list)

    scores = np.array(scores)
    scores_normal = np.array(scores_normal)
    cutoff = np.percentile(scores, percentile)

    anomalies: np.ndarray = scores > cutoff
    count_true = np.count_nonzero(anomalies)

    if plot_interval is not None and (epoch == 1 or epoch % plot_interval == 0):
        plt.figure()
        plt.hist(scores, bins=50, density=True, stacked=True,
                 alpha=0.7, color='blue', label='Loss Distribution')
        percentiles = [99, 98, 97, 96, 95]
        colors = ['red', 'green', 'purple', 'orange', 'cyan']
        for p, color in zip(percentiles, colors):
            p_value = np.percentile(scores, p)
            plt.axvline(p_value, color=color,
                        linestyle='dashed', label=f"{p}th Percentile")
        plt.xlabel("Loss")
        plt.ylabel("Density")
        plt.title(f"Loss Distribution with Percentiles (Epoch: {epoch}), #A={count_true}")
        plt.legend()
        plt.show()
        plt.close()

        if x_transformed is not None:
            plt.figure()
            plt.scatter(x_transformed[~anomalies, 0], x_transformed[~anomalies, 1], c='b', label="Normal")
            plt.scatter(x_transformed[anomalies, 0], x_transformed[anomalies, 1], c='r', marker='x', label="Anomaly")
            plt.title(f"2D Transformed data (Epoch {epoch}), #A={count_true})")
            plt.legend()
            plt.show()
            plt.close()

    # silhouette, db_score, ch_score = get_clustering_metrics(data_list, anomalies)

    sireos_score = sireos(data_list, normalize(scores))  # normalize the scores for sireos

    p_scores, normalized_p_scores, auc_roc, ap = None, None, None, None

    if has_labels:
        all_labeled_scores, labeled_scores, labels, all_labels = [], [], [], []
        labeled_anomalies = []
        labeled_normal_data = []
        for data, score, target, anomaly in zip(data_list, scores, target_list, anomalies):
            # if target != -1:
            all_labeled_scores.append(score)
            if target == -1:
                all_labels.append(0)
            else:
                all_labels.append(target)
            if target != -1:
                labeled_scores.append(score)
                labels.append(target)
                labeled_anomalies.append(anomaly)
            if target == 0:
                labeled_normal_data.append(data)

        all_labeled_scores = np.array(all_labeled_scores)
        labeled_scores = np.array(labeled_scores)
        labels = np.array(labels, dtype=int)
        labeled_anomalies = np.array(labeled_anomalies, dtype=int)
        all_labels = np.array(all_labels, dtype=int)

        if np.sum(labels) < 1 or len(labels) <= 1:
            msg = "Cannot Calculate classification metrics."
            if fl_note:
                msg = f"[%s] {msg}" % fl_note
            if epoch == 1:
                print(msg)
        else:
            # p_scores = dict()
            normalized_p_scores = dict()

            for k in kappas:
                # p_at_k = precision_at_k(y_true=labels, scores=labeled_scores, k=k, normalize=False)
                normalized_p_at_k = precision_at_k(y_true=labels, scores=labeled_scores, k=k, normalize=True)
                # p_scores[k] = p_at_k
                normalized_p_scores[k] = normalized_p_at_k

            auc_roc = roc_auc(labels, labeled_scores)
            ap = average_precision(labels, labeled_scores)

    if normalize_scores:
        scores = normalize(scores)

    return losses, sireos_score, normalized_p_scores, auc_roc, ap, anomalies, scores
