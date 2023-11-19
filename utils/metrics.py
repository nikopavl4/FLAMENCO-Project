"""
Methods for calculating several metrics.
"""

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import (pairwise_distances,
                             average_precision_score,
                             roc_auc_score,
                             roc_curve)


def sireos(
        data: np.ndarray,
        scores: np.ndarray
) -> float:
    """
    Compute the SIREOS score for a given data set as defined in:
        Marques, H. O., Zimek, A., Campello, R. J., & Sander, J. (2022, September).
        Similarity-Based Unsupervised Evaluation of Outlier Detection.
        In International Conference on Similarity Search and Applications (pp. 234-248).

    Args:
        data (np.ndarray): A 2-dimensional array where each row is a data point and each column is a feature.
        scores (ndarray): A 1-dimensional array containing scores for each data point.

    Returns:
        float: The computed SIREOS score for the given data.

    Note:
        The pairwise distance is computed for each data point pair.
        The 1st percentile of non-zero distances is used for normalization.
        The scores are normalized such that they sum to 1.
        The final score is computed as a weighted average of the pairwise distances, where the weights
        are given by the normalized scores.
    """
    n_samples, n_features = data.shape
    # pairwise distance
    D = pairwise_distances(data)
    t = np.quantile(D[np.nonzero(D)], 0.01)

    # score normalization
    scores = scores / scores.sum()

    score = 0
    for j in range(n_samples):
        score += np.mean(np.exp(
            -np.linalg.norm(data[None, j, :] - np.delete(data[None, :, :],
                                                         j, axis=1),
                            axis=-1) ** 2 / (2 * t * t))) * scores[j]
    return score


def precision_at_k(
        y_true: np.ndarray,
        scores: np.ndarray,
        k=10,
        normalize: bool = False,
) -> float:
    """
    Compute the Precision at K (P@K) for a set of predictions and true labels.

    Precision at K is used to determine the proportion of positive (or relevant)
    instances among the top K instances as ranked by the prediction scores.

    Args:
        y_true (np.ndarray): A 1D array of true labels. 1 indicates a positive instance and 0 otherwise.
        scores (np.ndarray): A 1D array of scores associated with each instance.
        Higher values indicate higher relevance or likelihood of being positive.
        k (int, default=10): The number of top instances to consider.

    Returns:
        float: The precision at K (P@K) value.

    Note:
        If there are fewer than K instances, the function computes P@K over all available instances.
    """
    assert len(y_true) == len(scores), "Length of y_true and scores must be the same."

    # get the indices sorted by score from high to low
    indices = np.argsort(scores)[::-1]

    # select top k
    top_k_indices = indices[:k]

    # select labels for top k indices
    top_k_labels = y_true[top_k_indices]

    # count the number of positive labels (anomalies)
    num_positive = np.sum(top_k_labels)

    # calculate P@k
    p_at_k = num_positive / k

    if normalize:
        # count the number of actual positive labels (anomalies)
        total_positive = np.sum(y_true)

        # calculate the best possible P@k
        if total_positive >= k:
            best_p_at_k = 1
        else:
            best_p_at_k = total_positive / k

        # calculate normalized P@k
        p_at_k = p_at_k / best_p_at_k

    return p_at_k


def adjusted_precision_at_k(
        y_true: np.ndarray,
        scores: np.ndarray,
        k=10
) -> float:
    """
    Compute the Adjusted Precision at K (Adjusted P@K) for a set of predictions and true labels.

    Adjusted P@k is used to account for the baseline precision in the dataset.
    It adjusts the precision at k by the fraction of positive instances in the entire dataset.

    Args:
        y_true (np.ndarray): A 1D array of true labels. 1 indicates a positive instance and 0 otherwise.
        scores (np.ndarray): A 1D array of scores associated with each instance.
        Higher values indicate higher relevance or likelihood of being positive.
        k (int, default=10): The number of top instances to consider.

    Returns:
        float: The adjusted precision at K (Adjusted P@K) value.
    """
    # precision at k
    p_at_k = precision_at_k(y_true, scores, k=k)

    # baseline precision
    baseline_precision = np.sum(y_true) / len(y_true)

    # adjusted P@k
    adjusted_p_at_k = (p_at_k - baseline_precision) / (1 - baseline_precision)

    return adjusted_p_at_k


def average_precision(
        y_true: np.ndarray,
        scores: np.ndarray
) -> float:
    """
    Compute the Average Precision (AP) score.

    Average Precision summarizes the precision-recall curve as the weighted mean of precisions achieved
    at each threshold, with the increase in recall from the previous threshold used as the weight.

    Args:
        y_true (np.ndarray): A 1D array of true labels. 1 indicates a positive instance and 0 otherwise.
        scores (np.ndarray): A 1D array of scores associated with each instance.
        Higher values indicate higher relevance or likelihood of being positive.

    Returns:
        float: The Average Precision score.
    """
    return average_precision_score(y_true, scores)


def adjusted_average_precision(
        y_true: np.ndarray,
        scores: np.ndarray
) -> float:
    """
    Compute the Adjusted Average Precision (AP) score.

    Args:
        y_true (np.ndarray): A 1D array of true labels. 1 indicates a positive instance and 0 otherwise.
        scores (np.ndarray): A 1D array of scores associated with each instance.
        Higher values indicate higher relevance or likelihood of being positive.

    Returns:
        float: The Adjusted Average Precision score.
    """
    ap = average_precision(y_true, scores)

    # baseline
    baseline_ap = np.sum(y_true) / len(y_true)

    adjusted_ap = (ap - baseline_ap) / (1 - baseline_ap)
    return adjusted_ap


def roc_auc(
        y_true,
        scores,
        average: str = "macro"
) -> float:
    """
    Compute the ROC AUC score based on true labels and predicted scores.

    Args:
        y_true: A 1D array of true labels. 1 indicates a positive instance and 0 otherwise.
        scores (np.ndarray): A 1D array of scores associated with each instance.

    Returns:
        float: The ROC AUC score.
    """
    return roc_auc_score(y_true, scores, average=average)


def plot_roc_curve(
        y_true: np.ndarray,
        scores: np.ndarray
) -> None:
    """
    Plots the Receiver Operating Characteristic (ROC) curve for the given true binary labels and their
    corresponding scores.

    Args:
        y_true (np.ndarray): A 1D array of true labels. 1 indicates a positive instance and 0 otherwise.
        scores (np.ndarray): A 1D array of scores associated with each instance.
        Higher values indicate higher relevance or likelihood of being positive.

    Returns:
        None: The function plots the ROC curve using matplotlib and does not return any value.

    Note:
        The area under the ROC curve (AUC) is also displayed in the plot's legend.
    """
    roc_auc_ = roc_auc(y_true, scores)
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    plt.figure()

    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_)

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')

    plt.show()
    plt.close()


if __name__ == "__main__":
    y_true = np.array([1, 0, 0, 1, 0, 0, 1])
    y_scores = np.array([0.9, 0.1, 0.01, 0.7, 0.2, 0.5, 0.4])

    print(precision_at_k(y_true, y_scores, 2, normalize=False))
    print(precision_at_k(y_true, y_scores, 2, normalize=True))
    print(precision_at_k(y_true, y_scores, 3, normalize=False))
    print(precision_at_k(y_true, y_scores, 3, normalize=True))
    print(precision_at_k(y_true, y_scores, 5, normalize=False))
    print(precision_at_k(y_true, y_scores, 5, normalize=True))
    print("\n\n\n")

    print(adjusted_precision_at_k(y_true, y_scores, 2))
    print(adjusted_precision_at_k(y_true, y_scores, 3))
    print(adjusted_precision_at_k(y_true, y_scores, 5))

    print(average_precision(y_true, y_scores))
    print(adjusted_average_precision(y_true, y_scores))

    print(roc_auc(y_true, y_scores))

    # plot_roc_curve(y_true, y_scores)
