"""
Score normalization
"""

import numpy as np
from scipy.stats import norm


def normalize(scores):
    """
    Normalize the obtained anomaly scores.

    Args:
        scores (np.ndarray): A 1D array of scores associated with each instance.

    Returns:
        np.ndarray: Normalized scores

    Note:
        Kriegel, H. P., Kroger, P., Schubert, E., & Zimek, A. (2011, April).
        Interpreting and unifying outlier scores.
        In Proceedings of the 2011 SIAM International Conference on Data Mining (pp. 13-24).
        Society for Industrial and Applied Mathematics.

        A score is normal if S is regular and the values are in [0, 1].
        A score is regular if S(o) >= 0 for any observation,
                              S(o) ~= 0 if o is an inlier,
                              S(o) >> 0 if o is an outlier.

        In AutoEncoders the reconstruction error is often used as a scoring function. In other words,
        when the reconstruction error is low (close to 0), the observation is considered as an inlier, while
        a higher error indicates an outlier. In addition, the error is always >= 0 and hence, using the reconstruction
        error as anomaly scoring is regular. Thus, we only normalize the values to the range [0, 1].
    """
    random_variable = _get_random_variable(scores)

    return np.maximum(
        0., 2. * random_variable.cdf(scores) - 1.
    )


def _get_random_variable(scores):
    """Get the RV object according to the derived anomaly scores."""
    loc, scale = norm.fit(scores)

    return norm(loc=loc, scale=scale)


if __name__ == "__main__":
    a = np.array([0.01, 0.1, 0.2, 0.3, 0.35,
                  0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 1., 2., 3.])
    normalized = normalize(a)
    print(a)
    print(normalized)
