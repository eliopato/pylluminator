"""Mathematical tools for samples processing """

import numpy as np
from statsmodels.robust import mad
from scipy.stats import norm
import patsy
from pylluminator.utils import get_logger


LOGGER = get_logger()


def iqr(data: np.array) -> float:
    """Get InterQuartile Range, defined by the difference between the 75th percentile (Q3) and the 25th percentile (Q1)

    :param data: input data
    :type data: numpy.ndarray

    :return: the IQR value
    :rtype: float
    """
    quartile_1 = np.percentile(data, 25)
    quartile_3 = np.percentile(data, 75)

    # Calculate IQR
    return quartile_3 - quartile_1


def huber(values: np.array, k=1.5, tol=1e-6) -> (float, float):
    """Perform Huber's M-estimator for robust estimation of mean and scale.

    :param values: Array of data points.
    :type values: numpy.ndarray
    :param k: Tuning parameter that controls the threshold for outlier rejection (default: 1.5).
    :type k: float
    :param tol: Convergence tolerance for the mean estimation (default: 1e-6).
    :type tol: float

    :return: A tuple containing the estimated mean and scale (mu, sigma).
    :rtype: tuple[float, float]
    """
    # remove NaN
    values = values[~np.isnan(values)]

    # initialize values
    mu = np.median(values)
    sigma = mad(values)  # median absolute deviation (MAD)

    if sigma == 0:
        LOGGER.warning('cannot estimate scale : MAD is zero for this sample')
        return None, None

    # iteratively refine the estimate of the mean (mu)
    while True:
        clipped_values = np.clip(values, a_min=mu-k*sigma, a_max=mu+k*sigma)
        mu_updated = np.mean(clipped_values)
        if abs(mu-mu_updated) < tol*sigma:
            break
        mu = mu_updated

    return mu, sigma


def background_correction_noob_fit(in_band_signal: np.array, out_of_band_signal: np.array) -> tuple:
    """
    Perform background correction using the Noob method.

    :param in_band_signal: Array of foreground signal intensities.
    :type in_band_signal: numpy.ndarray
    :param out_of_band_signal: Array of background signal intensities.
    :type out_of_band_signal: numpy.ndarray

    :return: estimated background mean (`mu`), background scale (`sigma`), and correction factor (`alpha`)
    :rtype: tuple[float, float, float]
    """
    # Calculate robust estimates for background
    background_mean, background_sigma = huber(out_of_band_signal)

    if background_mean is None or background_sigma is None:
        LOGGER.error("Failed to estimate mean and scale for background signal")
        return None, None, None

    # Calculate alpha
    foreground_mean, _ = huber(in_band_signal)

    if foreground_mean is None:
        LOGGER.error("Failed to estimate mean and scale for foreground signal")
        return None, None, None

    alpha = np.maximum(foreground_mean - background_mean, 10)

    return background_mean, background_sigma, alpha


def norm_exp_convolution(mu: float, sigma: float, alpha: float, signal_values: np.array, offset: int | float) -> np.array:
    """Perform normalization and background correction on signal values using a normal-exponential convolution model.

    :param mu: The mean (mu) of the background signal.
    :type mu: float
    :param sigma: The standard deviation (sigma) of the background signal.
    :type sigma: float
    :param alpha: The correction factor, representing the shift in signal.
    :type alpha: float
    :param signal_values: Array of observed signal values to be corrected.
    :type signal_values: numpy.ndarray
    :param offset: A constant value to add to the corrected signal for padding.
    :type offset: int | float

    :return: The background-corrected and normalized signal values.
    :rtype: numpy.ndarray
    """
    # Validate parameters
    if any(param is None for param in (alpha, sigma, mu)):
        LOGGER.warning(f'stopping - at least a parameter is not set: alpha {alpha}, sigma {sigma}, mu {mu}')
        return signal_values
    if alpha <= 0:
        LOGGER.warning(f'stopping - alpha must be > 0 (alpha: {alpha})')
        return signal_values
    if sigma <= 0:
        LOGGER.warning(f'stopping - sigma must be > 0 (sigma: {sigma})')
        return signal_values

    signal_values = signal_values.flatten()
    variance = sigma * sigma

    # Apply the normalization and background correction model
    signal_shifted = signal_values - mu - variance/alpha

    # Compute the signal correction using a normal-exponential convolution model
    adjusted_signal = (
            signal_shifted
            + variance * np.exp(norm(signal_shifted, sigma).logpdf(0)
                                - norm(signal_shifted, sigma).logsf(0))
    )

    if np.any(adjusted_signal < 0):
        LOGGER.warning('Limit of numerical accuracy reached with very low intensity or very high background: '
                       'setting adjusted intensities')
        adjusted_signal = np.clip(adjusted_signal, a_min=1e-06, a_max=None)

    # Add the offset to the corrected signal
    return (adjusted_signal + offset).astype(np.float32)


def quantile_normalization_using_target(source_array: np.array, target_array: np.array) -> np.array:
    """ Perform quantile normalization on the source_array using target_array as the target distribution, even if the
    arrays have different sizes.

    :param source_array: array to normalize
    :type source_array: numpy.ndarray
    :param target_array: array to use as target distribution
    :type target_array: numpy.ndarray

    :return: the quantile-normalized array, of the same size as source_array
    :rtype: numpy.ndarray
    """

    target_sorted = np.sort(target_array)
    source_ranks = source_array.argsort().argsort()

    interp_target = np.interp(
        np.linspace(0, 1, len(source_array)),
        np.linspace(0, 1, len(target_sorted)),
        target_sorted
    )

    return interp_target[source_ranks]

def get_factors_from_formula(formula: str) -> list[str]:
    """ Get the factors (column names) from a formula

    :param formula: the formula to parse
    :type formula: str

    :return: the list of factors
    :rtype: list[str]
    """
    md = patsy.ModelDesc.from_formula(formula)
    termlist = md.rhs_termlist + md.lhs_termlist

    factors = []
    for term in termlist:
        for factor in term.factors:
            factors.append(factor.name())

    return factors