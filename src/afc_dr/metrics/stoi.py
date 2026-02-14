import numpy as np
import pystoi


def stoi(
    x_ref: np.ndarray, y: np.ndarray, fs: int, extended: bool = False
) -> np.ndarray:
    """
    Compute the `stoi` metric for a given clean signal `x_ref` and a processed
    one `y`.

    Parameters
    ----------
    x_ref: np.ndarray
        Reference signal, allowed to be multidimensional, but assumed to have
        the time axis as the first one.

    y: np.ndarray
        The processed signal, assumed to be of the same shape as `x_ref`.

    fs: int
        Sampling frequency of the system.

    extended: bool
        Whether or not to use the extended measure.

    Returns
    -------
    A numpy array with shape `x_ref.shape[1:]` containing the `stoi` for each
    input signal.

    References
    ----------
    [1] C.H.Taal, R.C.Hendriks, R.Heusdens, J.Jensen 'A Short-Time Objective
        Intelligibility Measure for Time-Frequency Weighted Noisy Speech',
        ICASSP 2010, Texas, Dallas.
    [2] C.H.Taal, R.C.Hendriks, R.Heusdens, J.Jensen 'An Algorithm for
        Intelligibility Prediction of Time-Frequency Weighted Noisy Speech',
        IEEE Transactions on Audio, Speech, and Language Processing, 2011.
    [3] Jesper Jensen and Cees H. Taal, 'An Algorithm for Predicting the
        Intelligibility of Speech Masked by Modulated Noise Maskers', IEEE
        Transactions on Audio, Speech and Language Processing, 2016.
    """
    # flatten to iterate over
    x_prime = x_ref.reshape((x_ref.shape[0], -1))
    y_prime = y.reshape((y.shape[0], -1))
    res = np.array(
        [
            pystoi.stoi(x_prime[:, i], y_prime[:, i], fs, extended)
            for i in range(x_prime.shape[1])
        ]
    )

    return res.reshape(x_ref.shape[1:])
