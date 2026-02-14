import numpy as np
import pesq as pypesq


def pesq(x_ref: np.ndarray, y: np.ndarray, fs: int, mode: str) -> np.ndarray:
    """
    Wrapper around the `pesq` implementation of the equally named PyPi package
    to compute the narrow- or wideband PESQ scores [1].

    Parameters
    ----------
    x_ref: np.ndarray
        Clean reference signal of any dimension, the first dimension should be
        the time axis.

    y: np.ndarray
        The degraded signal, same shape as y.

    fs: int
        The sampling frequency in Hz

    mode: str
        The mode of the pesq computation, 'wb' or 'nb'.

    Returns
    -------
    A numpy array with shape `x_ref.shape[1:]` containing the pesq scores for
    each input-output pair.

    References
    ----------
    [1] A. W. Rix, J. G. Beerends, M. P. Hollier, and A. P. Hekstra,
        “Perceptual evaluation of speech quality (PESQ)-a new method for
        speech quality assessment of telephone networks and codecs,” in 2001
        IEEE International Conference on Acoustics, Speech, and Signal
        Processing. Proceedings (Cat. No.01CH37221), Salt Lake City, UT, USA:
        IEEE, 2001, pp. 749–752. doi: 10.1109/ICASSP.2001.941023.
    """
    x_ref_prime = x_ref.reshape((x_ref.shape[0], -1))
    y_prime = y.reshape((y.shape[0], -1))

    res = np.array(
        [
            pypesq.pesq(fs, x_ref_prime[:, i], y_prime[:, i], mode)
            for i in range(x_ref.shape[1])
        ]
    )

    return np.array(res).reshape(x_ref.shape[1:])
