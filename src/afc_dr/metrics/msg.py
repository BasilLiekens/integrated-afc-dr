import numpy as np
import scipy as sp


def msg(f: np.ndarray) -> float:
    """
    Given the feedback path `f` compute the maximal stable gain by linear
    interpolation. The result is returned as a scaling, not in dB!

    Parameters
    ----------
    f: np.ndarray, 1D
        The impulse response of the feedback path `f`.

    Returns
    -------
    The inverse of the maximal gain at unstable frequencies, the result is
    returned as a plain scalar, not in dB.
    """
    worN = 4096  # hardcoded value
    freqs, response = sp.signal.freqz(f, worN=worN)

    phase = np.unwrap(np.angle(response)) / (2 * np.pi)

    zero_crossings = np.nonzero(np.diff(np.ceil(phase)))[0]
    freqs_crossings = np.zeros((zero_crossings.shape[0],))
    for i in range(zero_crossings.shape[0]):
        zero_crossing = zero_crossings[i]
        if phase[zero_crossing + 1] > phase[zero_crossing]:
            offset = phase[zero_crossing + 1] - np.floor(phase[zero_crossing + 1])
        else:
            offset = np.ceil(phase[zero_crossing + 1]) - phase[zero_crossing + 1]

        freqs_crossings[i] = freqs[zero_crossing + 1] - offset * (
            freqs[zero_crossing + 1] - freqs[zero_crossing]
        ) / (np.abs(phase[zero_crossing + 1] - phase[zero_crossing]))

    # Re-evaluate to obtain magnitudes: nonlinear, so interpolation does not make sense
    _, magn_at_crossings = sp.signal.freqz(f, worN=np.asarray(freqs_crossings))
    if magn_at_crossings.size == 0:
        raise ValueError("System is always stable.")
    else:
        msg = 1 / np.max(np.abs(magn_at_crossings))

    return msg

