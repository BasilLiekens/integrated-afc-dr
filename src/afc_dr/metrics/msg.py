import numpy as np
import scipy as sp


def msg(f: np.ndarray) -> float:
    """
    Given the feedback path `f` and the delay in the forward path `dG`,
    compute the maximal stable gain by linear interpolation. The result
    is returned as a scaling, not in dB!

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
    freqs_crossings = freqs[zero_crossings + 1] - np.abs(
        phase[zero_crossings + 1] - np.ceil(phase[zero_crossings + 1])
    ) * (
        (freqs[zero_crossings + 1] - freqs[zero_crossings])
        / np.abs(phase[zero_crossings + 1] - phase[zero_crossings])
    )

    # Re-evaluate to obtain magnitudes: nonlinear, so interpolation does not make sense
    _, magn_at_crossings = sp.signal.freqz(f, worN=np.asarray(freqs_crossings))
    msg = 1 / np.max(np.abs(magn_at_crossings))

    return msg
