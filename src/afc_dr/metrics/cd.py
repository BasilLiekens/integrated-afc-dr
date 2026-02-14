import jax
import jax.numpy as jnp
import numpy as np
import scipy as sp


def cd(x: np.ndarray, y: np.ndarray, fs: int) -> np.ndarray:
    """
    Compute the cepstral distance between two 1D signals `x` and `y` with
    given stft parameters. The implementation is a literal translaton of
    the MATLAB files that were provided as part of the REVERB challenge. [1]

    This function is a wrapper around the actual implementation that provides
    the default arguments also used in the REVERB challenge.

    Parameters
    ----------
    x: jax.Array, 1D
        One of the two signals to use to compute the cepstral distance.

    y: jax.Array, 1D
        Similar to `x`, but now the second signal.

    fs: int
        Sampling frequency in Hz.

    References
    ----------
    [1] K. Kinoshita et al., “A summary of the REVERB challenge: state-of-the-art and
        remaining challenges in reverberant speech processing research,” EURASIP J.
        Adv. Signal Process., vol. 2016, no. 1, p. 7, Dec. 2016,
        doi: 10.1186/s13634-016-0306-6.
    """
    ## Default parameters of REVERB challenge
    frame_len = int(np.fix(0.025 * fs))
    hop = int(np.fix(0.01 * fs))
    win = sp.signal.windows.hann(frame_len, sym=False)
    N = 2 ** int(np.ceil(np.log2(frame_len)))
    order = 24

    return np.asarray(_cd(x, y, frame_len, hop, win, N, order))


@jax.jit(static_argnames=["frame_len", "hop", "order", "N"])
def _cd(
    x: jax.Array,
    y: jax.Array,
    frame_len: int,
    hop: int,
    win: jax.Array,
    N: int,
    order: int,
) -> jax.Array:
    """
    Inner body for cepstral distance computation. [1]

    Parameters
    ----------
    x: jax.Array, 1D
        One of the two signals to use to compute the cepstral distance.

    y: jax.Array, 1D
        Similar to `x`, but now the second signal.

    frame_len: int
        Size of frames used to compute the STFT representation.

    hop: int
        The hop between subsequent frames in the STFT.

    win: jax.Array, 1D
        The window to apply to frames in the STFT, should be of length `N`.

    N: int
        The DFT size to use when computing the cepstral coefficients. Is
        expected to be the next power of 2 of `frame_len`.

    order: int
        The order of cepstral coefficients that is taken into account.

    Returns
    -------
    The cepstral distance in a jax Array to facilitate later processing.

    References
    ----------
    [1] K. Kinoshita et al., “A summary of the REVERB challenge: state-of-the-art and
        remaining challenges in reverberant speech processing research,” EURASIP J.
        Adv. Signal Process., vol. 2016, no. 1, p. 7, Dec. 2016,
        doi: 10.1186/s13634-016-0306-6.
    """
    x = jax.lax.cond(x.shape[0] > y.shape[0], lambda: x[: y.shape[0]], lambda: x)
    y = jax.lax.cond(x.shape[0] > y.shape[0], lambda: y, lambda: y[: x.shape[0]])
    win = win.reshape((-1, 1))

    n_frames = (x.shape[0] - frame_len + hop) // hop

    idcs = jnp.arange(win.shape[0]).reshape((-1, 1)) + (
        jnp.arange(n_frames).reshape((1, -1)) * hop
    )

    X = x[idcs] * win
    Y = y[idcs] * win

    ceps_x = _real_cepstrum(X, N=N)
    ceps_y = _real_cepstrum(Y, N=N)

    ceps_x = ceps_x[: order + 1, :]
    ceps_y = ceps_y[: order + 1, :]

    ceps_x -= jnp.mean(ceps_x, axis=1, keepdims=True)
    ceps_y -= jnp.mean(ceps_y, axis=1, keepdims=True)

    e = (ceps_x - ceps_y) ** 2
    d = 10 / jnp.log(10) * jnp.sqrt(e[0, :] + 2 * jnp.sum(e[1:, :], axis=0))
    d = jnp.clip(d, min=0, max=10)

    return jnp.mean(d, keepdims=True)


@jax.jit(static_argnames=["N"])
def _real_cepstrum(X: jax.Array, N: int, flr: float = -100) -> jax.Array:
    """
    Compute the real cepstrum of `X` and floor the coefficients by the lower
    bound `flr`. Literal translation of source files provided as part of the
    REVERB challenge. [1]

    Parameters
    ----------
    X: jax.Array
        STFT coefficients to use to compute the cepstrum.

    N: int
        DFT size to use (is expected to be the next power of 2 w.r.t. X.shape[0])

    flr: float
        The relative lower bound (in dB) to apply to the coefficients.

    Returns
    -------
    The cepstrum as a jax Array.

    References
    ----------
    [1] K. Kinoshita et al., “A summary of the REVERB challenge: state-of-the-art and
        remaining challenges in reverberant speech processing research,” EURASIP J.
        Adv. Signal Process., vol. 2016, no. 1, p. 7, Dec. 2016,
        doi: 10.1186/s13634-016-0306-6.
    """
    Px = jnp.abs(jnp.fft.fft(X, n=N, axis=0))

    abs_flr = jnp.max(Px) * 10 ** (flr / 20)
    Px = jnp.maximum(Px, abs_flr)

    c = jnp.real(jnp.fft.ifft(jnp.log(Px), n=N, axis=0))
    return c
