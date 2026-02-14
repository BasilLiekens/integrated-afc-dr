from dataclasses import dataclass
from typing import Self

import jax
import jax.numpy as jnp
import scipy as sp


@jax.tree_util.register_pytree_node_class
@dataclass
class wpe_params:
    N: int  # DFT length
    hop: int  # hop between subsequent DFT frames
    win: jax.Array  # Window function, assumed to be identical for (i)stft

    D: int  # nb. mics
    Delta: int  # nb. frames before they are being used to predict, > 0
    K: int  # nb. frames used to predict the reverberation
    delta_L: int  # nb. frames that is used for estimation of the variance >= 0
    alpha: float  # forgetting factor

    inputBuff_e: jax.Array  # Buffer data until new frame collected (early data)
    inputBuff_l: jax.Array  # idem, but for late reflections + interference & noise
    outputBuff_e: jax.Array  # Buffer data that has not been sent as output
    outputBuff_l: jax.Array
    insIdx: int  # keeps track of how many samples have been inserted since updating

    # Delay lines (fd): [`N // 2 + 1` x (`Delta` + `K`) * `D` x 1]
    stftBuff_e: jax.Array
    stftBuff_l: jax.Array
    yBuff_e: jax.Array  # Similar to `stftBuff`, but now of length (`delta_L` + 1) * `D`
    yBuff_l: jax.Array

    G: jax.Array  # WPE prediction filter: [`N // 2 + 1` x `D` * `K` x `D`]
    R_inv: jax.Array  # Inverse of autocorrelation matrix

    def tree_flatten(self) -> tuple[tuple, tuple]:
        children = (
            self.inputBuff_e,
            self.inputBuff_l,
            self.outputBuff_e,
            self.outputBuff_l,
            self.insIdx,
            self.stftBuff_e,
            self.stftBuff_l,
            self.yBuff_e,
            self.yBuff_l,
            self.G,
            self.R_inv,
        )
        aux_data = (
            self.N,
            self.hop,
            self.win,
            self.D,
            self.Delta,
            self.K,
            self.delta_L,
            self.alpha,
        )

        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data: tuple, children: tuple) -> Self:
        return cls(*aux_data, *children)

    @classmethod
    def construct_params(
        cls,
        N: int,
        hop: int,
        win: jax.Array,
        D: int,
        Delta: int,
        K: int,
        delta_L: int,
        alpha: float,
    ) -> Self:
        """
        Convenience method that, given all required hyperparameters,
        constructs an instance of this that can be used to initialize
        simulations.
        """
        # Defensive check when constructing parameters to avoid issues due to not
        # updating the hop and DFT size simultaneously.
        if not sp.signal.check_COLA(win.squeeze() ** 2, nperseg=N, noverlap=hop):
            raise ValueError("Window does not satisfy COLA constraints.")

        inputBuff_e = jnp.zeros((N, D))
        inputBuff_l = jnp.zeros_like(inputBuff_e)
        outputBuff_e = jnp.zeros((N, D))
        outputBuff_l = jnp.zeros_like(outputBuff_e)
        insIdx = 0

        stftBuff_e = jnp.zeros((N // 2 + 1, (Delta + K) * D, 1), dtype=jnp.complex128)
        stftBuff_l = jnp.zeros_like(stftBuff_e)
        yBuff_e = jnp.zeros((N // 2 + 1, (delta_L + 1) * D, 1), dtype=jnp.complex128)
        yBuff_l = jnp.zeros_like(yBuff_e)

        G = jnp.zeros((N // 2 + 1, D * K, D), dtype=jnp.complex128)
        R_inv = jnp.stack(
            [jnp.eye(D * K, dtype=jnp.complex128) for _ in range(N // 2 + 1)]
        )

        return cls(
            N,
            hop,
            win,
            D,
            Delta,
            K,
            delta_L,
            alpha,
            inputBuff_e,
            inputBuff_l,
            outputBuff_e,
            outputBuff_l,
            insIdx,
            stftBuff_e,
            stftBuff_l,
            yBuff_e,
            yBuff_l,
            G,
            R_inv,
        )


@jax.jit
def step_wpe(
    p: wpe_params, x_k_e: jax.Array, x_k_l: jax.Array
) -> tuple[wpe_params, jax.Array, jax.Array]:
    """
    Perform one step of the adaptive, online, WPE algorithm [1], [2] in the
    STFT domain. This processing happens with the CTF assumption; no
    crossband filters are taken into account (in contrast to [2]).


    Parameters
    ----------
    p: wpe_params
        The state to start from.

    x_k_e: jax.Array
        The contribution of early reflections/desired signal to the input at
        this timestep.

    x_k_l: jax.Array
        Idem to `x_k_e`, but for the late reflections/noise/interference/...

    Returns
    -------
    A tuple containing the new, updated state, alongside the contributions of
    the desired signal and the interferers, respectively.

    References
    ----------
    [1] T. Yoshioka, H. Tachibana, T. Nakatani, and M. Miyoshi, “Adaptive
        dereverberation of speech signals with speaker-position change
        detection,” in 2009 IEEE International Conference on Acoustics, Speech
        and Signal Processing, Taipei, Taiwan: IEEE, Apr. 2009, pp. 3733–3736.

    [2] L. Drude, J. Heymann, C. Boeddeker, and R. Haeb-Umbach, “NARA-WPE: A
        Python package for weighted prediction error dereverberation in Numpy
        and Tensorﬂow for online and ofﬂine processing,” in Speech
        Communication; 13th ITG-Symposium, Oldenburg Germany, Oct. 2018,
        pp. 1–5.
    """
    ## Update inputbuffer
    inputBuff_e = jnp.roll(p.inputBuff_e, shift=-1, axis=0)
    inputBuff_e = inputBuff_e.at[-1, :].set(x_k_e)

    inputBuff_l = jnp.roll(p.inputBuff_l, shift=-1, axis=0)
    inputBuff_l = inputBuff_l.at[-1, :].set(x_k_l)

    insIdx = (p.insIdx + 1) % p.hop

    ## If enough samples collected, start new processing step.
    p = jax.lax.cond(
        insIdx == 0,
        lambda: _step_wpe_inner(p, inputBuff_e, inputBuff_l),
        lambda: p,
    )

    ## Update output buffer and return output
    out_e = p.outputBuff_e[0, :]
    outputBuff_e = jnp.roll(p.outputBuff_e, shift=-1, axis=0)
    outputBuff_e = outputBuff_e.at[-1, :].set(0)

    out_l = p.outputBuff_l[0, :]
    outputBuff_l = jnp.roll(p.outputBuff_l, shift=-1, axis=0)
    outputBuff_l = outputBuff_l.at[-1, :].set(0)

    p_new = wpe_params(
        p.N,
        p.hop,
        p.win,
        p.D,
        p.Delta,
        p.K,
        p.delta_L,
        p.alpha,
        inputBuff_e,
        inputBuff_l,
        outputBuff_e,
        outputBuff_l,
        insIdx,
        p.stftBuff_e,
        p.stftBuff_l,
        p.yBuff_e,
        p.yBuff_l,
        p.G,
        p.R_inv,
    )

    return p_new, out_e, out_l


@jax.jit
def _step_wpe_inner(
    p: wpe_params,
    inputBuff_e: jax.Array,
    inputBuff_l: jax.Array,
) -> wpe_params:
    """
    Perform the actual frequency domain processing.

    All necessary input is passed in via the `p`. However, `inputBuff` was
    already updated prior to calling this function. Therefore, to avoid having
    to allocate a new `wpe_params`, this array is passed in separately.

    Parameters
    ----------
    p: wpe_params
        "Old state"

    inputBuff_e: jax.Array, 2D
        Already updated input buffer for desired signal contribution ([`N` x `D`])

    inputBuff_l: jax.Array, 2D
        Idem, but now for the noise/interference/...


    Returns
    -------
    A new `wpe_params` struct containing the updated parameters (it will have
    to be reinstantiated at the end of `step_wpe()` once more, but done in
    this fashion to avoid having too many return arguments. The estimate of
    the early reflections in this frame is incorporated into the `outputBuff`
    attribute of `wpe_params`.
    """
    ## Prepare/update buffers
    fd_e = jnp.fft.rfft(p.win * inputBuff_e, axis=0)[:, :, None]  # broadcasting
    fd_l = jnp.fft.rfft(p.win * inputBuff_l, axis=0)[:, :, None]

    stftBuff_e = jnp.roll(p.stftBuff_e, shift=p.D, axis=1)  # insert micsigs up front
    stftBuff_e = stftBuff_e.at[:, : p.D, :].set(fd_e)
    stftBuff_l = jnp.roll(p.stftBuff_l, shift=p.D, axis=1)
    stftBuff_l = stftBuff_l.at[:, : p.D, :].set(fd_l)

    yBuff_e = jnp.roll(p.yBuff_e, shift=p.D, axis=1)
    yBuff_e = yBuff_e.at[:, : p.D, :].set(fd_e)
    yBuff_l = jnp.roll(p.yBuff_l, shift=p.D, axis=1)
    yBuff_l = yBuff_l.at[:, : p.D, :].set(fd_l)

    ## Perform the filtering, separately for desired and interferers
    y_e = stftBuff_e[:, : p.D, :]
    y_tilde_e = stftBuff_e[:, p.D * p.Delta :, :]

    x_hat_e = y_e - jnp.conj(p.G.transpose(0, 2, 1)) @ y_tilde_e

    y_l = stftBuff_l[:, : p.D, :]
    y_tilde_l = stftBuff_l[:, p.D * p.Delta :, :]

    x_hat_l = y_l - jnp.conj(p.G.transpose(0, 2, 1)) @ y_tilde_l

    ## Update filters
    yBuff = yBuff_e + yBuff_l
    y_tilde = y_tilde_e + y_tilde_l
    x_hat = x_hat_e + x_hat_l

    # keepdims to allow for broadcasting, no need to clip as the Kalman gain
    # denominator is bounded by alpha already + R_inv is positive semidefinite
    lmbd = jnp.mean(jnp.abs(yBuff) ** 2, axis=1, keepdims=True)

    K_k = (
        p.R_inv
        @ y_tilde
        / (p.alpha * lmbd + jnp.conj(y_tilde.transpose(0, 2, 1)) @ p.R_inv @ y_tilde)
    )
    R_inv_k = (
        p.R_inv / p.alpha
        - K_k @ jnp.conj(y_tilde.transpose(0, 2, 1)) @ p.R_inv / p.alpha
    )

    G_k = p.G + K_k @ jnp.conj(x_hat.transpose(0, 2, 1))

    ## Construct return objects, remove trailing "vector dimension" (for broadcasting)
    td_e = p.win * jnp.fft.irfft(x_hat_e, axis=0)[:, :, 0]
    outputBuff_e = p.outputBuff_e + td_e

    td_l = p.win * jnp.fft.irfft(x_hat_l, axis=0)[:, :, 0]
    outputBuff_l = p.outputBuff_l + td_l

    p_new = wpe_params(
        p.N,
        p.hop,
        p.win,
        p.D,
        p.Delta,
        p.K,
        p.delta_L,
        p.alpha,
        inputBuff_e,
        inputBuff_l,
        outputBuff_e,
        outputBuff_l,
        p.insIdx,
        stftBuff_e,
        stftBuff_l,
        yBuff_e,
        yBuff_l,
        G_k,
        R_inv_k,
    )

    return p_new


@jax.jit
def run_wpe(
    p: wpe_params, x_e: jax.Array, x_l: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """
    Given an input signal `x` and the initial state, run the wpe algorithm by
    repeatedly calling the `step_wpe()` function and keeping track of the
    internal state.
    """

    def _inner_loop(
        p: wpe_params, x_k: tuple[jax.Array, jax.Array]
    ) -> tuple[wpe_params, tuple[jax.Array, jax.Array]]:
        x_k_e, x_k_l = x_k
        p, y_k_e, y_k_l = step_wpe(p, x_k_e, x_k_l)
        return p, (y_k_e, y_k_l)

    _, output = jax.lax.scan(_inner_loop, init=p, xs=(x_e, x_l))

    return output
