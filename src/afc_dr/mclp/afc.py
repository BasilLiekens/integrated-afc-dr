from dataclasses import dataclass
from typing import Self

import jax
import jax.numpy as jnp
import scipy as sp


@jax.tree_util.register_pytree_node_class
@dataclass
class wpe_afc_params:
    N: int  # DFT length
    hop: int  # hop between subsequent DFT frames

    D: int  # nb. mics
    Delta: int  # nb. frames before they are being used to predict, hence > 0
    K: int  # nb. frames used for estimation, > 0
    delta_L: int  # nb. old frames used for estimation of the variance >= 0
    alpha: float  # forgetting factor

    # Related to the DFT/STFT, but located here to simplify the `tree_(un)flatten` funcs
    win: jax.Array  # Window function, assumed to be identical for (i)stft

    # Buffers, track early reflections, late reverberations and feedback separate
    inputBuff_e: jax.Array
    inputBuff_l: jax.Array
    inputBuff_f: jax.Array
    outputBuff_e: jax.Array
    outputBuff_l: jax.Array
    outputBuff_f: jax.Array
    insIdx: int  # keeps track of nb. of inserted samples to determine new frame time

    # Delay lines for filter computation (fd) [`N // 2 + 1` x (`Delta` + `K`) * `D` x 1]
    stftBuff_e: jax.Array
    stftBuff_l: jax.Array
    stftBuff_f: jax.Array

    # Similar, but now for variance estimation
    yBuff_e: jax.Array
    yBuff_l: jax.Array
    yBuff_f: jax.Array

    G: jax.Array
    R_inv: jax.Array

    def tree_flatten(self) -> tuple[tuple, tuple]:
        children = (
            self.win,
            self.inputBuff_e,
            self.inputBuff_l,
            self.inputBuff_f,
            self.outputBuff_e,
            self.outputBuff_l,
            self.outputBuff_f,
            self.insIdx,
            self.stftBuff_e,
            self.stftBuff_l,
            self.stftBuff_f,
            self.yBuff_e,
            self.yBuff_l,
            self.yBuff_f,
            self.G,
            self.R_inv,
        )

        aux_data = (
            self.N,
            self.hop,
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
        Convencience method that, given all required hyperparameters,
        constructs an instance of this that can be used to initialize
        simulations.
        """
        # Defensive check when constructing parameters to avoid issues due to not
        # updating the hop and DFT size simultaneously.
        if not sp.signal.check_COLA(win.squeeze() ** 2, nperseg=N, noverlap=hop):
            raise ValueError("Window does not satisfy COLA constraints.")

        inputBuff_e = jnp.zeros((N, D))
        inputBuff_l = jnp.zeros_like(inputBuff_e)
        inputBuff_f = jnp.zeros_like(inputBuff_e)

        outputBuff_e = jnp.zeros((N, D))
        outputBuff_l = jnp.zeros_like(outputBuff_e)
        outputBuff_f = jnp.zeros_like(outputBuff_e)

        insIdx = 0

        # append a trailing `1` to have an additional vector dimension for simplicity
        stftBuff_e = jnp.zeros((N // 2 + 1, (Delta + K) * D, 1), dtype=jnp.complex128)
        stftBuff_l = jnp.zeros_like(stftBuff_e)
        stftBuff_f = jnp.zeros_like(stftBuff_e)

        yBuff_e = jnp.zeros((N // 2 + 1, (delta_L + 1) * D, 1), dtype=jnp.complex128)
        yBuff_l = jnp.zeros_like(yBuff_e)
        yBuff_f = jnp.zeros_like(yBuff_e)

        ## Only the reference microphone of the output will be used.
        G = jnp.zeros((N // 2 + 1, D * K, D), dtype=jnp.complex128)
        R_inv = jnp.stack(
            [jnp.eye(D * K, dtype=jnp.complex128) for _ in range(N // 2 + 1)]
        )

        return cls(
            N,
            hop,
            D,
            Delta,
            K,
            delta_L,
            alpha,
            win,
            inputBuff_e,
            inputBuff_l,
            inputBuff_f,
            outputBuff_e,
            outputBuff_l,
            outputBuff_f,
            insIdx,
            stftBuff_e,
            stftBuff_l,
            stftBuff_f,
            yBuff_e,
            yBuff_l,
            yBuff_f,
            G,
            R_inv,
        )


@jax.tree_util.register_dataclass
@dataclass
class afc_params:
    """
    Necessities for simulating afc scenarios. It is assumed this class is used
    for SIMO simulations (i.e. one loudspeaker, multiple microphones).
    """

    gG: float  # gain in forward path
    clipG: float  # clipping in forward path
    dG: int  # delay in forward path (in addition to the delay of the filterbank)
    gBuff: jax.Array  # Buffer representing delay in forward path
    F: jax.Array  # feedback path
    uBuff: jax.Array  # buffer to compute microphone signals
    inloop: bool


@jax.jit
def step_wpe_afc(
    p_f: afc_params, p_w: wpe_afc_params, x_k_e: jax.Array, x_k_l: jax.Array
) -> tuple[afc_params, wpe_afc_params, jax.Array, jax.Array, jax.Array]:
    """
    Perform one (time domain) step of the adaptive, online WPE algorithm [1],
    [2], implemented in the STFT domain. However, the algorithm is now applied
    in an AFC context.

    Parameters
    ----------
    p_f: afc_params
        The state of the AFC scenario.

    p_w: wpe_afc_params
        The state of the algorithm itself.

    x_k_e: jax.Array
        The contribution of the early reflections/desired signal to the input
        at this timestep.

    x_k_l: jax.Array
        Idem to `x_k_e`, but for late reverberations/noise/interference/...

    Returns
    -------
    A tuple containing (in order), the new, updated state of both the afc and
    wpe parts of the simulation. In addition, the contributions from early and
    late reflections as well as the feedback component to the output are
    returned.

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
    ## Perform the afc simulation
    uBuff = jnp.roll(p_f.uBuff, shift=1, axis=0)
    uBuff = uBuff.at[0, :].set(p_f.gBuff[-1, :])
    f_k = (p_f.F.T @ uBuff)[:, 0]

    ## Update the wpe buffers with new input signals
    inputBuff_e = jnp.roll(p_w.inputBuff_e, shift=-1, axis=0)
    inputBuff_e = inputBuff_e.at[-1, :].set(x_k_e)
    inputBuff_l = jnp.roll(p_w.inputBuff_l, shift=-1, axis=0)
    inputBuff_l = inputBuff_l.at[-1, :].set(x_k_l)
    inputBuff_f = jnp.roll(p_w.inputBuff_f, shift=-1, axis=0)
    inputBuff_f = inputBuff_f.at[-1, :].set(f_k)

    insIdx = (p_w.insIdx + 1) % p_w.hop

    p_w = jax.lax.cond(
        insIdx == 0,
        lambda: _step_wpe_afc_inner(
            p_w, inputBuff_e, inputBuff_l, inputBuff_f, p_f.inloop
        ),
        lambda: p_w,
    )

    ## Update output buffer, return output. Only first microphone signal is taken.
    out_e = p_w.outputBuff_e[0, 0]
    outputBuff_e = jnp.roll(p_w.outputBuff_e, shift=-1, axis=0)
    outputBuff_e = outputBuff_e.at[-1, :].set(0)

    out_l = p_w.outputBuff_l[0, 0]
    outputBuff_l = jnp.roll(p_w.outputBuff_l, shift=-1, axis=0)
    outputBuff_l = outputBuff_l.at[-1, :].set(0)

    out_f = p_w.outputBuff_f[0, 0]
    outputBuff_f = jnp.roll(p_w.outputBuff_f, shift=-1, axis=0)
    outputBuff_f = outputBuff_f.at[-1, :].set(0)

    g_new = jnp.clip(p_f.gG * (out_e + out_l + out_f), min=-p_f.clipG, max=p_f.clipG)
    gBuff = jnp.roll(p_f.gBuff, shift=1, axis=0)
    gBuff = gBuff.at[0, :].set(g_new)

    p_f_new = afc_params(p_f.gG, p_f.clipG, p_f.dG, gBuff, p_f.F, uBuff, p_f.inloop)
    p_w_new = wpe_afc_params(
        p_w.N,
        p_w.hop,
        p_w.D,
        p_w.Delta,
        p_w.K,
        p_w.delta_L,
        p_w.alpha,
        p_w.win,
        inputBuff_e,
        inputBuff_l,
        inputBuff_f,
        outputBuff_e,
        outputBuff_l,
        outputBuff_f,
        insIdx,
        p_w.stftBuff_e,
        p_w.stftBuff_l,
        p_w.stftBuff_f,
        p_w.yBuff_e,
        p_w.yBuff_l,
        p_w.yBuff_f,
        p_w.G,
        p_w.R_inv,
    )

    return p_f_new, p_w_new, out_e, out_l, out_f


@jax.jit
def _step_wpe_afc_inner(
    p: wpe_afc_params,
    inputBuff_e: jax.Array,
    inputBuff_l: jax.Array,
    inputBuff_f: jax.Array,
    inloop: bool,
) -> wpe_afc_params:
    ## Prepare/update buffers
    fd_e = jnp.fft.rfft(p.win * inputBuff_e, axis=0)[:, :, None]  # broadcasting
    fd_l = jnp.fft.rfft(p.win * inputBuff_l, axis=0)[:, :, None]
    fd_f = jnp.fft.rfft(p.win * inputBuff_f, axis=0)[:, :, None]

    stftBuff_e = jnp.roll(p.stftBuff_e, shift=p.D, axis=1)
    stftBuff_e = stftBuff_e.at[:, : p.D, :].set(fd_e)
    stftBuff_l = jnp.roll(p.stftBuff_l, shift=p.D, axis=1)
    stftBuff_l = stftBuff_l.at[:, : p.D, :].set(fd_l)
    stftBuff_f = jnp.roll(p.stftBuff_f, shift=p.D, axis=1)
    stftBuff_f = stftBuff_f.at[:, : p.D, :].set(fd_f)

    yBuff_e = jnp.roll(p.yBuff_e, shift=p.D, axis=1)
    yBuff_e = yBuff_e.at[:, : p.D, :].set(fd_e)
    yBuff_l = jnp.roll(p.yBuff_l, shift=p.D, axis=1)
    yBuff_l = yBuff_l.at[:, : p.D, :].set(fd_l)
    yBuff_f = jnp.roll(p.yBuff_f, shift=p.D, axis=1)
    yBuff_f = yBuff_f.at[:, : p.D, :].set(fd_f)

    ## Compute individual output contributions
    G_H = jnp.conj(p.G.transpose(0, 2, 1))

    y_e = stftBuff_e[:, : p.D, :]
    y_tilde_e = stftBuff_e[:, p.D * p.Delta :, :]
    x_hat_e = y_e - G_H @ y_tilde_e

    y_l = stftBuff_l[:, : p.D, :]
    y_tilde_l = stftBuff_l[:, p.D * p.Delta :, :]
    x_hat_l = y_l - G_H @ y_tilde_l

    y_f = stftBuff_f[:, : p.D, :]
    y_tilde_f = stftBuff_f[:, p.D * p.Delta :, :]
    x_hat_f = y_f - G_H @ y_tilde_f

    ## Update filters
    yBuff = yBuff_e + yBuff_l + yBuff_f
    y_tilde = y_tilde_e + y_tilde_l + y_tilde_f
    x_hat = x_hat_e + x_hat_l + x_hat_f

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

    ## Construct return objects, trim trailing "vector dimension"
    td_e = jax.lax.cond(
        inloop,
        lambda: p.win * jnp.fft.irfft(x_hat_e, axis=0)[:, :, 0],
        lambda: p.win**2 * inputBuff_e,  # account for windowing still
    )
    outputBuff_e = p.outputBuff_e + td_e

    td_l = jax.lax.cond(
        inloop,
        lambda: p.win * jnp.fft.irfft(x_hat_l, axis=0)[:, :, 0],
        lambda: p.win**2 * inputBuff_l,
    )
    outputBuff_l = p.outputBuff_l + td_l

    td_f = jax.lax.cond(
        inloop,
        lambda: p.win * jnp.fft.irfft(x_hat_f, axis=0)[:, :, 0],
        lambda: p.win**2 * inputBuff_f,
    )
    outputBuff_f = p.outputBuff_f + td_f

    p_w_new = wpe_afc_params(
        p.N,
        p.hop,
        p.D,
        p.Delta,
        p.K,
        p.delta_L,
        p.alpha,
        p.win,
        inputBuff_e,
        inputBuff_l,
        inputBuff_f,
        outputBuff_e,
        outputBuff_l,
        outputBuff_f,
        p.insIdx,
        stftBuff_e,
        stftBuff_l,
        stftBuff_f,
        yBuff_e,
        yBuff_l,
        yBuff_f,
        G_k,
        R_inv_k,
    )

    return p_w_new


@jax.jit
def run_wpe_afc(
    p_f: afc_params, p_w: wpe_afc_params, x_e: jax.Array, x_l: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """
    Given initial state `p_f` and `p_w` alongside the individual contributions
    of early and late reflections `x_e` and `x_l`, apply the wpe algorithm in
    an AFC setting and return the individual contributions.

    Note, the returned results are the outputs before they are being fed into
    the delay buffer for the forward path. Hence, not scaled, clipped, ...
    """

    def _inner_loop(
        p: tuple[afc_params, wpe_afc_params], x_k: tuple[jax.Array, jax.Array]
    ) -> tuple[
        tuple[afc_params, wpe_afc_params], tuple[jax.Array, jax.Array, jax.Array]
    ]:
        p_f, p_w = p
        x_k_e, x_k_l = x_k
        p_f, p_w, y_k_e, y_k_l, y_k_f = step_wpe_afc(p_f, p_w, x_k_e, x_k_l)
        return (p_f, p_w), (y_k_e, y_k_l, y_k_f)

    _, output = jax.lax.scan(_inner_loop, init=(p_f, p_w), xs=(x_e, x_l))

    return output
