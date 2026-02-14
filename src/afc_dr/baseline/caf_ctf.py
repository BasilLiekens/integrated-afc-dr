from dataclasses import dataclass
from typing import Self

import jax
import jax.numpy as jnp
import numpy as np
import scipy as sp


@jax.tree_util.register_dataclass
@dataclass
class afc_params:
    """
    Necessities for simulating afc scenarios. It is assumed this class is used
    for SIMO simulations (i.e. one loudspeaker, multiple microphones).

    This class has some additional members that allow to store the different
    contributions to the output signal to allow for tracking metrics.
    """

    gG: float  # gain in forward path
    clipG: float  # clipping in forward path
    dG: int  # delay in forward path (in addition to the delay of the filterbank)

    # Buffer representing delay in forward path
    gBuff_e: jax.Array
    gBuff_l: jax.Array
    gBuff_f: jax.Array

    F: jax.Array  # feedback path
    uBuff: jax.Array  # buffer to compute microphone signals
    inloop: bool


@jax.tree_util.register_pytree_node_class
@dataclass
class caf_ctf_params:
    """
    Class that contains all the state of a "continuous adaptive filter" (CAF)
    using the convolutive transfer function approximation (CTF). Makes use of
    the recursive least squares (RLS) updating algorithm to be on par with
    what is used in the `wpe` algorithm. A feature flag `normalized` is
    provided which allows to choose whether to normalize the update of the
    (inverse of the) autocorrelation matrix.

    In the `construct_params()` method, it is assumed that there is only one
    input signal, hence the magic number `1`.
    """

    # "hyperparameters"
    N: int
    hop: int
    Delta: int  # nb. frames to leave between the current and ones used for prediction
    K: int  # nb. frames to use in ctf
    lmbd: float  # forgetting factor of RLS
    normalized: bool  # whether to include variance normalization

    # Related to "hyperparameters", but can not be passed as auxiliary
    win: jax.Array

    insIdx: int  # keep track of when a full frame was received
    # Microphone inputs
    inputBuff_e: jax.Array
    inputBuff_l: jax.Array
    inputBuff_f: jax.Array

    inputBuff_u: jax.Array  # loudspeaker buffer

    # For keeping track of the CAF part
    stftBuff_u: jax.Array

    # Output signals of the filterbank
    outputBuff_e: jax.Array
    outputBuff_l: jax.Array
    outputBuff_f: jax.Array

    # Filters
    R_inv: jax.Array
    W: jax.Array

    def tree_flatten(self) -> tuple[tuple, tuple]:
        children = (
            self.win,
            self.insIdx,
            self.inputBuff_e,
            self.inputBuff_l,
            self.inputBuff_f,
            self.inputBuff_u,
            self.stftBuff_u,
            self.outputBuff_e,
            self.outputBuff_l,
            self.outputBuff_f,
            self.R_inv,
            self.W,
        )

        aux_data = (
            self.N,
            self.hop,
            self.Delta,
            self.K,
            self.lmbd,
            self.normalized,
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
        Delta: int,
        K: int,
        lmbd: float,
        normalized: bool,
        win: np.ndarray,
    ) -> Self:
        """
        Given the necessary specifications, create the corresponding parameters.

        It is assumed a SISO system is constructed.
        """
        # Sanity check
        if not sp.signal.check_COLA(win.squeeze() ** 2, nperseg=N, noverlap=hop):
            raise ValueError("Window does not satisfy COLA constraints.")

        inputBuff_e = jnp.zeros((N, 1))
        inputBuff_l = jnp.zeros_like(inputBuff_e)
        inputBuff_f = jnp.zeros_like(inputBuff_e)

        inputBuff_u = jnp.zeros((N, 1))

        outputBuff_e = jnp.zeros((N, 1))
        outputBuff_l = jnp.zeros_like(outputBuff_e)
        outputBuff_f = jnp.zeros_like(outputBuff_e)

        # Assert single-sided DFT
        stftBuff_u = jnp.zeros((N // 2 + 1, Delta + K, 1), dtype=jnp.complex128)

        R_inv = jnp.stack([jnp.eye(K, dtype=jnp.complex128) for _ in range(N // 2 + 1)])
        W = jnp.zeros((N // 2 + 1, K, 1), dtype=jnp.complex128)

        return cls(
            N,
            hop,
            Delta,
            K,
            lmbd,
            normalized,
            jnp.asarray(win),
            0,
            inputBuff_e,
            inputBuff_l,
            inputBuff_f,
            inputBuff_u,
            stftBuff_u,
            outputBuff_e,
            outputBuff_l,
            outputBuff_f,
            R_inv,
            W,
        )


@jax.jit
def step_caf_ctf(
    p_s: afc_params, p_c: caf_ctf_params, x_k_e: jax.Array, x_k_l: jax.Array
) -> tuple[afc_params, caf_ctf_params, jax.Array, jax.Array, jax.Array]:
    """
    Given previous state `p_s` and `p_c` alongside new contributions of early
    reflections and late reverberations `x_k_e` and `x_k_l`, perform one time-
    domain step of the caf-ctf and return the individual contributions of
    early reflections, late reverberations and feedback signals. The "output"
    signals being the signals directly after the WOLA filterbank. Note that a
    SISO system is assumed in the derivation, but given the correct inputs,
    the code should also be able to deal with multichannel signals.
    """
    ## Compute new feedback contribution
    u_k = jnp.clip(
        p_s.gG * (p_s.gBuff_e[-1, :] + p_s.gBuff_l[-1, :] + p_s.gBuff_f[-1, :]),
        min=-p_s.clipG,
        max=p_s.clipG,
    )
    uBuff = jnp.roll(p_s.uBuff, shift=1, axis=0)
    uBuff = uBuff.at[0, :].set(u_k)
    x_k_f = p_s.F.T @ uBuff

    ## Update input buffers
    inputBuff_e = jnp.roll(p_c.inputBuff_e, shift=-1, axis=0)
    inputBuff_e = inputBuff_e.at[-1, :].set(x_k_e)
    inputBuff_l = jnp.roll(p_c.inputBuff_l, shift=-1, axis=0)
    inputBuff_l = inputBuff_l.at[-1, :].set(x_k_l)
    inputBuff_f = jnp.roll(p_c.inputBuff_f, shift=-1, axis=0)
    inputBuff_f = inputBuff_f.at[-1, :].set(x_k_f[0, :])

    inputBuff_u = jnp.roll(p_c.inputBuff_u, shift=-1, axis=0)
    inputBuff_u = inputBuff_u.at[-1, :].set(u_k)

    insIdx = (p_c.insIdx + 1) % p_c.hop

    p_c = jax.lax.cond(
        insIdx == 0,
        lambda: _step_caf_ctf_inner(
            p_c,
            inputBuff_e,
            inputBuff_l,
            inputBuff_f,
            inputBuff_u,
            p_s.inloop,
        ),
        lambda: p_c,
    )

    # update output buffers, do not clip the output signals yet, but only do
    # so after combining the three signal contributions.
    out_e = p_c.outputBuff_e[0, :]
    outputBuff_e = jnp.roll(p_c.outputBuff_e, shift=-1, axis=0)
    outputBuff_e = outputBuff_e.at[-1, :].set(0)

    out_l = p_c.outputBuff_l[0, :]
    outputBuff_l = jnp.roll(p_c.outputBuff_l, shift=-1, axis=0)
    outputBuff_l = outputBuff_l.at[-1, :].set(0)

    out_f = p_c.outputBuff_f[0, :]
    outputBuff_f = jnp.roll(p_c.outputBuff_f, shift=-1, axis=0)
    outputBuff_f = outputBuff_f.at[-1, :].set(0)

    gBuff_e = jnp.roll(p_s.gBuff_e, shift=1, axis=0)
    gBuff_e = gBuff_e.at[0, :].set(out_e)

    gBuff_l = jnp.roll(p_s.gBuff_l, shift=1, axis=0)
    gBuff_l = gBuff_l.at[0, :].set(out_l)

    gBuff_f = jnp.roll(p_s.gBuff_f, shift=1, axis=0)
    gBuff_f = gBuff_f.at[0, :].set(out_f)

    p_s_new = afc_params(
        p_s.gG, p_s.clipG, p_s.dG, gBuff_e, gBuff_l, gBuff_f, p_s.F, uBuff, p_s.inloop
    )
    p_c_new = caf_ctf_params(
        p_c.N,
        p_c.hop,
        p_c.Delta,
        p_c.K,
        p_c.lmbd,
        p_c.normalized,
        p_c.win,
        insIdx,
        inputBuff_e,
        inputBuff_l,
        inputBuff_f,
        inputBuff_u,
        p_c.stftBuff_u,
        outputBuff_e,
        outputBuff_l,
        outputBuff_f,
        p_c.R_inv,
        p_c.W,
    )

    return p_s_new, p_c_new, out_e, out_l, out_f


@jax.jit
def _step_caf_ctf_inner(
    p_c: caf_ctf_params,
    inputBuff_e: jax.Array,
    inputBuff_l: jax.Array,
    inputBuff_f: jax.Array,
    inputBuff_u: jax.Array,
    inloop: bool,
) -> caf_ctf_params:
    ## Transform inputs into frequency domain, `jnp.newaxis` for broadcasting
    fd_e = jnp.fft.rfft(inputBuff_e, n=p_c.N, axis=0)[:, :, jnp.newaxis]
    fd_l = jnp.fft.rfft(inputBuff_l, n=p_c.N, axis=0)[:, :, jnp.newaxis]
    fd_f = jnp.fft.rfft(inputBuff_f, n=p_c.N, axis=0)[:, :, jnp.newaxis]

    fd_u = jnp.fft.rfft(inputBuff_u, n=p_c.N, axis=0)

    ## Update estimation buffers
    stftBuff_u = jnp.roll(p_c.stftBuff_u, shift=1, axis=1)
    stftBuff_u = stftBuff_u.at[:, 0, :].set(fd_u)

    ## Actual estimation + filter update
    x_hat = jnp.conj(p_c.W.transpose(0, 2, 1)) @ stftBuff_u

    e = fd_e + fd_l + fd_f - x_hat

    stftBuff = stftBuff_u[:, p_c.Delta :, :]
    sigma = jnp.maximum(
        1e-10,  # scale-factor for normalization, based on the microphone signals
        jnp.mean(jnp.abs(fd_e + fd_l + fd_f) ** 2, axis=1, keepdims=True),
    )

    R_inv = jax.lax.cond(
        p_c.normalized,
        lambda: (
            p_c.R_inv / p_c.lmbd
            - (p_c.R_inv @ stftBuff @ jnp.conj(stftBuff.transpose(0, 2, 1)) @ p_c.R_inv)
            / (
                p_c.lmbd * p_c.lmbd * sigma
                + p_c.lmbd
                * jnp.conj(stftBuff.transpose(0, 2, 1))
                @ p_c.R_inv
                @ stftBuff
            )
        ),
        lambda: (
            p_c.R_inv / p_c.lmbd
            - (p_c.R_inv @ stftBuff @ jnp.conj(stftBuff.transpose(0, 2, 1)) @ p_c.R_inv)
            / (
                p_c.lmbd * p_c.lmbd
                + p_c.lmbd
                * jnp.conj(stftBuff.transpose(0, 2, 1))
                @ p_c.R_inv
                @ stftBuff
            )
        ),
    )

    W = jax.lax.cond(
        p_c.normalized,
        lambda: p_c.W + R_inv @ stftBuff @ jnp.conj(e.transpose(0, 2, 1)) / sigma,
        lambda: p_c.W + R_inv @ stftBuff @ jnp.conj(e.transpose(0, 2, 1)),
    )

    # Update output buffers, feedback cancellation method so the current signal
    # contributions just pass right through. (Pretend they only act on feedback)
    td_e = p_c.win**2 * inputBuff_e
    outputBuff_e = p_c.outputBuff_e + td_e

    td_l = p_c.win**2 * inputBuff_l
    outputBuff_l = p_c.outputBuff_l + td_l

    td_f = jax.lax.cond(
        inloop,
        lambda: p_c.win * jnp.fft.irfft((fd_f - x_hat)[:, :, 0], n=p_c.N, axis=0),
        lambda: p_c.win**2 * inputBuff_f,
    )
    outputBuff_f = p_c.outputBuff_f + td_f

    return caf_ctf_params(
        p_c.N,
        p_c.hop,
        p_c.Delta,
        p_c.K,
        p_c.lmbd,
        p_c.normalized,
        p_c.win,
        p_c.insIdx,
        inputBuff_e,
        inputBuff_l,
        inputBuff_f,
        inputBuff_u,
        stftBuff_u,
        outputBuff_e,
        outputBuff_l,
        outputBuff_f,
        R_inv,
        W,
    )


@jax.jit
def run_caf_ctf(
    p_s: afc_params, p_c: caf_ctf_params, x_e: jax.Array, x_l: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """
    Given initial state `p_s` and `p_c` alongside contributions of early
    reflections and late reverberations `x_e` and `x_l` respectively, run the
    caf-ctf for all of the input signal.
    """

    def _inner_loop(
        p: tuple[afc_params, caf_ctf_params], x_k: tuple[jax.Array, jax.Array]
    ) -> tuple[
        tuple[afc_params, caf_ctf_params], tuple[jax.Array, jax.Array, jax.Array]
    ]:
        p_s, p_c = p
        x_k_e, x_k_l = x_k
        p_s, p_c, y_k_e, y_k_l, y_k_f = step_caf_ctf(p_s, p_c, x_k_e, x_k_l)
        return (p_s, p_c), (y_k_e, y_k_l, y_k_f)

    _, (y_e, y_l, y_f) = jax.lax.scan(_inner_loop, init=(p_s, p_c), xs=(x_e, x_l))

    return y_e, y_l, y_f
