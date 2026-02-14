import jax
import jax.numpy as jnp


@jax.jit
def SNR(x: jax.Array, x1: jax.Array, x2: jax.Array) -> jax.Array:
    """
    Compute the signal to noise ratio (SNR) for a given signal `x`
    alongside the individual contributions of the early and late reflections
    `x_early` and `x_late`, respectively.

    Note that the same computation can be applied to things like signal to
    reverberant ratio, signal distortion etc.


    Parameters
    ----------
    x: jax.Array
        The total signal to be analyzed, can be of any dimensionality, but it
        is assumed the time axis is the first one. Additionally, it should
        hold that `jnp.allclose(x, x_early + x_late)`.

    x1: jax.Array
        The contribution of the desired signal to `x`.

    x2: jax.Array

    Returns
    -------
    The SNR in dB, shape is the same as `x`, but with the first dimension
    removed. If `x`, `x1` and `x2` do not exactly amount to the same
    signal, an array of `nan`s of the same shape is returned.
    """
    snr = jax.lax.cond(
        jnp.allclose(x, x1 + x2, atol=1e-8),
        lambda: 10 * jnp.log10(jnp.var(x1, axis=0) / jnp.var(x2, axis=0)),
        lambda: jnp.nan * jnp.zeros(x.shape[1:]),
    )
    return snr


@jax.jit
def SFRR(s: jax.Array, f: jax.Array) -> jax.Array:
    """
    Signal to feedback + reverberant ratio: since the goal of a joint DR and AFC
    filter is to model the late reverberant part of the joint source-to-microphone
    impulse response and feedback path the two contributions cannot be separated.
    This is just a wrapper around `SNR` provided for convenience.
    """
    return SNR(s + f, s, f)
