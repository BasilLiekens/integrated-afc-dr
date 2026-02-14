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
def SRR(x: jax.Array, x1: jax.Array, x2: jax.Array) -> jax.Array:
    """
    Similar to `SNR`, but now for the signal to reverberant ratio (SRR)
    """
    return SNR(x, x1, x2)


@jax.jit
def SFR(s: jax.Array, f: jax.Array) -> jax.Array:
    """
    Similar to `SFR`, but now for the signal to feedback ratio (SFR)

    This function only accepts the two individual contributions as there is
    no physical sense to also pass in the sum directly.
    """
    return SNR(s + f, s, f)


@jax.jit
def SDR(y: jax.Array, x_ref: jax.Array) -> jax.Array:
    """
    Similar to `SNR`, but now computes the ratio of the clean signal to
    the distorted (i.e. convolved, filtered etc.) signal.

    This function does not accept the joint input signal as it does not
    make physical sense to impose this.
    """
    return SNR(y + x_ref, y, x_ref)


@jax.jit
def SD(y: jax.Array, x_ref: jax.Array) -> jax.Array:
    """
    Signal distortion: defined as `10 log_10(|y - x_ref|^2)`, this metric is a
    different version of the `SDR` metric. Otherwise follows all the same
    conventions as `SDR`, `SNR` etc.

    However, this function does not accept the joint input signal as it does
    not make physical sense to impose a constraint on `y` and `x_ref` whereas
    it maybe would for things like `SNR` as there it is two parts of the same
    signal. This then also leads to the fact that no `nan`s are returned.
    """
    return 10 * jnp.log10(jnp.var(y - x_ref, axis=0) / jnp.var(x_ref, axis=0))
