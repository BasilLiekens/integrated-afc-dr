# Purpose of script:
# Main entry point for applying the dereverberation algorithms to afc.
#
# Context:
# Experiments where it is validated if `mclp` can be applied to the acoustic
# feedback cancellation problem as well.
#
# (c) Basil Liekens - ESAT/STADIUS - KU Leuven

import os
import sys

import jax
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

from afc_dr import mclp, metrics, plotting, signal_generation


def main():
    p = signal_generation.parameters.load_from_yaml(PATH_TO_CONFIG)
    audio_e, audio_l, _ = signal_generation.generate_signals(p)

    win = jnp.sqrt(sp.signal.windows.hann(p.N, sym=False).reshape((-1, 1)))

    F, msg = signal_generation.generate_feedback_rirs(p)
    F = jnp.asarray(F)
    gG = msg * 10 ** (-p.GM / 20)

    ## Simulation outside of the loop
    gBuff = jnp.zeros((p.dG + 1, 1))  # only one loudspeaker
    uBuff = jnp.zeros((p.rir_length, 1))
    p_f = mclp.afc_params(gG, p.clipG, p.dG, gBuff, F, uBuff, inloop=False, update=True)
    p_w = mclp.wpe_afc_params.construct_params(
        p.N, p.hop, win, audio_e.shape[1], p.Delta, p.K, p.alpha
    )

    _, _, y_e_init, y_l_init = mclp.run_wpe_afc(p_f, p_w, audio_e, audio_l)

    ## Simulation inside of the loop
    p_f.inloop = True
    p_w = mclp.wpe_afc_params.construct_params(
        p.N, p.hop, win, audio_e.shape[1], p.Delta, p.K, p.alpha
    )

    _, p_w_final, y_e_post, y_l_post = mclp.run_wpe_afc(p_f, p_w, audio_e, audio_l)

    ## Compute metrics
    y_e_init, y_l_init, y_e_post, y_l_post = (
        np.asarray(arr) for arr in (y_e_init, y_l_init, y_e_post, y_l_post)
    )
    offset = p.N + p.dG - 1  # compensate for the delay of STFT processing
    y_ref = audio_e[:-offset, [0]]
    y_init = (y_e_init + y_l_init)[offset:, jnp.newaxis]
    y_post = (y_e_post + y_l_post)[offset:, jnp.newaxis]

    sfrr_init = metrics.SFRR(y_e_init, y_l_init)
    sfrr_post = metrics.SFRR(y_e_post, y_l_post)

    pesq_init = metrics.pesq(y_ref, y_init, p.fs, mode="wb")[0]
    pesq_post = metrics.pesq(y_ref, y_post, p.fs, mode="wb")[0]

    estoi_init = metrics.stoi(y_ref, y_init, p.fs, extended=True)[0]
    estoi_post = metrics.stoi(y_ref, y_post, p.fs, extended=True)[0]

    cd_init = metrics.cd(y_ref[:, 0], y_init[:, 0], p.fs)[0]
    cd_post = metrics.cd(y_ref[:, 0], y_post[:, 0], p.fs)[0]

    print(
        f"Metrics\n{30 * '-'}\n\nSFRR\n{30 * '-'}\n- out of loop:\t{sfrr_init:.2f} dB\n"
        f"- inside loop:\t{sfrr_post:.2f} dB\n\nPESQ\n{30 * '-'}\n- out of loop:\t"
        f"{pesq_init:.2f}\n- inside loop:\t{pesq_post:.2f}\n\nESTOI\n{30 * '-'}\n- "
        f"out of loop:\t{estoi_init:.2f}\n- inside loop:\t{estoi_post:.2f}\n\nCD\n"
        f"{30 * '-'}\n- Initial:\t{cd_init:.2f}\n- After WPE:\t{cd_post:.2f}"
    )

    ## plot results
    _ = plotting.spectrogram((audio_e + audio_l)[:, 0], p.N, title="Input signal")
    _ = plotting.spectrogram_afc_contributions(
        y_e_init,
        y_l_init,
        p.N,
        title="Adaptation outside of the loop",
    )
    _ = plotting.spectrogram_afc_contributions(
        y_e_post,
        y_l_post,
        p.N,
        title="Adaptation inside of the loop",
    )

    plt.show(block=True)


if __name__ == "__main__":
    PATH_TO_CONFIG = os.path.join("config", "config.yml")
    jax.config.update("jax_enable_x64", True)
    mpl.use("TkAgg")  # avoid issues when plotting
    plt.ion()
    sys.exit(main())
