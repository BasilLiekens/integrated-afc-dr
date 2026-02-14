# Purpose of script:
# Compare the dereverberation algorithms applied to AFC with a continuously
# adapted WOLA-based adaptive filter.
#
# Context:
# Validating the performance of dereverberation algorithms for AFC.
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

from afc_dr import baseline, metrics, plotting, signal_generation


def main():
    ## Data loading & generation
    p = signal_generation.parameters.load_from_yaml(PATH_TO_CONFIG)
    audio_e, audio_l, _ = signal_generation.generate_signals(p)
    audio_e, audio_l = audio_e[:, [0]], audio_l[:, [0]]  # trim to one mic
    F, msg = signal_generation.generate_feedback_rirs(p)
    F = jnp.asarray(F[:, [0]])
    gG = msg * 10 ** (-p.GM / 20)

    ## Setup for simulations, magic number `1` due to SISO simulations
    lmbd = 0.99
    win = np.sqrt(sp.signal.windows.hann(p.N, sym=False))[:, jnp.newaxis]
    p_c = baseline.caf_ctf_params.construct_params(
        p.N,
        p.hop,
        p.Delta - 1,  # `-1` for different conventions
        p.K + 1,  # Similar; have the same temporal span
        lmbd,
        normalized=False,
        win=win,
    )

    gBuff_e = jnp.zeros((p.dG + 1, 1))
    gBuff_l = jnp.zeros_like(gBuff_e)
    gBuff_f = jnp.zeros_like(gBuff_e)

    uBuff = jnp.zeros((F.shape[0], 1))

    p_s = baseline.afc_params(
        gG, p.clipG, p.dG, gBuff_e, gBuff_l, gBuff_f, F, uBuff, inloop=False
    )

    y_e_init, y_l_init, y_f_init = baseline.run_caf_ctf(p_s, p_c, audio_e, audio_l)

    p_s = baseline.afc_params(
        gG, p.clipG, p.dG, gBuff_e, gBuff_l, gBuff_f, F, uBuff, inloop=True
    )
    y_e_post_reg, y_l_post_reg, y_f_post_reg = baseline.run_caf_ctf(
        p_s, p_c, audio_e, audio_l
    )

    p_c.normalized = True
    y_e_post_norm, y_l_post_norm, y_f_post_norm = baseline.run_caf_ctf(
        p_s, p_c, audio_e, audio_l
    )

    ## metrics and plotting
    offset = p.N + p.dG - 1
    y_ref = audio_e[:-offset:, :]
    y_init = np.asarray((y_e_init + y_l_init + y_f_init)[offset:, :])
    y_post_reg = np.asarray((y_e_post_reg + y_l_post_reg + y_f_post_reg)[offset:, :])
    y_post_norm = np.asarray(
        (y_e_post_norm + y_l_post_norm + y_f_post_norm)[offset:, :]
    )

    sfr_init = metrics.SFR(y_e_init, y_f_init)[0]
    sfr_post_reg = metrics.SFR(y_e_post_reg, y_f_post_reg)[0]
    sfr_post_norm = metrics.SFR(y_e_post_norm, y_f_post_norm)[0]

    pesq_init = metrics.pesq(y_ref, y_init, p.fs, mode="wb")[0]
    pesq_post_reg = metrics.pesq(y_ref, y_post_reg, p.fs, mode="wb")[0]
    pesq_post_norm = metrics.pesq(y_ref, y_post_norm, p.fs, mode="wb")[0]

    estoi_init = metrics.stoi(y_ref, y_init, p.fs, extended=True)[0]
    estoi_post_reg = metrics.stoi(y_ref, y_post_reg, p.fs, extended=True)[0]
    estoi_post_norm = metrics.stoi(y_ref, y_post_norm, p.fs, extended=True)[0]

    print(
        f"Metrics\n{30 * '-'}\nSFR\n- out of loop:\t\t{sfr_init:.2f}\n- "
        f"inside loop (reg):\t{sfr_post_reg:.2f}\n- inside loop (norm):\t"
        f"{sfr_post_norm:.2f}\n\nPESQ\n{30 * '-'}\n- out of loop:\t\t"
        f"{pesq_init:.2f}\n- inside loop (reg):\t{pesq_post_reg:.2f}\n- "
        f"inside loop (norm):\t{pesq_post_norm:.2f}\n\nESTOI\n{30 * '-'}\n- "
        f"out of loop:\t\t{estoi_init:.2f}\n- inside loop (reg):\t"
        f"{estoi_post_reg:.2f}\n- inside loop (norm):\t{estoi_post_norm:.2f}"
    )

    _ = plotting.spectrogram_afc_contributions(
        y_e_init[:, 0],
        y_l_init[:, 0],
        y_f_init[:, 0],
        p.N,
        title="Out of loop adaptation",
    )
    _ = plotting.spectrogram_afc_contributions(
        y_e_post_reg[:, 0],
        y_l_post_reg[:, 0],
        y_f_post_reg[:, 0],
        p.N,
        title="Inloop adaptation (not normalized)",
    )
    _ = plotting.spectrogram_afc_contributions(
        y_e_post_norm[:, 0],
        y_l_post_norm[:, 0],
        y_f_post_norm[:, 0],
        p.N,
        title="Inloop adaptation (normalized)",
    )

    plt.show(block=True)


if __name__ == "__main__":
    PATH_TO_CONFIG = os.path.join("config", "config.yml")
    jax.config.update("jax_enable_x64", True)

    mpl.use("Tkagg")
    plt.ion()
    sys.exit(main())
