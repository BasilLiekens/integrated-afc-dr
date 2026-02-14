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

    # In WPE `Delta` is introduced to avoid desired source cancellation. For `CAF-CTF`
    # having an initial delay is not always desirable. Hence, compensate while retaining
    # the temporal span.
    if p.Delta >= 1:
        p.Delta -= 1
        p.K += 1

    ## Setup for simulations, magic number `1` due to SISO simulations
    win = np.sqrt(sp.signal.windows.hann(p.N, sym=False))[:, np.newaxis]
    p_c = baseline.caf_ctf_params.construct_params(
        p.N,
        p.hop,
        p.Delta,
        p.K,
        p.alpha,
        normalized=False,
        win=win,
    )

    gBuff = jnp.zeros((p.dG + 1, 1))
    uBuff = jnp.zeros((F.shape[0], 1))

    # out-of-loop simulation = baseline
    p_s = baseline.afc_params(
        gG, p.clipG, p.dG, gBuff, F, uBuff, inloop=False, update=True
    )
    _, _, y_e_init, y_l_init = baseline.run_caf_ctf(p_s, p_c, audio_e, audio_l)

    # in-loop simulations without normalization
    p_s.inloop = True
    _, _, y_e_post_reg, y_l_post_reg = baseline.run_caf_ctf(p_s, p_c, audio_e, audio_l)

    # in-loop simulations with normalization
    p_c.normalized = True
    _, _, y_e_post_norm, y_l_post_norm = baseline.run_caf_ctf(
        p_s, p_c, audio_e, audio_l
    )

    ## metrics and plotting
    offset = p.N + p.dG - 1
    y_ref = audio_e[:-offset:, :]
    y_init = np.asarray((y_e_init + y_l_init)[offset:, :])
    y_post_reg = np.asarray((y_e_post_reg + y_l_post_reg)[offset:, :])
    y_post_norm = np.asarray((y_e_post_norm + y_l_post_norm)[offset:, :])

    sfrr_init = metrics.SFRR(y_e_init, y_l_init)[0]
    sfrr_post_reg = metrics.SFRR(y_e_post_reg, y_l_post_reg)[0]
    sfrr_post_norm = metrics.SFRR(y_e_post_norm, y_l_post_norm)[0]

    pesq_init = metrics.pesq(y_ref, y_init, p.fs, mode="wb")[0]
    pesq_post_reg = metrics.pesq(y_ref, y_post_reg, p.fs, mode="wb")[0]
    pesq_post_norm = metrics.pesq(y_ref, y_post_norm, p.fs, mode="wb")[0]

    estoi_init = metrics.stoi(y_ref, y_init, p.fs, extended=True)[0]
    estoi_post_reg = metrics.stoi(y_ref, y_post_reg, p.fs, extended=True)[0]
    estoi_post_norm = metrics.stoi(y_ref, y_post_norm, p.fs, extended=True)[0]

    cd_init = metrics.cd(y_ref[:, 0], y_init[:, 0], p.fs)[0]
    cd_post_reg = metrics.cd(y_ref[:, 0], y_post_reg[:, 0], p.fs)[0]
    cd_post_norm = metrics.cd(y_ref[:, 0], y_post_norm[:, 0], p.fs)[0]

    print(
        f"Metrics\n{30 * '-'}\n\nSFRR\n{30 * '-'}\n- out of loop:\t\t{sfrr_init:.2f} dB"
        f"\n- inside loop (reg):\t{sfrr_post_reg:.2f} dB\n- inside loop (norm):\t"
        f"{sfrr_post_norm:.2f} dB\n\nPESQ\n{30 * '-'}\n- out of loop:\t\t"
        f"{pesq_init:.2f}\n- inside loop (reg):\t{pesq_post_reg:.2f}\n- inside loop"
        f" (norm):\t{pesq_post_norm:.2f}\n\nESTOI\n{30 * '-'}\n- out of loop:\t\t"
        f"{estoi_init:.2f}\n- inside loop (reg):\t{estoi_post_reg:.2f}\n- inside loop"
        f" (norm):\t{estoi_post_norm:.2f}\n\nCD\n{30 * '-'}\n- out of loop:\t\t{cd_init:.2f}"
        f"\n- inside loop (reg):\t{cd_post_reg:.2f}\n- inside loop (norm):\t{cd_post_norm:.2f}"
    )

    _ = plotting.spectrogram_afc_contributions(
        y_e_init[:, 0],
        y_l_init[:, 0],
        p.N,
        title="Out of loop adaptation",
    )
    _ = plotting.spectrogram_afc_contributions(
        y_e_post_reg[:, 0],
        y_l_post_reg[:, 0],
        p.N,
        title="Inloop adaptation (not normalized)",
    )
    _ = plotting.spectrogram_afc_contributions(
        y_e_post_norm[:, 0],
        y_l_post_norm[:, 0],
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
