# Purpose of script:
# Main entry point for performing simulations for dereverberation.
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

import afc_dr


def main():
    ## Data generation
    p = afc_dr.signal_generation.parameters.load_from_yaml(PATH_TO_CONFIG)

    audio_e, audio_l, _ = afc_dr.signal_generation.generate_signals(p)
    audio = audio_e + audio_l

    ## Simulation parameters
    D: int = len(p.mics)
    win: jax.Array = jnp.asarray(  # reshape for broadcasting over multiple mics
        jnp.sqrt(sp.signal.windows.hann(p.N, sym=False).reshape((-1, 1)))
    )

    ## Processing
    w_p = afc_dr.mclp.wpe_params.construct_params(
        p.N, p.hop, win, D, p.Delta, p.K, p.alpha
    )

    _, y_e, y_l = afc_dr.mclp.run_wpe(w_p, audio_e, audio_l)

    ## Results
    # convert to numpy arrays for easier processing
    y_e = np.asarray(y_e)
    y_l = np.asarray(y_l)
    y = y_e + y_l

    offset = p.N - 1  # Compensates for the delay in the WOLA processing

    srr_init = afc_dr.metrics.SNR(audio, audio_e, audio_l)
    srr_wpe = afc_dr.metrics.SNR(y, y_e, y_l)

    stoi_init = afc_dr.metrics.stoi(audio_e, audio, p.fs, False)
    stoi_wpe = afc_dr.metrics.stoi(audio_e[:-offset, :], y[offset:, :], p.fs, False)

    pesq_init = afc_dr.metrics.pesq(audio_e, audio, p.fs, "wb")
    pesq_wpe = afc_dr.metrics.pesq(audio_e[:-offset, :], y[offset:, :], p.fs, "wb")

    cd_init = afc_dr.metrics.cd(audio_e[:, 0], audio[:, 0], p.fs)
    cd_post = afc_dr.metrics.cd(audio_e[:-offset, 0], y[offset:, 0], p.fs)

    print(
        f"SRR [dB]\n{30 * '-'}\n- Initial:\t{srr_init[0]:.2f}\n- "
        f"After WPE:\t{srr_wpe[0]:.2f}\n\nSTOI\n{30 * '-'}\n- Initial:\t"
        f"{stoi_init[0]:.2f}\n- After WPE\t{stoi_wpe[0]:.2f}\n\nPESQ\n"
        f"{30 * '-'}\n- Initial:\t{pesq_init[0]:.2f}\n- After WPE:\t"
        f"{pesq_wpe[0]:.2f}\n\nCD\n{30 * '-'}\n- Initial:\t{cd_init[0]:.2f}\n- "
        f"After WPE:\t{cd_post[0]:.2f}"
    )

    _ = afc_dr.plotting.spectrogram(audio[:, 0], p.N, "Input signal")
    _ = afc_dr.plotting.spectrogram_afc_contributions(
        y_e[:, 0], y_l[:, 0], p.N, "After WPE filtering"
    )

    plt.show(block=True)


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    PATH_TO_CONFIG = os.path.join("config", "config.yml")

    mpl.use("Tkagg")
    plt.ion()

    sys.exit(main())
