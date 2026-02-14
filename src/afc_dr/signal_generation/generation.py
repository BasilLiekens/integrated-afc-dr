import os

import numpy as np
import scipy as sp

from afc_dr.metrics import msg

from . import parameters


def generate_signals(p: parameters) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Given the parameters `p` extracted from a config file, create the
    corresponding microphone signals, split between desired signal
    contributions and interfering signals.

    Parameters
    ----------
    p: parameters
        The parameters on which the signal is based, potentially extracted from
        a config file as `p = parameters.load_from_yaml("path")`.

    Returns
    -------
    A tuple with three ndarrays, the first one being the contribution of the
    early reflections to the desired signal, the second one being the
    contribution of the late reflections and the third one being the noise.
    Shape of the returned arrays: [`T` * `fs` x `M`] with `M` the number of
    microphones.
    """
    audio_data = np.zeros((int(p.T * p.fs), len(p.audio_sources)))
    rir_data_early = np.zeros((p.rir_length, len(p.mics), len(p.sources)))
    rir_data_late = np.zeros_like(rir_data_early)
    micsigs_early = np.zeros((int(p.T * p.fs), len(p.mics)))
    micsigs_late = np.zeros_like(micsigs_early)

    ## read audio files
    for i, file in enumerate(p.audio_sources):
        fs, data = sp.io.wavfile.read(os.path.join(p.audio_base, file + ".wav"))
        data = (
            data / -np.iinfo(data.dtype).min
        )  # signed integers, so min larger than max

        if fs != p.fs:
            data = sp.signal.resample_poly(data, up=p.fs, down=fs)

        audio_data[:, i] = data[: int(p.T * p.fs)]

    ## read the rirs from each source to each mic and convolve
    early_length_samples = p.hop
    for i, source in enumerate(p.sources):
        for j, mic in enumerate(p.mics):
            fs, data = sp.io.wavfile.read(
                os.path.join(p.rir_base, p.scenario, source, mic + ".wav")
            )
            data = data / -np.iinfo(data.dtype).min

            if fs != p.fs:
                data = sp.signal.resample_poly(data, up=p.fs, down=fs)

            offset = np.nonzero(np.abs(data) >= 0.01 * np.max(np.abs(data)))[0][0]

            rir_data_early[: offset + early_length_samples, j, i] = data[
                : offset + early_length_samples
            ]
            rir_data_late[early_length_samples + offset :, j, i] = data[
                offset + early_length_samples : int(p.rir_length)
            ]

    ## Convolve the audio signals, normalize to unit variance
    for i in range(rir_data_early.shape[1]):  # loop over microphones
        for j in range(rir_data_early.shape[2]):  # loop over sources
            micsigs_early[:, i] += sp.signal.fftconvolve(
                rir_data_early[:, i, j], audio_data[:, j]
            )[: micsigs_early.shape[0]]

            micsigs_late[:, i] += sp.signal.fftconvolve(
                rir_data_late[:, i, j], audio_data[:, j]
            )[: micsigs_late.shape[0]]

    std_sig = np.std(micsigs_early[:, 0])
    micsigs_early /= std_sig
    micsigs_late /= std_sig

    ## Generate random noise
    rng = np.random.default_rng()  # do not set seed to prevent total determinism
    std_noise = 10 ** (-p.SNR / 20)
    noise = std_noise * rng.normal(size=micsigs_early.shape)

    return micsigs_early, micsigs_late, noise


def generate_feedback_rirs(p: parameters) -> tuple[np.ndarray, float]:
    """
    Construct the RIRs from the feedback source to the mics, rescaling it to
    to have a prespecified gain margin `margin`.

    Parameters
    ----------
    p: parameters
        The acoustic scenario parameters, potentially loaded from yaml. Should
        contain the locations of the source to mic RIRs, the sampling
        frequency, gain and delay in forward path and length of the RIRs.

    margin: float
        The gain margin [dB] to use to rescale the RIRs.

    Returns
    -------
    A tuple containing the RIRs of the requested length alongside the MSG (not
    in dB).
    """
    ## Load in RIRs
    rir_data = np.zeros((p.rir_length, len(p.mics)))

    for i, path in enumerate(p.mics):
        fs, data = sp.io.wavfile.read(
            os.path.join(p.rir_base, p.scenario, p.feedback_source, path + ".wav")
        )

        data = data / -np.iinfo(data.dtype).min

        if fs != p.fs:
            data = sp.signal.resample_poly(data, up=p.fs, down=fs, axis=0)

        rir_data[:, i] = data[: p.rir_length]

    # Compute the GM for each mic separately, the total margin is then
    # determine through the mic with the smallest margin.

    # Insert additional delays to mimic the delay in forward path
    rir_data_pad = np.concatenate((np.zeros((p.dG + 1, rir_data.shape[1])), rir_data))
    msg_ret = np.min([msg(rir_data_pad[:, i]) for i in range(rir_data.shape[1])])

    return rir_data, msg_ret
