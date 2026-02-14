import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from matplotlib.figure import Figure


def spectrogram(x: np.ndarray, N: int, title: str) -> Figure:
    """
    Given a 1D numpy array `x`, return a spectrogram of it with the given title.

    This spectrogram is always made by using a DFT length of `N`, using a
    square-root Hann window and a hop of 50%. In the `stft` object, the
    sampling frequency is set to 16e3, but that does not influence the result.

    Parameters
    ----------
    x: np.ndarray, 1D
        The signal to make a spectrogram of.

    N: int
        The number of points in the DFT.

    title: str
        The title of the plot.

    Returns
    -------
    The resulting figure.
    """
    win = np.sqrt(sp.signal.windows.hann(N, sym=False))
    _, _, X = sp.signal.stft(x, int(16e3), win, nperseg=N, noverlap=N // 2)

    fig, ax = plt.subplots()
    fig.set_size_inches(8.5, 5.5)
    img = ax.pcolormesh(
        20 * np.log10(np.maximum(np.abs(X), 1e-10)), cmap="plasma", vmin=-100, vmax=10
    )
    fig.colorbar(img)
    ax.set(xlabel="frame", ylabel="frequency bin", title=title)
    ax.autoscale(tight=True, axis="x")
    fig.tight_layout()

    return fig


def spectrogram_afc_contributions(
    x_e: np.ndarray, x_l: np.ndarray, x_f: np.ndarray, N: int, title: str
) -> Figure:
    """
    Utility function to plot all spectrograms of individual contributions to AFC
    output in one plot.

    Parameters
    ----------
    x_e: np.ndarray, 1D
        The contributions of the direct path + early reflections to the output.

    x_l: np.ndarray, 1D
        Similar, but now for the late reverberations.

    x_f: np.ndarray, 1D
        Similar, but now for the feedback component.

    N: int
        The number of points to use in the DFT.

    title: str
        The title of the figure to return.

    Returns
    -------
    The resulting figure
    """
    win = np.sqrt(sp.signal.windows.hann(N, sym=False))
    x = x_e + x_l + x_f

    _, _, X_e = sp.signal.stft(x_e, int(16e3), win, nperseg=N, noverlap=N // 2)
    _, _, X_l = sp.signal.stft(x_l, int(16e3), win, nperseg=N, noverlap=N // 2)
    _, _, X_f = sp.signal.stft(x_f, int(16e3), win, nperseg=N, noverlap=N // 2)
    _, _, X = sp.signal.stft(x, int(16e3), win, nperseg=N, noverlap=N // 2)

    fig = plt.figure()
    fig.set_size_inches(8.5, 5.5)
    ax = fig.add_subplot(221)
    img = ax.pcolormesh(
        20 * np.log10(np.maximum(np.abs(X), 1e-10)), cmap="plasma", vmin=-100, vmax=10
    )
    fig.colorbar(img)
    ax.set(xlabel="frame", ylabel="frequency bin", title="Total output signal")
    ax.autoscale(tight=True, axis="x")

    ax = fig.add_subplot(222)
    img = ax.pcolormesh(
        20 * np.log10(np.maximum(np.abs(X_e), 1e-10)), cmap="plasma", vmin=-100, vmax=10
    )
    fig.colorbar(img)
    ax.set(
        xlabel="frame",
        ylabel="frequency bin",
        title="Contribution of early reflections",
    )
    ax.autoscale(tight=True, axis="x")

    ax = fig.add_subplot(223)
    img = ax.pcolormesh(
        20 * np.log10(np.maximum(np.abs(X_f), 1e-10)), cmap="plasma", vmin=-100, vmax=10
    )
    fig.colorbar(img)
    ax.set(
        xlabel="frame",
        ylabel="frequency bin",
        title="Contribution of feedback",
    )
    ax.autoscale(tight=True, axis="x")

    ax = fig.add_subplot(224)
    img = ax.pcolormesh(
        20 * np.log10(np.maximum(np.abs(X_l), 1e-10)), cmap="plasma", vmin=-100, vmax=10
    )
    fig.colorbar(img)
    ax.set(
        xlabel="frame",
        ylabel="frequency bin",
        title="Contribution of late reverberations",
    )
    ax.autoscale(tight=True, axis="x")

    fig.suptitle(title)
    fig.tight_layout()
    return fig
