import os
from dataclasses import dataclass, field
from typing import Self

import yaml


@dataclass
class parameters:
    """
    Class containing the parameters required to create the desired signals.
    """

    ## Signal parameters
    fs: int = int(16e3)  # desired sampling frequency
    T: float = 10  # duration of the signal [s]
    SNR: float = 10  # SNR of measurement noise [dB]

    ## Data locations, catered towards `CSTR_VCTK` corpus and `MYRiAD` database
    audio_base: str = os.path.join("path", "to", "directory")  # where to find audio
    audio_sources: list[str] = field(  # what source files to pick
        default_factory=lambda: ["source_1", "source_2"]
    )

    rir_base: str = os.path.join("path", "to", "directory")  # where to find RIRs
    rir_length: int = 32000  # number of samples from RIR to take into account
    scenario: str = "scenario_name"  # which scenario to pick? Possibly empty
    sources: list[str] = field(  # Which are the sources (for RIRs)?
        default_factory=lambda: ["source_1", "source_2"]
    )
    feedback_source: str = "feedback_source_1"  # Which (single) source of feedback?
    mics: list[str] = field(default_factory=lambda: ["mic_1", "mic_2"])  # Mics to use

    ## Simulation settings
    N: int = 512  # DFT length
    overlap: float = 0.5  # Overlap between successive frames, used to set the hop
    hop: int = 256  # hop between successive frames; controlled by `N` and `overlap`

    Delta: int = 1  # Gap between frames to estimate and filter, should be >= 0
    K: int = 4  # Number of frames to filter in CTF approximation
    delta_L: int = 1  # Number of old frames variance estimation
    alpha: float = 0.99  # smoothing parameter for the time-recursive updates

    dG: int = 0  # Delay in forward path
    clipG: float = 1000  # Clipping value of the signal in the forward path
    GM: float = 6  # Gain margin w.r.t. MSG of the system

    def __post_init__(self):
        """Perform some sanity checks"""
        if len(self.audio_sources) != len(self.sources):
            raise ValueError(
                "Expected same number of audio sources and RIRs, got "
                f"{len(self.audio_sources)} and {len(self.sources)} instead"
            )

        self.hop = int((1 - self.overlap) * self.N)

    @classmethod
    def load_from_yaml(cls, path: str) -> Self:
        """Load the parameters from YAML file"""
        with open(path) as file:  # `open` is read-only by default
            data = yaml.safe_load(file)

        p = cls()

        for key, value in data.items():
            setattr(p, key, value)

        p.__post_init__()
        return p

    def __repr__(self):
        """Return string representation of the object"""
        return "Parameters:\n >> " + "\n >> ".join(
            [f"{key}: {value}" for key, value in self.__dict__.items()]
        )
