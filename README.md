# Integrated afc dr
This repository contains a toolbox for performing simulations that make use of the WPE [1]-[3] method to perform dereverberation and/or feedback cancellation.

## Setup
This repository is set up as a package, contained in `src/afc_dr` which can be installed as `pip install .`. This automatically downloads all required dependencies. 

Alternatively, a `uv.lock` file is provided which enables running scripts as `uv run scripts/main_xyz.py` without any prior setup if `uv` is installed.

Experiments rely on externally provided room impulse responses and audio files. The repository was created for the [MYRiAD database](https://link.springer.com/article/10.1186/s13636-023-00284-9) and the [CSTR VCTK corpus](https://datashare.ed.ac.uk/handle/10283/3443), but other datasources should also work if the paths are set correctly in the config file. 

## Folder structure
The project is set up with a src-layout: the main package providing all algorithms and utils can be found in `src/afc_dr`. 
The provided algorithms are the WPE algorithm in a pure dereverberation setting [1]-[3], the WPE algorithm in a setting with feedback and an implementation of a continuous adaptive filter in the STFT domain for feedback reduction [4], [5].

Example scripts using the package can be found in the `scripts` folder. 
These scripts are driven from a `config.yml` file, an example of which can be found under `config/config_example.yml`.

## References
[1] T. Yoshioka, H. Tachibana, T. Nakatani, and M. Miyoshi, “Adaptive dereverberation of speech signals with speaker-position change detection,” in 2009 IEEE International Conference on Acoustics, Speech and Signal Processing, Taipei, Taiwan: IEEE, Apr. 2009, pp. 3733–3736. doi: 10.1109/ICASSP.2009.4960438.

[2] T. Nakatani, T. Yoshioka, K. Kinoshita, M. Miyoshi, and Biing-Hwang Juang, “Speech Dereverberation Based on Variance-Normalized Delayed Linear Prediction,” IEEE Trans. Audio Speech Lang. Process., vol. 18, no. 7, pp. 1717–1731, Sep. 2010, doi: 10.1109/TASL.2010.2052251.

[3] L. Drude, J. Heymann, C. Boeddeker, and R. Haeb-Umbach, “NARA-WPE: A Python package for weighted prediction error dereverberation in Numpy and Tensorﬂow for online and ofﬂine processing,” in Speech Communication; 13th ITG-Symposium, Oldenburg Germany, Oct. 2018, pp. 1–5.

[4] A. Spriet, I. Proudler, M. Moonen, and J. Wouters, “Adaptive feedback cancellation in hearing aids with linear prediction of the desired signal,” IEEE Trans. Signal Process., vol. 53, no. 10, pp. 3749–3763, Oct. 2005, doi: 10.1109/TSP.2005.855108.

[5] M. Guo, S. H. Jensen, and J. Jensen, “Evaluation of State-of-the-Art Acoustic Feedback Cancellation Systems for Hearing Aids,” Journal of the Audio Engineering Society, vol. 61, pp. 125–137, 2013.
