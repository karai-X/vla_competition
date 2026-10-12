"""SPARC (spectral arc length) smoothness of an end-effector trajectory.

Implementation of Balasubramanian et al., "On the analysis of movement
smoothness", J. NeuroEng. Rehabil. 12:112 (2015): FFT of the speed profile
zero-padded to the next power of two times ``pad_level``, an adaptive cutoff at
the last frequency above ``amplitude_threshold``, and frequency normalised by
that cutoff. Larger (closer to zero) is smoother.
"""
from __future__ import annotations

import numpy as np


def compute_sparc(positions, dt, *, pad_level=4, cutoff_hz=10.0, amplitude_threshold=0.05, min_speed=1e-6):
    p = np.asarray(positions, dtype=np.float64)
    if p.ndim != 2 or len(p) < 2 or dt <= 0:
        raise ValueError("Require at least two positions and a positive dt")
    speed = np.linalg.norm(np.gradient(p, dt, axis=0), axis=1)
    if speed.max() < min_speed:
        return float("nan")  # undefined for a motionless trajectory
    n_fft = int(2 ** np.ceil(np.log2(len(speed))) * pad_level)
    spectrum = np.abs(np.fft.rfft(speed, n=n_fft))
    freq = np.fft.rfftfreq(n_fft, d=dt)
    magnitude = spectrum / spectrum.max()
    # Adaptive cutoff: the last frequency whose normalised amplitude still
    # reaches the threshold, but never above the fixed cutoff.
    above = np.nonzero(magnitude >= amplitude_threshold)[0]
    adaptive = min(float(freq[above[-1]]), cutoff_hz)
    keep = freq <= adaptive
    if keep.sum() < 2:
        return 0.0
    df = np.diff(freq[keep]) / adaptive
    dm = np.diff(magnitude[keep])
    return float(-np.sum(np.sqrt(df ** 2 + dm ** 2)))
