# ict_analysis.py

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# Calibration constants (CEA ICT)
VAL1 = 1.16025  # Microampere scaling factor
VAL2 = 0.80762  # Voltage scaling factor


def apply_filter(ivec: np.ndarray) -> np.ndarray:
    """
    Applies a zero-phase Butterworth low-pass filter to the input vector.

    Parameters:
    -----------
    ivec : ndarray
        Input current signal (in mA).

    Returns:
    --------
    ndarray
        Filtered current signal.
    """
    b, a = signal.butter(3, 0.01)
    zi = signal.lfilter_zi(b, a)
    z, _ = signal.lfilter(b, a, ivec, zi=zi * ivec[0])
    z2, _ = signal.lfilter(b, a, z, zi=zi * z[0])
    y = signal.filtfilt(b, a, ivec)
    return y


def get_boundary_minima(mins: np.ndarray, maxs: np.ndarray, startmax: int = 0) -> list:
    """
    Finds boundary minima (left and right) around each peak.

    Parameters:
    -----------
    mins : ndarray
        Indices of local minima in the filtered signal.
    maxs : ndarray
        Indices of local maxima in the filtered signal.
    startmax : int
        Index in maxs to start pairing minima (to skip spurious first peak).

    Returns:
    --------
    List[Tuple[int, int, int]]
        A list of (left_min_idx, peak_idx, right_min_idx) tuples.
    """
    boundaries = []
    for idx in range(startmax, len(maxs)):
        peak_idx = maxs[idx]
        left_candidates = mins[mins < peak_idx]
        left = int(left_candidates.max()) if left_candidates.size else 0
        right_candidates = mins[mins > peak_idx]
        right = int(right_candidates.min()) if right_candidates.size else len(mins) - 1
        boundaries.append((left, peak_idx, right))
    return boundaries


def analyse_vec(time_v: np.ndarray, V_v: np.ndarray, show_plot: bool = False) -> float:
    """
    Analyze ICT waveform data to compute the total integrated charge.

    Parameters:
    -----------
    time_v : ndarray
        Time vector in microseconds (µs).
    V_v : ndarray
        Voltage vector in volts (V).
    show_plot : bool, optional
        If True, plots the current waveform and integration spans.

    Returns:
    --------
    float
        Total integrated charge in picoCoulombs (pC).
    """
    # Convert voltage to current (mA)
    mA_v = VAL1 * 10 ** (V_v / VAL2)

    # Filter the current signal
    filtered = apply_filter(mA_v)

    # Find local minima and maxima
    mins = signal.argrelextrema(filtered, np.less)[0]
    maxs = signal.argrelextrema(filtered, np.greater)[0]

    # Determine integration boundaries around each peak
    peak_bounds = get_boundary_minima(mins, maxs, startmax=1)

    # Compute time step (µs)
    dt = np.mean(np.diff(time_v))

    # Integrate charge under each peak (mA * µs -> pC)
    charges = []
    for left, peak, right in peak_bounds:
        charge = filtered[left:right].sum() * dt
        charges.append(charge)
    total_charge = float(np.sum(charges))

    if show_plot:
        fig, ax = plt.subplots()
        ax.plot(time_v, mA_v, label="Original Signal")
        ax.plot(time_v, filtered, label="Filtered Signal")
        for i, (l, p, r) in enumerate(peak_bounds):
            ax.axvspan(time_v[l], time_v[r],
                       facecolor='green' if i % 2 == 0 else 'red', alpha=0.3)
        ax.set_title(f"Total Charge: {total_charge:.2f} pC")
        ax.set_xlabel("Time (µs)")
        ax.set_ylabel("Current (mA)")
        ax.legend()
        ax.grid(True)
        plt.show()

    return total_charge


def analyze_file(filepath: str, show_plot: bool = False) -> float:
    """
    Load a waveform file (CSV) and compute its integrated charge.

    Parameters:
    -----------
    filepath : str
        Path to the CSV file containing two columns: time (µs) and voltage (V).
    show_plot : bool, optional
        If True, shows the charge plot for this file.

    Returns:
    --------
    float
        Total integrated charge in picoCoulombs.
    """
    # Load data assuming two columns, header row optional
    data = np.loadtxt(filepath, delimiter=',', skiprows=2)
    time_v = data[:, 0]
    V_v = data[:, 1]
    return analyse_vec(time_v, V_v, show_plot)
