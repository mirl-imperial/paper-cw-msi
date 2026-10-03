"""Spatial resolution from a step-edge fit.

An intensity profile across a tissue edge is fitted to an ideal step blurred
by a Gaussian of standard deviation sigma (Zhang & Bergholm, Int. J. Comput.
Vis. 24, 219-250, 1997). Resolution is the 16-84% rise distance (Kompauer et
al., Nat. Methods 14, 90-96, 2017).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit
from scipy.special import erf, erfinv

LOW_FRAC, HIGH_FRAC = 0.16, 0.84


def _edge_model(x, low, high, x0, sigma):
    return low + (high - low) * 0.5 * (1 + erf((x - x0) / (np.sqrt(2) * sigma)))


def fit_psf_resolution(distance_um, intensity):
    """Fit a step-edge profile and compute the 16-84% spatial resolution.

    Parameters
    ----------
    distance_um : array_like
        Position along the sampled line, in micrometres.
    intensity : array_like
        Raw per-pixel intensity at each position.

    Returns
    -------
    popt : ndarray, shape (4,)
        Fitted parameters [low, high, x0, sigma] of the erf edge model.
    perr : ndarray, shape (4,)
        1-sigma parameter uncertainties (sqrt of the fit covariance diagonal).
    resolution_16_84_um : float
        16-84% (+/-1 sigma) rise distance, in micrometres:
        2 * sqrt(2) * sigma * erf^-1(0.68).
    """
    distance_um = np.asarray(distance_um, dtype=float)
    intensity = np.asarray(intensity, dtype=float)

    lo_guess, hi_guess = intensity.min(), intensity.max()
    x0_guess = distance_um[np.argmin(np.abs(intensity - 0.5 * (lo_guess + hi_guess)))]
    sigma_guess = (distance_um.max() - distance_um.min()) / 8.0
    if sigma_guess <= 0:
        sigma_guess = 1.0
    p0 = [lo_guess, hi_guess, x0_guess, sigma_guess]
    bounds = (
        [-np.inf, -np.inf, distance_um.min(), 1e-3],
        [np.inf, np.inf, distance_um.max(), max(distance_um.max() - distance_um.min(), 1.0)],
    )
    popt, pcov = curve_fit(_edge_model, distance_um, intensity, p0=p0, bounds=bounds, maxfev=10000)
    perr = np.sqrt(np.diag(pcov))

    sigma = popt[3]
    z = erfinv(HIGH_FRAC - LOW_FRAC)  # erf(z) = 0.68 for 16-84%
    resolution_16_84_um = 2 * np.sqrt(2) * sigma * z

    return popt, perr, resolution_16_84_um


if __name__ == "__main__":
    import csv
    import sys
    from pathlib import Path

    if len(sys.argv) != 2:
        print("Usage: python line_spread_resolution.py <line_profile.csv>")
        print("  CSV must have columns: distance_um, intensity")
        sys.exit(1)

    data_path = Path(sys.argv[1])
    with open(data_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = [(float(r["distance_um"]), float(r["intensity"])) for r in reader]
    distance_um, intensity = zip(*rows)

    popt, perr, resolution_16_84_um = fit_psf_resolution(distance_um, intensity)
    print(f"fit [low, high, x0, sigma] = {popt}")
    print(f"fit uncertainties          = {perr}")
    print(f"16-84% resolution = {resolution_16_84_um:.3f} um")
