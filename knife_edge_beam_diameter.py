"""Focused-beam diameter from a knife-edge power-vs-position scan.

A razor edge is translated across the focused beam in ~1 um steps while
transmitted power is logged. Transmitted power vs edge position follows an
error function whose 1/e^2 radius w gives the beam diameter D = 2w (the
derivative of that erf is a Gaussian of 1/e^2 full-width D).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit
from scipy.special import erf


def _erf_edge(x, a, b, x0, w):
    """Power vs position: baseline a, swing b, edge centre x0, 1/e^2 radius w."""
    return a + b * 0.5 * (1.0 + erf(np.sqrt(2.0) * (x - x0) / w))


def fit_knife_edge(position_mm, transmitted_power_w):
    """Fit a knife-edge scan and compute the focused-beam 1/e^2 diameter.

    Parameters
    ----------
    position_mm : array_like
        Knife-edge stage position, in millimetres.
    transmitted_power_w : array_like
        Transmitted power at each position, in watts.

    Returns
    -------
    x0_mm : float
        Fitted edge-centre position, in the same frame as `position_mm`.
    w_mm : float
        Fitted 1/e^2 beam radius, in millimetres.
    beam_diameter_um : float
        1/e^2 beam diameter, D = 2w, in micrometres.
    diameter_sigma_um : float
        1-sigma fit uncertainty on the beam diameter, in micrometres.
    """
    position_mm = np.asarray(position_mm, dtype=float)
    transmitted_power_w = np.asarray(transmitted_power_w, dtype=float)

    p0 = [
        transmitted_power_w.min(),
        np.ptp(transmitted_power_w),
        position_mm[np.argmin(np.abs(transmitted_power_w - transmitted_power_w.mean()))],
        5e-3,
    ]
    popt, pcov = curve_fit(_erf_edge, position_mm, transmitted_power_w, p0=p0, maxfev=20000)
    a, b, x0_mm, w_mm = popt
    w_mm = abs(w_mm)
    w_sig_mm = np.sqrt(np.diag(pcov))[3]

    beam_diameter_um = 2.0 * w_mm * 1e3
    diameter_sigma_um = 2.0 * w_sig_mm * 1e3
    return x0_mm, w_mm, beam_diameter_um, diameter_sigma_um


if __name__ == "__main__":
    import ast
    import sys
    from pathlib import Path

    if len(sys.argv) != 2:
        print("Usage: python knife_edge_beam_diameter.py <knife_edge_measurements.txt>")
        sys.exit(1)

    data_path = Path(sys.argv[1])
    lines = data_path.read_text().splitlines()
    pts = np.array(ast.literal_eval(lines[1]), dtype=float)
    position_mm, transmitted_power_w = pts[:, 0], pts[:, 1]

    x0_mm, w_mm, beam_diameter_um, diameter_sigma_um = fit_knife_edge(
        position_mm, transmitted_power_w
    )
    print(f"edge centre x0 = {x0_mm:.6f} mm")
    print(f"1/e^2 radius w = {w_mm * 1e3:.3f} um")
    print(f"1/e^2 diameter = {beam_diameter_um:.3f} +/- {diameter_sigma_um:.3f} um")
