"""Thin-plate-spline coregistration fit from paired landmark points.

Fits the transform used to warp an MSI ion image into the pixel frame of the
post-ablation H&E image of the same section, from manually placed BigWarp
landmark pairs. `calibrate_pixel_pitch` derives the H&E pixel pitch from the
same landmark pairs, given the MSI raster step.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import RBFInterpolator


def load_bigwarp_landmarks(path):
    """Read active landmark pairs from a BigWarp CSV export.

    BigWarp columns (no header row): name, active, moving_x, moving_y,
    target_x, target_y.

    Returns
    -------
    (moving_points, target_points) : two ndarrays, shape (N, 2), of (x, y)
    """
    rows = []
    with open(path) as f:
        for line in f:
            parts = [p.strip().strip('"') for p in line.strip().split(",")]
            if len(parts) >= 6 and parts[1].lower() == "true":
                rows.append([float(p) for p in parts[2:6]])
    landmarks = np.array(rows)
    return landmarks[:, :2], landmarks[:, 2:]


def fit_coregistration(moving_points, target_points):
    """Fit a thin-plate-spline transform from target-space to moving-space.

    Parameters
    ----------
    moving_points : array_like, shape (N, 2)
        (x, y) landmark coordinates in the moving image (the image being
        warped, e.g. the MSI ion image), each row paired with the
        corresponding row of `target_points`.
    target_points : array_like, shape (N, 2)
        (x, y) landmark coordinates in the target image (the fixed
        reference frame being warped into, e.g. the H&E image).

    Returns
    -------
    dict with keys "tps_x", "tps_y"
        Each a fitted `scipy.interpolate.RBFInterpolator` mapping
        target-space (x, y) coordinates to the corresponding moving-space
        x or y coordinate. Pass this dict as the `tps_transform` argument of
        the functions in `he_overlay_render`. Swapping the two arguments
        gives the inverse direction (moving to target).
    """
    moving_points = np.asarray(moving_points, dtype=float)
    target_points = np.asarray(target_points, dtype=float)

    tps_x = RBFInterpolator(target_points, moving_points[:, 0], kernel="thin_plate_spline")
    tps_y = RBFInterpolator(target_points, moving_points[:, 1], kernel="thin_plate_spline")
    return {"tps_x": tps_x, "tps_y": tps_y}


def calibrate_pixel_pitch(moving_points, target_points, moving_pitch_um):
    """Target-image pixel pitch from landmark-pair distances.

    For every pair of landmarks, the physical distance between them is the
    moving-image pixel distance times `moving_pitch_um`; dividing by the
    target-image pixel distance gives one pitch estimate. The median over
    all pairs is returned.

    Returns
    -------
    (pitch_um, mad_um) : median pitch and median absolute deviation, in um per pixel
    """
    moving_points = np.asarray(moving_points, dtype=float)
    target_points = np.asarray(target_points, dtype=float)
    ratios = []
    n = len(moving_points)
    for i in range(n):
        for j in range(i + 1, n):
            d_moving = float(np.linalg.norm(moving_points[i] - moving_points[j]))
            d_target = float(np.linalg.norm(target_points[i] - target_points[j]))
            if d_target > 1e-6:
                ratios.append(d_moving * moving_pitch_um / d_target)
    ratios = np.array(ratios)
    pitch = float(np.median(ratios))
    return pitch, float(np.median(np.abs(ratios - pitch)))


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python he_overlay_coregistration.py <bigwarp_landmarks.csv>")
        sys.exit(1)

    moving_points, target_points = load_bigwarp_landmarks(sys.argv[1])
    print(f"Loaded {len(moving_points)} active landmarks.")
    transform = fit_coregistration(moving_points, target_points)

    # The thin-plate spline interpolates the landmarks exactly, so each target
    # landmark should map back onto its paired moving landmark.
    pred_x = transform["tps_x"](target_points)
    pred_y = transform["tps_y"](target_points)
    residual = np.hypot(pred_x - moving_points[:, 0], pred_y - moving_points[:, 1])
    print(f"Landmark residual (px): mean={residual.mean():.3g}, max={residual.max():.3g}")

    pitch, mad = calibrate_pixel_pitch(moving_points, target_points, moving_pitch_um=3.0)
    print(f"H&E pixel pitch (3 um MSI raster): {pitch:.4f} um/px (MAD {mad:.4f})")
