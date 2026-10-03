"""Ion images warped into the H&E frame: RGB composites (Fig. 4a,b) and
single-ion overlays on a greyscale H&E backdrop (Fig. 4d-f).
"""
from __future__ import annotations

import numpy as np
from matplotlib import colormaps
from matplotlib.colors import Normalize
from scipy.ndimage import label as cc_label
from scipy.ndimage import map_coordinates

from he_panel_contrast import enhance_he_panel

# (m/z, label, RGB channel in Fig. 4a,b, overlay panel in Fig. 4d-f)
FIG4_CHANNELS = [
    (790.5367, "PE 40:6 [M-H]-", "R", "f"),
    (134.0465, "adenine [M-H]-", "G", "d"),
    (772.5833, "PC 36:1 [M-CH3]-", "B", "e"),
]


def read_hdi_ion_images(path, target_mz, mz_tol=0.002):
    """Read selected channels of a Waters HDImaging peak-picked export (.txt).

    The raster width is the number of pixels before the stage y position
    first changes. Each target m/z is matched to the nearest peak-list
    channel within `mz_tol` (Da).

    Returns
    -------
    dict {target_mz: ndarray (height, width), float32}
    """
    with open(path, encoding="utf-8", errors="replace") as fh:
        header = [fh.readline() for _ in range(4)]
        cells = header[3].rstrip("\n").split("\t")[3:]
        mz = np.array([float(c) for c in cells if c.strip() != ""])
        cols = {}
        for t in target_mz:
            i = int(np.argmin(np.abs(mz - t)))
            if abs(mz[i] - t) > mz_tol:
                raise ValueError(f"no channel within {mz_tol} Da of m/z {t}; closest {mz[i]:.4f}")
            cols[t] = 3 + i
        values = {t: [] for t in target_mz}
        y_mm = []
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3 + mz.size:
                continue
            y_mm.append(float(parts[2]))
            for t, c in cols.items():
                values[t].append(float(parts[c]))
    y_mm = np.array(y_mm)
    changes = np.flatnonzero(y_mm != y_mm[0])
    width = int(changes[0]) if changes.size else y_mm.size
    height = y_mm.size // width
    return {
        t: np.array(v, dtype=np.float32)[: height * width].reshape(height, width)
        for t, v in values.items()
    }


def ion_image_rgba(raster, saturation_pct=99.5, rot90_k=1):
    """8-bit greyscale render of an ion image, as used to build Fig. 4.

    The raster is rotated by `rot90_k` quarter turns anticlockwise (the
    orientation the landmarks were placed in) and scaled linearly from 0 to
    the `saturation_pct` percentile of its non-zero pixels, clipped above.

    Returns
    -------
    ndarray, shape (H, W, 4), uint8
    """
    img = np.rot90(np.asarray(raster), rot90_k)
    vmax = float(np.percentile(img[img > 0], saturation_pct))
    return colormaps["gray"](Normalize(vmin=0.0, vmax=vmax)(img), bytes=True)


def _extract_intensity(ion_image):
    """2D intensity from an ion image: the array itself, a non-constant alpha
    channel, or the Rec. 709 luminance of the RGB channels."""
    ion_image = np.asarray(ion_image)
    if ion_image.ndim == 2:
        return ion_image.astype(np.float32)
    if ion_image.shape[-1] == 4:
        alpha = ion_image[..., 3]
        if alpha.max() > alpha.min() and alpha.max() > 0:
            return alpha.astype(np.float32)
    rgb = ion_image[..., :3].astype(np.float32)
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def _apply_intensity_transform(intensity, transform, log_offset=1.0):
    """Return (transformed intensity, value that zero signal maps to)."""
    if transform == "sqrt":
        return np.sqrt(np.maximum(intensity, 0.0)).astype(np.float32), 0.0
    if transform == "log":
        return (
            np.log10(intensity + log_offset).astype(np.float32),
            float(np.log10(log_offset)),
        )
    if transform == "linear":
        return intensity.astype(np.float32), 0.0
    raise ValueError(f"unknown intensity transform {transform!r}")


def warp_ion_channel(
    ion_image,
    tps_transform,
    output_origin_xy,
    output_shape,
    intensity_pct_clip=99.0,
    intensity_transform="sqrt",
):
    """Transform, normalise and warp one ion image into the target frame.

    Parameters
    ----------
    ion_image : array_like, shape (Hm, Wm) or (Hm, Wm, 3) or (Hm, Wm, 4)
        Ion image in the pixel grid the landmarks were placed on.
    tps_transform : dict
        Output of `he_overlay_coregistration.fit_coregistration`, mapping
        target-space (x, y) to ion-image (x, y) coordinates.
    output_origin_xy : (float, float)
        (x0, y0) of the top-left output pixel in the target (H&E) frame.
    output_shape : (int, int)
        (height, width) of the output region, in target pixels.
    intensity_pct_clip : float
        Percentile (0-100) of the transformed intensity, taken over all
        ion-image pixels with non-zero signal, that maps to 1.
    intensity_transform : {"sqrt", "linear", "log"}
        Applied in the ion image's own grid, before normalisation and warping.

    Returns
    -------
    ndarray, shape (H, W), float32 in [0, 1]
        Normalised intensity in the target frame; 0 where the target pixel
        maps outside the ion image.
    """
    intensity = _extract_intensity(ion_image)
    msi_h, msi_w = intensity.shape
    intensity_x, floor = _apply_intensity_transform(intensity, intensity_transform)

    nz_mask = intensity > 0
    if nz_mask.any():
        vmax = float(np.percentile(intensity_x[nz_mask], intensity_pct_clip))
    else:
        vmax = floor + 1.0

    x0, y0 = output_origin_xy
    out_h, out_w = output_shape
    yy, xx = np.mgrid[y0 : y0 + out_h, x0 : x0 + out_w]
    target_xy = np.stack([xx.ravel(), yy.ravel()], axis=1)
    msi_x = tps_transform["tps_x"](target_xy).reshape(out_h, out_w)
    msi_y = tps_transform["tps_y"](target_xy).reshape(out_h, out_w)

    sampled = map_coordinates(intensity_x, [msi_y, msi_x], order=1, mode="constant", cval=floor)
    in_bounds = (msi_x >= 0) & (msi_x < msi_w - 1) & (msi_y >= 0) & (msi_y < msi_h - 1)
    normed = (sampled - floor) / max(vmax - floor, 1e-9)
    return np.clip(normed, 0.0, 1.0) * in_bounds


def render_ion_overlay(
    ion_image,
    tps_transform,
    output_origin_xy,
    output_shape,
    intensity_pct_clip=99.0,
    signal_floor=0.12,
    alpha_max=0.85,
    colormap="viridis",
    intensity_transform="sqrt",
):
    """Warp, normalise and colour-map a single ion image into an RGBA overlay.

    Parameters are as for `warp_ion_channel`, plus:

    signal_floor : float
        Normalised-intensity threshold (0-1). Pixels below it are fully
        transparent; above it, intensity is rescaled to span the full
        colour and opacity range.
    alpha_max : float
        Opacity (0-1) at full intensity.
    colormap : str
        Matplotlib colormap name.

    Returns
    -------
    ndarray, shape (H, W, 4), float64 in [0, 1]
        RGBA overlay. Composite it over `he_greyscale_backdrop` with
        `composite_over_backdrop`.
    """
    normed = warp_ion_channel(
        ion_image,
        tps_transform,
        output_origin_xy,
        output_shape,
        intensity_pct_clip=intensity_pct_clip,
        intensity_transform=intensity_transform,
    )
    above = np.clip((normed - signal_floor) / max(1.0 - signal_floor, 1e-9), 0.0, 1.0)
    overlay = colormaps[colormap](above)
    overlay[..., 3] = above * alpha_max
    return overlay


def he_greyscale_backdrop(image_rgb, lighten=0.45):
    """Luminance of the contrast-stretched H&E, lightened toward white.

    Parameters
    ----------
    image_rgb : array_like, shape (H, W, 3)
        H&E export (uint8 range), already cropped with
        `he_panel_contrast.crop_bottom_fraction`. It is passed through
        `he_panel_contrast.enhance_he_panel` with its default settings.
    lighten : float
        The luminance v in [0, 1] is mapped to v * (1 - lighten) + lighten.

    Returns
    -------
    ndarray, shape (H, W), dtype uint8
    """
    colour = enhance_he_panel(image_rgb)
    lum = (
        0.2126 * colour[..., 0] + 0.7152 * colour[..., 1] + 0.0722 * colour[..., 2]
    ).astype(np.float32) / 255.0
    lightened = lum * (1.0 - lighten) + lighten
    return (lightened * 255).astype(np.uint8)


def composite_over_backdrop(grey_backdrop, overlay):
    """Alpha-composite an RGBA overlay over a greyscale backdrop.

    Returns
    -------
    ndarray, shape (H, W, 3), float64 in [0, 1]
    """
    bg = np.asarray(grey_backdrop, dtype=np.float64)[..., None] / 255.0
    alpha = overlay[..., 3:4]
    return bg * (1.0 - alpha) + overlay[..., :3] * alpha


def tissue_bbox(he_rgb, threshold=215, margin_frac=0.04):
    """Bounding box of the largest tissue region in an H&E image.

    Tissue is every pixel with Rec. 709 luminance below `threshold`; the
    largest connected region is kept, and its bounding box is padded on all
    sides by `margin_frac` of its shorter side (clipped to the image).

    Returns
    -------
    (x0, y0, height, width) : tuple of int
        Use as `output_origin_xy=(x0, y0)`, `output_shape=(height, width)`.
    """
    he_rgb = np.asarray(he_rgb)
    he_h, he_w = he_rgb.shape[:2]
    grey = (
        0.2126 * he_rgb[..., 0] + 0.7152 * he_rgb[..., 1] + 0.0722 * he_rgb[..., 2]
    ).astype(np.float32)
    labeled, _ = cc_label(grey < threshold)
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    ys, xs = np.where(labeled == int(sizes.argmax()))
    ty0, ty1 = int(ys.min()), int(ys.max())
    tx0, tx1 = int(xs.min()), int(xs.max())
    margin = int(round(min(ty1 - ty0, tx1 - tx0) * margin_frac))
    y0, y1 = max(0, ty0 - margin), min(he_h, ty1 + margin)
    x0, x1 = max(0, tx0 - margin), min(he_w, tx1 + margin)
    return x0, y0, y1 - y0, x1 - x0


def render_rgb_composite(red, green, blue):
    """Stack three normalised channels into an RGB image, values in [0, 1]."""
    return np.stack([np.asarray(c) for c in (red, green, blue)], axis=-1)


def region_outline(region_shape, region_transform, whole_inverse_transform, crop_origin_xy):
    """Footprint of the region export on the whole-section crop (Fig. 4a box).

    Parameters
    ----------
    region_shape : (int, int)
        (height, width) of the region export after cropping.
    region_transform : dict
        `fit_coregistration(region_moving, region_target)`.
    whole_inverse_transform : dict
        `fit_coregistration(whole_target, whole_moving)` (arguments swapped).
    crop_origin_xy : (int, int)
        (x0, y0) of the whole-section crop, from `tissue_bbox`.

    Returns
    -------
    ndarray, shape (4, 2)
        (x, y) corners in the pixel frame of the Fig. 4a crop.
    """
    h, w = region_shape
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=float)
    ion_xy = np.stack(
        [region_transform["tps_x"](corners), region_transform["tps_y"](corners)], axis=1
    )
    whole_xy = np.stack(
        [whole_inverse_transform["tps_x"](ion_xy), whole_inverse_transform["tps_y"](ion_xy)],
        axis=1,
    )
    return whole_xy - np.asarray(crop_origin_xy, dtype=float)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    from PIL import Image

    from he_overlay_coregistration import fit_coregistration, load_bigwarp_landmarks
    from he_panel_contrast import crop_bottom_fraction

    if len(sys.argv) != 2:
        print("Usage: python he_overlay_render.py <unzipped Zenodo data record>")
        print("  Writes Fig. 4a, 4b and 4d-f image arrays as PNGs to ./fig4_panels/")
        sys.exit(1)

    root = Path(sys.argv[1])
    export = root / "01_MSI_HDI_processed_data" / "3um_olfactory_bulb" / "MB1_OLF_2_3um_6.txt"
    coreg = root / "09_Fig4_HE_exports_and_landmarks"
    out_dir = Path("fig4_panels")
    out_dir.mkdir(exist_ok=True)

    def save(arr, name):
        Image.fromarray((np.asarray(arr) * 255).round().astype(np.uint8)).save(out_dir / name)
        print(f"  wrote {out_dir / name}")

    print("Reading the 3 um HDImaging export (~1 min) ...")
    rasters = read_hdi_ion_images(export, [ch[0] for ch in FIG4_CHANNELS])
    ions = {mz: ion_image_rgba(raster) for mz, raster in rasters.items()}
    by_rgb = {ch[2]: ch[0] for ch in FIG4_CHANNELS}

    whole_he = np.asarray(Image.open(coreg / "HE_exports" / "OLF_Ablated.jpg").convert("RGB"))
    whole_moving, whole_target = load_bigwarp_landmarks(
        coreg / "bigwarp_landmarks" / "landmarksCW_OLF.csv"
    )
    whole_t = fit_coregistration(whole_moving, whole_target)
    x0, y0, crop_h, crop_w = tissue_bbox(whole_he)
    fig4a = render_rgb_composite(*[
        warp_ion_channel(ions[by_rgb[c]], whole_t, (x0, y0), (crop_h, crop_w),
                         intensity_transform="linear")
        for c in "RGB"
    ])
    save(fig4a, "fig4a_rgb_composite.png")

    zoom_he = crop_bottom_fraction(
        np.asarray(Image.open(coreg / "HE_exports" / "WILL_HE_MBI_OLF_2_Zoom.jpg").convert("RGB")),
        0.07,
    )
    zoom_moving, zoom_target = load_bigwarp_landmarks(
        coreg / "bigwarp_landmarks" / "landmarksZoom.csv"
    )
    zoom_t = fit_coregistration(zoom_moving, zoom_target)
    zoom_shape = zoom_he.shape[:2]

    box = region_outline(zoom_shape, zoom_t, fit_coregistration(whole_target, whole_moving), (x0, y0))
    print("  Fig. 4a box corners (x, y px in the 4a crop):", np.round(box, 1).tolist())

    fig4b = render_rgb_composite(*[
        warp_ion_channel(ions[by_rgb[c]], zoom_t, (0, 0), zoom_shape, intensity_transform="linear")
        for c in "RGB"
    ])
    save(fig4b, "fig4b_rgb_composite_zoom.png")

    grey = he_greyscale_backdrop(zoom_he)
    for mz, _, _, panel in FIG4_CHANNELS:
        overlay = render_ion_overlay(ions[mz], zoom_t, (0, 0), zoom_shape)
        save(composite_over_backdrop(grey, overlay), f"fig4{panel}_{mz:.4f}_overlay.png")
