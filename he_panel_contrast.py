"""Display processing of the H&E region export (Fig. 4c).

`crop_bottom_fraction` removes the scale-bar strip that NDP.view burns into
the bottom of an export (the bottom 7% of the image rows). `enhance_he_panel`
then rescales each RGB channel independently so that its 1st-percentile
intensity maps to 0 and its 99th-percentile intensity maps to 255 (linear
between, clipped outside). No other per-pixel operation is applied.
"""
from __future__ import annotations

import numpy as np


def crop_bottom_fraction(image, frac=0.07):
    """Drop the bottom `frac` of the image rows (keeps round(H * (1 - frac)) rows)."""
    image = np.asarray(image)
    keep_rows = int(round(image.shape[0] * (1.0 - frac)))
    return image[:keep_rows]


def enhance_he_panel(image_rgb, pct_lo=1.0, pct_hi=99.0):
    """Per-channel linear percentile contrast stretch.

    Parameters
    ----------
    image_rgb : array_like, shape (H, W, 3)
        RGB image, uint8 range (0-255).
    pct_lo, pct_hi : float
        Percentiles (0-100) mapped to 0 and 255 in each channel.

    Returns
    -------
    ndarray, shape (H, W, 3), dtype uint8
    """
    out = np.asarray(image_rgb, dtype=np.float32).copy()
    for c in range(3):
        channel = out[..., c]
        lo, hi = np.percentile(channel, (pct_lo, pct_hi))
        out[..., c] = np.clip((channel - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    return (out * 255).astype(np.uint8)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    from PIL import Image

    if len(sys.argv) != 2:
        print("Usage: python he_panel_contrast.py <WILL_HE_MBI_OLF_2_Zoom.jpg>")
        print("  Writes the contrast-enhanced H&E of Fig. 4c.")
        sys.exit(1)

    he = crop_bottom_fraction(np.asarray(Image.open(sys.argv[1]).convert("RGB")), 0.07)
    enhanced = enhance_he_panel(he)
    out_path = Path("fig4c_he_preview.png")
    Image.fromarray(enhanced).save(out_path)
    print(f"Wrote {out_path} ({enhanced.shape[1]} x {enhanced.shape[0]} px)")
