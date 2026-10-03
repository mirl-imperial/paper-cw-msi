"""Convert a Waters HDImaging peak-picked pixel export (.txt) to centroid imzML.

HDImaging's own imzML export writes full profile spectra. The tab-delimited
.txt export from the Process tab instead holds one shared m/z peak list and
each pixel's intensity at those peaks; this script writes it to a
continuous-mode centroid imzML + .ibd pair without changing any values.

.txt format (HDImaging v1.4):
    line 1    title or blank (unused)
    line 2    summed spectrum (unused)
    line 3    peak indices 1..N
    line 4    the N peak m/z values
    line 5+   one row per pixel: pixel index, stage x (mm), stage y (mm),
              N intensities, then two MassLynx function/scan columns, which
              are not pixel coordinates.

Pixel positions come from the acquisition order and the raster geometry
(imzML scan-settings conventions; defaults are HDImaging's top-down, one-way,
left-to-right horizontal raster). --width is required; it is checked against
the stage x column, which resets at the start of each scan line.

    python hdi_txt_to_imzml.py INPUT.txt OUTPUT_STEM \\
        --width 1033 --pixel-size 3 --polarity negative
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
from pyimzml.ImzMLWriter import ImzMLWriter


def pixel_index_to_xy(i0, width, height, scan_direction, line_scan_direction,
                       scan_pattern, scan_type):
    """i0: 0-based pixel index in acquisition order. Returns 1-based (x, y)."""
    if scan_type == "horizontal_line":
        line_len, n_lines = width, height
    elif scan_type == "vertical_line":
        line_len, n_lines = height, width
    else:
        raise ValueError(f"unknown scan_type: {scan_type}")

    line_no = i0 // line_len
    pos_in_line = i0 % line_len

    base_reversed = line_scan_direction in ("line_right_left", "line_bottom_up")
    if scan_pattern == "meandering" and line_no % 2 == 1:
        reversed_this_line = not base_reversed
    else:
        reversed_this_line = base_reversed
    pos = (line_len - 1 - pos_in_line) if reversed_this_line else pos_in_line

    line_reversed = scan_direction == "bottom_up"
    line = (n_lines - 1 - line_no) if line_reversed else line_no

    if scan_type == "horizontal_line":
        x, y = pos, line
    else:
        x, y = line, pos
    return x + 1, y + 1


def _count_and_detect_width(f):
    """Return (n_pixels, width), with width taken from the stage x resets.

    width is None if the resets are not evenly spaced.
    """
    n_pixels = 0
    prev_t = None
    resets = []
    for line in f:
        if not line.strip():
            continue
        t = float(line.split("\t", 3)[1])
        if prev_t is not None and t < prev_t - 1e-9:
            resets.append(n_pixels)
        prev_t = t
        n_pixels += 1
    if len(resets) < 2:
        return n_pixels, None
    gaps = {resets[i + 1] - resets[i] for i in range(len(resets) - 1)}
    if len(gaps) != 1:
        return n_pixels, None
    return n_pixels, resets[0]


def convert(txt_path, out_path, width, pixel_size, height=None,
            pixel_size_y=None, polarity="negative", scan_direction="top_down",
            line_scan_direction="line_left_right", scan_pattern="one_way",
            scan_type="horizontal_line", limit=None):
    t0 = time.time()

    def open_data_rows():
        """Open the export past its header; return (file, n_peaks, mzs)."""
        f = open(txt_path, "r", encoding="utf-8", errors="strict")
        f.readline()
        f.readline()
        row_idx = f.readline().rstrip("\n").split("\t")
        row_mz = f.readline().rstrip("\n").split("\t")
        n_peaks = len([c for c in row_idx[3:] if c != ""])
        mzs = np.array([float(v) for v in row_mz[3:3 + n_peaks]], dtype=np.float64)
        assert len(mzs) == n_peaks
        return f, n_peaks, mzs

    f, n_peaks, mzs = open_data_rows()
    print(f"n_peaks={n_peaks}  mz range={mzs.min():.4f}-{mzs.max():.4f}")

    if limit is not None:
        n_pixels = limit
        for _ in range(limit):
            f.readline()
    else:
        n_pixels, detected_width = _count_and_detect_width(f)
        if detected_width is not None and detected_width != width:
            print(
                f"WARNING: --width {width} does not match the raster width "
                f"{detected_width} detected from the stage x column."
            )
    f.close()

    if height is None:
        assert n_pixels % width == 0, (
            f"{n_pixels} pixels is not an exact multiple of --width {width}; "
            "pass the correct raster width (or --height explicitly)."
        )
        height = n_pixels // width
    else:
        assert width * height == n_pixels, (
            f"width*height ({width}*{height}={width*height}) != pixel count "
            f"({n_pixels}); check --width/--height."
        )
    print(f"n_pixels={n_pixels}  width={width}  height={height}")

    pixel_size_y = pixel_size if pixel_size_y is None else pixel_size_y

    max_x = max_y = 0
    f, _, _ = open_data_rows()
    with f, ImzMLWriter(
        out_path,
        mode="continuous",
        spec_type="centroid",
        polarity=polarity,
        scan_direction=scan_direction,
        line_scan_direction=line_scan_direction,
        scan_pattern=scan_pattern,
        scan_type=scan_type,
    ) as writer:
        i0 = 0
        for line in f:
            if not line.strip():
                continue
            if i0 >= n_pixels:
                break
            cols = line.rstrip("\n").split("\t")
            intens = np.array(cols[3:3 + n_peaks], dtype=np.float64)
            x, y = pixel_index_to_xy(
                i0, width, height, scan_direction, line_scan_direction,
                scan_pattern, scan_type,
            )
            max_x, max_y = max(max_x, x), max(max_y, y)
            writer.addSpectrum(mzs, intens, (x, y, 1))
            i0 += 1
            if i0 % 200000 == 0:
                print(f"  {i0} pixels written ({time.time() - t0:.0f}s)")

    print(f"Done: {n_pixels} pixels, max_x={max_x}, max_y={max_y}, "
          f"{time.time() - t0:.0f}s")
    _inject_pixel_size(out_path, pixel_size, pixel_size_y)


def _inject_pixel_size(out_path, pixel_size_x, pixel_size_y):
    # pyimzML does not write the pixel size (IMS:1000046/47); add it to scanSettings.
    imzml_path = out_path if out_path.endswith(".imzML") else out_path + ".imzML"
    with open(imzml_path, "r", encoding="ISO-8859-1") as f:
        xml = f.read()
    marker = 'name="max count of pixels y"'
    idx = xml.index(marker)
    line_end = xml.index("\n", idx)
    insertion = (
        f'\n      <cvParam cvRef="IMS" accession="IMS:1000046" name="pixel size (x)" '
        f'value="{pixel_size_x}" unitCvRef="UO" unitAccession="UO:0000015" unitName="micrometer"/>'
        f'\n      <cvParam cvRef="IMS" accession="IMS:1000047" name="pixel size y" '
        f'value="{pixel_size_y}" unitCvRef="UO" unitAccession="UO:0000015" unitName="micrometer"/>'
    )
    xml = xml[:line_end] + insertion + xml[line_end:]
    with open(imzml_path, "w", encoding="ISO-8859-1") as f:
        f.write(xml)
    print(f"Injected pixel size {pixel_size_x} x {pixel_size_y} um into {imzml_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input_txt", help="HDImaging .txt pixel export")
    p.add_argument("output_stem", help="output path without extension")
    p.add_argument("--width", type=int, required=True, help="raster width in pixels")
    p.add_argument("--height", type=int, default=None,
                   help="raster height in pixels (default: pixel count / width)")
    p.add_argument("--pixel-size", type=float, required=True,
                   help="pixel size in micrometres (x)")
    p.add_argument("--pixel-size-y", type=float, default=None,
                   help="pixel size in micrometres (y), default: --pixel-size")
    p.add_argument("--polarity", choices=["positive", "negative"], required=True)
    p.add_argument("--scan-direction", choices=["top_down", "bottom_up"],
                   default="top_down")
    p.add_argument("--line-scan-direction",
                   choices=["line_left_right", "line_right_left",
                            "line_top_down", "line_bottom_up"],
                   default="line_left_right")
    p.add_argument("--scan-pattern", choices=["one_way", "meandering"],
                   default="one_way")
    p.add_argument("--scan-type", choices=["horizontal_line", "vertical_line"],
                   default="horizontal_line")
    p.add_argument("--limit", type=int, default=None,
                   help="convert only the first N pixels")
    args = p.parse_args()

    convert(args.input_txt, args.output_stem, args.width, args.pixel_size,
            height=args.height, pixel_size_y=args.pixel_size_y,
            polarity=args.polarity, scan_direction=args.scan_direction,
            line_scan_direction=args.line_scan_direction,
            scan_pattern=args.scan_pattern, scan_type=args.scan_type,
            limit=args.limit)


if __name__ == "__main__":
    sys.exit(main())
