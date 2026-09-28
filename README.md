# Code for: Continuous-wave mid-infrared laser ablation enables high-resolution ambient mass spectrometry imaging

William J. C. Francis, Milena Micic, Lucy Noyes, David Blair, Anna Chauvet, Yayue Song,
Lauren Ford, Stefania Maneta-Stavrakaki, Dániel Simon, Ioannis Bitharas, Zoltan Takáts,
Robert T. Murray

This repository contains the custom code behind specific numbers and image panels in the
manuscript (submitted to *Nature Communications*). The data are in Zenodo at
https://doi.org/TK.

**Scope.** The repository holds only the calculations whose results cannot be read directly
from the deposited data: the Fig. 4 image processing, the beam-diameter and
line-spread-function fits, and the conversion that produced the deposited imzML files. It is
not a one-command figure-reproduction pipeline. Figure assembly, layout, labelling and
styling code is not included. Two kinds of file are provided:

- **Calculation functions** (`he_panel_contrast.py`, `he_overlay_coregistration.py`,
  `he_overlay_render.py`, `knife_edge_beam_diameter.py`, `line_spread_resolution.py`). NumPy
  arrays in, NumPy arrays or values out, with no file I/O or plotting inside the functions
  (the HDImaging reader in `he_overlay_render.py` and the landmark reader in
  `he_overlay_coregistration.py` excepted). A demo at the bottom of each file, under
  `if __name__ == "__main__":`, runs it on the deposited data.
- **Data-conversion utility** (`hdi_txt_to_imzml.py`). A command-line tool whose job is file
  I/O, reading one format and writing another.

Figure numbers below refer to the submitted manuscript.

## Getting the data

Download the Zenodo record (https://doi.org/TK) and unzip each section into one folder, e.g.
`cw_msi_data/`, giving `cw_msi_data/01_MSI_HDI_processed_data/`,
`cw_msi_data/05_beam_profile_knife_edge/`, `cw_msi_data/06_line_spread_function/`,
`cw_msi_data/09_Fig4_HE_exports_and_landmarks/`, and so on. The paths below refer to that layout.

## Installation

```
pip install -r requirements.txt
```

Tested with Python 3.12.1, NumPy 2.4.6, SciPy 1.16.3, Matplotlib 3.11.1, Pillow 10.2.0 and
pyimzML 1.5.5 (Windows 10).

## Files

### `he_overlay_coregistration.py` (Fig. 4a,b,d–f)
```
load_bigwarp_landmarks(path) -> (moving_points, target_points)
fit_coregistration(moving_points, target_points) -> {"tps_x", "tps_y"}
calibrate_pixel_pitch(moving_points, target_points, moving_pitch_um) -> (pitch_um, mad_um)
```
Fits the thin-plate-spline transform, from BigWarp landmark pairs, that maps H&E (target)
pixel coordinates to ion-image (moving) pixel coordinates. `calibrate_pixel_pitch` gives the
H&E pixel pitch as the median over landmark pairs of the ion-image distance times the 3 µm
raster step, divided by the H&E distance. For `landmarksZoom.csv` this is the 0.668 µm pitch
reported in the Methods.
```
python he_overlay_coregistration.py cw_msi_data/09_Fig4_HE_exports_and_landmarks/bigwarp_landmarks/landmarksZoom.csv
```

### `he_overlay_render.py` (Fig. 4a,b,d–f)
```
read_hdi_ion_images(path, target_mz, mz_tol=0.002) -> {mz: raster}
ion_image_rgba(raster, saturation_pct=99.5, rot90_k=1) -> rgba_uint8
warp_ion_channel(ion_image, tps_transform, output_origin_xy, output_shape,
                 intensity_pct_clip=99, intensity_transform="sqrt") -> normalised
render_ion_overlay(ion_image, tps_transform, output_origin_xy, output_shape,
                   intensity_pct_clip=99, signal_floor=0.12, alpha_max=0.85,
                   colormap="viridis", intensity_transform="sqrt") -> rgba
he_greyscale_backdrop(image_rgb, lighten=0.45) -> grey_uint8
composite_over_backdrop(grey_backdrop, overlay) -> rgb
tissue_bbox(he_rgb, threshold=215, margin_frac=0.04) -> (x0, y0, height, width)
render_rgb_composite(red, green, blue) -> rgb
region_outline(region_shape, region_transform, whole_inverse_transform, crop_origin_xy) -> corners
```
The Fig. 4 ion images are 8-bit greyscale renders of the 3 µm olfactory-bulb export
(`01_MSI_HDI_processed_data/3um_olfactory_bulb/`). `read_hdi_ion_images` rebuilds each
channel's raster from the export, and `ion_image_rgba` renders it as used for the figure:
linear from 0 to the 99.5th percentile of the non-zero pixels, clipped above, and rotated a
quarter turn anticlockwise into HDImaging's display orientation. The landmarks in
`09_Fig4_HE_exports_and_landmarks/bigwarp_landmarks/` were placed in this orientation.

Each ion image is then normalised to the 99th percentile of its non-zero pixels and resampled
into the H&E frame by bilinear interpolation through the co-registration transform.

- **RGB composites (Fig. 4a,b).** Linear normalisation. The three channels (R, PE 40:6,
  *m/z* 790.5367; G, adenine, *m/z* 134.0465; B, PC 36:1 as [M−CH3]−, *m/z* 772.5833) are
  used directly as the red, green and blue values, on black. Fig. 4a is warped into the
  tissue bounding box of `OLF_Ablated.jpg` with `landmarksCW_OLF.csv`, and Fig. 4b into
  `WILL_HE_MBI_OLF_2_Zoom.jpg` with `landmarksZoom.csv`. `region_outline` maps the footprint
  of the zoom export onto Fig. 4a through both transforms, giving the boxed region.
- **Single-ion overlays (Fig. 4d–f).** Square-root transform in the ion image's own grid
  before normalisation. Pixels below 0.12 of the normalised range are transparent. Above it,
  the values are rescaled to 0–1 and drawn in viridis with opacity rising to 0.85. The
  backdrop is the luminance of the contrast-stretched H&E of Fig. 4c, lightened by a uniform
  45% blend toward white.

The demo reads the 3 µm export (about a minute) and writes Fig. 4a, 4b and 4d–f to
`./fig4_panels/`, and prints the corners of the Fig. 4a box:
```
python he_overlay_render.py cw_msi_data
```

### `he_panel_contrast.py` (Fig. 4c)
```
crop_bottom_fraction(image, frac=0.07) -> image
enhance_he_panel(image_rgb, pct_lo=1, pct_hi=99) -> enhanced_rgb_uint8
```
The bottom 7% of the NDP.view export, which carries a burned-in scale bar, is cropped. Each
RGB channel is then stretched linearly so that its 1st percentile maps to 0 and its 99th
percentile to 255, with no other per-pixel operation. The unprocessed export is
Supplementary Fig. 6.
```
python he_panel_contrast.py cw_msi_data/09_Fig4_HE_exports_and_landmarks/HE_exports/WILL_HE_MBI_OLF_2_Zoom.jpg
```

### `knife_edge_beam_diameter.py` (Supplementary Fig. 2)
```
fit_knife_edge(position_mm, transmitted_power_w) -> (x0_mm, w_mm, beam_diameter_um, diameter_sigma_um)
```
Fits the knife-edge power-versus-position scan to an error-function edge and returns the
1/e² diameter of the focused CW beam, reported as 10.5 ± 0.1 µm.
```
python knife_edge_beam_diameter.py cw_msi_data/05_beam_profile_knife_edge/knife_edge_measurements_20251219_161447.txt
```

### `line_spread_resolution.py` (Supplementary Fig. 7)
```
fit_psf_resolution(distance_um, intensity) -> (popt, perr, resolution_16_84_um)
```
Fits an intensity profile across a tissue step edge to an error-function edge model (an ideal
step blurred by a Gaussian point-spread function of standard deviation σ) and returns the
16–84% rise distance, 2√2·σ·erf⁻¹(0.68). The ten deposited profiles are
`06_line_spread_function/line_01.csv` to `line_10.csv`. Their mean ± s.d. is the reported
10.3 ± 1.1 µm (n = 10), and `line_fit_summary.csv` lists the value for each line.
```
python line_spread_resolution.py cw_msi_data/06_line_spread_function/line_01.csv
```

### `hdi_txt_to_imzml.py` (`02_MSI_IMZML_data/`)
```
python hdi_txt_to_imzml.py INPUT.txt OUTPUT_STEM --width W --pixel-size P --polarity {positive,negative}
```
Converts a Waters HDImaging (Maldichrom) peak-picked pixel `.txt` export into a centroid,
continuous-mode imzML + `.ibd` pair. The centroided imzML files in the data record were made
this way. HDImaging's own imzML export re-reads the raw file and writes full, uncentroided
profile spectra (tens to 100+ GB), so this script works from the much smaller peak-picked
export instead.

**The script performs no centroiding, peak-picking or binning of its own, and applies no
smoothing, filtering, resampling or intensity transform.** Peak picking was done upstream in
HDImaging (Methods: 0.02 Da *m/z* window, mass-resolution setting 20,000), and the export
already holds one shared peak list with each pixel's intensity at each of those *m/z* values.
The script writes that peak list and the per-pixel intensities to imzML/ibd unchanged. Its
only computation is each pixel's (x, y) raster position, reconstructed from the acquisition
order.

`--width` is always required. It is cross-checked against the raster width detected from the
stage x column, which resets at the start of each scan line, and a mismatch prints a warning.
Optional flags cover other raster geometries (scan direction, line-scan direction, scan
pattern, scan type). The module docstring describes the `.txt` format, including the two
trailing columns that look like pixel coordinates but are MassLynx bookkeeping. The deposited
files used:

| Dataset | `--width` | `--pixel-size` |
|---|---|---|
| 6 µm hippocampus (CW) | 1265 | 6 |
| 3 µm olfactory bulb (CW) | 1033 | 3 |
| 25 µm hippocampus (ns-pulsed) | 296 | 25 |

all with `--polarity negative`.

## Dependencies

NumPy and SciPy throughout. `he_overlay_render.py` also uses Matplotlib, for colour-map
lookup only. The demos use Pillow to read and write images. `hdi_txt_to_imzml.py` requires
pyimzML. See `requirements.txt`.

## Licence

MIT (see `LICENSE`).

## Citation

Please cite the associated article (see `CITATION.cff`). Software DOI: https://doi.org/TK.
