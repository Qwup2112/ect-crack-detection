---
name: contrast
description: Contrast enhancement (Vietnamese "bật tông") for ECT maps so cracks stand out. Use when the user asks to boost contrast, make cracks clearer or brighter, or when a scan looks flat, faint or washed out.
---

# Contrast boost — making cracks visible

Enhancement is for **seeing**. Measurements (measure_crack, z-scores, CNR) always use the
unenhanced data, so never quote values read off an enhanced map.

## Methods (enhance_contrast / draw_scan enhance=...)

| method | what it does | use when |
|---|---|---|
| stretch | colour spans the 2nd-98th percentile | default display; a few extreme pixels washed everything out |
| detrend | subtract an 8 px Gaussian blur (the plate background), then stretch | a slope or bright region across the plate hides cracks |
| local | local contrast normalisation (each area scaled by its own mean and spread) | uneven scans, faint cracks next to strong ones |
| gamma | stretch then gamma 0.7 | lift the dim half of the scale a little |
| strong | background removal + local normalisation + 1-99 % stretch | weak scans: low frequency (10 kHz pair), low excitation, early_session |

Start with `detrend`; go to `strong` only if cracks are still faint. Show before/after with
`enhance_contrast` so the user sees the effect, then keep using the chosen method for that scan.

## Sawtooth (Vietnamese "răng cưa") - already removed, the lab's way

Every view except `raw` goes through the lab's anti-sawtooth method (scripts/dataloader.py,
as used in plot_all_1Hall_heatmaps.py):
1. `reshape_raster`: odd scan lines reversed (the scanner runs back and forth);
2. the return lines aligned with the forward lines (probe lag, 1-5 px) and level-matched;
3. `LPF`: Hamming FIR low-pass, order 24, padded 50 samples, group delay removed - applied
   down every column (Y, where the sawtooth lives), then a 3 px median;
4. drawn with bilinear interpolation.
Measured on three scans: steps 1-2 leave 47-57 % of the sawtooth, step 3 leaves 20-30 %.
If the user still sees a comb on a map, check it is not the `raw` view; then draw `filtered`.

## Colour maps

- Amplitude/phase maps: `magma` (default) or `viridis` for raw data; `gray` if the user wants a
  neutral look; `jet` is the lab's own convention (plot_all_1Hall_heatmaps.py) - use it when the
  user wants figures matching the lab's.
- Derivatives (dx, dx_inv) and z-maps: diverging `RdBu_r`, centred on zero. Do not enhance them;
  they are already contrast maps.

## What a crack looks like

An ECT crack signature is a **dipole**: one edge gives a positive lobe, the other a negative one,
and which is which depends on sensor and frequency. The X derivative (`dx`) and its inverse
(`dx_inv`) make the dipole obvious. If an enhanced amplitude map still hides a crack, draw `dx`.

## R versus P

R (amplitude) values are ~0.01, P (phase) ~30. Never compare raw values between them; compare
z_top5 (compare_scans) instead.
