---
name: cat-vien
description: Cắt viền (border cropping) of ECT scans. Use before cropping, when the user asks to trim or remove noisy edges, or when edge artifacts appear in a map or detection.
---

# Cắt viền — removing the noisy scan border

## Why the border is noisy

Two different faults, both at the edges of a scan:
- **Level**: rows or columns whose median is offset, e.g. the bright band where each scan line
  starts.
- **Spread**: streaks where the probe turns around. Their median looks normal but their scatter is
  large, so measuring the level alone misses them.

On top of that, the plate edge itself appears inside some scans (early_session), and it looks
like a crack to any detector.

## How the auto crop works (crop_border mode=auto)

1. Remove the per-row trend, then compute a robust z-score (median/MAD) of every row's and
   column's level and spread.
2. Walk inward from each edge while either z stays above 3, and stop at the first normal line.
   Only the contiguous border band is cut; nothing in the middle is touched.
3. Each side is capped at 15 % of the image. If almost everything looks abnormal, it falls back
   to a fixed 5 px.

The result is **asymmetric** and that is correct. Measured examples: cambien_4lop_2 R 1.7 V 300
kHz gives top 10, bottom 6, left 5, right 12 px; the old R 1.3 V 200 kHz gave top 14, bottom 3. A
symmetric 5 px crop leaves noise on one side and throws away good data on the other.

## Procedure

1. Call `crop_border` with mode `auto` and let the user see the shaded bands on the right.
2. Report the four margins in px and mm, and the remaining grid size.
3. Use `fixed` only when the user asks for it or when auto clearly fails (cuts into the cracks, or
   leaves a bright band). Lower `z_thresh` (e.g. 2.5) crops more; raise it (e.g. 4) crops less.
4. The choice is kept for this scan in the chat; every later figure of that scan uses it.

## Cautions

- A crack near the plate edge also touches the border: if a detection box sits on the crop line,
  say it may be a real crack or the edge.
- Coordinates from the tools are always in mm of the original grid, before the crop, so they stay
  comparable whatever crop is used.
- Truncated files lost rows at the bottom; do not mistake that for a crop.
