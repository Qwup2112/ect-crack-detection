---
name: crack-detection
description: Crack detection (Vietnamese "nhận diện vết nứt") in ECT scans and how to interpret and report the result. Use whenever the user asks where the cracks are, how many, to detect, count, score, measure or compare cracks.
---

# Crack detection — detecting and reporting cracks

## Procedure

1. Pick the scan (the PreToolUse hook resolves names; if it lists several, ask).
2. The border is handled for you: detect_cracks uses the auto crop unless the chat chose
   another one, so do not call crop_border first just for detection. If edge boxes appear,
   look at the crop (skill border-crop) and re-run.
3. `detect_cracks` with method `yolo` (default; runs on onnxruntime in about a second). If YOLO
   cannot run, the tool falls back to the threshold detector and says so - pass that on.
4. Report: the number of cracks, then each as `#n: X, Y (mm), box size, confidence`. Keep the
   numbering; the user will refer to "#3".
5. If the scan has labels, report found / false / missed **and** that the labels are a draft.
   If the scan was in YOLO's training set, add that its score is optimistic.
6. Check the result yourself before answering: 15 expected (3 x 5); if the count is far off
   (< 12 or > 18), also run `method=threshold` and compare. If the user asked about size, run
   `measure_crack` yourself. For a full analysis follow skill auto-analysis.
7. End with at most one optional next step as an offer (measure a crack, the dipole view
   `draw_scan dx`), not a question the user must answer first.

## Interpreting

- The coupon has **15 cracks in a 3 x 5 grid**. Fewer than 15 on a full-coupon scan usually means
  faint cracks (deep row, low frequency) or a crop cutting them; more means noise or the plate
  edge. Say which you think it is and why, from the figure.
- YOLO found 15/15 on every scan of the two labelled sessions; on early_session (a placement it
  never saw) it finds 3-13. It fires on pure noise, so a lone low-confidence box (< 0.6) away
  from the grid is suspect.
- The threshold detector only says "deviates from the background". Its boxes are candidates,
  never ground truth. Its z_top5 ranks scans fairly across sensors.
- A crack is a dipole; a box covering both lobes is right.
- `measure_crack`: length is reliable when the signature is clear (SNR >= 3); width includes the
  probe footprint (~6 mm), so the real crack is far narrower; depth cannot be estimated without a
  calibration. If the signature spills out of its box, the length is a lower bound.

## Never

- Never invent or adjust crack positions, counts or scores.
- Never present threshold boxes or draft labels as ground truth.
- Never edit labels: they come from a person.
