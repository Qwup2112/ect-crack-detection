# Project memory of the ECT chat agent

Fixed knowledge, loaded by the harness at the start of every session. What the agent learns
while chatting goes to the long-term memory (remember / recall tools), not here.

## The specimen and the scans

- A calibration coupon with **15 artificial cracks in a 3 x 5 grid** at different depths. All
  sessions scan the same coupon, at different plate positions.
- 2D raster scans in TDMS files: X is the fast scan axis, Y the slow one. The scanner runs back
  and forth (serpentine); odd lines are un-mirrored and the return-line lag (cam_4lop_1: 2 px) is
  corrected when a file is read.
- Sensors: **R = amplitude**, **P = phase** (values ~0.01 vs ~30 scale), plus PECT/Differential/
  Feedback templates. `_default` files are empty templates.
- Filenames carry the parameters: `4lop_R_amp_1.7V_fre_300k_lf_1mm_155x100mm` = sensor R,
  excitation 1.7 V, 300 kHz, lift-off 1 mm, plate area 155 x 100 mm. Early files use `freq_..K`,
  `liffoff`, and `_tron` (a different defect shape).
- Sessions in data/: cam_4lop_1, cambien_4lop_2, early_session (101 readable files).

## Known data faults

- `early_session/R_amp_5V_freq_100K_liffoff_1mm` is empty (never recorded).
- 10 files are truncated (worst: the P/R 5 V 160 kHz pair, 22 rows lost). They are analysed on
  their complete rows; tools report it as a warning.
- Files without a plate size in the name use an assumed pitch of 1 mm per point.

## Best configurations (z_top5, default pipeline, interior candidates)

| session | sensor | configuration | z_top5 |
|---|---|---|---|
| cambien_4lop_2 | R | 1.7 V, 300 kHz | 10.35 |
| cambien_4lop_2 | P | 0.3 V, 20 kHz | 9.17 |
| cam_4lop_1 | R | 4 V, 250 kHz | 8.27 |
| cam_4lop_1 | P | 4 V, 40 kHz | 7.21 |
| early_session | R | 4 V, 60 kHz | 3.67 |
| early_session | P | 3 V, 30 kHz | 2.85 |

## The YOLO model

- YOLO11n trained on the 56 labelled scans (cam_4lop_1 + cambien_4lop_2). Held-out session
  scores against the DRAFT labels: recall 1.00 on both folds, precision 0.97-0.99; the threshold
  detector reaches recall 0.30-0.35.
- It still fires on pure noise (17 boxes per noise image at conf 0.5): every training image had
  cracks. Boxes on the scan border are dropped as plate edge.
- It has only seen this coupon: results do not transfer to other specimens without new labels.
- It runs on onnxruntime (results/model/crack_yolo.onnx, identical boxes to the PyTorch model on
  all 101 readable scans), which needs little memory. If it still cannot run, the tools fall back
  to the threshold detector - say so when it happens.

## Your tools beyond the fixed ones

- run_python runs your own Python (numpy, scipy, matplotlib, the app's ect modules) in a
  separate guarded process for analysis the fixed tools cannot do. It can write only in the
  chat's work folder under app/state/work/; data/ and results/labels/ are read-only.
