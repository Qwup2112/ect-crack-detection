---
name: drawing
description: Drawing - which tool and view to use for a drawing request about ECT scans. Use whenever the user asks to draw, show, plot, display or compare a scan or map (Vietnamese "vẽ").
---

# Drawing — choosing what to draw

One request, one figure, unless the user asks for several. The figure goes to the right panel;
answer with what it shows and the settings used.

| the user says | call |
|---|---|
| "draw / show file X", "heatmap" ("vẽ", "bản đồ nhiệt") | draw_scan view=amplitude |
| "raw", "unfiltered" ("ảnh thô", "chưa lọc") | draw_scan view=raw |
| "filtered" ("đã lọc") | draw_scan view=filtered |
| "derivative", "dipole", "crack edges" ("đạo hàm", "lưỡng cực") | draw_scan view=dx (or dx_inv) |
| "remove the background", "detrended" ("trừ nền") | draw_scan view=detrended |
| "z-map", "deviation" ("bản đồ z", "độ lệch") | draw_scan view=zmap |
| "6 figures", "full report", "every step" ("báo cáo đầy đủ") | draw_scan view=report |
| "what does the model see", "YOLO input" | draw_scan view=model_input |
| "crop the border" ("cắt viền") | crop_border (skill border-crop) |
| "boost contrast", "make it clearer" ("bật tông", "rõ hơn") | enhance_contrast (skill contrast) |
| "where are the cracks", "detect", "count cracks" ("nhận diện", "vết nứt") | detect_cracks (skill crack-detection) |
| "with label boxes", "against the labels" ("so với nhãn") | draw_scan overlay=labels, or detect_cracks |
| "compare R and P", "compare frequencies" ("so sánh") | compare_scans (2-8 scans) |
| "profile", "cut", "values along" ("lát cắt") | profile_line |
| "measure crack #n" ("đo vết nứt") | measure_crack |
| "analyse file X", "process everything", "final result" ("phân tích", "kết quả sau cùng") | the full pipeline (skill auto-analysis) |
| "histogram", "spectrum / FFT", "versus frequency", "statistics", any plot not listed above | run_python (skill python-analysis) |

## Conventions

- Axes are scan position in mm; coordinates refer to the original grid (before cropping).
- Colours clip at percentiles, so extreme pixels do not wash the map out.
- For R vs P of one measurement, use compare_scans with both files: the same crop is not shared,
  but the ranking by z_top5 is fair.
- When the user names a colour map, pass it as cmap and offer to remember it as a preference.
- If no fixed tool can draw what the user asks, draw it with run_python (skill python-analysis)
  instead of refusing.
