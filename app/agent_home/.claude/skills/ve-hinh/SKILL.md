---
name: ve-hinh
description: Vẽ hình (drawing) - which tool and view to use for a drawing request about ECT scans. Use whenever the user asks to draw, show, plot, display or compare a scan or map.
---

# Vẽ hình — choosing what to draw

One request, one figure, unless the user asks for several. The figure goes to the right panel;
answer with what it shows and the settings used.

| the user says | call |
|---|---|
| "vẽ / hiển thị file X", "bản đồ nhiệt", "heatmap" | draw_scan view=amplitude |
| "ảnh thô", "chưa lọc" | draw_scan view=raw |
| "đã lọc" | draw_scan view=filtered |
| "đạo hàm", "lưỡng cực", "dipole", "cạnh vết nứt" | draw_scan view=dx (or dx_inv) |
| "trừ nền", "bỏ nền" | draw_scan view=detrended |
| "bản đồ z", "độ lệch" | draw_scan view=zmap |
| "6 hình", "báo cáo đầy đủ", "tất cả các bước" | draw_scan view=report |
| "mô hình nhìn thấy gì", "đầu vào YOLO" | draw_scan view=model_input |
| "cắt viền" | crop_border (skill cat-vien) |
| "bật tông", "rõ hơn", "tăng tương phản" | enhance_contrast (skill bat-tong) |
| "vết nứt ở đâu", "nhận diện", "đếm vết nứt" | detect_cracks (skill nhan-dien-vet-nut) |
| "kèm khung nhãn", "so với nhãn" | draw_scan overlay=labels, or detect_cracks |
| "so sánh R và P", "so sánh tần số" | compare_scans (2-8 scans) |
| "lát cắt", "profile", "giá trị dọc theo" | profile_line |
| "đo vết nứt số n" | measure_crack |
| "phân tích file X", "xử lý hết", "kết quả sau cùng" | the full pipeline (skill phan-tich-tu-dong) |
| "histogram", "phổ / FFT", "theo tần số", "thống kê", any plot not listed above | run_python (skill phan-tich-python) |

## Conventions

- Axes are scan position in mm; coordinates refer to the original grid (before cropping).
- Colours clip at percentiles, so extreme pixels do not wash the map out.
- For R vs P of one measurement, use compare_scans with both files: the same crop is not shared,
  but the ranking by z_top5 is fair.
- When the user names a colour map, pass it as cmap and offer to remember it as a preference.
- If no fixed tool can draw what the user asks, draw it with run_python (skill phan-tich-python)
  instead of refusing.
