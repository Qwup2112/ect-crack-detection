# ECT Crack Chat

Ứng dụng chat để phân tích ảnh quét dòng xoáy (ECT) lưu trong file TDMS. Bên trái là khung chat
và câu trả lời, bên phải là hình kết quả. Bạn hỏi hoặc ra lệnh bằng tiếng Việt, ví dụ
"Vẽ bản đồ nhiệt file R 1.7V 300k", "Cắt viền", "Bật tông", "Nhận diện vết nứt", "So sánh R và
P". Trợ lý gọi các công cụ xử lý rồi vẽ kết quả sang khung phải.

Bộ não là **Claude Agent SDK**, tức harness của Claude Code đóng gói thành thư viện. Nó chạy
bằng tài khoản Claude Code đã đăng nhập trên máy, **không cần và không dùng API key**.

## Chạy

Cài một lần (trên ổ D:, trong thư mục dự án):

```bat
python -m venv --system-site-packages .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

Sau đó bấm đúp **`run_app.bat`**. Backend chạy ở cửa sổ thu nhỏ "ECT backend", app mở trong một
cửa sổ Edge không có thanh địa chỉ. Đóng cửa sổ "ECT backend" để tắt app.

Máy cần đã đăng nhập Claude Code (lệnh `claude` chạy được).

## Trợ lý biết làm gì

| Công cụ | Việc |
|---|---|
| `list_scans`, `scan_info` | Tra danh mục 110 file (101 file đọc được), thông số, cảnh báo |
| `draw_scan` | Vẽ: thô, đã lọc, đã cắt viền, bản đồ biên độ/pha, trừ nền, đạo hàm dA/dx, bản đồ z, báo cáo 6 hình, ảnh đầu vào YOLO |
| `crop_border` | Cắt viền tự động (đo riêng từng cạnh, thường không đối xứng) hoặc cố định, hiện vùng bị cắt |
| `enhance_contrast` | Bật tông: kéo giãn, trừ nền, chuẩn hóa cục bộ, gamma, mạnh |
| `detect_cracks` | Nhận diện vết nứt bằng YOLO (tự chuyển sang bộ dò ngưỡng nếu không chạy được), so với nhãn |
| `measure_crack` | Đo chiều dài, chiều rộng, góc, SNR của một vết nứt |
| `compare_scans` | So sánh 2–8 file, xếp hạng theo z_top5 |
| `profile_line` | Lát cắt 1D theo X hoặc Y |
| `remember`, `recall`, `forget` | Bộ nhớ dài hạn (sở thích, điều đã học) |

Hình không bị răng cưa nhờ phương pháp của phòng lab (`scripts/dataloader.py`):
đảo dòng lẻ của quét zig-zag (`reshape_raster`), căn lệch chiều đi/về, lọc thông thấp `LPF`
(FIR Hamming bậc 24) dọc mỗi cột, rồi hiển thị nội suy bilinear. Đo trên 3 file, phần răng cưa
còn lại 20–30% so với chỉ đảo dòng.

## Harness: memory, tools, hooks, skills

- **Memory**: `app/agent_home/CLAUDE.md` (kiến thức cố định về bộ dữ liệu) +
  `app/state/memory.json` (điều trợ lý học khi chat, qua `remember`) + phiên chat hiện tại.
- **Tools**: máy chủ MCP trong tiến trình (`app/backend/agent/tools.py`), chỉ các tool trên;
  Bash, đọc/ghi file… bị tắt.
- **Hooks** (`app/backend/agent/hooks.py`): PreToolUse chặn tool lạ và đổi tên file gõ tắt
  thành id chuẩn (hoặc từ chối kèm danh sách nếu mơ hồ); PostToolUse báo trạng thái lên giao
  diện và ghi nhật ký; UserPromptSubmit nạp bộ nhớ vào mỗi câu hỏi.
- **Skills** (`app/agent_home/.claude/skills/`): dạy trợ lý cách **cắt viền** (`cat-vien`),
  **bật tông** (`bat-tong`), **nhận diện vết nứt** (`nhan-dien-vet-nut`), **chọn hình để vẽ**
  (`ve-hinh`).

Khi phát triển bằng Claude Code: `CLAUDE.md` ở gốc dự án, `.claude/settings.json` chặn ghi vào
`data/` và `results/labels/`, và một hook chạy smoke test sau mỗi lần sửa `app/backend/ect/`.

## Giao diện

`app/frontend/` là giao diện tạm. Prompt để Google AI Studio thiết kế giao diện thật nằm trong
[`docs/google_ai_studio_prompt.md`](docs/google_ai_studio_prompt.md). Lưu file `index.html`
nhận được vào `app/frontend/` là chạy, không cần sửa backend.

## Lưu ý về dữ liệu

- Nhãn trong `results/labels/` là **bản nháp**, chưa người duyệt. Mọi điểm số so với nhãn là tạm.
- Mô hình YOLO chỉ học trên tấm chuẩn 15 vết nứt (lưới 3 × 5); vẫn báo sai trên ảnh nhiễu.
- 1 file rỗng, 10 file bị cắt cụt, 18 file theo quy ước tên cũ: công cụ đọc được và có cảnh báo.

## Cấu trúc

```
data/                 file TDMS gốc (chỉ đọc)
results/labels/       nhãn nháp;  results/model/  trọng số YOLO
scripts/              code của phòng lab (dataloader.py, plot_all_1Hall_heatmaps.py)
app/backend/ect/      xử lý: đọc, lọc, cắt viền, bật tông, nhận diện, vẽ
app/backend/agent/    trợ lý: tools, hooks, memory, phiên chat, system prompt
app/agent_home/       CLAUDE.md + skills của trợ lý
app/backend/server.py backend FastAPI (127.0.0.1:8765)
app/frontend/         giao diện tạm
app/state/            ảnh, bộ nhớ, nhật ký khi chạy (không đưa vào git)
docs/                 prompt Google AI Studio
```

Phần mềm cũ (app Tkinter) lấy lại được bằng `git checkout legacy-app`.
