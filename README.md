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

Sau đó bấm đúp **`ECT Crack Chat`** (shortcut có icon trong thư mục dự án) hoặc `run_app.bat`.
Đây là một phần mềm có cửa sổ riêng (WebView2 có sẵn trong Windows 11), không mở trình duyệt,
không có đường link. Bộ xử lý chạy ngầm bên trong chương trình; đóng cửa sổ là tắt hết.
Muốn đặt shortcut ra Desktop:
`powershell -ExecutionPolicy Bypass -File app\assets\make_shortcut.ps1 -Dest "$env:USERPROFILE\Desktop"`.

Máy cần đã đăng nhập Claude Code (lệnh `claude` chạy được). Nhật ký: `app/state/desktop.log`.

## Trợ lý biết làm gì

| Công cụ | Việc |
|---|---|
| `list_scans`, `scan_info` | Tra danh mục 110 file (101 file đọc được), thông số, cảnh báo |
| `draw_scan` | Vẽ: thô, đã lọc, đã cắt viền, bản đồ biên độ/pha, trừ nền, đạo hàm dA/dx, bản đồ z, báo cáo 6 hình, ảnh đầu vào YOLO |
| `crop_border` | Cắt viền tự động (đo riêng từng cạnh, thường không đối xứng) hoặc cố định, hiện vùng bị cắt |
| `enhance_contrast` | Bật tông: kéo giãn, trừ nền, chuẩn hóa cục bộ, gamma, mạnh |
| `detect_cracks` | Nhận diện vết nứt bằng YOLO (chạy bằng ONNX, ~0,4 s, nhẹ bộ nhớ; tự chuyển sang bộ dò ngưỡng nếu không chạy được), so với nhãn |
| `measure_crack` | Đo chiều dài, chiều rộng, góc, SNR của một vết nứt |
| `compare_scans` | So sánh 2–8 file, xếp hạng theo z_top5 |
| `profile_line` | Lát cắt 1D theo X hoặc Y |
| `run_python` | **Phân tích sâu**: trợ lý tự viết và chạy Python (thống kê nhiều file, xu hướng theo tần số, FFT, histogram, đồ thị tùy ý) |
| `remember`, `recall`, `forget` | Bộ nhớ dài hạn (sở thích, điều đã học) |

Trợ lý **tự làm trọn việc**: "phân tích file X", "detect vết nứt", "gửi kết quả sau cùng" thì nó
tự cắt viền, bật tông, nhận diện, đo và báo cáo trong một lần trả lời, không dừng lại hỏi từng bước.

**Quyền của `run_python`:** mã chạy trong một tiến trình riêng (tối đa 180 s), chỉ được ghi file
vào `app/state/work/<cuộc chat>/`; `data/` và `results/labels/` chỉ đọc; không chạy chương trình
khác, không mạng, không ghi registry. Mọi đoạn mã đều được ghi lại trong
`app/state/logs/tools.jsonl`. Muốn tắt công cụ này: đặt biến môi trường `ECT_PYTHON_TOOL=0`.

Hình không bị răng cưa nhờ phương pháp của phòng lab (`scripts/dataloader.py`):
đảo dòng lẻ của quét zig-zag (`reshape_raster`), căn lệch chiều đi/về, lọc thông thấp `LPF`
(FIR Hamming bậc 24) dọc mỗi cột, rồi hiển thị nội suy bilinear. Đo trên 3 file, phần răng cưa
còn lại 20–30% so với chỉ đảo dòng.

## Harness: memory, tools, hooks, skills

- **Memory**: `app/agent_home/CLAUDE.md` (kiến thức cố định về bộ dữ liệu) +
  `app/state/memory.json` (điều trợ lý học khi chat, qua `remember`) + phiên chat hiện tại.
- **Tools**: máy chủ MCP trong tiến trình (`app/backend/agent/tools.py`), chỉ các tool trên;
  Bash, đọc/ghi file… của Claude Code bị tắt. Phân tích tự do đi qua `run_python` (có rào chắn).
- **Hooks** (`app/backend/agent/hooks.py`): PreToolUse chặn tool lạ và đổi tên file gõ tắt
  thành id chuẩn (hoặc từ chối kèm danh sách nếu mơ hồ); PostToolUse báo trạng thái lên giao
  diện và ghi nhật ký; UserPromptSubmit nạp bộ nhớ vào mỗi câu hỏi.
- **Skills** (`app/agent_home/.claude/skills/`): dạy trợ lý **phân tích tự động trọn quy trình**
  (`phan-tich-tu-dong`), **cắt viền** (`cat-vien`), **bật tông** (`bat-tong`), **nhận diện vết
  nứt** (`nhan-dien-vet-nut`), **chọn hình để vẽ** (`ve-hinh`), **phân tích sâu bằng Python**
  (`phan-tich-python`).

Khi phát triển bằng Claude Code: `CLAUDE.md` ở gốc dự án, `.claude/settings.json` chặn ghi vào
`data/` và `results/labels/`, và một hook chạy smoke test sau mỗi lần sửa `app/backend/ect/`.

## Giao diện

`app/frontend/index.html` là thiết kế làm bằng Google AI Studio
(prompt: [`docs/google_ai_studio_prompt.md`](docs/google_ai_studio_prompt.md)), đã nối với bộ
xử lý: chữ trả lời chạy dần, trạng thái từng công cụ, hình hiện bên phải, "Mở ảnh gốc" mở bằng
trình xem ảnh của Windows, "Tải ảnh" mở hộp thoại lưu. Phần "mô phỏng" của bản AI Studio (tự vẽ
ảnh giả và bịa số liệu khi mất kết nối) đã bị bỏ: mọi hình và con số đều do công cụ thật tạo ra.

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
app/backend/server.py backend FastAPI (chạy ngầm trong phần mềm)
app/desktop.py        phần mềm desktop: cửa sổ WebView2 + khởi động backend
app/frontend/         giao diện (thiết kế Google AI Studio, đã nối backend)
app/assets/           icon, script tạo shortcut
app/state/            ảnh, bộ nhớ, nhật ký khi chạy (không đưa vào git)
docs/                 prompt Google AI Studio
```

Phần mềm cũ (app Tkinter) lấy lại được bằng `git checkout legacy-app`.
