"""Project locations. Raw scans live in data/ and are never written; everything the app
produces goes under app/state/ (images, memory, logs)."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
LABELS_DIR = RESULTS_DIR / "labels"
MODEL_DIR = RESULTS_DIR / "model"
WEIGHTS = MODEL_DIR / "crack_yolo.pt"
ONNX_WEIGHTS = MODEL_DIR / "crack_yolo.onnx"     # same model for onnxruntime (app/tools/export_onnx.py)
SPLIT_FILE = MODEL_DIR / "split.json"

APP_DIR = PROJECT_ROOT / "app"
STATE_DIR = APP_DIR / "state"
IMAGES_DIR = STATE_DIR / "images"
LOG_DIR = STATE_DIR / "logs"
MEMORY_FILE = STATE_DIR / "memory.json"
AGENT_HOME = APP_DIR / "agent_home"


def ensure_state_dirs():
    for d in (STATE_DIR, IMAGES_DIR, LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)


def inside(path, root):
    """True when `path` resolves to a location under `root` (no escaping via ..)."""
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False
