"""
config.py — Runtime constants and device setup
เป็น module ที่ import ก่อนสุด ไม่มี side-effects นอกจาก detect device
"""
from __future__ import annotations

import gc
import os
import glob
import logging
from functools import lru_cache

import numpy as np
import torch

log = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════
# Device
# ══════════════════════════════════════════════════════════════

def _detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE: str = _detect_device()

# dtype: float16 บน GPU ลด VRAM ครึ่งหนึ่ง ความแม่นยำต่างกันน้อยมากสำหรับ cosine
DTYPE: torch.dtype = (
    torch.float16 if DEVICE == "cuda"
    else torch.float32
)

# Batch size ต่อ device — ปรับตาม VRAM จริงหลัง benchmark
_BATCH_SIZE: dict[str, int] = {
    "cuda": 512,
    "mps":  128,
    "cpu":   32,
}
BATCH_SIZE: int = _BATCH_SIZE[DEVICE]

def vram_info() -> str:
    """Human-readable VRAM usage string"""
    if DEVICE != "cuda":
        return f"device={DEVICE}"
    used  = torch.cuda.memory_allocated()  / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    return f"VRAM {used:.2f}/{total:.2f} GB"


def free_memory() -> None:
    """Force garbage collection และคืน GPU cache"""
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()


# ══════════════════════════════════════════════════════════════
# Thai Font
# ══════════════════════════════════════════════════════════════

@lru_cache(maxsize=1)
def get_thai_font_path() -> str | None:
    """หา Thai .ttf ที่ใช้ได้ในระบบ (cached — เรียกซ้ำไม่เสียเวลา)"""
    patterns = [
        "/usr/share/fonts/**/*Sarabun*.ttf",
        "/usr/share/fonts/**/*Thai*.ttf",
        "/usr/share/fonts/**/*Garuda*.ttf",
        "/usr/share/fonts/**/*Noto*Thai*.ttf",
        "/root/.fonts/*Thai*.ttf",
        "/tmp/*Thai*.ttf",
        "/tmp/*Sarabun*.ttf",
    ]
    for pat in patterns:
        found = glob.glob(pat, recursive=True)
        if found:
            return found[0]
    return None


def setup_matplotlib_thai(font_path: str | None = None) -> str:
    """
    ตั้งค่า matplotlib ให้แสดงภาษาไทยได้
    Returns font name ที่ใช้ หรือ raise RuntimeError ถ้าไม่พบ font
    """
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm

    path = font_path or get_thai_font_path()
    if path is None:
        raise RuntimeError(
            "Thai font not found.\n"
            "Install via:  !apt-get install -y fonts-thai-tlwg\n"
            "Or download Sarabun from https://fonts.google.com/specimen/Sarabun"
        )

    fm.fontManager.addfont(path)
    font_name = fm.FontProperties(fname=path).get_name()

    plt.rcParams.update({
        "font.family":        font_name,
        "axes.unicode_minus": False,
    })

    # Clear matplotlib font cache
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "matplotlib")
    for f in glob.glob(os.path.join(cache_dir, "*.json")):
        try:
            os.remove(f)
        except OSError:
            pass

    log.info("Thai font registered: %s (%s)", font_name, path)
    return font_name


# ══════════════════════════════════════════════════════════════
# Visualization constants
# ══════════════════════════════════════════════════════════════

SET_COLORS: dict[str, str] = {
    "X": "#2563EB",
    "Y": "#DB2777",
    "A": "#059669",
    "B": "#D97706",
}

GROUP_COLORS: dict[str, str] = {
    "A": "#7C3AED",
    "B": "#059669",
    "C": "#2563EB",
}

DEFAULT_PLOT_STYLE: dict = {
    "figure.dpi":       120,
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "font.size":        10,
}

# ══════════════════════════════════════════════════════════════
# Default SEAT templates
# ══════════════════════════════════════════════════════════════

DEFAULT_TEMPLATES: list[str] = [
    "{}",
    "ฉันเป็น{}",
    "เขาเป็น{}",
    "นี่คือ{}",
    "{}เป็นอาชีพที่น่านับถือ",
]
