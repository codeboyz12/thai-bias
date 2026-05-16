"""
pipeline.py — One-model pipeline: load → encode → metric → free → plot
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import matplotlib.pyplot as plt

from .config import GROUP_COLORS, vram_info
from .embedders import EmbedderFactory
from .metrics   import run_seat, gender_axis, project, direct_bias, template_sensitivity
from .visualize import (
    plot_scatter, plot_scatter_with_g,
    plot_seat_bar, plot_occ_bar,
)

log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
# Result
# ══════════════════════════════════════════════════════════════

@dataclass
class ModelResult:
    """ผลการทดลองทั้งหมดของ model หนึ่งตัว"""
    model_name:  str
    short:       str
    group:       str
    vecs:        dict[str, np.ndarray]   # X, Y, A, B, occ
    seat_score:  float
    effect_d:    float
    p_value:     float
    db_mean:     float
    occ_projs:   np.ndarray
    tmpl_std:    float = 0.0

    @property
    def is_significant(self) -> bool:
        return self.p_value < 0.05

    @property
    def g_unit(self) -> np.ndarray:
        return gender_axis(self.vecs["X"], self.vecs["Y"])


# ══════════════════════════════════════════════════════════════
# Run
# ══════════════════════════════════════════════════════════════

def run_model(
    model_name:  str,
    word_sets:   dict[str, list[str]],   # {"X", "Y", "A", "B"}
    occupations: list[str],
    n_perm:      int = 10_000,
    seed:        int = 42,
) -> ModelResult:
    """
    Full pipeline สำหรับ model เดียว
    model ถูก free จาก memory ทันทีหลัง encode เสร็จ

    Parameters
    ----------
    model_name  : HuggingFace model ID
    word_sets   : {"X": [...], "Y": [...], "A": [...], "B": [...]}
    occupations : list[str] — คำอาชีพสำหรับ direct bias
    n_perm      : จำนวน permutation สำหรับ p-value
    """
    X, Y, A, B = (word_sets[k] for k in ("X", "Y", "A", "B"))

    # ── Load + Encode ─────────────────────────────────────────
    log.info("[%s] Loading  %s", vram_info(), model_name)
    emb = EmbedderFactory.create(model_name)

    log.info("[%s] Encoding", vram_info())
    X_v   = emb.encode(X)
    Y_v   = emb.encode(Y)
    A_v   = emb.encode(A)
    B_v   = emb.encode(B)
    occ_v = emb.encode(occupations)

    # ── Free model ────────────────────────────────────────────
    # ทำก่อน metric เพราะ metric คำนวณบน CPU ด้วย numpy ไม่ต้องการ model แล้ว
    g_tmp  = gender_axis(X_v, Y_v)           # คำนวณก่อน free
    t_std  = template_sensitivity(X + Y, emb, g_tmp)  # ต้องการ emb.model
    emb.free()
    log.info("[%s] Model freed", vram_info())

    # ── Metrics ───────────────────────────────────────────────
    seat   = run_seat(X_v, Y_v, A_v, B_v, n_perm=n_perm, seed=seed)
    g_unit = gender_axis(X_v, Y_v)
    projs  = project(occ_v, g_unit)
    db     = direct_bias(occ_v, g_unit)

    log.info(
        "[%s] d=%.3f  p=%.4f  DB=%.4f",
        EmbedderFactory.get_short(model_name), seat.effect_d, seat.p_value, db,
    )

    return ModelResult(
        model_name = model_name,
        short      = EmbedderFactory.get_short(model_name),
        group      = EmbedderFactory.get_group(model_name),
        vecs       = {"X": X_v, "Y": Y_v, "A": A_v, "B": B_v, "occ": occ_v},
        seat_score = seat.score,
        effect_d   = seat.effect_d,
        p_value    = seat.p_value,
        db_mean    = db,
        occ_projs  = projs,
        tmpl_std   = t_std,
    )


# ══════════════════════════════════════════════════════════════
# Plot
# ══════════════════════════════════════════════════════════════

def plot_model(
    result:     ModelResult,
    word_sets:  dict[str, list[str]],
    occ_labels: list[str],
) -> None:
    """4-panel figure สำหรับ model เดียว"""
    g   = result.g_unit
    vd  = {k: result.vecs[k] for k in ["X", "Y", "A", "B"]}
    ld  = {k: word_sets[k]   for k in ["X", "Y", "A", "B"]}

    col = GROUP_COLORS[result.group]
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    fig.suptitle(
        f"{result.short}  [Group {result.group}]  "
        f"d={result.effect_d:.3f}  {'p<0.05 ✓' if result.is_significant else 'n.s.'}",
        fontsize=12, fontweight="bold", color=col,
    )

    plot_scatter(vd, ld, "1. Word Positions", axes[0])
    plot_scatter_with_g(vd, ld, g, result.vecs["occ"], occ_labels,
                        "2. Gender Axis + Occupations", axes[1])
    plot_seat_bar(result.effect_d, result.p_value, axes[2])
    plot_occ_bar(result.occ_projs, occ_labels, axes[3])

    plt.tight_layout()
    plt.show()
