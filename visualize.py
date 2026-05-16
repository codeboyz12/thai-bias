"""
visualize.py — Plot functions (ไม่มี metric logic)
ทุก function รับ ax จากภายนอก → testable และ composable
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.decomposition import PCA

from .config import SET_COLORS, GROUP_COLORS


# ══════════════════════════════════════════════════════════════
# PCA helpers
# ══════════════════════════════════════════════════════════════

def _pca_2d(vecs: np.ndarray) -> tuple[np.ndarray, PCA]:
    pca    = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(vecs)
    return coords, pca


def _project_g_to_2d(g_unit: np.ndarray, pca: PCA, ref: np.ndarray) -> np.ndarray:
    """แปลง g_unit (high-dim) ไปยัง 2D space ของ PCA"""
    v = (pca.transform((ref + g_unit).reshape(1, -1))[0]
         - pca.transform(ref.reshape(1, -1))[0])
    return v / np.linalg.norm(v)


def _axis_line(origin: np.ndarray, unit: np.ndarray, coords: np.ndarray):
    """หา p1, p2 ที่ทำให้เส้น g ยาวถึงขอบ plot"""
    margin  = 0.4
    x0, x1  = coords[:, 0].min() - margin, coords[:, 0].max() + margin
    y0, y1  = coords[:, 1].min() - margin, coords[:, 1].max() + margin
    scales  = []
    for ux, lo, hi, ox in [
        (unit[0], x0, x1, origin[0]),
        (unit[1], y0, y1, origin[1]),
    ]:
        if abs(ux) > 1e-9:
            scales += [(lo - ox) / ux, (hi - ox) / ux]
    pos = [s for s in scales if s > 0]
    neg = [s for s in scales if s < 0]
    sf  = min(pos) if pos else 1.0
    sb  = max(neg) if neg else -1.0
    return origin + sb * unit, origin + sf * unit


# ══════════════════════════════════════════════════════════════
# Panel 1 — Word scatter
# ══════════════════════════════════════════════════════════════

def plot_scatter(
    vecs_dict:   dict[str, np.ndarray],
    labels_dict: dict[str, list[str]],
    title:       str,
    ax:          plt.Axes,
) -> None:
    """Scatter ทุก word set บน PCA 2D"""
    all_vecs   = np.concatenate(list(vecs_dict.values()))
    coords, _  = _pca_2d(all_vecs)

    start = 0
    for key, vecs in vecs_dict.items():
        n = len(vecs)
        c = coords[start:start + n]
        start += n
        ax.scatter(c[:, 0], c[:, 1], s=80, color=SET_COLORS[key],
                   label=key, zorder=4, edgecolors="white", linewidths=0.5)
        for i, lbl in enumerate(labels_dict[key]):
            ax.annotate(lbl, c[i], fontsize=7.5, color=SET_COLORS[key],
                        xytext=(4, 3), textcoords="offset points")

    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.legend(fontsize=7, markerscale=0.8)
    ax.grid(True, alpha=0.2, linestyle="--")


# ══════════════════════════════════════════════════════════════
# Panel 2 — Scatter + gender axis + projections
# ══════════════════════════════════════════════════════════════

def plot_scatter_with_g(
    vecs_dict:   dict[str, np.ndarray],
    labels_dict: dict[str, list[str]],
    g_unit:      np.ndarray,
    occ_vecs:    np.ndarray,
    occ_labels:  list[str],
    title:       str,
    ax:          plt.Axes,
) -> None:
    """Scatter + gender axis arrow + occupation projection lines"""
    all_vecs    = np.concatenate(list(vecs_dict.values()) + [occ_vecs])
    coords, pca = _pca_2d(all_vecs)
    n_set       = sum(len(v) for v in vecs_dict.values())
    sc, oc      = coords[:n_set], coords[n_set:]

    # word sets
    start = 0
    for key, vecs in vecs_dict.items():
        n = len(vecs)
        c = sc[start:start + n]
        start += n
        ax.scatter(c[:, 0], c[:, 1], s=80, color=SET_COLORS[key],
                   label=key, zorder=4, edgecolors="white", linewidths=0.5)
        for i, lbl in enumerate(labels_dict[key]):
            ax.annotate(lbl, c[i], fontsize=7, color=SET_COLORS[key],
                        xytext=(4, 3), textcoords="offset points")

    # occupations
    ax.scatter(oc[:, 0], oc[:, 1], s=40, color="#94A3B8", marker="^",
               zorder=3, label="อาชีพ", edgecolors="white", linewidths=0.4)
    for i, lbl in enumerate(occ_labels):
        ax.annotate(lbl, oc[i], fontsize=6.5, color="#64748B",
                    xytext=(3, 3), textcoords="offset points")

    # gender axis
    ref    = list(vecs_dict.values())[0][0]
    origin = sc.mean(axis=0)
    g2     = _project_g_to_2d(g_unit, pca, ref)
    p1, p2 = _axis_line(origin, g2, coords)

    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="#EF4444",
            linestyle="--", alpha=0.4, lw=1.2, zorder=1)
    ax.annotate("", xy=p2, xytext=p2 - g2 * 0.3,
                arrowprops=dict(arrowstyle="->", color="#EF4444", lw=1.5))
    ax.text(p2[0] + 0.04, p2[1] + 0.04, "g (male→)",
            fontsize=7.5, color="#EF4444", fontstyle="italic")

    # projection lines
    for i in range(len(occ_vecs)):
        v    = oc[i] - origin
        foot = origin + np.dot(v, g2) * g2
        ax.plot([oc[i, 0], foot[0]], [oc[i, 1], foot[1]],
                color="#94A3B8", lw=0.5, alpha=0.4)

    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.legend(fontsize=7, markerscale=0.8, ncol=2)
    ax.grid(True, alpha=0.2, linestyle="--")


# ══════════════════════════════════════════════════════════════
# Panel 3 — SEAT bar
# ══════════════════════════════════════════════════════════════

def plot_seat_bar(d_val: float, p_val: float, ax: plt.Axes) -> None:
    """Single horizontal bar แสดง SEAT effect size"""
    sig   = p_val < 0.05
    color = (
        "#EF4444" if (sig and d_val > 0)
        else "#3B82F6" if (sig and d_val < 0)
        else "#CBD5E1"
    )
    ax.barh(["d"], [d_val], color=color, height=0.4)
    ax.axvline(0,    color="black", lw=0.8)
    for v in [0.2, -0.2]:
        ax.axvline(v, color="#F59E0B", lw=1, linestyle="--", alpha=0.7)
    ax.set_xlim(-2, 2)
    ax.set_title(
        f"Effect size  d = {d_val:.3f}\n{'p < 0.05 ✓' if sig else 'n.s.'}",
        fontsize=9,
    )
    ax.set_xlabel("Cohen's d")


# ══════════════════════════════════════════════════════════════
# Panel 4 — Occupation projection bar
# ══════════════════════════════════════════════════════════════

def plot_occ_bar(
    projs:      np.ndarray,
    occ_labels: list[str],
    ax:         plt.Axes,
) -> None:
    """Horizontal bars แสดง projection ของแต่ละอาชีพ"""
    colors = ["#2563EB" if p > 0 else "#DB2777" for p in projs]
    ax.barh(occ_labels, projs, color=colors, height=0.7, alpha=0.8)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_title("Projection on g (อาชีพ)", fontsize=9)
    ax.set_xlabel("Scalar projection")
    ax.tick_params(axis="y", labelsize=7.5)


# ══════════════════════════════════════════════════════════════
# Comparison plots
# ══════════════════════════════════════════════════════════════

def plot_group_bars(
    names:     list[str],
    d_vals:    list[float],
    p_vals:    list[float],
    db_vals:   list[float],
    t_stds:    list[float],
    group:     str,
    axes:      list[plt.Axes],
) -> None:
    """Bar charts สำหรับ within-group comparison (3-4 panels)"""
    col = GROUP_COLORS[group]

    # 1. Effect size d
    bar_c = [col if p < 0.05 else "#CBD5E1" for p in p_vals]
    axes[0].barh(names, d_vals, color=bar_c, height=0.65)
    axes[0].axvline(0, color="black", lw=0.8)
    for v in [0.2, -0.2]:
        axes[0].axvline(v, color="#F59E0B", lw=1, linestyle="--", alpha=0.6)
    axes[0].set_title("SEAT Effect Size d\n(สี = p<0.05)", fontsize=9)
    axes[0].set_xlabel("Cohen's d")

    # 2. p-value
    p_c = ["#EF4444" if p < 0.05 else "#94A3B8" for p in p_vals]
    axes[1].barh(names, p_vals, color=p_c, height=0.65)
    axes[1].axvline(0.05, color="#EF4444", lw=1, linestyle="--", label="p=0.05")
    axes[1].set_title("p-value (permutation test)", fontsize=9)
    axes[1].set_xlabel("p-value")
    axes[1].legend(fontsize=7)

    # 3. Direct Bias
    axes[2].barh(names, db_vals, color=col, alpha=0.75, height=0.65)
    axes[2].set_title("Direct Bias\n(mean |projection|)", fontsize=9)
    axes[2].set_xlabel("Direct Bias")

    # 4. Template σ (group C only)
    if len(axes) > 3:
        axes[3].barh(names, t_stds, color="#F59E0B", alpha=0.8, height=0.65)
        axes[3].set_title("Template Sensitivity\n(std ข้าม templates)", fontsize=9)
        axes[3].set_xlabel("Std deviation")


def plot_occ_heatmap(
    projs:       np.ndarray,
    model_names: list[str],
    occ_labels:  list[str],
    title:       str,
    ax:          plt.Axes,
) -> None:
    """Occupation projection heatmap"""
    sns.heatmap(
        projs,
        xticklabels = occ_labels,
        yticklabels = model_names,
        cmap        = "RdBu_r",
        center      = 0,
        vmin        = -1.5,
        vmax        =  1.5,
        linewidths  = 0.3,
        annot       = len(model_names) <= 10,
        fmt         = ".2f",
        annot_kws   = {"size": 7},
        ax          = ax,
    )
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right", fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)


def plot_cross_scatter(
    records:  list[dict],   # [{"short", "group", "d", "db", "sig"}]
    ax:       plt.Axes,
) -> None:
    """Scatter: Direct Bias vs SEAT d แยกสีตามกลุ่ม"""
    groups = sorted({r["group"] for r in records})
    for grp in groups:
        sub = [r for r in records if r["group"] == grp]
        ax.scatter(
            [r["db"] for r in sub],
            [r["d"]  for r in sub],
            color       = GROUP_COLORS[grp],
            s           = 70,
            label       = f"Group {grp}",
            edgecolors  = "white",
            linewidths  = 0.5,
            zorder      = 4,
        )
        for r in sub:
            ax.annotate(r["short"], (r["db"], r["d"]),
                        fontsize=6.5, xytext=(3, 3),
                        textcoords="offset points")
    ax.axhline(0,    color="black",   lw=0.6)
    ax.axhline(0.2,  color="#F59E0B", lw=1, linestyle="--", alpha=0.5)
    ax.axhline(-0.2, color="#F59E0B", lw=1, linestyle="--", alpha=0.5)
    ax.set_xlabel("Direct Bias")
    ax.set_ylabel("SEAT Effect size d")
    ax.set_title("Direct Bias vs SEAT d")
    ax.legend(fontsize=8)


def make_group_legend() -> list[mpatches.Patch]:
    return [mpatches.Patch(color=GROUP_COLORS[g], label=f"Group {g}") for g in "ABC"]
