"""
analysis.py — Multi-model comparison and ranking
ทุก function รับ list[ModelResult] → produce figures + DataFrame
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from .config   import GROUP_COLORS
from .pipeline import ModelResult
from .visualize import (
    plot_group_bars, plot_occ_heatmap,
    plot_cross_scatter, make_group_legend,
)


# ══════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════

def _results_to_df(results: list[ModelResult]) -> pd.DataFrame:
    rows = [{
        "model":       r.short,
        "group":       r.group,
        "d":           r.effect_d,
        "p":           r.p_value,
        "db":          r.db_mean,
        "tmpl_std":    r.tmpl_std,
        "sig":         r.is_significant,
        "biased_occ":  int(np.sum(np.abs(r.occ_projs) > 0.2)),
    } for r in results]
    return pd.DataFrame(rows)


def _filter_group(results: list[ModelResult], group: str) -> list[ModelResult]:
    return sorted(
        [r for r in results if r.group == group],
        key=lambda r: abs(r.effect_d), reverse=True,
    )


# ══════════════════════════════════════════════════════════════
# Within-group comparison
# ══════════════════════════════════════════════════════════════

def plot_within_group(
    results:    list[ModelResult],
    group:      str,
    occ_labels: list[str],
) -> None:
    """
    Bar charts + occupation heatmap สำหรับ model ในกลุ่มเดียวกัน
    กลุ่ม C มี 4 panels (เพิ่ม Template Sensitivity)
    """
    rs = _filter_group(results, group)
    if not rs:
        print(f"No results for group {group}")
        return

    names   = [r.short      for r in rs]
    d_vals  = [r.effect_d   for r in rs]
    p_vals  = [r.p_value    for r in rs]
    db_vals = [r.db_mean    for r in rs]
    t_stds  = [r.tmpl_std   for r in rs]

    n_panels = 4 if group == "C" else 3
    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(6 * n_panels, max(4, len(rs) * 0.5 + 1.5)),
    )
    fig.suptitle(
        f"Within-Group Comparison — Group {group}",
        fontsize=12, fontweight="bold",
    )

    plot_group_bars(names, d_vals, p_vals, db_vals, t_stds, group, list(axes))
    plt.tight_layout()
    plt.show()

    # Occupation heatmap
    projs = np.array([r.occ_projs for r in rs])
    fig2, ax2 = plt.subplots(figsize=(13, max(3, len(rs) * 0.5 + 1.5)))
    plot_occ_heatmap(
        projs, names, occ_labels,
        f"Occupation Projection — Group {group}", ax2,
    )
    plt.tight_layout()
    plt.show()


# ══════════════════════════════════════════════════════════════
# Cross-group comparison
# ══════════════════════════════════════════════════════════════

def plot_cross_group(
    results:    list[ModelResult],
    occ_labels: list[str],
) -> None:
    """Boxplot + scatter + % significant + full heatmap"""
    if not results:
        return

    df      = _results_to_df(results)
    records = df.to_dict("records")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Cross-Group Comparison: A vs B vs C",
                 fontsize=13, fontweight="bold")

    # ── 1. Boxplot + jitter ───────────────────────────────────
    rng = np.random.default_rng(42)
    for i, grp in enumerate(["A", "B", "C"]):
        vals = df[df.group == grp]["d"].values
        axes[0].boxplot(
            vals, positions=[i], widths=0.4, patch_artist=True,
            boxprops    = dict(facecolor=GROUP_COLORS[grp], alpha=0.5),
            medianprops = dict(color="black", lw=2),
        )
        jitter = rng.uniform(-0.1, 0.1, len(vals))
        axes[0].scatter(np.full(len(vals), i) + jitter, vals,
                        color=GROUP_COLORS[grp], s=45, zorder=5,
                        edgecolors="white", linewidths=0.5)
    for v in [0.2, -0.2]:
        axes[0].axhline(v, color="#F59E0B", lw=1, linestyle="--", alpha=0.6)
    axes[0].set_xticks([0, 1, 2])
    axes[0].set_xticklabels(["A\n(Instruct)", "B\n(Sentence)", "C\n(Base BERT)"])
    axes[0].set_ylabel("Effect size d")
    axes[0].set_title("Distribution of SEAT d")

    # ── 2. Scatter d vs DB ────────────────────────────────────
    plot_cross_scatter(records, axes[1])

    # ── 3. % significant per group ────────────────────────────
    sig_pct = df.groupby("group")["sig"].mean() * 100
    bars    = axes[2].bar(
        sig_pct.index, sig_pct.values,
        color=[GROUP_COLORS[g] for g in sig_pct.index],
        width=0.5, alpha=0.85,
    )
    for bar, val in zip(bars, sig_pct.values):
        axes[2].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 1,
                     f"{val:.0f}%", ha="center", fontsize=9, fontweight="bold")
    axes[2].set_ylim(0, 115)
    axes[2].set_ylabel("% significant (p < 0.05)")
    axes[2].set_title("Proportion of Significant Bias")
    axes[2].set_xticklabels(["A\n(Instruct)", "B\n(Sentence)", "C\n(Base BERT)"])

    plt.tight_layout()
    plt.show()

    # ── Full occupation heatmap (ทุก model) ───────────────────
    sorted_r = sorted(results, key=lambda r: (r.group, -abs(r.effect_d)))
    projs    = np.array([r.occ_projs for r in sorted_r])
    ylabels  = [f"[{r.group}] {r.short}" for r in sorted_r]

    fig2, ax2 = plt.subplots(
        figsize=(14, max(4, len(sorted_r) * 0.45 + 1.5))
    )
    plot_occ_heatmap(projs, ylabels, occ_labels,
                     "All Models — Occupation Projection on g", ax2)
    plt.tight_layout()
    plt.show()


# ══════════════════════════════════════════════════════════════
# Final ranking
# ══════════════════════════════════════════════════════════════

def final_ranking(
    results:    list[ModelResult],
    occ_labels: list[str],
) -> pd.DataFrame:
    """
    สร้าง ranking table + bar + bubble chart

    Returns
    -------
    pd.DataFrame — sorted by |effect_d| descending
    """
    if not results:
        print("No results.")
        return pd.DataFrame()

    rows = [{
        "Model":           r.short,
        "Group":           r.group,
        "SEAT d":          round(r.effect_d, 4),
        "p-value":         round(r.p_value, 4),
        "Sig":             "✓" if r.is_significant else "–",
        "Direct Bias":     round(r.db_mean, 4),
        "Biased Occ >0.2": int(np.sum(np.abs(r.occ_projs) > 0.2)),
        "Template σ":      round(r.tmpl_std, 4),
    } for r in results]

    df = (pd.DataFrame(rows)
          .assign(**{"|d|": lambda x: x["SEAT d"].abs()})
          .sort_values("|d|", ascending=False)
          .drop(columns="|d|")
          .reset_index(drop=True))
    df.index += 1
    df.index.name = "Rank"

    # ── Print ─────────────────────────────────────────────────
    sep = "=" * 72
    print(f"\n{sep}")
    print("  FINAL RANKING — Thai Gender Bias (SEAT TH-1)")
    print(sep)
    print(df.to_string())
    print(sep)

    # ── Figures ───────────────────────────────────────────────
    rs_sorted = sorted(results, key=lambda r: r.effect_d)
    names     = [f"[{r.group}] {r.short}" for r in rs_sorted]
    d_vals    = [r.effect_d for r in rs_sorted]
    bar_colors = [
        "#EF4444" if (r.is_significant and r.effect_d > 0)
        else "#3B82F6" if (r.is_significant and r.effect_d < 0)
        else "#CBD5E1"
        for r in rs_sorted
    ]

    fig, axes = plt.subplots(
        1, 2,
        figsize=(18, max(5, len(rs_sorted) * 0.45 + 2)),
    )
    fig.suptitle("Final Ranking — Thai Gender Bias",
                 fontsize=13, fontweight="bold")

    # Bar chart
    axes[0].barh(names, d_vals, color=bar_colors, height=0.7)
    axes[0].axvline(0,    color="black", lw=0.8)
    for v in [0.2, -0.2]:
        axes[0].axvline(v, color="#F59E0B", lw=1, linestyle="--", alpha=0.7)
    axes[0].set_title(
        "SEAT Effect Size d\n🔴 male bias  🔵 female bias  ⬜ n.s.",
        fontsize=9,
    )
    axes[0].set_xlabel("Cohen's d")
    axes[0].legend(handles=make_group_legend(), fontsize=8, loc="lower right")

    # Bubble scatter
    records = [{"short": r.short, "group": r.group,
                "d": r.effect_d, "db": r.db_mean} for r in results]
    for r in results:
        sz = np.sum(np.abs(r.occ_projs) > 0.2) * 20 + 30
        axes[1].scatter(r.db_mean, r.effect_d,
                        s=sz, color=GROUP_COLORS[r.group],
                        edgecolors="white", linewidths=0.6,
                        alpha=0.85, zorder=4)
        axes[1].annotate(r.short, (r.db_mean, r.effect_d),
                         fontsize=7, xytext=(4, 4),
                         textcoords="offset points")
    axes[1].axhline(0,    color="black", lw=0.6)
    for v in [0.2, -0.2]:
        axes[1].axhline(v, color="#F59E0B", lw=1, linestyle="--", alpha=0.5)
    axes[1].set_xlabel("Direct Bias (mean |projection|)")
    axes[1].set_ylabel("SEAT Effect size d")
    axes[1].set_title("Bias Landscape\n(bubble = จำนวนอาชีพ biased >0.2)")
    axes[1].legend(handles=make_group_legend(), fontsize=8)

    plt.tight_layout()
    plt.show()

    return df
