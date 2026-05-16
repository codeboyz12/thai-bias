"""
metrics.py — Vectorized SEAT metrics
ทุกการคำนวณใช้ numpy matrix operations ไม่มี Python loop ใน hot path
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass


# ══════════════════════════════════════════════════════════════
# Low-level — vectorized cosine
# ══════════════════════════════════════════════════════════════

def cosine_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Pairwise cosine similarity ระหว่างทุกคู่ใน A และ B
    รับ L2-normalized vectors (จาก SentenceTransformer normalize=True)

    Parameters
    ----------
    A : (m, d)
    B : (n, d)

    Returns
    -------
    (m, n) — cosine[i, j] = cos(A[i], B[j])
    """
    return A @ B.T   # ← เร็วสุด: BLAS dgemm, O(m·n·d)


def _s_scores(
    targets: np.ndarray,
    A_vecs:  np.ndarray,
    B_vecs:  np.ndarray,
) -> np.ndarray:
    """
    Vectorized s(w, A, B) สำหรับทุก w ใน targets พร้อมกัน

    Parameters
    ----------
    targets : (n, d)
    A_vecs  : (na, d)
    B_vecs  : (nb, d)

    Returns
    -------
    (n,) — s score ของแต่ละ target word
    """
    mean_A = cosine_matrix(targets, A_vecs).mean(axis=1)   # (n,)
    mean_B = cosine_matrix(targets, B_vecs).mean(axis=1)   # (n,)
    return mean_A - mean_B


# ══════════════════════════════════════════════════════════════
# SEAT Score + Effect Size
# ══════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class SEATResult:
    score:       float
    effect_d:    float
    p_value:     float
    sx:          np.ndarray   # s scores ของ X (ใช้ต่อใน permutation ได้)
    sy:          np.ndarray   # s scores ของ Y


def seat_score(
    X_vecs: np.ndarray,
    Y_vecs: np.ndarray,
    A_vecs: np.ndarray,
    B_vecs: np.ndarray,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """
    Returns (score, effect_size_d, sx, sy)
    แยก sx/sy ออกมาเพื่อให้ permutation_test reuse ได้โดยไม่คำนวณซ้ำ
    """
    sx = _s_scores(X_vecs, A_vecs, B_vecs)   # (nx,)
    sy = _s_scores(Y_vecs, A_vecs, B_vecs)   # (ny,)

    score = float(sx.sum() - sy.sum())

    all_s       = np.concatenate([sx, sy])
    pooled_std  = float(all_s.std())
    effect_d    = float((sx.mean() - sy.mean()) / max(pooled_std, 1e-10))

    return score, effect_d, sx, sy


# ══════════════════════════════════════════════════════════════
# Permutation Test — vectorized
# ══════════════════════════════════════════════════════════════

def permutation_test(
    sx:     np.ndarray,
    sy:     np.ndarray,
    n_perm: int = 10_000,
    seed:   int = 42,
) -> float:
    """
    One-sided permutation test บน pre-computed s scores
    รับ sx, sy จาก seat_score() ไม่คำนวณซ้ำ

    Vectorized: สร้าง permutation matrix ทีเดียว (n_perm, nx+ny)
    แทนที่จะวน loop n_perm รอบ

    Parameters
    ----------
    sx     : (nx,)
    sy     : (ny,)

    Returns
    -------
    p-value (float)
    """
    observed = float(sx.sum() - sy.sum())
    combined = np.concatenate([sx, sy])
    n        = len(sx)
    rng      = np.random.default_rng(seed)

    # สร้าง permutation indices ทั้งหมดพร้อมกัน
    # shape: (n_perm, n_total) — memory: 10000 × 16 × 8 bytes = ~1.3 MB (เล็กมาก)
    perms    = rng.permuted(
        np.tile(np.arange(len(combined)), (n_perm, 1)),
        axis=1,
    )
    perm_sx_sum = combined[perms[:, :n]].sum(axis=1)   # (n_perm,)
    perm_sy_sum = combined[perms[:, n:]].sum(axis=1)   # (n_perm,)
    perm_stats  = perm_sx_sum - perm_sy_sum             # (n_perm,)

    return float((perm_stats >= observed).mean())


def run_seat(
    X_vecs: np.ndarray,
    Y_vecs: np.ndarray,
    A_vecs: np.ndarray,
    B_vecs: np.ndarray,
    n_perm: int = 10_000,
    seed:   int = 42,
) -> SEATResult:
    """
    One-stop function: คำนวณ score + d + p-value ครบในครั้งเดียว
    sx/sy คำนวณแค่ครั้งเดียว แล้ว reuse ใน permutation test
    """
    score, d, sx, sy = seat_score(X_vecs, Y_vecs, A_vecs, B_vecs)
    p = permutation_test(sx, sy, n_perm=n_perm, seed=seed)
    return SEATResult(score=score, effect_d=d, p_value=p, sx=sx, sy=sy)


# ══════════════════════════════════════════════════════════════
# Gender Axis + Projection
# ══════════════════════════════════════════════════════════════

def gender_axis(X_vecs: np.ndarray, Y_vecs: np.ndarray) -> np.ndarray:
    """g_unit = normalize(mean(X) - mean(Y))"""
    g    = X_vecs.mean(axis=0) - Y_vecs.mean(axis=0)
    norm = np.linalg.norm(g)
    return g / max(norm, 1e-10)


def project(vecs: np.ndarray, g_unit: np.ndarray) -> np.ndarray:
    """Scalar projection: shape (n,)"""
    return vecs @ g_unit


def direct_bias(occ_vecs: np.ndarray, g_unit: np.ndarray) -> float:
    """Mean absolute projection ของคำอาชีพบน g"""
    return float(np.abs(project(occ_vecs, g_unit)).mean())


# ══════════════════════════════════════════════════════════════
# Template Sensitivity (กลุ่ม C)
# ══════════════════════════════════════════════════════════════

def template_sensitivity(
    words:    list[str],
    embedder,                   # EmbedderBase (หลีกเลี่ยง circular import)
    g_unit:   np.ndarray,
    templates: list[str] | None = None,
) -> float:
    """
    std ของ mean|projection| ข้าม templates
    ยิ่งสูง = model sensitive ต่อ wording มาก → ผลน่าเชื่อถือน้อย

    คืนค่า 0.0 สำหรับ group A และ B (ไม่ใช้ multi-template)
    """
    if embedder.config.group != "C":
        return 0.0

    from .config import DEFAULT_TEMPLATES
    tmpl_list = templates or DEFAULT_TEMPLATES

    scores: list[float] = []
    for tmpl in tmpl_list:
        sentences = [tmpl.format(w) for w in words]
        with __import__("torch").inference_mode():
            vecs = embedder.model.encode(
                sentences,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            ).astype("float32")
        scores.append(float(np.abs(project(vecs, g_unit)).mean()))

    return float(np.std(scores))
