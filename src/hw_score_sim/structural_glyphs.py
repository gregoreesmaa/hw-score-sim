"""54 visually distinct handwritten structural glyph archetypes (Clefs, Key Signatures, Barlines).

Self-contained module implementing:
  - 18 Clef archetypes (`CLEF_ARCHETYPES`), each supporting G, F, C, G8vb, and G8va
  - 18 Key Signature & Accidental archetypes (`KEYSIG_ARCHETYPES`), varying both
    accidental glyph morphology (#, b, natural, double_sharp, double_flat) and
    key-signature horizontal spacing, staggering, and baseline drift
  - 18 Barline archetypes (`BARLINE_ARCHETYPES`), varying single, double, final,
    repeat_start, repeat_end, and repeat_both geometry, stroke physics, and repeat dots

Designed to plug cleanly into `glyph_variants.py` and `styles.py` without modifying
them directly. All rendering uses `stroke_engine.py` primitives (`StrokeCanvas`,
`draw_calligraphic_polyline`, `draw_filled_blob`).
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .stroke_engine import (
    RGBLike,
    StrokeCanvas,
    draw_calligraphic_polyline,
    draw_filled_blob,
)


# ---------------------------------------------------------------------------
# Archetype Registries (18 + 18 + 18 = 54)
# ---------------------------------------------------------------------------

CLEF_ARCHETYPES: Tuple[str, ...] = (
    "classic_engraved",
    "baroque_ornate_scroll",
    "hurried_single_loop",
    "tall_spindly_g",
    "stubby_compact",
    "minimal_s_slash",
    "figure_8_treble",
    "beethoven_wild_slash",
    "chopin_delicate_spiral",
    "jazz_realbook_bold",
    "sharp_angular_modern",
    "detached_dot_bass",
    "open_c_hook_bass",
    "double_bar_c_clef",
    "bracket_c_clef",
    "calligraphic_broad_nib",
    "tremor_shaky_clef",
    "backslant_lefty_clef",
)

KEYSIG_ARCHETYPES: Tuple[str, ...] = (
    "classic_neat",
    "tall_narrow_sharp",
    "wide_slanted_hash",
    "hurried_x_sharp",
    "open_loop_flat",
    "pointed_teardrop_flat",
    "tiny_pinprick_acc",
    "oversized_bold_acc",
    "crammed_tight_cluster",
    "wide_spread_keysig",
    "ascending_drift_keysig",
    "descending_drift_keysig",
    "double_stroke_sketch",
    "calligraphic_nib_acc",
    "baroque_vertical_flat",
    "jazz_heavy_marker_acc",
    "hairline_needle_acc",
    "jagged_zigzag_acc",
)

BARLINE_ARCHETYPES: Tuple[str, ...] = (
    "crisp_vertical",
    "tall_overshoot",
    "short_clipped_inside",
    "steep_forward_slant",
    "back_slant_lefty",
    "bowed_curved_bar",
    "s_wobble_freehand",
    "heavy_thick_marker",
    "hairline_faint",
    "tapered_quill_bar",
    "double_pass_sketch",
    "gapped_broken_bar",
    "hooked_top_bottom_serif",
    "wide_double_gap",
    "tight_double_gap",
    "oversized_repeat_blobs",
    "tiny_pinprick_repeat_dots",
    "slashed_repeat_ticks",
)

# Key-signature diatonic staff steps (0 = bottom line, 8 = top line)
_SHARP_STEPS: Dict[str, Tuple[int, ...]] = {
    "G": (8, 5, 9, 6, 3, 7, 4),
    "F": (6, 3, 7, 4, 1, 5, 2),
    "C": (7, 4, 8, 5, 2, 6, 3),
    "C-4": (2, 6, 3, 7, 4, 8, 5),
}

_FLAT_STEPS: Dict[str, Tuple[int, ...]] = {
    "G": (4, 7, 3, 6, 2, 5, 1),
    "F": (2, 5, 1, 4, 0, 3, -1),
    "C": (3, 6, 2, 5, 1, 4, 0),
    "C-4": (5, 8, 4, 7, 3, 6, 2),
}


# ---------------------------------------------------------------------------
# Internal Helpers
# ---------------------------------------------------------------------------


def _resolve_canvas(engine: Any) -> StrokeCanvas:
    """Accept either a StrokeCanvas directly or an engine wrapper exposing `.canvas`."""
    if isinstance(engine, StrokeCanvas):
        return engine
    if hasattr(engine, "canvas") and isinstance(engine.canvas, StrokeCanvas):
        return engine.canvas
    return engine


def _resolve_ink_rgb(style: Optional[dict], color_rgb: Optional[RGBLike] = None) -> RGBLike:
    if color_rgb is not None:
        return color_rgb
    if style and "ink_rgb" in style:
        return style["ink_rgb"]
    return (0.10, 0.10, 0.12)


def _eval_cubic(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, n: int = 16) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n)[:, None]
    mt = 1.0 - t
    return (mt**3) * p0 + 3.0 * (mt**2) * t * p1 + 3.0 * mt * (t**2) * p2 + (t**3) * p3


def _eval_quad(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, n: int = 12) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n)[:, None]
    mt = 1.0 - t
    return (mt**2) * p0 + 2.0 * mt * t * p1 + (t**2) * p2


def _eval_catmull_rom(ctrl: np.ndarray, samples_per_seg: int = 8) -> np.ndarray:
    pts = np.asarray(ctrl, dtype=np.float64)
    if len(pts) <= 2:
        return pts
    p_ext = np.vstack((2.0 * pts[0] - pts[1], pts, 2.0 * pts[-1] - pts[-2]))
    out: List[np.ndarray] = []
    t = np.linspace(0.0, 1.0, samples_per_seg, endpoint=False)[:, None]
    t2 = t * t
    t3 = t2 * t
    for i in range(len(pts) - 1):
        p0, p1, p2, p3 = p_ext[i], p_ext[i + 1], p_ext[i + 2], p_ext[i + 3]
        seg = 0.5 * (
            (2.0 * p1)
            + (-p0 + p2) * t
            + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
            + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
        )
        out.append(seg)
    out.append(pts[-1:])
    return np.vstack(out)


def _transform_pts(
    pts_norm: np.ndarray,
    cx: float,
    cy: float,
    sp: float,
    rng: np.random.Generator,
    sx: float = 1.0,
    sy: float = 1.0,
    slant_deg: float = 0.0,
    jitter_sp: float = 0.04,
) -> np.ndarray:
    """Scale, slant, jitter, and place normalized (in sp units) control points."""
    pts = np.asarray(pts_norm, dtype=np.float64).copy()
    if jitter_sp > 0.0:
        pts += rng.normal(0.0, jitter_sp, size=pts.shape)
    shear = math.tan(math.radians(float(np.clip(slant_deg, -30.0, 30.0))))
    x = pts[:, 0] * sx - pts[:, 1] * shear
    y = pts[:, 1] * sy
    return np.column_stack((cx + x * sp, cy + y * sp))


def _stable_index(key: str, modulo: int) -> int:
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % modulo


def _normalize_clef_kind(clef_kind: str) -> Tuple[str, int]:
    """Return (base_clef, octave_shift) where base_clef in {'G', 'F', 'C', 'C-4'}."""
    raw = str(clef_kind).strip().upper().replace("_", "-")
    shift = 0
    if "8VB" in raw or "-8" in raw:
        shift = -1
    elif "8VA" in raw or "+8" in raw:
        shift = 1
    if raw.startswith("G") or "TREBLE" in raw:
        return "G", shift
    if raw.startswith("F") or "BASS" in raw:
        return "F", shift
    if "TENOR" in raw or raw.startswith("C-4") or raw == "C4":
        return "C-4", shift
    return "C", shift


def _normalize_acc_kind(acc_kind: str) -> str:
    raw = str(acc_kind).strip().lower().replace("-", "_")
    if raw in ("#", "sharp", "is", "s"):
        return "sharp"
    if raw in ("b", "flat", "es", "f"):
        return "flat"
    if raw in ("n", "nat", "natural", "♮"):
        return "natural"
    if raw in ("x", "##", "double_sharp", "doublesharp", "isis"):
        return "double_sharp"
    if raw in ("bb", "double_flat", "doubleflat", "eses"):
        return "double_flat"
    return "sharp"


def _normalize_bar_kind(bar_kind: str) -> str:
    raw = str(bar_kind).strip().lower().replace("-", "_")
    aliases = {
        "single": "single",
        "regular": "single",
        "double": "double",
        "final": "final",
        "end": "final",
        "light_heavy": "final",
        "repeat_start": "repeat_start",
        "repeat_left": "repeat_start",
        "start_repeat": "repeat_start",
        "repeat_end": "repeat_end",
        "repeat_right": "repeat_end",
        "end_repeat": "repeat_end",
        "repeat_both": "repeat_both",
        "repeat_both_sides": "repeat_both",
    }
    return aliases.get(raw, "single")


# ---------------------------------------------------------------------------
# Style -> Structural Archetype Resolver
# ---------------------------------------------------------------------------


def resolve_structural_archetypes(style: dict) -> Tuple[str, str, str]:
    """Deterministically resolve (clef_archetype, keysig_archetype, barline_archetype).

    Respects explicit `clef_archetype`, `keysig_archetype`, and `barline_archetype`
    keys when valid; otherwise maps deterministically from style metadata (`name`,
    `era`, and nib/slant/tremor cues).
    """
    st = style or {}
    name = str(st.get("name", "")).strip().lower()
    era = str(st.get("era", "")).strip().lower()
    slant = float(st.get("slant_deg", 0.0))
    tremor = float(st.get("tremor_px", 0.25))
    stroke_scale = float(st.get("stroke_scale", 1.0))
    overshoot = float(st.get("barline_overshoot_sp", 0.16))
    ornate = float(st.get("clef_ornate", 0.5))

    # 1. Clef archetype
    clef = st.get("clef_archetype")
    if clef not in CLEF_ARCHETYPES:
        if "baroque" in name or "bach" in name or "baroque" in era:
            clef = "baroque_ornate_scroll"
        elif "chopin" in name or "delicate" in name or "salon" in name:
            clef = "chopin_delicate_spiral"
        elif ornate >= 0.85:
            clef = "baroque_ornate_scroll"
        elif "tremor" in name or "shaky" in name or tremor >= 0.9:
            clef = "tremor_shaky_clef"
        elif "realbook" in name or "jazz" in name or "marker" in name:
            clef = "jazz_realbook_bold"
        elif "lefty" in name or slant <= -10.0:
            clef = "backslant_lefty_clef"
        elif "beethoven" in name or "wild" in name or "frantic" in name:
            clef = "beethoven_wild_slash"
        elif "calligraph" in name or "quill" in name:
            clef = "calligraphic_broad_nib"
        elif "modern" in name or "angular" in name:
            clef = "sharp_angular_modern"
        elif "sketch" in name or "hurried" in name or ornate <= 0.18:
            clef = "hurried_single_loop"
        elif name:
            clef = CLEF_ARCHETYPES[_stable_index(f"clef:{name}:{era}", len(CLEF_ARCHETYPES))]
        else:
            clef = "classic_engraved"

    # 2. Key-signature / accidental archetype
    keysig = st.get("keysig_archetype")
    if keysig not in KEYSIG_ARCHETYPES:
        if "baroque" in name or "bach" in name or "baroque" in era:
            keysig = "baroque_vertical_flat"
        elif "realbook" in name or "jazz" in name or "marker" in name:
            keysig = "jazz_heavy_marker_acc"
        elif "calligraph" in name or "quill" in name:
            keysig = "calligraphic_nib_acc"
        elif "chopin" in name or "hairline" in name or stroke_scale <= 0.65:
            keysig = "hairline_needle_acc"
        elif "beethoven" in name or "sketch" in name or "hurried" in name:
            keysig = "hurried_x_sharp"
        elif stroke_scale >= 1.45:
            keysig = "jazz_heavy_marker_acc"
        elif "cram" in name or float(st.get("end_bar_cram", 0.0)) >= 0.5:
            keysig = "crammed_tight_cluster"
        elif name:
            keysig = KEYSIG_ARCHETYPES[_stable_index(f"keysig:{name}:{era}", len(KEYSIG_ARCHETYPES))]
        else:
            keysig = "classic_neat"

    # 3. Barline archetype
    barline = st.get("barline_archetype")
    if barline not in BARLINE_ARCHETYPES:
        if "lefty" in name or slant <= -10.0:
            barline = "back_slant_lefty"
        elif "baroque" in name or "bach" in name or "baroque" in era:
            barline = "tapered_quill_bar"
        elif "realbook" in name or "jazz" in name or "marker" in name:
            barline = "heavy_thick_marker"
        elif "tremor" in name or "shaky" in name or tremor >= 0.9:
            barline = "s_wobble_freehand"
        elif slant >= 11.0 or "beethoven" in name:
            barline = "steep_forward_slant"
        elif stroke_scale >= 1.45:
            barline = "heavy_thick_marker"
        elif "hairline" in name or "faint" in name or stroke_scale <= 0.65:
            barline = "hairline_faint"
        elif "quill" in name:
            barline = "tapered_quill_bar"
        elif overshoot >= 0.45:
            barline = "tall_overshoot"
        elif name:
            barline = BARLINE_ARCHETYPES[_stable_index(f"barline:{name}:{era}", len(BARLINE_ARCHETYPES))]
        else:
            barline = "crisp_vertical"

    return str(clef), str(keysig), str(barline)


# ---------------------------------------------------------------------------
# 1. CLEF ARCHETYPES (18 distinct styles x G, F, C, G8vb, G8va)
# ---------------------------------------------------------------------------


def _clef_style_params(archetype: str, style: dict) -> Tuple[dict, dict]:
    """Return (local_stroke_style, morph_cfg) for a clef archetype."""
    local = dict(style or {})
    base_slant = float(local.get("slant_deg", 0.0))
    cfg: Dict[str, Any] = {
        "sx": 1.0,
        "sy": 1.0,
        "slant": base_slant,
        "w_mult": 1.0,
        "jitter": 0.04,
        "smooth": True,
        "dot_r": 0.21,
        "dot_dx": 1.06,
        "mode": archetype,
    }

    if archetype == "classic_engraved":
        cfg.update(sx=1.0, sy=1.0, slant=0.0, w_mult=1.05, jitter=0.025)
    elif archetype == "baroque_ornate_scroll":
        cfg.update(sx=1.08, sy=1.10, slant=4.0, w_mult=1.08, dot_r=0.25)
        local["nib_angle_deg"] = 38.0
        local["nib_min_ratio"] = 0.24
    elif archetype == "hurried_single_loop":
        cfg.update(sx=0.95, sy=0.96, slant=10.0, w_mult=0.95, jitter=0.06)
    elif archetype == "tall_spindly_g":
        cfg.update(sx=0.68, sy=1.38, slant=2.0, w_mult=0.72, jitter=0.03)
    elif archetype == "stubby_compact":
        cfg.update(sx=1.32, sy=0.72, slant=0.0, w_mult=1.22, jitter=0.035)
    elif archetype == "minimal_s_slash":
        cfg.update(sx=0.92, sy=1.02, slant=8.0, w_mult=1.05, jitter=0.045)
    elif archetype == "figure_8_treble":
        cfg.update(sx=1.04, sy=1.05, slant=3.0, w_mult=1.05, jitter=0.03)
    elif archetype == "beethoven_wild_slash":
        cfg.update(sx=1.18, sy=1.22, slant=19.0, w_mult=1.28, jitter=0.075)
        local["wobble_px"] = max(float(local.get("wobble_px", 0.7)), 1.35)
    elif archetype == "chopin_delicate_spiral":
        cfg.update(sx=0.94, sy=1.14, slant=9.0, w_mult=0.58, jitter=0.025, dot_r=0.14)
        local["nib_min_ratio"] = 0.25
    elif archetype == "jazz_realbook_bold":
        cfg.update(sx=1.12, sy=0.98, slant=5.0, w_mult=1.72, jitter=0.04, dot_r=0.29)
        local["nib_min_ratio"] = 0.88
    elif archetype == "sharp_angular_modern":
        cfg.update(sx=1.02, sy=1.06, slant=0.0, w_mult=1.05, jitter=0.02, smooth=False)
    elif archetype == "detached_dot_bass":
        cfg.update(sx=0.96, sy=1.00, slant=2.0, w_mult=1.02, dot_r=0.33, dot_dx=1.48)
    elif archetype == "open_c_hook_bass":
        cfg.update(sx=1.16, sy=1.04, slant=4.0, w_mult=1.08, dot_dx=1.18)
    elif archetype == "double_bar_c_clef":
        cfg.update(sx=1.05, sy=1.04, slant=0.0, w_mult=1.12)
    elif archetype == "bracket_c_clef":
        cfg.update(sx=1.04, sy=1.04, slant=0.0, w_mult=1.10, smooth=False)
    elif archetype == "calligraphic_broad_nib":
        cfg.update(sx=1.06, sy=1.06, slant=6.0, w_mult=1.42, jitter=0.03)
        local["nib_angle_deg"] = 44.0
        local["nib_min_ratio"] = 0.13
    elif archetype == "tremor_shaky_clef":
        cfg.update(sx=1.00, sy=1.02, slant=2.0, w_mult=0.96, jitter=0.095)
        local["tremor_px"] = max(float(local.get("tremor_px", 0.25)), 1.85)
        local["wobble_px"] = max(float(local.get("wobble_px", 0.7)), 1.65)
    elif archetype == "backslant_lefty_clef":
        cfg.update(sx=1.02, sy=1.02, slant=-20.0, w_mult=1.06, jitter=0.04)
        local["nib_angle_deg"] = -36.0
    return local, cfg


def _draw_small_octave_8(
    canvas: StrokeCanvas,
    cx: float,
    cy: float,
    sp: float,
    color_rgb: RGBLike,
    local_style: dict,
    cfg: dict,
    rng: np.random.Generator,
) -> None:
    ctrl = np.array([
        [0.02, -0.58],
        [-0.30, -0.32],
        [0.02, -0.02],
        [0.32, 0.30],
        [0.00, 0.58],
        [-0.32, 0.30],
        [-0.02, -0.02],
        [0.30, -0.32],
        [0.02, -0.58],
    ], dtype=np.float64)
    pts = _transform_pts(
        ctrl, cx, cy, sp, rng,
        sx=0.85 * cfg["sx"], sy=0.85, slant_deg=cfg["slant"], jitter_sp=0.03,
    )
    path = _eval_catmull_rom(pts, 6) if cfg["smooth"] else pts
    w = max(0.85, sp * 0.13 * cfg["w_mult"])
    draw_calligraphic_polyline(canvas, path, color_rgb, w, local_style, kind="clef")


def _draw_g_clef_arch(
    canvas: StrokeCanvas,
    cx: float,
    cy: float,
    sp: float,
    archetype: str,
    color_rgb: RGBLike,
    local_style: dict,
    cfg: dict,
    rng: np.random.Generator,
) -> None:
    base_w = max(0.85, sp * float(local_style.get("clef_stroke_ratio", 0.19)) * cfg["w_mult"])
    sx, sy, slant, jit = cfg["sx"], cfg["sy"], cfg["slant"], cfg["jitter"]

    def _emit(ctrl: np.ndarray, width_mult: float = 1.0, smooth: Optional[bool] = None) -> np.ndarray:
        use_smooth = cfg["smooth"] if smooth is None else smooth
        pts = _transform_pts(ctrl, cx, cy, sp, rng, sx=sx, sy=sy, slant_deg=slant, jitter_sp=jit)
        path = _eval_catmull_rom(pts, 8) if (use_smooth and len(pts) >= 3) else pts
        draw_calligraphic_polyline(canvas, path, color_rgb, base_w * width_mult, local_style, kind="clef")
        return pts

    if archetype == "baroque_ornate_scroll":
        # Extra inner spiral turn + scrolled top loop + sweeping ornate bottom scroll with big finial dot.
        ctrl = np.array([
            [0.06, 0.12], [-0.14, 0.28], [0.02, 0.44], [0.46, 0.26], [0.64, -0.18],
            [0.18, -0.68], [-0.54, -0.35], [-0.66, 0.42], [-0.12, 0.96], [0.58, 0.56],
            [0.64, -0.45], [-0.24, -1.35], [0.48, -2.25], [0.22, -3.15], [-0.30, -2.50],
            [-0.12, -1.00], [0.06, 0.90], [0.12, 2.25], [-0.32, 2.75], [-0.72, 2.35], [-0.46, 2.02],
        ], dtype=np.float64)
        pts = _emit(ctrl, 1.05)
        r = max(1.3, sp * 0.26)
        draw_filled_blob(canvas, pts[-1, 0], pts[-1, 1], r, r, 0.0, color_rgb, local_style, hollow=False)

    elif archetype == "hurried_single_loop":
        # Rapid open single-stroke loop without bottom curl.
        ctrl = np.array([
            [0.25, 0.35], [0.52, -0.10], [-0.18, -0.58], [-0.52, 0.20], [0.02, 0.62],
            [0.44, -0.95], [0.12, -2.55], [-0.22, -1.85], [-0.04, 0.40], [0.04, 2.15],
        ], dtype=np.float64)
        _emit(ctrl, 0.96)

    elif archetype == "minimal_s_slash":
        # Minimalist modern S-curve crossed by a single bold vertical slash.
        s_ctrl = np.array([
            [0.45, -1.55], [-0.28, -1.30], [-0.36, -0.45], [0.46, 0.20],
            [0.36, 0.98], [-0.48, 0.85],
        ], dtype=np.float64)
        slash = np.array([[0.12, -2.55], [-0.08, 2.25]], dtype=np.float64)
        _emit(s_ctrl, 1.08)
        _emit(slash, 1.12, smooth=False)

    elif archetype == "figure_8_treble":
        # Two balanced stacked figure-8 loops + central vertical spine.
        fig8 = np.array([
            [0.00, -0.25], [0.52, -0.95], [0.46, -2.15], [0.00, -2.60],
            [-0.48, -2.05], [-0.42, -0.95], [0.02, -0.20],
            [0.58, 0.30], [0.54, 1.02], [0.00, 1.24],
            [-0.56, 0.98], [-0.54, 0.26], [0.00, -0.22],
        ], dtype=np.float64)
        spine = np.array([[0.02, -2.75], [-0.02, 2.30], [-0.44, 2.55]], dtype=np.float64)
        _emit(fig8, 1.02)
        _emit(spine, 1.05, smooth=True)

    elif archetype == "beethoven_wild_slash":
        # Aggressive sweeping Beethoven stroke with huge overshooting diagonal spine & spiky belly.
        loop = np.array([
            [0.18, 0.10], [0.72, -0.25], [0.58, 0.88], [-0.62, 0.72],
            [-0.74, -0.42], [0.55, -1.95], [0.16, -3.25], [-0.32, -2.10],
        ], dtype=np.float64)
        slash = np.array([[0.28, -3.35], [-0.02, 0.15], [-0.24, 2.95], [-0.72, 2.55]], dtype=np.float64)
        _emit(loop, 1.15)
        _emit(slash, 1.32, smooth=False)

    elif archetype == "chopin_delicate_spiral":
        # Graceful wispy multi-turn spiral with delicate long spire.
        ctrl = np.array([
            [0.00, 0.08], [0.34, -0.14], [0.48, 0.32], [0.08, 0.68], [-0.44, 0.44],
            [-0.56, -0.24], [-0.14, -1.25], [0.38, -2.35], [0.14, -3.10],
            [-0.20, -2.35], [-0.06, -0.60], [0.04, 1.35], [0.08, 2.35], [-0.36, 2.60],
        ], dtype=np.float64)
        _emit(ctrl, 0.95)

    elif archetype == "sharp_angular_modern":
        # Faceted geometric/polygonal G-clef made of crisp straight segments.
        poly = np.array([
            [0.05, 0.18], [0.46, -0.15], [0.58, 0.52], [0.00, 0.95], [-0.62, 0.48],
            [-0.58, -0.42], [0.44, -1.85], [0.14, -2.90], [-0.24, -2.25],
            [-0.04, 0.25], [0.06, 2.15], [-0.46, 2.48],
        ], dtype=np.float64)
        _emit(poly, 1.05, smooth=False)

    elif archetype == "detached_dot_bass":
        # G-clef with broken open inner curl and prominent detached finial dots.
        body = np.array([
            [0.48, -0.15], [0.62, 0.45], [0.10, 0.92], [-0.58, 0.56], [-0.62, -0.25],
            [-0.12, -1.25], [0.42, -2.15], [0.16, -2.92], [-0.24, -2.25],
            [-0.04, 0.25], [0.06, 2.05],
        ], dtype=np.float64)
        _emit(body, 1.02)
        r = max(1.4, sp * cfg["dot_r"])
        # Detached center G-dot and bottom finial dot
        p_center = _transform_pts(np.array([[0.02, 0.14]]), cx, cy, sp, rng, sx=sx, sy=sy, slant_deg=slant, jitter_sp=0.02)[0]
        p_bot = _transform_pts(np.array([[-0.52, 2.35]]), cx, cy, sp, rng, sx=sx, sy=sy, slant_deg=slant, jitter_sp=0.02)[0]
        draw_filled_blob(canvas, p_center[0], p_center[1], r * 0.9, r * 0.9, 0.0, color_rgb, local_style, hollow=False)
        draw_filled_blob(canvas, p_bot[0], p_bot[1], r, r, 0.0, color_rgb, local_style, hollow=False)

    elif archetype == "open_c_hook_bass":
        # Wide open C-belly + sweeping open upper hook.
        belly = np.array([
            [0.62, -0.45], [-0.15, -0.68], [-0.74, 0.05], [-0.36, 0.95], [0.58, 0.82],
        ], dtype=np.float64)
        stem = np.array([
            [0.42, -2.15], [0.16, -2.92], [-0.26, -2.20], [-0.04, 0.20], [0.06, 2.15], [-0.42, 2.48],
        ], dtype=np.float64)
        _emit(belly, 1.12)
        _emit(stem, 1.02)

    elif archetype == "double_bar_c_clef":
        # G-clef with bold doubled parallel vertical spine rails.
        curl = np.array([
            [0.00, 0.12], [0.46, -0.18], [0.62, 0.42], [0.16, 0.90],
            [-0.52, 0.62], [-0.66, -0.18], [-0.14, -1.25], [0.44, -2.15], [0.18, -2.85],
        ], dtype=np.float64)
        rail1 = np.array([[-0.18, -2.65], [-0.12, 2.30]], dtype=np.float64)
        rail2 = np.array([[0.12, -2.65], [0.18, 2.30]], dtype=np.float64)
        _emit(curl, 1.0)
        _emit(rail1, 1.35, smooth=False)
        _emit(rail2, 0.82, smooth=False)

    elif archetype == "bracket_c_clef":
        # Architectural squared-off bracket top & bottom serifs around G-clef.
        spine = np.array([
            [-0.45, -2.65], [0.35, -2.65], [0.35, -1.95], [-0.05, -0.95],
            [-0.05, 2.25], [-0.55, 2.25], [-0.55, 1.85],
        ], dtype=np.float64)
        belly = np.array([
            [0.00, 0.05], [0.55, 0.05], [0.55, 0.88], [-0.58, 0.88], [-0.58, -0.35], [0.38, -0.35],
        ], dtype=np.float64)
        _emit(spine, 1.12, smooth=False)
        _emit(belly, 1.05, smooth=False)

    else:
        # Classic / tall_spindly / stubby / jazz / calligraphic / tremor / backslant
        curl = np.array([
            [-0.05, 0.12],
            [0.44, -0.18],
            [0.64, 0.36],
            [0.24, 0.88],
            [-0.48, 0.68],
            [-0.68, -0.12],
            [-0.20, -1.12],
            [0.44, -2.05],
            [0.22, -2.92],
            [-0.24, -2.32],
            [-0.10, -0.95],
            [0.04, 0.85],
            [0.10, 2.08],
            [-0.26, 2.48],
            [-0.56, 2.16],
        ], dtype=np.float64)
        pts = _emit(curl, 1.06)
        if archetype in ("classic_engraved", "calligraphic_broad_nib", "jazz_realbook_bold"):
            r = max(1.1, sp * cfg["dot_r"])
            draw_filled_blob(canvas, pts[-1, 0], pts[-1, 1], r, r, 0.0, color_rgb, local_style, hollow=False)


def _draw_f_clef_arch(
    canvas: StrokeCanvas,
    cx: float,
    cy: float,
    sp: float,
    archetype: str,
    color_rgb: RGBLike,
    local_style: dict,
    cfg: dict,
    rng: np.random.Generator,
) -> None:
    base_w = max(0.85, sp * float(local_style.get("clef_stroke_ratio", 0.19)) * cfg["w_mult"])
    sx, sy, slant, jit = cfg["sx"], cfg["sy"], cfg["slant"], cfg["jitter"]

    def _emit(ctrl: np.ndarray, width_mult: float = 1.12, smooth: Optional[bool] = None) -> np.ndarray:
        use_smooth = cfg["smooth"] if smooth is None else smooth
        pts = _transform_pts(ctrl, cx, cy, sp, rng, sx=sx, sy=sy, slant_deg=slant, jitter_sp=jit)
        path = _eval_catmull_rom(pts, 8) if (use_smooth and len(pts) >= 3) else pts
        draw_calligraphic_polyline(canvas, path, color_rgb, base_w * width_mult, local_style, kind="clef")
        return pts

    if archetype == "baroque_ornate_scroll":
        body = np.array([
            [-0.36, 0.14], [-0.58, -0.10], [-0.44, -0.84], [0.28, -1.04],
            [0.68, -0.35], [0.46, 0.72], [-0.14, 1.65], [-0.74, 2.25], [-0.92, 1.98],
        ], dtype=np.float64)
        pts = _emit(body, 1.15)
        r0 = max(1.4, sp * 0.30)
        draw_filled_blob(canvas, pts[0, 0], pts[0, 1], r0, r0, 0.0, color_rgb, local_style, hollow=False)

    elif archetype == "hurried_single_loop":
        body = np.array([[-0.30, -0.78], [0.32, -0.86], [0.55, -0.22], [0.22, 0.85], [-0.52, 1.85]], dtype=np.float64)
        _emit(body, 1.0)

    elif archetype == "minimal_s_slash":
        body = np.array([[-0.22, -0.85], [0.45, -0.45], [-0.45, 1.75]], dtype=np.float64)
        _emit(body, 1.08)

    elif archetype == "figure_8_treble":
        # Looped figure-8 head descending into bass tail.
        body = np.array([
            [-0.35, -0.10], [-0.62, -0.58], [-0.18, -0.98], [0.48, -0.72],
            [0.52, -0.10], [-0.08, 0.15], [0.46, 0.55], [0.16, 1.40], [-0.58, 2.02],
        ], dtype=np.float64)
        _emit(body, 1.10)

    elif archetype == "beethoven_wild_slash":
        body = np.array([
            [-0.45, -0.45], [0.42, -1.08], [0.76, -0.18], [0.18, 1.18], [-0.78, 2.38],
        ], dtype=np.float64)
        _emit(body, 1.32)

    elif archetype == "sharp_angular_modern":
        body = np.array([
            [-0.50, -0.10], [-0.50, -0.92], [0.54, -0.92], [0.64, -0.15],
            [0.24, 0.95], [-0.65, 2.05],
        ], dtype=np.float64)
        _emit(body, 1.15, smooth=False)

    elif archetype == "detached_dot_bass":
        body = np.array([
            [-0.18, -0.86], [0.42, -0.92], [0.64, -0.22], [0.36, 0.78], [-0.52, 1.95],
        ], dtype=np.float64)
        _emit(body, 1.10)
        # Detached starting head blob sitting well to the left
        p0 = _transform_pts(np.array([[-0.68, -0.05]]), cx, cy, sp, rng, sx=sx, sy=sy, slant_deg=slant, jitter_sp=0.02)[0]
        r0 = max(1.5, sp * 0.34)
        draw_filled_blob(canvas, p0[0], p0[1], r0, r0, 0.0, color_rgb, local_style, hollow=False)

    elif archetype == "open_c_hook_bass":
        # Wide open C-shaped arch sweeping outward.
        body = np.array([
            [-0.62, -0.82], [0.38, -1.02], [0.82, -0.18], [0.62, 0.92], [-0.32, 1.98],
        ], dtype=np.float64)
        _emit(body, 1.16)

    elif archetype == "double_bar_c_clef":
        # Bass clef with doubled vertical left stem rail + sweeping hook.
        rail = np.array([[-0.56, -0.95], [-0.56, 1.95]], dtype=np.float64)
        body = np.array([
            [-0.40, -0.85], [0.36, -0.94], [0.66, -0.22], [0.34, 0.85], [-0.42, 1.95],
        ], dtype=np.float64)
        _emit(rail, 1.30, smooth=False)
        _emit(body, 1.05)

    elif archetype == "bracket_c_clef":
        # Squared architectural bracket bass clef.
        body = np.array([
            [-0.52, -0.15], [-0.52, -0.96], [0.58, -0.96], [0.58, 0.65], [-0.55, 2.02],
        ], dtype=np.float64)
        _emit(body, 1.15, smooth=False)

    else:
        body = np.array([
            [-0.48, -0.05],
            [-0.36, -0.78],
            [0.28, -0.96],
            [0.62, -0.35],
            [0.42, 0.65],
            [-0.18, 1.55],
            [-0.65, 2.08],
        ], dtype=np.float64)
        pts = _emit(body, 1.16)
        r0 = max(1.2, sp * (0.26 if archetype != "chopin_delicate_spiral" else 0.16))
        draw_filled_blob(canvas, pts[0, 0], pts[0, 1], r0, r0 * 0.95, 0.0, color_rgb, local_style, hollow=False)

    # Two F-clef dots straddling cy
    dot_x_norm = cfg["dot_dx"]
    dr = max(1.0, sp * cfg["dot_r"])
    for sign in (-1.0, 1.0):
        dp = _transform_pts(
            np.array([[dot_x_norm, sign * 0.48]]), cx, cy, sp, rng,
            sx=sx, sy=sy, slant_deg=slant, jitter_sp=0.03,
        )[0]
        if archetype == "sharp_angular_modern":
            # Diamond dot
            d_poly = np.array([
                [dp[0] - dr, dp[1]], [dp[0], dp[1] - dr],
                [dp[0] + dr, dp[1]], [dp[0], dp[1] + dr], [dp[0] - dr, dp[1]],
            ], dtype=np.float64)
            draw_calligraphic_polyline(canvas, d_poly, color_rgb, max(0.8, dr * 0.8), local_style, kind="clef")
        else:
            draw_filled_blob(canvas, dp[0], dp[1], dr, dr * 0.96, 0.0, color_rgb, local_style, hollow=False)


def _draw_c_clef_arch(
    canvas: StrokeCanvas,
    cx: float,
    cy: float,
    sp: float,
    archetype: str,
    color_rgb: RGBLike,
    local_style: dict,
    cfg: dict,
    rng: np.random.Generator,
) -> None:
    base_w = max(0.85, sp * float(local_style.get("clef_stroke_ratio", 0.19)) * cfg["w_mult"])
    sx, sy, slant, jit = cfg["sx"], cfg["sy"], cfg["slant"], cfg["jitter"]

    def _emit(ctrl: np.ndarray, width_mult: float = 1.1, smooth: Optional[bool] = None) -> np.ndarray:
        use_smooth = cfg["smooth"] if smooth is None else smooth
        pts = _transform_pts(ctrl, cx, cy, sp, rng, sx=sx, sy=sy, slant_deg=slant, jitter_sp=jit)
        path = _eval_catmull_rom(pts, 8) if (use_smooth and len(pts) >= 3) else pts
        draw_calligraphic_polyline(canvas, path, color_rgb, base_w * width_mult, local_style, kind="clef")
        return pts

    if archetype == "bracket_c_clef":
        # Square architectural `[` outer bar + angular `>` central pointer.
        outer = np.array([[-0.22, -1.88], [-0.66, -1.88], [-0.66, 1.88], [-0.22, 1.88]], dtype=np.float64)
        upper = np.array([[-0.22, -1.55], [0.58, -1.55], [0.58, -0.45], [-0.15, 0.00]], dtype=np.float64)
        lower = np.array([[-0.15, 0.00], [0.58, 0.45], [0.58, 1.55], [-0.22, 1.55]], dtype=np.float64)
        _emit(outer, 1.45, smooth=False)
        _emit(upper, 1.12, smooth=False)
        _emit(lower, 1.12, smooth=False)
        return

    if archetype == "double_bar_c_clef":
        # Heavy twin parallel vertical bars with cross-ties + compact twin curls.
        bar1 = np.array([[-0.74, -1.95], [-0.74, 1.95]], dtype=np.float64)
        bar2 = np.array([[-0.30, -1.95], [-0.30, 1.95]], dtype=np.float64)
        tie_t = np.array([[-0.82, -1.88], [-0.22, -1.88]], dtype=np.float64)
        tie_b = np.array([[-0.82, 1.88], [-0.22, 1.88]], dtype=np.float64)
        upper = np.array([[-0.12, -1.75], [0.58, -1.55], [0.62, -0.75], [-0.10, -0.02]], dtype=np.float64)
        lower = np.array([[-0.10, 0.02], [0.62, 0.75], [0.58, 1.55], [-0.12, 1.75]], dtype=np.float64)
        _emit(bar1, 1.85, smooth=False)
        _emit(bar2, 1.45, smooth=False)
        _emit(tie_t, 1.0, smooth=False)
        _emit(tie_b, 1.0, smooth=False)
        _emit(upper, 1.12)
        _emit(lower, 1.12)
        return

    if archetype == "minimal_s_slash":
        # Minimalist single bar + quick '3' stroke.
        bar = np.array([[-0.54, -1.80], [-0.54, 1.80]], dtype=np.float64)
        three = np.array([[-0.12, -1.55], [0.52, -1.15], [-0.10, 0.0], [0.52, 1.15], [-0.12, 1.55]], dtype=np.float64)
        _emit(bar, 1.25, smooth=False)
        _emit(three, 1.05)
        return

    if archetype == "hurried_single_loop":
        # Single vertical bar + fast continuous cursive 3.
        bar = np.array([[-0.56, -1.75], [-0.56, 1.75]], dtype=np.float64)
        three = np.array([[-0.15, -1.60], [0.48, -1.40], [0.46, -0.58], [-0.16, 0.0], [0.48, 0.62], [0.44, 1.42], [-0.18, 1.62]], dtype=np.float64)
        _emit(bar, 1.35, smooth=False)
        _emit(three, 1.02)
        return

    if archetype == "open_c_hook_bass":
        # Single left rail + wide sweeping open C-arms.
        bar = np.array([[-0.65, -1.85], [-0.65, 1.85]], dtype=np.float64)
        upper = np.array([[0.72, -1.75], [0.15, -1.80], [-0.30, -0.95], [0.22, -0.08]], dtype=np.float64)
        lower = np.array([[0.22, 0.08], [-0.30, 0.95], [0.15, 1.80], [0.72, 1.75]], dtype=np.float64)
        _emit(bar, 1.55, smooth=False)
        _emit(upper, 1.12)
        _emit(lower, 1.12)
        return

    if archetype == "sharp_angular_modern":
        # K-like faceted modernist C-clef.
        bar = np.array([[-0.60, -1.85], [-0.60, 1.85]], dtype=np.float64)
        upper = np.array([[0.58, -1.80], [-0.28, 0.0], [0.58, 1.80]], dtype=np.float64)
        _emit(bar, 1.60, smooth=False)
        _emit(upper, 1.18, smooth=False)
        return

    if archetype == "figure_8_treble":
        # Left bar + figure-8 double loop.
        bar = np.array([[-0.62, -1.85], [-0.62, 1.85]], dtype=np.float64)
        fig8 = np.array([
            [-0.12, 0.0], [0.55, -0.55], [0.48, -1.55], [-0.12, -1.75],
            [-0.30, -0.95], [-0.10, 0.0], [0.55, 0.55], [0.48, 1.55],
            [-0.12, 1.75], [-0.30, 0.95], [-0.12, 0.0],
        ], dtype=np.float64)
        _emit(bar, 1.55, smooth=False)
        _emit(fig8, 1.05)
        return

    # Classic / baroque / tall / stubby / beethoven / chopin / jazz / detached_dot / calligraphic / tremor / backslant
    bar1 = np.array([[-0.64, -1.85], [-0.64, 1.85]], dtype=np.float64)
    bar2 = np.array([[-0.28, -1.82], [-0.28, 1.82]], dtype=np.float64)
    _emit(bar1, 1.68, smooth=False)
    _emit(bar2, 0.86, smooth=False)

    upper = np.array([[-0.12, -1.78], [0.54, -1.68], [0.64, -0.82], [-0.08, -0.04]], dtype=np.float64)
    lower = np.array([[-0.08, 0.04], [0.64, 0.82], [0.54, 1.68], [-0.12, 1.78]], dtype=np.float64)
    u_pts = _emit(upper, 1.15)
    l_pts = _emit(lower, 1.15)

    if archetype == "baroque_ornate_scroll":
        r = max(1.2, sp * 0.22)
        draw_filled_blob(canvas, u_pts[0, 0], u_pts[0, 1], r, r, 0.0, color_rgb, local_style, hollow=False)
        draw_filled_blob(canvas, l_pts[-1, 0], l_pts[-1, 1], r, r, 0.0, color_rgb, local_style, hollow=False)
    elif archetype == "detached_dot_bass":
        r = max(1.4, sp * cfg["dot_r"])
        for sign in (-1.0, 1.0):
            dp = _transform_pts(np.array([[1.08, sign * 0.55]]), cx, cy, sp, rng, sx=sx, sy=sy, slant_deg=slant, jitter_sp=0.02)[0]
            draw_filled_blob(canvas, dp[0], dp[1], r, r, 0.0, color_rgb, local_style, hollow=False)


def draw_clef_variant(
    engine: Any,
    x: float,
    y_center: float,
    sp: float,
    clef_kind: str,
    archetype: str,
    style: dict,
    rng: np.random.Generator,
    color_rgb: Optional[RGBLike] = None,
) -> None:
    """Draw one of 18 visually distinct handwritten clef archetypes.

    Supports clef_kind in {'G', 'F', 'C', 'G8vb', 'G8va'} (plus 'G-2', 'F-4', 'C-3', 'C-4').
    """
    canvas = _resolve_canvas(engine)
    ink = _resolve_ink_rgb(style, color_rgb)
    arch = archetype if archetype in CLEF_ARCHETYPES else "classic_engraved"
    base, octave_shift = _normalize_clef_kind(clef_kind)
    local_style, cfg = _clef_style_params(arch, style)

    if base == "G":
        # G-clef curl centers on line 2 (y_center + 1.0 * sp if y_center is staff middle line 3).
        _draw_g_clef_arch(canvas, x, y_center, sp, arch, ink, local_style, cfg, rng)
    elif base == "F":
        _draw_f_clef_arch(canvas, x, y_center, sp, arch, ink, local_style, cfg, rng)
    else:
        _draw_c_clef_arch(canvas, x, y_center, sp, arch, ink, local_style, cfg, rng)

    if octave_shift != 0:
        y8 = y_center + (sp * 3.10 * cfg["sy"] if octave_shift < 0 else -sp * 3.25 * cfg["sy"])
        _draw_small_octave_8(canvas, x, y8, sp, ink, local_style, cfg, rng)


# ---------------------------------------------------------------------------
# 2. KEY SIGNATURE & ACCIDENTAL ARCHETYPES (18 distinct styles)
# ---------------------------------------------------------------------------


def _keysig_params(archetype: str, style: dict) -> Tuple[dict, dict]:
    """Return (local_stroke_style, cfg) for a key-signature/accidental archetype."""
    local = dict(style or {})
    base_slant = float(local.get("slant_deg", 0.0))
    cfg: Dict[str, Any] = {
        "sx": 1.0,
        "sy": 1.0,
        "slant": base_slant + 3.0,
        "w_mult": 1.0,
        "spacing_sp": 0.96,
        "y_drift_sp": 0.0,
        "stagger_sp": 0.0,
        "double_pass": False,
        "jagged": False,
        "open_loop": False,
        "pointed": False,
        "hurried_x": False,
        "baroque_upright": False,
        "crossbar_mult": 1.60,
        "mode": archetype,
    }

    if archetype == "classic_neat":
        cfg.update(sx=1.0, sy=1.0, slant=2.0, w_mult=1.0, spacing_sp=0.96)
    elif archetype == "tall_narrow_sharp":
        cfg.update(sx=0.66, sy=1.44, slant=2.0, w_mult=0.92, spacing_sp=0.80)
    elif archetype == "wide_slanted_hash":
        cfg.update(sx=1.38, sy=0.96, slant=20.0, w_mult=1.08, spacing_sp=1.18)
    elif archetype == "hurried_x_sharp":
        cfg.update(sx=1.05, sy=1.02, slant=12.0, w_mult=1.06, spacing_sp=0.94, hurried_x=True)
    elif archetype == "open_loop_flat":
        cfg.update(sx=1.06, sy=1.04, slant=4.0, w_mult=1.00, spacing_sp=1.04, open_loop=True)
    elif archetype == "pointed_teardrop_flat":
        cfg.update(sx=1.02, sy=1.08, slant=5.0, w_mult=1.04, spacing_sp=0.98, pointed=True)
    elif archetype == "tiny_pinprick_acc":
        cfg.update(sx=0.62, sy=0.64, slant=2.0, w_mult=0.70, spacing_sp=0.72)
    elif archetype == "oversized_bold_acc":
        cfg.update(sx=1.36, sy=1.36, slant=3.0, w_mult=1.48, spacing_sp=1.28)
    elif archetype == "crammed_tight_cluster":
        cfg.update(sx=0.90, sy=1.00, slant=6.0, w_mult=1.02, spacing_sp=0.56, stagger_sp=0.14)
    elif archetype == "wide_spread_keysig":
        cfg.update(sx=1.02, sy=1.02, slant=2.0, w_mult=1.00, spacing_sp=1.58)
    elif archetype == "ascending_drift_keysig":
        cfg.update(sx=0.98, sy=1.00, slant=6.0, w_mult=1.00, spacing_sp=0.96, y_drift_sp=-0.16)
    elif archetype == "descending_drift_keysig":
        cfg.update(sx=0.98, sy=1.00, slant=-3.0, w_mult=1.00, spacing_sp=0.96, y_drift_sp=0.16)
    elif archetype == "double_stroke_sketch":
        cfg.update(sx=1.04, sy=1.04, slant=5.0, w_mult=0.88, spacing_sp=1.02, double_pass=True)
    elif archetype == "calligraphic_nib_acc":
        cfg.update(sx=1.06, sy=1.08, slant=6.0, w_mult=1.28, spacing_sp=1.04, crossbar_mult=2.25)
        local["nib_angle_deg"] = 42.0
        local["nib_min_ratio"] = 0.14
    elif archetype == "baroque_vertical_flat":
        cfg.update(sx=0.90, sy=1.16, slant=0.0, w_mult=0.95, spacing_sp=0.94, baroque_upright=True)
        local["nib_angle_deg"] = 35.0
        local["nib_min_ratio"] = 0.28
    elif archetype == "jazz_heavy_marker_acc":
        cfg.update(sx=1.14, sy=1.04, slant=7.0, w_mult=1.82, spacing_sp=1.14, crossbar_mult=1.30)
        local["nib_min_ratio"] = 0.88
    elif archetype == "hairline_needle_acc":
        cfg.update(sx=0.92, sy=1.14, slant=3.0, w_mult=0.46, spacing_sp=0.90, crossbar_mult=1.25)
        local["stroke_opacity"] = min(float(local.get("stroke_opacity", 0.92)), 0.78)
    elif archetype == "jagged_zigzag_acc":
        cfg.update(sx=1.06, sy=1.04, slant=4.0, w_mult=1.06, spacing_sp=1.02, jagged=True)
    return local, cfg


def _emit_acc_stroke(
    canvas: StrokeCanvas,
    pts: np.ndarray,
    color_rgb: RGBLike,
    width: float,
    local_style: dict,
    cfg: dict,
    sp: float,
    rng: np.random.Generator,
) -> None:
    if cfg["jagged"] and len(pts) >= 2:
        # Insert crisp sawtooth mid-kinks along segments for jagged zigzag morphology.
        out = [pts[0]]
        for i in range(len(pts) - 1):
            p0, p1 = pts[i], pts[i + 1]
            d = p1 - p0
            seg_len = float(math.hypot(d[0], d[1]))
            if seg_len > sp * 0.45:
                nx, ny = -d[1] / seg_len, d[0] / seg_len
                amp = sp * 0.11
                out.append(p0 + 0.35 * d + np.array([nx * amp, ny * amp]))
                out.append(p0 + 0.68 * d - np.array([nx * amp, ny * amp]))
            out.append(p1)
        pts = np.asarray(out, dtype=np.float64)

    draw_calligraphic_polyline(canvas, pts, color_rgb, width, local_style, kind="accidental")
    if cfg["double_pass"]:
        off = rng.normal(0.0, 0.065 * sp, size=(1, 2))
        draw_calligraphic_polyline(canvas, pts + off, color_rgb, width * 0.82, local_style, kind="accidental")


def draw_accidental_variant(
    engine: Any,
    x: float,
    y: float,
    sp: float,
    acc_kind: str,
    archetype: str,
    style: dict,
    rng: np.random.Generator,
    color_rgb: Optional[RGBLike] = None,
) -> None:
    """Draw one of 18 visually distinct handwritten accidental variants.

    Supports acc_kind in {'sharp', 'flat', 'natural', 'double_sharp', 'double_flat'}.
    """
    canvas = _resolve_canvas(engine)
    ink = _resolve_ink_rgb(style, color_rgb)
    arch = archetype if archetype in KEYSIG_ARCHETYPES else "classic_neat"
    kind = _normalize_acc_kind(acc_kind)
    local_style, cfg = _keysig_params(arch, style)

    sx, sy = cfg["sx"], cfg["sy"]
    slant = cfg["slant"]
    base_w = max(0.55, sp * float(local_style.get("acc_stroke_ratio", 0.145)) * cfg["w_mult"])

    def _xform(arr: np.ndarray, jit: float = 0.03) -> np.ndarray:
        return _transform_pts(arr, x, y, sp, rng, sx=sx, sy=sy, slant_deg=slant, jitter_sp=jit)

    if kind == "double_sharp":
        r = 0.46
        s1 = _xform(np.array([[-r, -r], [r, r]], dtype=np.float64))
        s2 = _xform(np.array([[-r, r], [r, -r]], dtype=np.float64))
        _emit_acc_stroke(canvas, s1, ink, base_w * 1.10, local_style, cfg, sp, rng)
        _emit_acc_stroke(canvas, s2, ink, base_w * 1.10, local_style, cfg, sp, rng)
        dr = max(0.9, sp * 0.14 * min(sx, sy) * (1.2 if cfg["w_mult"] > 1.3 else 1.0))
        for cx_s in (-r * 0.85, r * 0.85):
            for cy_s in (-r * 0.85, r * 0.85):
                pt = _xform(np.array([[cx_s, cy_s]], dtype=np.float64), jit=0.015)[0]
                draw_filled_blob(canvas, pt[0], pt[1], dr, dr, 0.0, ink, local_style, hollow=False)
        return

    if kind == "double_flat":
        dx = sp * 0.34 * sx
        draw_accidental_variant(canvas, x - dx, y, sp * 0.96, "flat", arch, style, rng, color_rgb=ink)
        draw_accidental_variant(canvas, x + dx * 0.72, y, sp * 0.96, "flat", arch, style, rng, color_rgb=ink)
        return

    if kind == "sharp":
        if cfg["hurried_x"]:
            # Fast X-crossed hurried sharp
            v1 = _xform(np.array([[-0.38, -1.12], [0.22, 1.14]], dtype=np.float64))
            v2 = _xform(np.array([[0.36, -1.18], [-0.24, 1.08]], dtype=np.float64))
            h1 = _xform(np.array([[-0.60, -0.18], [0.62, -0.52]], dtype=np.float64))
            h2 = _xform(np.array([[-0.62, 0.50], [0.60, 0.16]], dtype=np.float64))
            _emit_acc_stroke(canvas, v1, ink, base_w * 0.96, local_style, cfg, sp, rng)
            _emit_acc_stroke(canvas, v2, ink, base_w * 0.96, local_style, cfg, sp, rng)
            _emit_acc_stroke(canvas, h1, ink, base_w * 1.45, local_style, cfg, sp, rng)
            _emit_acc_stroke(canvas, h2, ink, base_w * 1.45, local_style, cfg, sp, rng)
            return

        dx_v = 0.25 if not cfg["open_loop"] else 0.32
        h_v = 1.18
        for sign in (-1.0, 1.0):
            seg = _xform(np.array([[sign * dx_v, -h_v - sign * 0.08], [sign * dx_v, h_v - sign * 0.08]], dtype=np.float64))
            _emit_acc_stroke(canvas, seg, ink, base_w * 0.92, local_style, cfg, sp, rng)
        w_h = 0.58 if not cfg["open_loop"] else 0.44
        dy_h = 0.36
        tilt = 0.24 if not cfg["baroque_upright"] else 0.10
        for sign in (-1.0, 1.0):
            bar = _xform(np.array([[-w_h, sign * dy_h + tilt * 0.5], [w_h, sign * dy_h - tilt * 0.5]], dtype=np.float64))
            _emit_acc_stroke(canvas, bar, ink, base_w * cfg["crossbar_mult"], local_style, cfg, sp, rng)
            if cfg["baroque_upright"]:
                # Decorative tiny serif dots at crossbar tips
                r = max(0.8, sp * 0.09)
                draw_filled_blob(canvas, bar[0, 0], bar[0, 1], r, r, 0.0, ink, local_style, hollow=False)
                draw_filled_blob(canvas, bar[-1, 0], bar[-1, 1], r, r, 0.0, ink, local_style, hollow=False)
        return

    if kind == "flat":
        stem_top = -1.68 if not cfg["baroque_upright"] else -1.85
        stem_bot = 0.62
        stem = _xform(np.array([[-0.24, stem_top], [-0.24, stem_bot]], dtype=np.float64))
        _emit_acc_stroke(canvas, stem, ink, base_w * 1.04, local_style, cfg, sp, rng)

        if cfg["open_loop"]:
            # Deliberately unclosed open-C belly loop
            belly = _xform(np.array([
                [-0.14, -0.26], [0.46, -0.44], [0.64, 0.04], [0.22, 0.46],
            ], dtype=np.float64))
            _emit_acc_stroke(canvas, _eval_catmull_rom(belly, 7), ink, base_w * 1.12, local_style, cfg, sp, rng)
        elif cfg["pointed"]:
            # Sharp gothic pointed teardrop belly
            belly = _xform(np.array([
                [-0.22, -0.24], [0.66, -0.48], [0.46, 0.24], [-0.24, stem_bot],
            ], dtype=np.float64))
            _emit_acc_stroke(canvas, belly, ink, base_w * 1.14, local_style, cfg, sp, rng)
        elif cfg["hurried_x"]:
            # Fast single-slash check belly
            belly = _xform(np.array([
                [-0.20, -0.18], [0.52, -0.32], [-0.22, stem_bot],
            ], dtype=np.float64))
            _emit_acc_stroke(canvas, belly, ink, base_w * 1.10, local_style, cfg, sp, rng)
        elif cfg["baroque_upright"]:
            # Small high-set baroque round loop
            p0 = _xform(np.array([[-0.24, -0.30]]))[0]
            p1 = _xform(np.array([[0.46, -0.58]]))[0]
            p2 = _xform(np.array([[0.44, 0.22]]))[0]
            p3 = _xform(np.array([[-0.24, 0.48]]))[0]
            _emit_acc_stroke(canvas, _eval_cubic(p0, p1, p2, p3, 14), ink, base_w * 1.12, local_style, cfg, sp, rng)
        else:
            p0 = _xform(np.array([[-0.24, -0.24]]))[0]
            p1 = _xform(np.array([[0.64, -0.56]]))[0]
            p2 = _xform(np.array([[0.60, 0.36]]))[0]
            p3 = _xform(np.array([[-0.22, stem_bot]]))[0]
            curve = _eval_cubic(p0, p1, p2, p3, 14) if not cfg["jagged"] else np.vstack((p0, p1, p2, p3))
            _emit_acc_stroke(canvas, curve, ink, base_w * 1.16, local_style, cfg, sp, rng)
        return

    # "natural"
    dx = 0.28
    dy_box = 0.38
    tilt = 0.20 if not cfg["baroque_upright"] else 0.04
    gap = 0.14 if cfg["open_loop"] else 0.0
    s1 = _xform(np.array([
        [-dx, -1.32],
        [-dx, dy_box + tilt * 0.4],
        [dx - gap, dy_box - tilt * 0.6],
    ], dtype=np.float64))
    s2 = _xform(np.array([
        [-dx + gap, -dy_box + tilt * 0.5],
        [dx, -dy_box - tilt * 0.5],
        [dx, 1.32],
    ], dtype=np.float64))
    _emit_acc_stroke(canvas, s1, ink, base_w * 1.10, local_style, cfg, sp, rng)
    _emit_acc_stroke(canvas, s2, ink, base_w * 1.10, local_style, cfg, sp, rng)


def _step_to_y(y_lines: Any, x: float, step: int, sp: float) -> float:
    """Resolve staff step (0 = bottom line, 8 = top line) to pixel y."""
    if callable(y_lines):
        # Supports renderer's y_fn(x, line_idx) where line_idx in {-2,-1,0,1,2}
        # line_idx = 2 is bottom line (step 0), line_idx = -2 is top line (step 8).
        yc = float(y_lines(x, 0))
        return yc - (step - 4) * (sp * 0.5)
    if isinstance(y_lines, (int, float, np.floating)):
        return float(y_lines) - (step - 4) * (sp * 0.5)
    arr = [float(v) for v in y_lines]
    if not arr:
        return 0.0
    # Determine bottom line y (largest y in image coordinates).
    y_bot = max(arr[0], arr[-1])
    if len(arr) >= 2:
        step_px = abs(arr[-1] - arr[0]) / max(1, (len(arr) - 1) * 2)
    else:
        step_px = sp * 0.5
    return y_bot - step * step_px


def draw_keysig_variant(
    engine: Any,
    x_start: float,
    y_lines: Any,
    sp: float,
    fifths: int,
    clef_kind: str,
    archetype: str,
    style: dict,
    rng: np.random.Generator,
    color_rgb: Optional[RGBLike] = None,
) -> float:
    """Draw a key signature (fifths in -7..+7) using one of 18 archetypes.

    Returns the ending x coordinate after the final accidental.
    """
    n_acc = int(np.clip(fifths, -7, 7))
    if n_acc == 0:
        return float(x_start)

    canvas = _resolve_canvas(engine)
    ink = _resolve_ink_rgb(style, color_rgb)
    arch = archetype if archetype in KEYSIG_ARCHETYPES else "classic_neat"
    _, cfg = _keysig_params(arch, style)
    base, _ = _normalize_clef_kind(clef_kind)

    acc_kind = "sharp" if n_acc > 0 else "flat"
    count = abs(n_acc)
    step_table = _SHARP_STEPS if n_acc > 0 else _FLAT_STEPS
    steps = step_table.get(base, step_table["G"])

    x_cur = float(x_start)
    spacing = sp * cfg["spacing_sp"]
    for idx in range(count):
        step = steps[idx % len(steps)]
        y_base = _step_to_y(y_lines, x_cur, step, sp)
        # Archetype-specific vertical drift & crammed stagger
        y_drift = idx * cfg["y_drift_sp"] * sp
        y_stagger = ((1.0 if idx % 2 == 0 else -1.0) * cfg["stagger_sp"] * sp) if cfg["stagger_sp"] else 0.0
        y_pos = y_base + y_drift + y_stagger + float(rng.normal(0.0, 0.025 * sp))
        draw_accidental_variant(canvas, x_cur, y_pos, sp, acc_kind, arch, style, rng, color_rgb=ink)
        x_cur += spacing * float(rng.uniform(0.95, 1.05))

    return x_cur


# ---------------------------------------------------------------------------
# 3. BARLINE ARCHETYPES (18 distinct styles x 6 bar kinds)
# ---------------------------------------------------------------------------


def _barline_params(archetype: str, style: dict) -> Tuple[dict, dict]:
    """Return (local_stroke_style, cfg) for a barline archetype."""
    local = dict(style or {})
    cfg: Dict[str, Any] = {
        "over_top_sp": 0.05,
        "over_bot_sp": 0.05,
        "slant_deg": 0.0,
        "bow_sp": 0.0,
        "s_wave_sp": 0.0,
        "w_mult": 1.0,
        "thick_mult": 1.0,
        "gap_sp": 0.46,
        "dot_offset_sp": 0.48,
        "dot_r_sp": 0.20,
        "dot_style": "blob",
        "double_pass": False,
        "gapped": False,
        "serif_hooks": False,
        "tapered_quill": False,
        "mode": archetype,
    }

    if archetype == "crisp_vertical":
        cfg.update(over_top_sp=0.02, over_bot_sp=0.02, slant_deg=0.0, w_mult=1.0)
        local["wobble_px"] = min(float(local.get("wobble_px", 0.7)), 0.20)
        local["tremor_px"] = min(float(local.get("tremor_px", 0.25)), 0.12)
    elif archetype == "tall_overshoot":
        cfg.update(over_top_sp=0.72, over_bot_sp=0.72)
    elif archetype == "short_clipped_inside":
        cfg.update(over_top_sp=-0.42, over_bot_sp=-0.42)
    elif archetype == "steep_forward_slant":
        cfg.update(slant_deg=15.0, over_top_sp=0.14, over_bot_sp=0.14)
    elif archetype == "back_slant_lefty":
        cfg.update(slant_deg=-15.0, over_top_sp=0.14, over_bot_sp=0.14)
    elif archetype == "bowed_curved_bar":
        cfg.update(bow_sp=0.48, over_top_sp=0.10, over_bot_sp=0.10)
    elif archetype == "s_wobble_freehand":
        cfg.update(s_wave_sp=0.34, over_top_sp=0.12, over_bot_sp=0.12)
        local["tremor_px"] = max(float(local.get("tremor_px", 0.25)), 1.45)
    elif archetype == "heavy_thick_marker":
        cfg.update(w_mult=2.15, thick_mult=1.60, dot_r_sp=0.28)
        local["nib_min_ratio"] = 0.90
    elif archetype == "hairline_faint":
        cfg.update(w_mult=0.45, thick_mult=0.58, dot_r_sp=0.14)
        local["stroke_opacity"] = 0.56
    elif archetype == "tapered_quill_bar":
        cfg.update(w_mult=1.35, over_top_sp=0.22, over_bot_sp=0.28, tapered_quill=True)
        local["ink_pooling"] = 0.65
    elif archetype == "double_pass_sketch":
        cfg.update(w_mult=0.92, double_pass=True, over_top_sp=0.18, over_bot_sp=0.18)
    elif archetype == "gapped_broken_bar":
        cfg.update(w_mult=1.05, gapped=True)
    elif archetype == "hooked_top_bottom_serif":
        cfg.update(w_mult=1.08, serif_hooks=True, over_top_sp=0.12, over_bot_sp=0.12)
    elif archetype == "wide_double_gap":
        cfg.update(gap_sp=0.98, dot_offset_sp=0.78)
    elif archetype == "tight_double_gap":
        cfg.update(gap_sp=0.22, dot_offset_sp=0.28)
    elif archetype == "oversized_repeat_blobs":
        cfg.update(dot_r_sp=0.40, dot_offset_sp=0.62, w_mult=1.18)
    elif archetype == "tiny_pinprick_repeat_dots":
        cfg.update(dot_r_sp=0.10, dot_offset_sp=0.40, w_mult=0.82)
    elif archetype == "slashed_repeat_ticks":
        cfg.update(dot_style="slash", dot_offset_sp=0.54, dot_r_sp=0.26, over_top_sp=0.15, over_bot_sp=0.15)
    return local, cfg


def _draw_single_bar_stroke(
    canvas: StrokeCanvas,
    bx: float,
    y_top: float,
    y_bot: float,
    sp: float,
    width: float,
    ink: RGBLike,
    local_style: dict,
    cfg: dict,
    rng: np.random.Generator,
) -> None:
    yt = y_top - cfg["over_top_sp"] * sp
    yb = y_bot + cfg["over_bot_sp"] * sp
    if yb <= yt + 2.0:
        ym = 0.5 * (y_top + y_bot)
        yt, yb = ym - sp * 0.6, ym + sp * 0.6
    h = yb - yt
    shear = math.tan(math.radians(float(np.clip(cfg["slant_deg"], -28.0, 28.0))))

    n_pts = max(14, int(abs(h) / 6.0))
    t = np.linspace(0.0, 1.0, n_pts)
    ys = yt + t * h
    xs = bx + (0.5 - t) * h * shear

    if abs(cfg["bow_sp"]) > 1e-4:
        xs = xs + cfg["bow_sp"] * sp * (4.0 * t * (1.0 - t))
    if abs(cfg["s_wave_sp"]) > 1e-4:
        xs = xs + cfg["s_wave_sp"] * sp * np.sin(2.0 * math.pi * t)

    pts = np.column_stack((xs, ys))

    if cfg["gapped"]:
        # Render 3 broken vertical segments with crisp dry-lift gaps.
        splits = [(0.0, 0.34), (0.45, 0.68), (0.78, 1.0)]
        for s0, s1 in splits:
            i0 = int(round(s0 * (n_pts - 1)))
            i1 = max(i0 + 2, int(round(s1 * (n_pts - 1))) + 1)
            seg = pts[i0:min(n_pts, i1)]
            if len(seg) >= 2:
                draw_calligraphic_polyline(canvas, seg, ink, width, local_style, kind="barline")
    elif cfg["tapered_quill"]:
        # Heavy top half tapering to a fine bottom point (quill lift).
        mid = n_pts // 2
        st_top = dict(local_style)
        st_top["taper_start"] = 0.04
        st_top["taper_end"] = 0.35
        st_bot = dict(local_style)
        st_bot["taper_start"] = 0.10
        st_bot["taper_end"] = 0.55
        draw_calligraphic_polyline(canvas, pts[: mid + 2], ink, width * 1.45, st_top, kind="barline")
        draw_calligraphic_polyline(canvas, pts[mid:], ink, max(0.55, width * 0.52), st_bot, kind="barline")
        r_top = max(1.2, width * 0.72)
        draw_filled_blob(canvas, pts[0, 0], pts[0, 1], r_top, r_top * 0.9, 0.0, ink, local_style, hollow=False)
    else:
        draw_calligraphic_polyline(canvas, pts, ink, width, local_style, kind="barline")
        if cfg["double_pass"]:
            dx_off = sp * float(rng.uniform(0.08, 0.14))
            pass2 = pts + np.array([[dx_off, float(rng.normal(0.0, 0.05 * sp))]])
            draw_calligraphic_polyline(canvas, pass2, ink, width * 0.80, local_style, kind="barline")

    if cfg["serif_hooks"]:
        hook_w = sp * 0.42
        top_hook = np.array([
            [pts[0, 0] - hook_w * 0.55, pts[0, 1] - sp * 0.08],
            [pts[0, 0] + hook_w * 0.55, pts[0, 1] + sp * 0.08],
        ], dtype=np.float64)
        bot_hook = np.array([
            [pts[-1, 0] - hook_w * 0.55, pts[-1, 1] + sp * 0.08],
            [pts[-1, 0] + hook_w * 0.55, pts[-1, 1] - sp * 0.08],
        ], dtype=np.float64)
        draw_calligraphic_polyline(canvas, top_hook, ink, width * 0.92, local_style, kind="barline")
        draw_calligraphic_polyline(canvas, bot_hook, ink, width * 0.92, local_style, kind="barline")


def _draw_repeat_dots_arch(
    canvas: StrokeCanvas,
    dx: float,
    y_top: float,
    y_bot: float,
    sp: float,
    ink: RGBLike,
    local_style: dict,
    cfg: dict,
    rng: np.random.Generator,
) -> None:
    ym = 0.5 * (y_top + y_bot)
    r = max(0.85, sp * cfg["dot_r_sp"])
    for sign in (-1.0, 1.0):
        px = dx + float(rng.normal(0.0, 0.025 * sp))
        py = ym + sign * 0.5 * sp + float(rng.normal(0.0, 0.025 * sp))
        if cfg["dot_style"] == "slash":
            slash = np.array([
                [px - sp * 0.22, py + sp * 0.20],
                [px + sp * 0.22, py - sp * 0.20],
            ], dtype=np.float64)
            draw_calligraphic_polyline(canvas, slash, ink, max(1.0, sp * 0.15), local_style, kind="barline")
        else:
            draw_filled_blob(
                canvas, px, py, r, r * float(rng.uniform(0.90, 1.10)),
                float(rng.uniform(-20.0, 20.0)), ink, local_style, hollow=False,
            )


def draw_barline_variant(
    engine: Any,
    x: float,
    y_top: float,
    y_bot: float,
    sp: float,
    bar_kind: str,
    archetype: str,
    style: dict,
    rng: np.random.Generator,
    color_rgb: Optional[RGBLike] = None,
) -> None:
    """Draw one of 18 visually distinct handwritten barline archetypes.

    Supports bar_kind in {'single', 'double', 'final', 'repeat_start',
    'repeat_end', 'repeat_both'}.
    """
    canvas = _resolve_canvas(engine)
    ink = _resolve_ink_rgb(style, color_rgb)
    arch = archetype if archetype in BARLINE_ARCHETYPES else "crisp_vertical"
    kind = _normalize_bar_kind(bar_kind)
    local_style, cfg = _barline_params(arch, style)

    thin_w = max(0.60, sp * float(local_style.get("barline_width_ratio", 0.145)) * cfg["w_mult"])
    thick_w = max(1.30, sp * float(local_style.get("barline_thick_ratio", 0.42)) * cfg["thick_mult"])
    gap = sp * cfg["gap_sp"]
    dot_off = sp * cfg["dot_offset_sp"]

    if kind == "single":
        _draw_single_bar_stroke(canvas, x, y_top, y_bot, sp, thin_w, ink, local_style, cfg, rng)
    elif kind == "double":
        _draw_single_bar_stroke(canvas, x - 0.5 * gap, y_top, y_bot, sp, thin_w, ink, local_style, cfg, rng)
        _draw_single_bar_stroke(canvas, x + 0.5 * gap, y_top, y_bot, sp, thin_w, ink, local_style, cfg, rng)
    elif kind == "final":
        _draw_single_bar_stroke(canvas, x - gap, y_top, y_bot, sp, thin_w, ink, local_style, cfg, rng)
        _draw_single_bar_stroke(canvas, x, y_top, y_bot, sp, thick_w, ink, local_style, cfg, rng)
    elif kind == "repeat_start":
        _draw_single_bar_stroke(canvas, x, y_top, y_bot, sp, thick_w, ink, local_style, cfg, rng)
        _draw_single_bar_stroke(canvas, x + gap, y_top, y_bot, sp, thin_w, ink, local_style, cfg, rng)
        _draw_repeat_dots_arch(canvas, x + gap + dot_off, y_top, y_bot, sp, ink, local_style, cfg, rng)
    elif kind == "repeat_end":
        _draw_single_bar_stroke(canvas, x - gap, y_top, y_bot, sp, thin_w, ink, local_style, cfg, rng)
        _draw_single_bar_stroke(canvas, x, y_top, y_bot, sp, thick_w, ink, local_style, cfg, rng)
        _draw_repeat_dots_arch(canvas, x - gap - dot_off, y_top, y_bot, sp, ink, local_style, cfg, rng)
    elif kind == "repeat_both":
        _draw_single_bar_stroke(canvas, x - gap, y_top, y_bot, sp, thin_w, ink, local_style, cfg, rng)
        _draw_single_bar_stroke(canvas, x, y_top, y_bot, sp, thick_w, ink, local_style, cfg, rng)
        _draw_single_bar_stroke(canvas, x + gap, y_top, y_bot, sp, thin_w, ink, local_style, cfg, rng)
        _draw_repeat_dots_arch(canvas, x - gap - dot_off, y_top, y_bot, sp, ink, local_style, cfg, rng)
        _draw_repeat_dots_arch(canvas, x + gap + dot_off, y_top, y_bot, sp, ink, local_style, cfg, rng)
    else:
        _draw_single_bar_stroke(canvas, x, y_top, y_bot, sp, thin_w, ink, local_style, cfg, rng)
