"""30 historical & vernacular handwritten musical pause (rest & fermata) archetypes.

Standalone module providing 30 brand-new, visually unmistakable pause variations
with zero overlap against the legacy 10 ``REST_ARCHETYPES`` in ``styles.py`` /
``glyph_variants.py``:

- ``NEW_QUARTER_REST_ARCHETYPES`` (18 quarter-rest archetypes)
- ``FLAGGED_REST_ARCHETYPES`` (6 eighth / 16th / 32nd rest archetypes)
- ``BAR_REST_ARCHETYPES`` (6 whole / half / multi-bar rest & fermata archetypes)

Every glyph instance is procedurally simulated fresh from Bezier / Catmull-Rom /
polyline nib strokes with per-instance motor jitter — zero glyph caching or
bitmap stamping.
"""

from __future__ import annotations

from functools import lru_cache
import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from .stroke_engine import (
    RGBLike,
    StrokeCanvas,
    draw_calligraphic_polyline,
    draw_filled_blob,
)


# ---------------------------------------------------------------------------
# Public Registries (18 + 6 + 6 = 30 brand-new pause archetypes)
# ---------------------------------------------------------------------------

NEW_QUARTER_REST_ARCHETYPES: Tuple[str, ...] = (
    "baroque_reversed_r",
    "beethoven_violent_zig",
    "chopin_gossamer_3",
    "schubert_vertical_drop",
    "brahms_bold_chevron",
    "liszt_sweeping_s_hook",
    "debussy_floating_tilde",
    "stravinsky_geometric_z",
    "shostakovich_looped_2",
    "realbook_chisel_bolt",
    "bebop_napkin_tick",
    "studio_copyist_clean_3",
    "student_awkward_e",
    "arthritis_tremor_zig",
    "lefty_backslant_hook",
    "hairline_needle_s",
    "open_double_loop_r",
    "pinprick_minimal_v",
)

FLAGGED_REST_ARCHETYPES: Tuple[str, ...] = (
    "classic_dot_slash",
    "hurried_7_stroke",
    "open_loop_bulb",
    "baroque_curved_flag",
    "detached_dot_tick",
    "heavy_marker_7",
)

BAR_REST_ARCHETYPES: Tuple[str, ...] = (
    "crisp_engraved_block",
    "thick_ink_smudge_bar",
    "hurried_horizontal_dash",
    "hollow_box_sketch",
    "slanted_wedge_bar",
    "double_pass_scribble_bar",
)

ALL_NEW_PAUSE_ARCHETYPES: Tuple[str, ...] = (
    NEW_QUARTER_REST_ARCHETYPES + FLAGGED_REST_ARCHETYPES + BAR_REST_ARCHETYPES
)

PAUSE_ARCHETYPE_DESCRIPTIONS: Dict[str, str] = {
    # 18 quarter rests
    "baroque_reversed_r": "17th/18th-c. Bach/Handel reversed-r / gamma quarter rest",
    "beethoven_violent_zig": "Sharp slashing downward Z with heavy ink-blot tail",
    "chopin_gossamer_3": "Delicate tiny spider-leg cursive 3",
    "schubert_vertical_drop": "Hurried nearly-vertical squiggle",
    "brahms_bold_chevron": "Heavy broad-nib double-chevron",
    "liszt_sweeping_s_hook": "Tall dramatic sweeping S with long lower hook",
    "debussy_floating_tilde": "Light airy tilted wave/tilde",
    "stravinsky_geometric_z": "Razor-sharp angular Z with crisp corners",
    "shostakovich_looped_2": "Hurried Russian ballpoint looped figure-2",
    "realbook_chisel_bolt": "Thick 1970s felt-tip lightning bolt",
    "bebop_napkin_tick": "Tiny hurried 2-stroke diagonal flick",
    "studio_copyist_clean_3": "Wide neat Hollywood copyist reverse-3",
    "student_awkward_e": "Hand-drawn uneven backwards-E / squiggle",
    "arthritis_tremor_zig": "Shaky trembling multi-segment rest",
    "lefty_backslant_hook": "Left-handed back-slanted sharp hook",
    "hairline_needle_s": "Ultra-slim 0.25mm needle-thin S-curve",
    "open_double_loop_r": "Baroque ornate double-looped figure",
    "pinprick_minimal_v": "Microscopic V-tick shorthand",
    # 6 flagged rests
    "classic_dot_slash": "Standard bulb + diagonal stem",
    "hurried_7_stroke": "Rapid single-stroke figure-7 without bulb",
    "open_loop_bulb": "Open circular loop + sweeping diagonal tail",
    "baroque_curved_flag": "Calligraphic curved S-stem with broad-nib flag",
    "detached_dot_tick": "Tiny separate dot floating left of thin slash",
    "heavy_marker_7": "Bold thick chisel-marker 7",
    # 6 bar rests & fermatas
    "crisp_engraved_block": "Tidy rectangular hanging/sitting bar",
    "thick_ink_smudge_bar": "Heavy rounded ink-blotted bar",
    "hurried_horizontal_dash": "Quick thin horizontal pen-slash",
    "hollow_box_sketch": "Outlined hollow/half-filled sketch rectangle",
    "slanted_wedge_bar": "Angled calligraphic wedge bar",
    "double_pass_scribble_bar": "Back-and-forth scribbled bar rest",
}


# ---------------------------------------------------------------------------
# Local Bezier / Spline / Transform Helpers
# ---------------------------------------------------------------------------


@lru_cache(maxsize=64)
def _quad_basis(n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = np.linspace(0.0, 1.0, n)[:, None]
    mt = 1.0 - t
    return mt**2, 2.0 * mt * t, t**2


@lru_cache(maxsize=64)
def _cubic_basis(n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    t = np.linspace(0.0, 1.0, n)[:, None]
    mt = 1.0 - t
    return mt**3, 3.0 * (mt**2) * t, 3.0 * mt * (t**2), t**3


def _eval_quad(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, n: int = 12) -> np.ndarray:
    b0, b1, b2 = _quad_basis(n)
    return b0 * p0 + b1 * p1 + b2 * p2


def _eval_cubic(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, n: int = 16) -> np.ndarray:
    b0, b1, b2, b3 = _cubic_basis(n)
    return b0 * p0 + b1 * p1 + b2 * p2 + b3 * p3


def _eval_catmull_rom(ctrl: np.ndarray, samples_per_seg: int = 9) -> np.ndarray:
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


def _transform_local(
    pts_norm: np.ndarray,
    cx: float,
    cy: float,
    sp: float,
    rng: np.random.Generator,
    style: dict,
    jitter_sp: float = 0.06,
    scale_x_range: Tuple[float, float] = (0.90, 1.12),
    scale_y_range: Tuple[float, float] = (0.90, 1.12),
    extra_slant_deg: float = 0.0,
    slant_jitter_deg: float = 3.5,
) -> np.ndarray:
    """Perturb normalized (in sp units) control points with per-instance motor jitter."""
    pts = np.asarray(pts_norm, dtype=np.float64).copy()
    if jitter_sp > 0.0:
        pts += rng.normal(0.0, jitter_sp, size=pts.shape)

    sx = float(rng.uniform(*scale_x_range))
    sy = float(rng.uniform(*scale_y_range))
    slant_deg = (
        float(style.get("slant_deg", 0.0))
        + extra_slant_deg
        + float(rng.normal(0.0, slant_jitter_deg))
    )
    shear = math.tan(math.radians(float(np.clip(slant_deg, -32.0, 32.0))))

    x = pts[:, 0] * sx - pts[:, 1] * shear
    y = pts[:, 1] * sy

    ox = float(rng.normal(0.0, 0.045 * sp))
    oy = float(rng.normal(0.0, 0.045 * sp))
    return np.column_stack((cx + ox + x * sp, cy + oy + y * sp))


def _local_style(style: dict, **overrides: float) -> dict:
    st = dict(style)
    st.update(overrides)
    return st


# ---------------------------------------------------------------------------
# 1. 18 Brand-New Quarter-Rest Archetypes
# ---------------------------------------------------------------------------


def _draw_quarter_rest_variant(
    canvas: StrokeCanvas,
    archetype: str,
    cx: float,
    cy: float,
    sp: float,
    color_rgb: RGBLike,
    style: dict,
    rng: np.random.Generator,
) -> None:
    base_w = max(1.0, sp * float(style.get("glyph_stroke_ratio", 0.21)))

    if archetype == "baroque_reversed_r":
        # Authentic 17th/18th-c. Bach/Handel reversed-r / gamma quarter rest:
        # slightly slanted vertical spine with a crisp rightward beak/arm at top + serif foot.
        spine_ctrl = np.array([
            [-0.08, -1.28],
            [-0.02, -0.45],
            [-0.10, 0.48],
            [-0.18, 1.30],
        ], dtype=np.float64)
        arm_ctrl = np.array([
            [-0.08, -1.24],
            [0.32, -1.34],
            [0.64, -1.16],
            [0.44, -0.84],
        ], dtype=np.float64)
        foot_ctrl = np.array([
            [-0.38, 1.32],
            [-0.18, 1.30],
            [0.06, 1.26],
        ], dtype=np.float64)
        all_pts = np.vstack((spine_ctrl, arm_ctrl, foot_ctrl))
        pts = _transform_local(
            all_pts, cx, cy, sp, rng, style,
            jitter_sp=0.055, scale_x_range=(0.90, 1.12), scale_y_range=(0.92, 1.10),
        )
        spine = _eval_catmull_rom(pts[:4], samples_per_seg=8)
        arm = _eval_cubic(pts[4], pts[5], pts[6], pts[7], n=14)
        foot = _eval_quad(pts[8], pts[9], pts[10], n=8)
        st = _local_style(style, nib_angle_deg=28.0, nib_min_ratio=0.26, ink_pooling=0.35)
        draw_calligraphic_polyline(canvas, spine, color_rgb, base_w * 1.18, st, kind="rest")
        draw_calligraphic_polyline(canvas, arm, color_rgb, base_w * 1.12, st, kind="rest")
        draw_calligraphic_polyline(canvas, foot, color_rgb, base_w * 0.72, st, kind="rest")
        return

    if archetype == "beethoven_violent_zig":
        # Sharp slashing downward Z with heavy ink-blot tail pooling at the bottom.
        ctrl = np.array([
            [-0.48, -1.52],
            [0.54, -0.78],
            [-0.50, 0.22],
            [0.44, 0.88],
            [0.24, 1.38],
        ], dtype=np.float64)
        pts = _transform_local(
            ctrl, cx, cy, sp, rng, style,
            jitter_sp=0.09, scale_x_range=(0.90, 1.16), scale_y_range=(0.92, 1.14),
            extra_slant_deg=8.0,
        )
        st = _local_style(
            style,
            nib_angle_deg=24.0,
            nib_min_ratio=0.38,
            ink_pooling=0.80,
            ink_feather=max(float(style.get("ink_feather", 0.0)), 0.22),
            wobble_px=max(float(style.get("wobble_px", 0.5)), 0.65),
        )
        draw_calligraphic_polyline(canvas, pts, color_rgb, base_w * 1.55, st, kind="rest")
        # Heavy ink-blot tail at bottom terminal
        blot_c = pts[-1] + np.array([rng.normal(0.0, 0.05 * sp), rng.normal(0.04 * sp, 0.05 * sp)])
        rx = sp * float(rng.uniform(0.24, 0.34))
        ry = sp * float(rng.uniform(0.19, 0.28))
        draw_filled_blob(canvas, blot_c[0], blot_c[1], rx, ry, float(rng.uniform(-35, 25)), color_rgb, st, hollow=False)
        return

    if archetype == "chopin_gossamer_3":
        # Delicate tiny spider-leg cursive 3: compact, airy, ultra-fine hairlines.
        ctrl = np.array([
            [-0.24, -0.82],
            [0.24, -0.74],
            [0.18, -0.26],
            [-0.12, -0.02],
            [0.26, 0.26],
            [0.16, 0.74],
            [-0.28, 0.86],
        ], dtype=np.float64)
        pts = _transform_local(
            ctrl, cx, cy, sp, rng, style,
            jitter_sp=0.045, scale_x_range=(0.88, 1.10), scale_y_range=(0.88, 1.10),
            extra_slant_deg=6.0,
        )
        curve = _eval_catmull_rom(pts, samples_per_seg=8)
        st = _local_style(
            style,
            nib_min_ratio=0.45,
            taper_start=0.24,
            taper_end=0.28,
            wobble_px=min(float(style.get("wobble_px", 0.4)), 0.35),
            stroke_opacity=float(style.get("stroke_opacity", 0.90)) * 0.85,
        )
        draw_calligraphic_polyline(canvas, curve, color_rgb, max(0.65, base_w * 0.50), st, kind="rest")
        return

    if archetype == "schubert_vertical_drop":
        # Hurried nearly-vertical squiggle: tall, very narrow horizontal excursion.
        ctrl = np.array([
            [0.04, -1.56],
            [0.14, -0.78],
            [-0.11, -0.08],
            [0.15, 0.62],
            [-0.08, 1.54],
        ], dtype=np.float64)
        pts = _transform_local(
            ctrl, cx, cy, sp, rng, style,
            jitter_sp=0.05, scale_x_range=(0.85, 1.15), scale_y_range=(0.94, 1.12),
            extra_slant_deg=2.0, slant_jitter_deg=2.0,
        )
        curve = _eval_catmull_rom(pts, samples_per_seg=9)
        st = _local_style(style, nib_min_ratio=0.55, taper_start=0.16, taper_end=0.22)
        draw_calligraphic_polyline(canvas, curve, color_rgb, base_w * 0.96, st, kind="rest")
        return

    if archetype == "brahms_bold_chevron":
        # Heavy broad-nib double-chevron (> stacked over >).
        ctrl = np.array([
            [-0.38, -1.36],
            [0.46, -0.68],
            [-0.34, -0.02],
            [0.48, 0.64],
            [-0.34, 1.34],
        ], dtype=np.float64)
        pts = _transform_local(
            ctrl, cx, cy, sp, rng, style,
            jitter_sp=0.055, scale_x_range=(0.92, 1.12), scale_y_range=(0.92, 1.10),
        )
        st = _local_style(
            style,
            nib_angle_deg=22.0,
            nib_min_ratio=0.24,
            ink_pooling=0.55,
            taper_start=0.06,
            taper_end=0.08,
        )
        draw_calligraphic_polyline(canvas, pts, color_rgb, base_w * 1.80, st, kind="rest")
        return

    if archetype == "liszt_sweeping_s_hook":
        # Tall dramatic sweeping S with long scooping lower J-hook.
        ctrl = np.array([
            [-0.18, -1.80],
            [0.48, -1.16],
            [-0.38, -0.22],
            [0.42, 0.62],
            [-0.38, 1.48],
            [0.14, 1.76],
            [0.62, 1.16],
        ], dtype=np.float64)
        pts = _transform_local(
            ctrl, cx, cy, sp, rng, style,
            jitter_sp=0.065, scale_x_range=(0.90, 1.14), scale_y_range=(0.94, 1.12),
            extra_slant_deg=6.0,
        )
        curve = _eval_catmull_rom(pts, samples_per_seg=9)
        st = _local_style(style, nib_angle_deg=35.0, nib_min_ratio=0.28, taper_end=0.26)
        draw_calligraphic_polyline(canvas, curve, color_rgb, base_w * 1.18, st, kind="rest")
        return

    if archetype == "debussy_floating_tilde":
        # Light airy tilted wave/tilde floating across the middle staff.
        ctrl = np.array([
            [-0.52, -0.72],
            [-0.10, -1.04],
            [0.02, 0.00],
            [0.14, 1.04],
            [0.54, 0.72],
        ], dtype=np.float64)
        pts = _transform_local(
            ctrl, cx, cy, sp, rng, style,
            jitter_sp=0.055, scale_x_range=(0.90, 1.14), scale_y_range=(0.88, 1.12),
        )
        curve = _eval_catmull_rom(pts, samples_per_seg=10)
        st = _local_style(
            style,
            nib_min_ratio=0.45,
            taper_start=0.28,
            taper_end=0.28,
            stroke_opacity=float(style.get("stroke_opacity", 0.90)) * 0.72,
        )
        draw_calligraphic_polyline(canvas, curve, color_rgb, base_w * 0.78, st, kind="rest")
        return

    if archetype == "stravinsky_geometric_z":
        # Razor-sharp angular Z with crisp drafting-pen corners & geometric serif foot.
        ctrl = np.array([
            [-0.44, -1.26],
            [0.44, -1.26],
            [-0.44, 0.52],
            [0.44, 0.52],
            [0.10, 1.28],
        ], dtype=np.float64)
        pts = _transform_local(
            ctrl, cx, cy, sp, rng, style,
            jitter_sp=0.028, scale_x_range=(0.94, 1.08), scale_y_range=(0.94, 1.08),
            slant_jitter_deg=1.5,
        )
        st = _local_style(
            style,
            nib_min_ratio=0.78,
            wobble_px=min(float(style.get("wobble_px", 0.4)), 0.14),
            tremor_px=min(float(style.get("tremor_px", 0.18)), 0.06),
            taper_start=0.04,
            taper_end=0.04,
        )
        draw_calligraphic_polyline(canvas, pts, color_rgb, base_w * 0.92, st, kind="rest")
        return

    if archetype == "shostakovich_looped_2":
        # Hurried Russian ballpoint looped figure-2.
        ctrl = np.array([
            [-0.34, -0.96],
            [0.08, -1.32],
            [0.40, -0.82],
            [0.06, -0.02],
            [-0.42, 0.88],
            [-0.44, 1.24],
            [-0.08, 1.26],
            [-0.04, 0.92],
            [0.50, 1.06],
        ], dtype=np.float64)
        pts = _transform_local(
            ctrl, cx, cy, sp, rng, style,
            jitter_sp=0.055, scale_x_range=(0.90, 1.12), scale_y_range=(0.90, 1.12),
            extra_slant_deg=7.0,
        )
        curve = _eval_catmull_rom(pts, samples_per_seg=8)
        st = _local_style(
            style,
            nib_min_ratio=0.84,
            ink_pooling=0.10,
            taper_start=0.10,
            taper_end=0.18,
        )
        draw_calligraphic_polyline(canvas, curve, color_rgb, base_w * 0.80, st, kind="rest")
        return

    if archetype == "realbook_chisel_bolt":
        # Thick 1970s felt-tip chisel-marker lightning bolt.
        ctrl = np.array([
            [-0.26, -1.38],
            [0.46, -0.52],
            [-0.32, 0.14],
            [0.44, 0.82],
            [-0.14, 1.32],
        ], dtype=np.float64)
        pts = _transform_local(
            ctrl, cx, cy, sp, rng, style,
            jitter_sp=0.06, scale_x_range=(0.92, 1.14), scale_y_range=(0.92, 1.10),
        )
        st = _local_style(
            style,
            nib_angle_deg=42.0,
            nib_min_ratio=0.42,
            taper_start=0.03,
            taper_end=0.05,
            ink_pooling=0.20,
            stroke_opacity=min(1.0, float(style.get("stroke_opacity", 0.92)) * 1.06),
        )
        draw_calligraphic_polyline(canvas, pts, color_rgb, base_w * 2.05, st, kind="rest")
        return

    if archetype == "bebop_napkin_tick":
        # Tiny hurried 2-stroke diagonal flick (compact slash + upper-right cross-tick).
        s1 = np.array([
            [-0.30, -0.64],
            [0.24, 0.66],
        ], dtype=np.float64)
        s2 = np.array([
            [-0.08, -0.14],
            [0.44, -0.52],
        ], dtype=np.float64)
        pts = _transform_local(
            np.vstack((s1, s2)), cx, cy, sp, rng, style,
            jitter_sp=0.055, scale_x_range=(0.88, 1.14), scale_y_range=(0.88, 1.14),
        )
        st = _local_style(style, nib_min_ratio=0.55, taper_start=0.14, taper_end=0.22)
        draw_calligraphic_polyline(canvas, pts[:2], color_rgb, base_w * 1.10, st, kind="rest")
        draw_calligraphic_polyline(canvas, pts[2:], color_rgb, base_w * 0.92, st, kind="rest")
        return

    if archetype == "studio_copyist_clean_3":
        # Wide neat Hollywood copyist reverse-3 (bellies bulging LEFT, opening RIGHT).
        ctrl = np.array([
            [0.44, -1.24],
            [-0.36, -1.08],
            [-0.46, -0.46],
            [0.16, -0.04],
            [-0.48, 0.44],
            [-0.32, 1.10],
            [0.44, 1.24],
        ], dtype=np.float64)
        pts = _transform_local(
            ctrl, cx, cy, sp, rng, style,
            jitter_sp=0.04, scale_x_range=(0.94, 1.08), scale_y_range=(0.94, 1.08),
            slant_jitter_deg=2.0,
        )
        curve = _eval_catmull_rom(pts, samples_per_seg=9)
        st = _local_style(
            style,
            nib_angle_deg=30.0,
            nib_min_ratio=0.35,
            wobble_px=min(float(style.get("wobble_px", 0.4)), 0.25),
        )
        draw_calligraphic_polyline(canvas, curve, color_rgb, base_w * 1.24, st, kind="rest")
        return

    if archetype == "student_awkward_e":
        # Hand-drawn uneven backwards-E / 3-pronged squiggle.
        spine = np.array([
            [0.24, -1.16],
            [0.30, -0.02],
            [0.18, 1.16],
        ], dtype=np.float64)
        p_top = np.array([[-0.42, -1.08], [0.28, -1.14]], dtype=np.float64)
        p_mid = np.array([[-0.34, 0.04], [0.30, -0.04]], dtype=np.float64)
        p_bot = np.array([[-0.44, 1.12], [0.22, 1.14]], dtype=np.float64)
        pts = _transform_local(
            np.vstack((spine, p_top, p_mid, p_bot)), cx, cy, sp, rng, style,
            jitter_sp=0.085, scale_x_range=(0.88, 1.15), scale_y_range=(0.88, 1.14),
        )
        st = _local_style(
            style,
            nib_min_ratio=0.72,
            wobble_px=max(float(style.get("wobble_px", 0.5)), 0.75),
            tremor_px=max(float(style.get("tremor_px", 0.2)), 0.35),
        )
        draw_calligraphic_polyline(canvas, _eval_quad(pts[0], pts[1], pts[2], n=10), color_rgb, base_w * 0.96, st, kind="rest")
        draw_calligraphic_polyline(canvas, pts[3:5], color_rgb, base_w * 0.92, st, kind="rest")
        draw_calligraphic_polyline(canvas, pts[5:7], color_rgb, base_w * 0.88, st, kind="rest")
        draw_calligraphic_polyline(canvas, pts[7:9], color_rgb, base_w * 0.94, st, kind="rest")
        return

    if archetype == "arthritis_tremor_zig":
        # Shaky trembling multi-segment rest with pronounced high-frequency hand oscillation.
        base_ctrl = np.array([
            [-0.22, -1.40],
            [0.36, -0.58],
            [-0.30, 0.16],
            [0.32, 0.82],
            [-0.20, 1.38],
        ], dtype=np.float64)
        pts = _transform_local(
            base_ctrl, cx, cy, sp, rng, style,
            jitter_sp=0.07, scale_x_range=(0.90, 1.12), scale_y_range=(0.90, 1.12),
        )
        # Subdivide into many shaky micro-segments with alternating lateral tremor kicks.
        shaky_pts: List[np.ndarray] = []
        for i in range(len(pts) - 1):
            p0, p1 = pts[i], pts[i + 1]
            dv = p1 - p0
            seg_len = float(np.hypot(dv[0], dv[1]))
            n_sub = max(5, int(seg_len / max(2.2, 0.16 * sp)))
            nx_v, ny_v = (-dv[1] / max(seg_len, 1e-6), dv[0] / max(seg_len, 1e-6))
            for k in range(n_sub):
                u = k / n_sub
                base_p = p0 * (1.0 - u) + p1 * u
                sign = 1.0 if (k % 2 == 0) else -1.0
                amp = sign * float(rng.uniform(0.06, 0.16)) * sp
                shaky_pts.append(base_p + np.array([nx_v * amp, ny_v * amp]))
        shaky_pts.append(pts[-1])
        st = _local_style(
            style,
            nib_min_ratio=0.65,
            tremor_px=max(1.5, 0.12 * sp),
            ink_dropout=max(float(style.get("ink_dropout", 0.04)), 0.14),
        )
        draw_calligraphic_polyline(canvas, np.asarray(shaky_pts, dtype=np.float64), color_rgb, base_w * 0.88, st, kind="rest")
        return

    if archetype == "lefty_backslant_hook":
        # Left-handed back-slanted sharp hook leaning unmistakably backward (\).
        ctrl = np.array([
            [-0.56, -1.24],
            [-0.12, -1.36],
            [0.24, -0.34],
            [0.44, 0.64],
            [-0.22, 1.32],
        ], dtype=np.float64)
        pts = _transform_local(
            ctrl, cx, cy, sp, rng, style,
            jitter_sp=0.06, scale_x_range=(0.90, 1.12), scale_y_range=(0.90, 1.12),
            extra_slant_deg=-20.0,
        )
        st = _local_style(style, nib_angle_deg=-35.0, nib_min_ratio=0.35, taper_end=0.22)
        draw_calligraphic_polyline(canvas, pts, color_rgb, base_w * 1.14, st, kind="rest")
        return

    if archetype == "hairline_needle_s":
        # Ultra-slim 0.25mm technical-pen needle-thin S-curve.
        ctrl = np.array([
            [0.24, -1.46],
            [-0.28, -0.96],
            [0.22, -0.22],
            [-0.26, 0.42],
            [0.24, 1.02],
            [-0.22, 1.46],
        ], dtype=np.float64)
        pts = _transform_local(
            ctrl, cx, cy, sp, rng, style,
            jitter_sp=0.04, scale_x_range=(0.90, 1.10), scale_y_range=(0.92, 1.10),
        )
        curve = _eval_catmull_rom(pts, samples_per_seg=9)
        st = _local_style(
            style,
            nib_min_ratio=0.92,
            wobble_px=min(float(style.get("wobble_px", 0.4)), 0.22),
            tremor_px=min(float(style.get("tremor_px", 0.16)), 0.08),
            ink_pooling=0.05,
        )
        draw_calligraphic_polyline(canvas, curve, color_rgb, max(0.55, base_w * 0.34), st, kind="rest")
        return

    if archetype == "open_double_loop_r":
        # Baroque ornate double-looped figure (figure-8 with top & bottom loops + flourish tail).
        ctrl = np.array([
            [0.16, -1.28],
            [-0.36, -1.32],
            [-0.44, -0.68],
            [0.22, -0.44],
            [0.36, 0.22],
            [-0.34, 0.52],
            [-0.32, 1.18],
            [0.22, 1.18],
            [0.54, 0.72],
        ], dtype=np.float64)
        pts = _transform_local(
            ctrl, cx, cy, sp, rng, style,
            jitter_sp=0.055, scale_x_range=(0.90, 1.12), scale_y_range=(0.90, 1.12),
        )
        curve = _eval_catmull_rom(pts, samples_per_seg=8)
        st = _local_style(style, nib_angle_deg=32.0, nib_min_ratio=0.28, ink_pooling=0.28)
        draw_calligraphic_polyline(canvas, curve, color_rgb, base_w * 1.02, st, kind="rest")
        return

    if archetype == "pinprick_minimal_v":
        # Microscopic V-tick shorthand centered in the staff.
        ctrl = np.array([
            [-0.28, -0.44],
            [0.02, 0.46],
            [0.32, -0.24],
        ], dtype=np.float64)
        pts = _transform_local(
            ctrl, cx, cy, sp, rng, style,
            jitter_sp=0.045, scale_x_range=(0.86, 1.14), scale_y_range=(0.86, 1.14),
        )
        st = _local_style(style, nib_min_ratio=0.65, taper_start=0.14, taper_end=0.20)
        draw_calligraphic_polyline(canvas, pts, color_rgb, max(0.75, base_w * 0.72), st, kind="rest")
        return

    raise ValueError(f"Unknown quarter rest archetype: {archetype!r}")


# ---------------------------------------------------------------------------
# 2. 6 Brand-New Flagged-Rest Archetypes (8th / 16th / 32nd)
# ---------------------------------------------------------------------------


def _draw_flagged_rest_variant(
    canvas: StrokeCanvas,
    n_hooks: int,
    archetype: str,
    cx: float,
    cy: float,
    sp: float,
    color_rgb: RGBLike,
    style: dict,
    rng: np.random.Generator,
) -> None:
    base_w = max(0.95, sp * float(style.get("glyph_stroke_ratio", 0.19)))
    slant_deg = float(style.get("slant_deg", 0.0)) + float(rng.normal(7.0, 3.5))
    shear = math.tan(math.radians(float(np.clip(slant_deg, -18.0, 24.0))))

    top_y = cy - (0.58 + 0.36 * (n_hooks - 1)) * sp
    bot_y = cy + (1.18 + 0.24 * (n_hooks - 1)) * sp
    stem_top = np.array([cx + 0.30 * sp + rng.normal(0.0, 0.05 * sp), top_y + rng.normal(0.0, 0.05 * sp)])
    stem_bot = np.array([
        stem_top[0] - (bot_y - top_y) * (0.28 + 0.22 * shear) + rng.normal(0.0, 0.06 * sp),
        bot_y + rng.normal(0.0, 0.05 * sp),
    ])

    if archetype == "classic_dot_slash":
        # Standard solid ink bulb + graceful concave arm into crisp slanted stem.
        draw_calligraphic_polyline(canvas, np.vstack((stem_top, stem_bot)), color_rgb, base_w * 1.05, style, kind="rest")
        for k in range(n_hooks):
            frac = 0.05 + k * (0.34 if n_hooks == 2 else 0.27)
            attach = stem_top * (1.0 - frac) + stem_bot * frac
            head_pt = attach + np.array([-(0.58 + rng.uniform(-0.06, 0.08)) * sp, (-0.10 + rng.uniform(-0.06, 0.06)) * sp])
            mid = 0.5 * (attach + head_pt) + np.array([0.06 * sp, 0.26 * sp + rng.normal(0.0, 0.04 * sp)])
            arm = _eval_quad(head_pt, mid, attach, n=10)
            draw_calligraphic_polyline(canvas, arm, color_rgb, base_w * 0.88, style, kind="rest")
            r = max(1.2, sp * float(rng.uniform(0.20, 0.26)))
            draw_filled_blob(canvas, head_pt[0], head_pt[1], r, r * float(rng.uniform(0.88, 1.12)), float(rng.uniform(-25, 25)), color_rgb, style, hollow=False)
        return

    if archetype == "hurried_7_stroke":
        # Rapid single-stroke figure-7 without any bulb!
        st = _local_style(style, nib_min_ratio=0.65, taper_start=0.10, taper_end=0.22)
        top_left = stem_top + np.array([-(0.66 + rng.uniform(-0.06, 0.08)) * sp, float(rng.normal(0.04 * sp, 0.05 * sp))])
        pts_main = np.vstack((top_left, stem_top, stem_bot))
        draw_calligraphic_polyline(canvas, pts_main, color_rgb, base_w * 0.96, st, kind="rest")
        for k in range(1, n_hooks):
            frac = k * (0.34 if n_hooks == 2 else 0.27)
            attach = stem_top * (1.0 - frac) + stem_bot * frac
            bar_left = attach + np.array([-(0.58 + rng.uniform(-0.06, 0.08)) * sp, float(rng.normal(0.06 * sp, 0.05 * sp))])
            draw_calligraphic_polyline(canvas, np.vstack((bar_left, attach)), color_rgb, base_w * 0.90, st, kind="rest")
        return

    if archetype == "open_loop_bulb":
        # Open circular cursive loop at tip + sweeping curved diagonal tail.
        mid_stem = 0.5 * (stem_top + stem_bot) + np.array([0.12 * sp, 0.0])
        stem_curve = _eval_quad(stem_top, mid_stem, stem_bot, n=12)
        draw_calligraphic_polyline(canvas, stem_curve, color_rgb, base_w * 0.95, style, kind="rest")
        for k in range(n_hooks):
            frac = 0.06 + k * (0.34 if n_hooks == 2 else 0.27)
            attach = stem_top * (1.0 - frac) + stem_bot * frac
            head_pt = attach + np.array([-(0.60 + rng.uniform(-0.06, 0.08)) * sp, (-0.08 + rng.uniform(-0.05, 0.05)) * sp])
            r = max(1.4, sp * float(rng.uniform(0.22, 0.28)))
            start_a = float(rng.uniform(0.15, 0.45)) * math.pi
            phis = np.linspace(start_a, start_a + 1.82 * math.pi, 16)
            loop = np.column_stack((
                head_pt[0] + r * np.cos(phis),
                head_pt[1] + r * 0.90 * np.sin(phis),
            ))
            mid = 0.5 * (attach + head_pt) + np.array([0.05 * sp, 0.24 * sp])
            arm = _eval_quad(loop[-1], mid, attach, n=8)
            draw_calligraphic_polyline(canvas, np.vstack((loop, arm[1:])), color_rgb, base_w * 0.78, style, kind="rest")
        return

    if archetype == "baroque_curved_flag":
        # Calligraphic curved S-stem with broad-nib ribbon flag(s).
        p1 = stem_top + np.array([0.16 * sp, 0.35 * (bot_y - top_y)])
        p2 = stem_bot + np.array([-0.18 * sp, -0.32 * (bot_y - top_y)])
        s_stem = _eval_cubic(stem_top, p1, p2, stem_bot, n=16)
        st = _local_style(style, nib_angle_deg=22.0, nib_min_ratio=0.20, ink_pooling=0.35)
        draw_calligraphic_polyline(canvas, s_stem, color_rgb, base_w * 1.25, st, kind="rest")
        for k in range(n_hooks):
            frac = 0.06 + k * (0.33 if n_hooks == 2 else 0.26)
            idx = min(len(s_stem) - 1, int(round(frac * (len(s_stem) - 1))))
            attach = s_stem[idx]
            tip = attach + np.array([-0.68 * sp, 0.28 * sp + rng.normal(0.0, 0.05 * sp)])
            c1 = attach + np.array([-0.24 * sp, -0.34 * sp])
            c2 = attach + np.array([-0.66 * sp, -0.18 * sp])
            flag_curve = _eval_cubic(attach, c1, c2, tip, n=14)
            draw_calligraphic_polyline(canvas, flag_curve, color_rgb, base_w * 1.38, st, kind="rest")
        return

    if archetype == "detached_dot_tick":
        # Tiny separate dot floating cleanly to the LEFT of a thin crisp diagonal slash (visible gap!).
        st = _local_style(
            style,
            nib_min_ratio=0.80,
            wobble_px=min(float(style.get("wobble_px", 0.4)), 0.25),
            taper_start=0.14,
            taper_end=0.18,
        )
        draw_calligraphic_polyline(canvas, np.vstack((stem_top, stem_bot)), color_rgb, max(0.70, base_w * 0.62), st, kind="rest")
        for k in range(n_hooks):
            frac = 0.12 + k * (0.34 if n_hooks == 2 else 0.27)
            attach = stem_top * (1.0 - frac) + stem_bot * frac
            # Deliberately detached gap (~0.52 sp to the left) with no connecting bar.
            dot_pt = attach + np.array([-(0.52 + rng.uniform(0.0, 0.10)) * sp, float(rng.normal(-0.04 * sp, 0.04 * sp))])
            r = max(1.0, sp * float(rng.uniform(0.14, 0.19)))
            draw_filled_blob(canvas, dot_pt[0], dot_pt[1], r, r * float(rng.uniform(0.85, 1.15)), float(rng.uniform(-30, 30)), color_rgb, st, hollow=False)
        return

    if archetype == "heavy_marker_7":
        # Bold thick chisel-marker 7 (heavy top bar + chunky slanted stem).
        st = _local_style(
            style,
            nib_angle_deg=38.0,
            nib_min_ratio=0.45,
            taper_start=0.03,
            taper_end=0.05,
            ink_pooling=0.18,
            stroke_opacity=min(1.0, float(style.get("stroke_opacity", 0.92)) * 1.06),
        )
        top_left = stem_top + np.array([-(0.72 + rng.uniform(-0.06, 0.08)) * sp, float(rng.normal(-0.04 * sp, 0.04 * sp))])
        draw_calligraphic_polyline(canvas, np.vstack((top_left, stem_top, stem_bot)), color_rgb, base_w * 2.05, st, kind="rest")
        for k in range(1, n_hooks):
            frac = k * (0.36 if n_hooks == 2 else 0.28)
            attach = stem_top * (1.0 - frac) + stem_bot * frac
            bar_left = attach + np.array([-(0.64 + rng.uniform(-0.05, 0.07)) * sp, float(rng.normal(0.0, 0.04 * sp))])
            draw_calligraphic_polyline(canvas, np.vstack((bar_left, attach)), color_rgb, base_w * 1.85, st, kind="rest")
        return

    raise ValueError(f"Unknown flagged rest archetype: {archetype!r}")


# ---------------------------------------------------------------------------
# 3. 6 Brand-New Bar-Rest & Fermata Archetypes (whole / half / fermata)
# ---------------------------------------------------------------------------


def _draw_fermata_variant(
    canvas: StrokeCanvas,
    archetype: str,
    cx: float,
    cy: float,
    sp: float,
    color_rgb: RGBLike,
    style: dict,
    rng: np.random.Generator,
) -> None:
    """Draw a handwritten fermata (arch + center dot) matched to the bar-rest archetype."""
    base_w = max(0.95, sp * float(style.get("glyph_stroke_ratio", 0.20)))
    ox = float(rng.normal(0.0, 0.04 * sp))
    oy = float(rng.normal(0.0, 0.04 * sp))
    fx, fy = cx + ox, cy + oy

    if archetype == "crisp_engraved_block":
        # Classic graceful calligraphic semicircle arch + crisp dot.
        phis = np.linspace(math.pi * 1.06, -math.pi * 0.06, 20)
        rx = sp * float(rng.uniform(0.95, 1.08))
        ry = sp * float(rng.uniform(0.72, 0.84))
        arch = np.column_stack((fx + rx * np.cos(phis), fy + 0.18 * sp - ry * np.sin(phis)))
        st = _local_style(style, nib_angle_deg=0.0, nib_min_ratio=0.32, taper_start=0.22, taper_end=0.22)
        draw_calligraphic_polyline(canvas, arch, color_rgb, base_w * 1.35, st, kind="rest")
        r = max(1.2, sp * float(rng.uniform(0.16, 0.21)))
        draw_filled_blob(canvas, fx, fy - 0.05 * sp, r, r, 0.0, color_rgb, st, hollow=False)
        return

    if archetype == "thick_ink_smudge_bar":
        # Heavy wet-ink blotted arch + oversized ink-blot dot.
        phis = np.linspace(math.pi * 1.04, -math.pi * 0.04, 18)
        rx = sp * float(rng.uniform(0.96, 1.12))
        ry = sp * float(rng.uniform(0.70, 0.86))
        arch = np.column_stack((fx + rx * np.cos(phis), fy + 0.20 * sp - ry * np.sin(phis)))
        st = _local_style(
            style,
            nib_min_ratio=0.68,
            ink_pooling=0.75,
            ink_feather=max(float(style.get("ink_feather", 0.0)), 0.30),
        )
        draw_calligraphic_polyline(canvas, arch, color_rgb, base_w * 2.05, st, kind="rest")
        r = max(1.6, sp * float(rng.uniform(0.24, 0.32)))
        draw_filled_blob(canvas, fx, fy - 0.04 * sp, r * 1.1, r * 0.92, float(rng.uniform(-20, 20)), color_rgb, st, hollow=False)
        return

    if archetype == "hurried_horizontal_dash":
        # Quick shallow flat-ish single-flick arc + tiny quick slash dot.
        rx = sp * float(rng.uniform(1.02, 1.18))
        ry = sp * float(rng.uniform(0.42, 0.56))
        p0 = np.array([fx - rx, fy + 0.10 * sp + rng.normal(0.0, 0.05 * sp)])
        p1 = np.array([fx + rng.normal(0.0, 0.08 * sp), fy - ry])
        p2 = np.array([fx + rx, fy + 0.06 * sp + rng.normal(0.0, 0.05 * sp)])
        arch = _eval_quad(p0, p1, p2, n=14)
        st = _local_style(style, nib_min_ratio=0.72, taper_start=0.22, taper_end=0.26)
        draw_calligraphic_polyline(canvas, arch, color_rgb, max(0.70, base_w * 0.62), st, kind="rest")
        tick = np.array([[fx - 0.14 * sp, fy + 0.02 * sp], [fx + 0.16 * sp, fy - 0.06 * sp]])
        draw_calligraphic_polyline(canvas, tick, color_rgb, max(0.75, base_w * 0.72), st, kind="rest")
        return

    if archetype == "hollow_box_sketch":
        # Sketched double-arc outline + open tiny hollow ring dot.
        rx = sp * float(rng.uniform(0.95, 1.08))
        ry = sp * float(rng.uniform(0.70, 0.82))
        phis = np.linspace(math.pi * 1.04, -math.pi * 0.04, 16)
        outer = np.column_stack((fx + rx * np.cos(phis), fy + 0.18 * sp - ry * np.sin(phis)))
        inner = np.column_stack((fx + (rx * 0.82) * np.cos(phis[::-1]), fy + 0.18 * sp - (ry * 0.72) * np.sin(phis[::-1])))
        st = _local_style(style, nib_min_ratio=0.75)
        draw_calligraphic_polyline(canvas, outer, color_rgb, max(0.75, base_w * 0.65), st, kind="rest")
        draw_calligraphic_polyline(canvas, inner, color_rgb, max(0.70, base_w * 0.58), st, kind="rest")
        r = max(1.3, sp * float(rng.uniform(0.18, 0.24)))
        draw_filled_blob(canvas, fx, fy - 0.02 * sp, r, r, 0.0, color_rgb, st, hollow=True, open_gap=0.14)
        return

    if archetype == "slanted_wedge_bar":
        # Asymmetric tilted calligraphic wedge-arch + angled diamond dot.
        rx = sp * float(rng.uniform(0.96, 1.12))
        ry = sp * float(rng.uniform(0.74, 0.88))
        p0 = np.array([fx - rx, fy + 0.24 * sp])
        p1 = np.array([fx - 0.45 * rx, fy - ry * 1.15])
        p2 = np.array([fx + 0.75 * rx, fy - ry * 0.85])
        p3 = np.array([fx + rx, fy + 0.04 * sp])
        arch = _eval_cubic(p0, p1, p2, p3, n=18)
        st = _local_style(style, nib_angle_deg=24.0, nib_min_ratio=0.22, taper_start=0.25, taper_end=0.12)
        draw_calligraphic_polyline(canvas, arch, color_rgb, base_w * 1.65, st, kind="rest")
        r = max(1.2, sp * 0.18)
        dia = np.array([[fx, fy - 0.06 * sp - r], [fx + r * 1.1, fy - 0.06 * sp], [fx, fy - 0.06 * sp + r], [fx - r * 1.1, fy - 0.06 * sp], [fx, fy - 0.06 * sp - r]])
        draw_calligraphic_polyline(canvas, dia, color_rgb, base_w * 0.85, st, kind="rest")
        return

    if archetype == "double_pass_scribble_bar":
        # Back-and-forth double-pass scribbled arch + quick scribbled dot.
        rx = sp * float(rng.uniform(0.95, 1.10))
        ry = sp * float(rng.uniform(0.68, 0.82))
        phis = np.linspace(math.pi * 1.05, -math.pi * 0.05, 14)
        pass1 = np.column_stack((fx + rx * np.cos(phis), fy + 0.18 * sp - ry * np.sin(phis)))
        pass2 = np.column_stack((
            fx + (rx * float(rng.uniform(0.94, 1.06))) * np.cos(phis[::-1]) + rng.normal(0.0, 0.04 * sp, len(phis)),
            fy + 0.14 * sp - (ry * float(rng.uniform(0.90, 1.08))) * np.sin(phis[::-1]) + rng.normal(0.0, 0.04 * sp, len(phis)),
        ))
        draw_calligraphic_polyline(canvas, np.vstack((pass1, pass2)), color_rgb, base_w * 0.92, style, kind="rest")
        dot_z = np.array([
            [fx - 0.16 * sp, fy - 0.10 * sp],
            [fx + 0.16 * sp, fy - 0.04 * sp],
            [fx - 0.14 * sp, fy + 0.06 * sp],
            [fx + 0.14 * sp, fy + 0.08 * sp],
        ]) + rng.normal(0.0, 0.03 * sp, size=(4, 2))
        draw_calligraphic_polyline(canvas, dot_z, color_rgb, base_w * 0.80, style, kind="rest")
        return

    raise ValueError(f"Unknown fermata/bar archetype: {archetype!r}")


def _draw_bar_rest_variant(
    canvas: StrokeCanvas,
    bar_kind: str,
    archetype: str,
    cx: float,
    cy: float,
    sp: float,
    color_rgb: RGBLike,
    style: dict,
    rng: np.random.Generator,
) -> None:
    """Draw a whole (hanging below line) or half (sitting above line) rest."""
    if bar_kind == "fermata":
        _draw_fermata_variant(canvas, archetype, cx, cy, sp, color_rgb, style, rng)
        return

    is_whole = bar_kind == "whole"
    # Whole rest hangs below the line (+y); half rest sits above the line (-y).
    sign = 1.0 if is_whole else -1.0
    xc = cx + float(rng.normal(0.0, 0.04 * sp))
    yc = cy + sign * 0.26 * sp + float(rng.normal(0.0, 0.03 * sp))

    if archetype == "crisp_engraved_block":
        # Tidy rectangular hanging/sitting bar with crisp squared ends (3 parallel passes).
        w = sp * float(rng.uniform(1.02, 1.14))
        h = sp * float(rng.uniform(0.42, 0.48))
        st = _local_style(
            style,
            nib_min_ratio=0.92,
            wobble_px=min(float(style.get("wobble_px", 0.4)), 0.14),
            tremor_px=min(float(style.get("tremor_px", 0.16)), 0.06),
            taper_start=0.0,
            taper_end=0.0,
        )
        # 3 crisp horizontal strips stacked tightly to form a sharp-cornered rectangle.
        strip_w = h / 2.2
        for frac in (-0.32, 0.0, 0.32):
            yy = yc + frac * h
            pts = np.array([[xc - 0.5 * w, yy], [xc + 0.5 * w, yy]], dtype=np.float64)
            draw_calligraphic_polyline(canvas, pts, color_rgb, strip_w, st, kind="beam")
        return

    if archetype == "thick_ink_smudge_bar":
        # Heavy rounded ink-blotted bar with wet ink feathering and swollen belly.
        rx = sp * float(rng.uniform(0.62, 0.74))
        ry = sp * float(rng.uniform(0.32, 0.40))
        st = _local_style(
            style,
            ink_pooling=0.80,
            ink_feather=max(float(style.get("ink_feather", 0.0)), 0.38),
            stroke_opacity=min(1.0, float(style.get("stroke_opacity", 0.92)) * 1.06),
        )
        draw_filled_blob(canvas, xc, yc + sign * 0.04 * sp, rx, ry, float(rng.normal(0.0, 3.0)), color_rgb, st, hollow=False)
        pts = np.array([[xc - rx * 0.85, yc], [xc + rx * 0.85, yc]], dtype=np.float64)
        draw_calligraphic_polyline(canvas, pts, color_rgb, ry * 1.55, st, kind="beam")
        return

    if archetype == "hurried_horizontal_dash":
        # Quick thin horizontal pen-slash (longer, slender, tapered ends).
        w = sp * float(rng.uniform(1.18, 1.48))
        tilt = math.radians(float(rng.normal(-2.5, 3.5)))
        dx = 0.5 * w * math.cos(tilt)
        dy = 0.5 * w * math.sin(tilt)
        pts = np.array([
            [xc - dx, yc - dy],
            [xc + float(rng.normal(0.0, 0.06 * sp)), yc + float(rng.normal(0.0, 0.04 * sp))],
            [xc + dx, yc + dy],
        ], dtype=np.float64)
        st = _local_style(style, nib_min_ratio=0.72, taper_start=0.22, taper_end=0.26)
        draw_calligraphic_polyline(canvas, pts, color_rgb, max(0.85, sp * 0.16), st, kind="rest")
        return

    if archetype == "hollow_box_sketch":
        # Outlined hollow/half-filled sketch rectangle with clear white interior + light hatch.
        w = sp * float(rng.uniform(1.04, 1.24))
        h = sp * float(rng.uniform(0.48, 0.58))
        x0, x1 = xc - 0.5 * w, xc + 0.5 * w
        y0, y1 = yc - 0.5 * h, yc + 0.5 * h
        box = np.array([
            [x0, y0],
            [x1, y0],
            [x1, y1],
            [x0, y1],
            [x0, y0],
        ], dtype=np.float64)
        box += rng.normal(0.0, 0.03 * sp, size=box.shape)
        st = _local_style(style, nib_min_ratio=0.75, taper_start=0.06, taper_end=0.06)
        draw_calligraphic_polyline(canvas, box, color_rgb, max(0.85, sp * 0.13), st, kind="rest")
        # Single quick diagonal sketch hatch inside so it reads unmistakably as hand-sketched.
        hatch = np.array([
            [x0 + 0.22 * w, y1 - 0.14 * h],
            [x1 - 0.22 * w, y0 + 0.14 * h],
        ], dtype=np.float64) + rng.normal(0.0, 0.025 * sp, size=(2, 2))
        draw_calligraphic_polyline(canvas, hatch, color_rgb, max(0.70, sp * 0.09), st, kind="rest")
        return

    if archetype == "slanted_wedge_bar":
        # Angled calligraphic wedge bar: tilted ~13 deg, thicker on one side tapering to a chisel point.
        w = sp * float(rng.uniform(1.05, 1.26))
        tilt = math.radians(float(rng.normal(-13.0, 3.0)))
        dx = 0.5 * w * math.cos(tilt)
        dy = 0.5 * w * math.sin(tilt)
        st = _local_style(style, nib_angle_deg=26.0, nib_min_ratio=0.30, taper_start=0.04, taper_end=0.38)
        # Main wedge stroke + upper flared chisel edge
        p_left = np.array([xc - dx, yc - dy])
        p_right = np.array([xc + dx, yc + dy])
        draw_calligraphic_polyline(canvas, np.vstack((p_left, p_right)), color_rgb, sp * 0.44, st, kind="beam")
        p_flare = p_left + np.array([0.08 * sp, -sign * 0.16 * sp])
        draw_calligraphic_polyline(canvas, np.vstack((p_flare, p_right)), color_rgb, sp * 0.24, st, kind="rest")
        return

    if archetype == "double_pass_scribble_bar":
        # Back-and-forth scribbled bar rest with visible turnaround loops.
        w = sp * float(rng.uniform(1.04, 1.26))
        h = sp * float(rng.uniform(0.38, 0.48))
        ctrl = np.array([
            [xc - 0.54 * w, yc - 0.34 * h],
            [xc + 0.52 * w, yc - 0.26 * h],
            [xc - 0.50 * w, yc + 0.02 * h],
            [xc + 0.54 * w, yc + 0.08 * h],
            [xc - 0.48 * w, yc + 0.34 * h],
            [xc + 0.50 * w, yc + 0.32 * h],
        ], dtype=np.float64)
        ctrl += rng.normal(0.0, 0.04 * sp, size=ctrl.shape)
        st = _local_style(style, nib_min_ratio=0.68, taper_start=0.08, taper_end=0.14)
        draw_calligraphic_polyline(canvas, ctrl, color_rgb, max(0.95, sp * 0.17), st, kind="rest")
        return

    raise ValueError(f"Unknown bar rest archetype: {archetype!r}")


# ---------------------------------------------------------------------------
# Public Unified Entry Point
# ---------------------------------------------------------------------------


_PAUSE_KIND_ALIASES: Dict[str, str] = {
    "1": "whole",
    "semibreve": "whole",
    "whole": "whole",
    "2": "half",
    "minim": "half",
    "half": "half",
    "4": "quarter",
    "crotchet": "quarter",
    "quarter": "quarter",
    "8": "eighth",
    "quaver": "eighth",
    "eighth": "eighth",
    "16": "16th",
    "semiquaver": "16th",
    "16th": "16th",
    "32": "32nd",
    "demisemiquaver": "32nd",
    "32nd": "32nd",
    "fermata": "fermata",
    "pause": "fermata",
    "hold": "fermata",
}


def draw_pause_variant(
    canvas: StrokeCanvas,
    pause_kind: str,
    cx: float,
    cy: float,
    sp: float,
    color_rgb: RGBLike,
    style: dict,
    rng: np.random.Generator,
    archetype: Optional[str] = None,
) -> str:
    """Draw one of the 30 brand-new handwritten pause/rest variations onto ``canvas``.

    Args:
        canvas: Target ``StrokeCanvas``.
        pause_kind: One of ``"quarter"``, ``"eighth"``, ``"16th"``, ``"32nd"``,
            ``"half"``, ``"whole"``, ``"fermata"`` (plus standard duration aliases).
        cx, cy: Center coordinates in pixels.
        sp: Staff-space height in pixels.
        color_rgb: Base ink RGB color.
        style: Style dict (nib physics, wobble, slant, opacity, etc.).
        rng: NumPy Generator for per-instance motor jitter (never cached).
        archetype: Explicit archetype name, or ``None`` to sample uniformly from
            the matching registry for ``pause_kind``.

    Returns:
        The archetype name that was rendered.
    """
    pk = _PAUSE_KIND_ALIASES.get(str(pause_kind).strip().lower(), str(pause_kind).strip().lower())

    if pk == "quarter":
        if archetype is None:
            arch = NEW_QUARTER_REST_ARCHETYPES[int(rng.integers(0, len(NEW_QUARTER_REST_ARCHETYPES)))]
        else:
            if archetype not in NEW_QUARTER_REST_ARCHETYPES:
                raise ValueError(
                    f"Archetype {archetype!r} is not in NEW_QUARTER_REST_ARCHETYPES"
                )
            arch = archetype
        _draw_quarter_rest_variant(canvas, arch, cx, cy, sp, color_rgb, style, rng)
        return arch

    if pk in ("eighth", "16th", "32nd"):
        n_hooks = {"eighth": 1, "16th": 2, "32nd": 3}[pk]
        if archetype is None:
            arch = FLAGGED_REST_ARCHETYPES[int(rng.integers(0, len(FLAGGED_REST_ARCHETYPES)))]
        else:
            if archetype not in FLAGGED_REST_ARCHETYPES:
                raise ValueError(
                    f"Archetype {archetype!r} is not in FLAGGED_REST_ARCHETYPES"
                )
            arch = archetype
        _draw_flagged_rest_variant(canvas, n_hooks, arch, cx, cy, sp, color_rgb, style, rng)
        return arch

    if pk in ("whole", "half", "fermata"):
        if archetype is None:
            arch = BAR_REST_ARCHETYPES[int(rng.integers(0, len(BAR_REST_ARCHETYPES)))]
        else:
            if archetype not in BAR_REST_ARCHETYPES:
                raise ValueError(
                    f"Archetype {archetype!r} is not in BAR_REST_ARCHETYPES"
                )
            arch = archetype
        _draw_bar_rest_variant(canvas, pk, arch, cx, cy, sp, color_rgb, style, rng)
        return arch

    raise ValueError(
        f"Unsupported pause_kind {pause_kind!r}; expected one of "
        f"'quarter', 'eighth', '16th', '32nd', 'half', 'whole', 'fermata'."
    )
