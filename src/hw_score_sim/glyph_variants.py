"""Polymorphic handwritten musical glyph generator.

Draws rests ("pauses"), noteheads, stems/flags, beams, clefs, accidentals,
time signatures, barlines, and braces directly onto a StrokeCanvas using
parametric Bezier/polyline strokes with continuous control-point perturbation
and multi-archetype polymorphism.
"""

from __future__ import annotations

from functools import lru_cache
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .rest_variants import (
    BAR_REST_ARCHETYPES,
    FLAGGED_REST_ARCHETYPES,
    NEW_QUARTER_REST_ARCHETYPES,
    draw_pause_variant,
)
from .stroke_engine import (
    RGBLike,
    StrokeCanvas,
    draw_calligraphic_polyline,
    draw_filled_blob,
)
from .structural_glyphs import (
    BARLINE_ARCHETYPES,
    CLEF_ARCHETYPES,
    KEYSIG_ARCHETYPES,
    draw_accidental_variant,
    draw_barline_variant,
    draw_clef_variant,
    draw_keysig_variant,
    resolve_structural_archetypes,
)
from .styles import (
    BEAM_ARCHETYPES,
    BEAM_GROUPING_MODES,
    BEAM_SLOPE_STYLES,
    FLAG_ARCHETYPES,
    LEDGER_STYLES,
    STEM_ARCHETYPES,
    resolve_beam_archetype,
    resolve_beam_grouping_mode,
    resolve_beam_slope_style,
    resolve_flag_archetype,
    resolve_ledger_style,
    resolve_stem_archetype,
)


QUARTER_REST_ARCHETYPES: Tuple[str, ...] = (
    "lightning_z",
    "backwards_7_hook",
    "cursive_3",
    "hurried_slash_zig",
    "baroque_c_hook",
    "calligraphic_s_curl",
    "sharp_v_tick",
    "baroque_flag_loop",
    "modern_tall_z",
    "scribble_s",
)

EIGHTH_REST_ARCHETYPES: Tuple[str, ...] = (
    "dot_head",
    "loop_head",
    "quick_tick_hook",
)

NOTEHEAD_ARCHETYPES: Tuple[str, ...] = (
    "oval",
    "slash",
    "teardrop",
    "compact",
    "heavy_wedge",
    "hollow_diamond",
    "slim_oval",
    "needle_slash",
    "hairline_dash",
    "vertical_tick",
    "pinprick_dot",
    "slim_diamond",
    "boomerang_hook",
    "open_loop",
    "zigzag_fill",
    "broad_nib_lozenge",
    "ink_blot",
    "chevron_wedge",
)

_LEGACY_CLEF_ARCHETYPE_MAP: Dict[str, str] = {
    "ornate": "baroque_ornate_scroll",
    "simplified": "classic_engraved",
    "hurried": "hurried_single_loop",
}


# ---------------------------------------------------------------------------
# Geometry & Sampling Helpers
# ---------------------------------------------------------------------------


@lru_cache(maxsize=64)
def _cubic_basis(n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    t = np.linspace(0.0, 1.0, n)[:, None]
    mt = 1.0 - t
    return mt**3, 3.0 * (mt**2) * t, 3.0 * mt * (t**2), t**3


@lru_cache(maxsize=64)
def _quad_basis(n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = np.linspace(0.0, 1.0, n)[:, None]
    mt = 1.0 - t
    return mt**2, 2.0 * mt * t, t**2


def _eval_cubic(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, n: int = 16) -> np.ndarray:
    b0, b1, b2, b3 = _cubic_basis(n)
    return b0 * p0 + b1 * p1 + b2 * p2 + b3 * p3


def _eval_quad(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, n: int = 12) -> np.ndarray:
    b0, b1, b2 = _quad_basis(n)
    return b0 * p0 + b1 * p1 + b2 * p2


def _eval_catmull_rom(ctrl: np.ndarray, samples_per_seg: int = 10) -> np.ndarray:
    """Evaluate a smooth Catmull-Rom spline passing through all control points."""
    pts = np.asarray(ctrl, dtype=np.float64)
    if len(pts) <= 2:
        return pts
    # Pad endpoints by reflection for natural tangents.
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
    jitter_sp: float = 0.08,
    scale_x_range: Tuple[float, float] = (0.88, 1.14),
    scale_y_range: Tuple[float, float] = (0.88, 1.14),
    extra_slant_deg: float = 0.0,
) -> np.ndarray:
    """Perturb normalized (in sp units) control points with jitter, scale, slant, and offset."""
    pts = np.asarray(pts_norm, dtype=np.float64).copy()
    if jitter_sp > 0.0:
        pts += rng.normal(0.0, jitter_sp, size=pts.shape)

    sx = float(rng.uniform(*scale_x_range))
    sy = float(rng.uniform(*scale_y_range))
    slant_deg = float(style.get("slant_deg", 0.0)) + extra_slant_deg + float(rng.normal(0.0, 4.5))
    shear = math.tan(math.radians(float(np.clip(slant_deg, -28.0, 28.0))))

    x = pts[:, 0] * sx - pts[:, 1] * shear
    y = pts[:, 1] * sy

    ox = float(rng.normal(0.0, 0.04 * sp))
    oy = float(rng.normal(0.0, 0.04 * sp))
    return np.column_stack((cx + ox + x * sp, cy + oy + y * sp))


def _choose_weighted(
    choices: Sequence[str],
    weights_spec: Optional[Union[Dict[str, float], Sequence[float]]],
    rng: np.random.Generator,
    default_weights: Optional[Sequence[float]] = None,
) -> str:
    """Pick an item from choices using dict/sequence weights or defaults."""
    if isinstance(weights_spec, dict) and weights_spec:
        w = np.array([max(0.0, float(weights_spec.get(c, 0.0))) for c in choices], dtype=np.float64)
    elif isinstance(weights_spec, (list, tuple, np.ndarray)) and len(weights_spec) == len(choices):
        w = np.maximum(0.0, np.asarray(weights_spec, dtype=np.float64))
    elif default_weights is not None:
        w = np.maximum(0.0, np.asarray(default_weights, dtype=np.float64))
    else:
        w = np.ones(len(choices), dtype=np.float64)

    total = float(np.sum(w))
    if total <= 1e-12:
        w = np.ones(len(choices), dtype=np.float64)
        total = float(len(choices))
    probs = w / total
    idx = int(rng.choice(len(choices), p=probs))
    return choices[idx]


# ---------------------------------------------------------------------------
# Rests ("Pauses")
# ---------------------------------------------------------------------------


def _draw_quarter_rest_archetype(
    canvas: StrokeCanvas,
    archetype: str,
    cx: float,
    cy: float,
    sp: float,
    color_rgb: RGBLike,
    style: dict,
    rng: np.random.Generator,
) -> None:
    """Render one of 6 distinct handwritten quarter-rest archetypes."""
    base_w = max(1.2, sp * float(style.get("glyph_stroke_ratio", 0.22)))

    if archetype == "lightning_z":
        # Classic sharp manuscript Z / lightning bolt with top flick & lower tail.
        ctrl = np.array([
            [-0.22, -1.42],
            [0.36, -0.58],
            [-0.30, 0.14],
            [0.32, 0.84],
            [-0.22, 1.38],
        ], dtype=np.float64)
        # Sharp zig with slight curved bottom tail.
        pts = _transform_local(ctrl, cx, cy, sp, rng, style, jitter_sp=0.11, scale_x_range=(0.82, 1.22), scale_y_range=(0.86, 1.18))
        # Slightly round only the final hook while keeping upper corners brisk.
        upper = pts[:4]
        tail = _eval_quad(pts[3], pts[3] + np.array([-0.38 * sp, 0.18 * sp]), pts[4], n=10)
        full = np.vstack((upper, tail[1:]))
        draw_calligraphic_polyline(canvas, full, color_rgb, base_w * 1.15, style, kind="rest")

    elif archetype == "backwards_7_hook":
        # Tall slanted backwards-7 / reversed-r stem with bottom curved hook.
        top_flick = np.array([
            [-0.36, -1.25],
            [0.24, -1.36],
            [0.02, -0.35],
            [-0.18, 0.72],
            [0.28, 1.28],
        ], dtype=np.float64)
        pts = _transform_local(top_flick, cx, cy, sp, rng, style, jitter_sp=0.10, scale_x_range=(0.80, 1.20), scale_y_range=(0.88, 1.16))
        # Draw top bar + slanted spine as one brisk stroke, lower curl as smooth curve.
        spine = np.vstack((pts[:2], _eval_quad(pts[1], pts[2], pts[3], n=12)[1:]))
        hook = _eval_quad(pts[3], pts[3] + np.array([0.05 * sp, 0.55 * sp]), pts[4], n=10)
        draw_calligraphic_polyline(canvas, np.vstack((spine, hook[1:])), color_rgb, base_w * 1.1, style, kind="rest")

    elif archetype == "cursive_3":
        # Flowing handwritten '3' / double-belly cursive pause.
        ctrl = np.array([
            [-0.28, -1.32],
            [0.36, -1.10],
            [0.32, -0.45],
            [-0.16, -0.05],
            [0.38, 0.38],
            [0.22, 1.12],
            [-0.32, 1.32],
        ], dtype=np.float64)
        pts = _transform_local(ctrl, cx, cy, sp, rng, style, jitter_sp=0.10, scale_x_range=(0.84, 1.20), scale_y_range=(0.85, 1.18))
        curve = _eval_catmull_rom(pts, samples_per_seg=8)
        draw_calligraphic_polyline(canvas, curve, color_rgb, base_w * 1.05, style, kind="rest")

    elif archetype == "hurried_slash_zig":
        # Fast Beethoven/sketch diagonal slash with tiny top nick & bottom kick.
        ctrl = np.array([
            [0.12, -1.40],
            [0.36, -1.02],
            [-0.26, 0.58],
            [0.22, 1.22],
        ], dtype=np.float64)
        pts = _transform_local(ctrl, cx, cy, sp, rng, style, jitter_sp=0.12, scale_x_range=(0.78, 1.25), scale_y_range=(0.85, 1.20), extra_slant_deg=8.0)
        draw_calligraphic_polyline(canvas, pts, color_rgb, base_w * 1.22, style, kind="rest")

    elif archetype == "baroque_c_hook":
        # Classic engraved-inspired hand: upper zig into pronounced open C-belly.
        upper_ctrl = np.array([
            [-0.16, -1.48],
            [0.34, -0.78],
            [-0.22, -0.04],
            [0.28, 0.56],
        ], dtype=np.float64)
        lower_ctrl = np.array([
            [0.28, 0.56],
            [-0.24, 0.36],
            [-0.38, 0.92],
            [0.08, 1.44],
        ], dtype=np.float64)
        all_ctrl = np.vstack((upper_ctrl, lower_ctrl[1:]))
        pts = _transform_local(all_ctrl, cx, cy, sp, rng, style, jitter_sp=0.09, scale_x_range=(0.85, 1.18), scale_y_range=(0.88, 1.16))
        # Upper zig + smooth C-hook at bottom.
        upper = pts[:4]
        lower = _eval_cubic(pts[3], pts[4], pts[5], pts[6], n=16)
        draw_calligraphic_polyline(canvas, upper, color_rgb, base_w * 1.25, style, kind="rest")
        draw_calligraphic_polyline(canvas, lower, color_rgb, base_w * 1.0, style, kind="rest")

    elif archetype == "calligraphic_s_curl":
        # Sweeping calligraphic S-curve with top serif flick and graceful bottom curl.
        ctrl = np.array([
            [-0.10, -1.52],
            [0.34, -0.95],
            [0.02, -0.22],
            [-0.30, 0.38],
            [0.22, 0.92],
            [-0.28, 1.42],
        ], dtype=np.float64)
        pts = _transform_local(ctrl, cx, cy, sp, rng, style, jitter_sp=0.10, scale_x_range=(0.84, 1.20), scale_y_range=(0.86, 1.18))
        curve = _eval_catmull_rom(pts, samples_per_seg=9)
        draw_calligraphic_polyline(canvas, curve, color_rgb, base_w * 1.12, style, kind="rest")

    elif archetype == "sharp_v_tick":
        # Rapid contemporary / student angular V-tick check mark with brisk lower flick.
        ctrl = np.array([
            [-0.34, -1.28],
            [0.38, -0.42],
            [-0.28, 0.34],
            [0.14, 1.34],
        ], dtype=np.float64)
        pts = _transform_local(ctrl, cx, cy, sp, rng, style, jitter_sp=0.10, scale_x_range=(0.82, 1.22), scale_y_range=(0.86, 1.16))
        draw_calligraphic_polyline(canvas, pts, color_rgb, base_w * 1.18, style, kind="rest")

    elif archetype == "baroque_flag_loop":
        # 18th-century reverse-flag / closed upper loop descending into a sweeping C-tail.
        loop_ctrl = np.array([
            [0.26, -1.18],
            [-0.22, -1.36],
            [-0.36, -0.78],
            [0.24, -0.58],
            [0.10, 0.12],
            [-0.34, 0.76],
            [0.18, 1.38],
        ], dtype=np.float64)
        pts = _transform_local(loop_ctrl, cx, cy, sp, rng, style, jitter_sp=0.09, scale_x_range=(0.84, 1.18), scale_y_range=(0.88, 1.16))
        curve = _eval_catmull_rom(pts, samples_per_seg=8)
        draw_calligraphic_polyline(canvas, curve, color_rgb, base_w * 1.14, style, kind="rest")

    elif archetype == "modern_tall_z":
        # Elongated jazz / conductor tall narrow Z with horizontal top bar & tight lower bracket.
        ctrl = np.array([
            [-0.36, -1.48],
            [0.36, -1.44],
            [-0.32, 0.52],
            [0.34, 0.56],
            [-0.12, 1.46],
        ], dtype=np.float64)
        pts = _transform_local(ctrl, cx, cy, sp, rng, style, jitter_sp=0.09, scale_x_range=(0.82, 1.18), scale_y_range=(0.90, 1.18))
        upper = pts[:4]
        tail = _eval_quad(pts[3], pts[3] + np.array([-0.26 * sp, 0.36 * sp]), pts[4], n=9)
        draw_calligraphic_polyline(canvas, np.vstack((upper, tail[1:])), color_rgb, base_w * 1.16, style, kind="rest")

    else:  # "scribble_s"
        # Loose continuous cursive S-squiggly loop with double-inflection and trailing whip.
        ctrl = np.array([
            [0.30, -1.42],
            [-0.30, -1.04],
            [0.34, -0.44],
            [-0.34, 0.16],
            [0.32, 0.74],
            [-0.18, 1.18],
            [0.24, 1.48],
        ], dtype=np.float64)
        pts = _transform_local(ctrl, cx, cy, sp, rng, style, jitter_sp=0.11, scale_x_range=(0.80, 1.24), scale_y_range=(0.85, 1.18))
        curve = _eval_catmull_rom(pts, samples_per_seg=8)
        draw_calligraphic_polyline(canvas, curve, color_rgb, base_w * 1.04, style, kind="rest")


def _draw_flagged_rest(
    canvas: StrokeCanvas,
    n_hooks: int,
    cx: float,
    cy: float,
    sp: float,
    color_rgb: RGBLike,
    style: dict,
    rng: np.random.Generator,
) -> None:
    """Draw an eighth (1), 16th (2), or 32nd (3) rest with 3 polymorphic archetypes."""
    archetype = _choose_weighted(
        EIGHTH_REST_ARCHETYPES,
        style.get("eighth_rest_archetype_weights"),
        rng,
        default_weights=(0.48, 0.28, 0.24),
    )
    base_w = max(1.0, sp * float(style.get("glyph_stroke_ratio", 0.19)))
    slant_deg = float(style.get("slant_deg", 0.0)) + float(rng.normal(8.0, 5.0))
    shear = math.tan(math.radians(float(np.clip(slant_deg, -20.0, 26.0))))

    top_y = cy - (0.55 + 0.35 * (n_hooks - 1)) * sp
    bot_y = cy + (1.15 + 0.25 * (n_hooks - 1)) * sp
    stem_top = np.array([cx + 0.28 * sp + rng.normal(0, 0.06 * sp), top_y + rng.normal(0, 0.06 * sp)])
    stem_bot = np.array([
        stem_top[0] - (bot_y - top_y) * (0.28 + 0.25 * shear) + rng.normal(0, 0.08 * sp),
        bot_y + rng.normal(0, 0.06 * sp),
    ])
    draw_calligraphic_polyline(canvas, np.vstack((stem_top, stem_bot)), color_rgb, base_w * 1.05, style, kind="rest")

    # Attach n_hooks along the slanted stem.
    for k in range(n_hooks):
        frac = 0.06 + k * (0.34 if n_hooks == 2 else 0.26)
        attach = stem_top * (1.0 - frac) + stem_bot * frac
        arm_dx = (0.58 + rng.uniform(-0.08, 0.10)) * sp
        arm_dy = (-0.12 + rng.uniform(-0.08, 0.08)) * sp
        head_pt = attach + np.array([-arm_dx, arm_dy])

        if archetype == "dot_head":
            # Sweeping concave arm + filled ink dot at left tip.
            mid = 0.5 * (attach + head_pt) + np.array([0.05 * sp, 0.24 * sp + rng.normal(0, 0.05 * sp)])
            arm = _eval_quad(head_pt, mid, attach, n=10)
            draw_calligraphic_polyline(canvas, arm, color_rgb, base_w * 0.9, style, kind="rest")
            dot_r = max(1.1, sp * rng.uniform(0.18, 0.25))
            draw_filled_blob(canvas, head_pt[0], head_pt[1], dot_r, dot_r * rng.uniform(0.85, 1.15), rng.uniform(-30, 30), color_rgb, style, hollow=False)

        elif archetype == "loop_head":
            # Open/closed cursive loop at tip flowing into the arm.
            r = sp * rng.uniform(0.18, 0.26)
            phis = np.linspace(math.pi * 0.2, math.pi * 2.15, 14)
            loop = np.column_stack((head_pt[0] + r * np.cos(phis), head_pt[1] + r * 0.85 * np.sin(phis)))
            mid = 0.5 * (attach + head_pt) + np.array([0.04 * sp, 0.20 * sp])
            arm = _eval_quad(loop[-1], mid, attach, n=8)
            draw_calligraphic_polyline(canvas, np.vstack((loop, arm[1:])), color_rgb, base_w * 0.88, style, kind="rest")

        else:  # "quick_tick_hook"
            # Brisk V/7 tick hook.
            tip = head_pt + np.array([0.05 * sp, -0.18 * sp + rng.normal(0, 0.05 * sp)])
            elbow = head_pt + np.array([0.14 * sp, 0.16 * sp])
            tick = np.vstack((tip, elbow, attach))
            draw_calligraphic_polyline(canvas, tick, color_rgb, base_w * 0.98, style, kind="rest")


def draw_rest(
    canvas: StrokeCanvas,
    rest_type: str,
    cx: float,
    cy: float,
    sp: float,
    color_rgb: RGBLike,
    style: dict,
    rng: np.random.Generator,
) -> None:
    """Draw a handwritten musical rest onto canvas.

    rest_type in {"whole", "half", "quarter", "eighth", "16th", "32nd"}.
    """
    rt = str(rest_type).strip().lower()
    alias_map = {
        "1": "whole",
        "semibreve": "whole",
        "2": "half",
        "minim": "half",
        "4": "quarter",
        "crotchet": "quarter",
        "8": "eighth",
        "quaver": "eighth",
        "16": "16th",
        "semiquaver": "16th",
        "32": "32nd",
        "demisemiquaver": "32nd",
    }
    rt = alias_map.get(rt, rt)

    if rt in ("whole", "half"):
        bar_arch = style.get("bar_rest_archetype")
        if bar_arch in BAR_REST_ARCHETYPES:
            draw_pause_variant(canvas, rt, cx, cy, sp, color_rgb, style, rng, archetype=str(bar_arch))
            return
        # Slightly uneven hand-drawn ink block hanging below (whole) or sitting above (half) the line.
        w = sp * float(rng.uniform(0.92, 1.18))
        h = sp * float(rng.uniform(0.38, 0.52))
        y_center = cy + 0.24 * sp if rt == "whole" else cy - 0.24 * sp
        y_center += float(rng.normal(0.0, 0.04 * sp))
        x_center = cx + float(rng.normal(0.0, 0.04 * sp))
        tilt = float(rng.normal(0.0, 3.5))
        dx = 0.5 * w * math.cos(math.radians(tilt))
        dy = 0.5 * w * math.sin(math.radians(tilt))
        pts = np.array([
            [x_center - dx, y_center - dy],
            [x_center, y_center + float(rng.normal(0.0, 0.04 * sp))],
            [x_center + dx, y_center + dy],
        ], dtype=np.float64)
        local_style = dict(style)
        local_style["taper_start"] = 0.04
        local_style["taper_end"] = 0.04
        local_style["nib_min_ratio"] = max(_get_style_min_ratio(style), 0.78)
        draw_calligraphic_polyline(canvas, pts, color_rgb, h, local_style, kind="beam")
        return

    if rt == "fermata":
        bar_arch = style.get("bar_rest_archetype")
        arch_arg = str(bar_arch) if bar_arch in BAR_REST_ARCHETYPES else None
        draw_pause_variant(canvas, "fermata", cx, cy, sp, color_rgb, style, rng, archetype=arch_arg)
        return

    if rt == "quarter":
        forced = style.get("force_rest_archetype")
        if forced in NEW_QUARTER_REST_ARCHETYPES:
            draw_pause_variant(canvas, "quarter", cx, cy, sp, color_rgb, style, rng, archetype=str(forced))
            return
        if forced in QUARTER_REST_ARCHETYPES:
            arch = str(forced)
        else:
            pause_arch = style.get("pause_archetype")
            if pause_arch in NEW_QUARTER_REST_ARCHETYPES:
                draw_pause_variant(canvas, "quarter", cx, cy, sp, color_rgb, style, rng, archetype=str(pause_arch))
                return
            arch = _choose_weighted(QUARTER_REST_ARCHETYPES, style.get("rest_archetype_weights"), rng)
        _draw_quarter_rest_archetype(canvas, arch, cx, cy, sp, color_rgb, style, rng)
        return

    if rt in ("eighth", "16th", "32nd"):
        flagged_arch = style.get("flagged_rest_archetype")
        if flagged_arch in FLAGGED_REST_ARCHETYPES:
            draw_pause_variant(canvas, rt, cx, cy, sp, color_rgb, style, rng, archetype=str(flagged_arch))
            return
        n_hooks = {"eighth": 1, "16th": 2, "32nd": 3}[rt]
        _draw_flagged_rest(canvas, n_hooks, cx, cy, sp, color_rgb, style, rng)
        return

    # Fallback: draw quarter rest if unknown string.
    arch = _choose_weighted(QUARTER_REST_ARCHETYPES, style.get("rest_archetype_weights"), rng)
    _draw_quarter_rest_archetype(canvas, arch, cx, cy, sp, color_rgb, style, rng)


def _get_style_min_ratio(style: dict) -> float:
    for k in ("nib_min_ratio", "min_width_ratio", "min_ratio"):
        if k in style:
            return float(style[k])
    return 0.35


# ---------------------------------------------------------------------------
# Noteheads
# ---------------------------------------------------------------------------

_DEFAULT_NOTEHEAD_WEIGHTS: Tuple[float, ...] = (
    0.20,  # oval
    0.09,  # slash
    0.09,  # teardrop
    0.08,  # compact
    0.06,  # heavy_wedge
    0.06,  # hollow_diamond
    0.05,  # slim_oval
    0.04,  # needle_slash
    0.04,  # hairline_dash
    0.03,  # vertical_tick
    0.03,  # pinprick_dot
    0.04,  # slim_diamond
    0.03,  # boomerang_hook
    0.04,  # open_loop
    0.03,  # zigzag_fill
    0.03,  # broad_nib_lozenge
    0.03,  # ink_blot
    0.03,  # chevron_wedge
)


def _rot_pts(pts_local: np.ndarray, nx: float, ny: float, rad: float) -> np.ndarray:
    """Rotate local (N, 2) points by ``rad`` radians and translate to ``(nx, ny)``."""
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    return np.column_stack((
        nx + pts_local[:, 0] * cos_a - pts_local[:, 1] * sin_a,
        ny + pts_local[:, 0] * sin_a + pts_local[:, 1] * cos_a,
    ))


def _notehead_aspect_scales(aspect: float) -> Tuple[float, float, float]:
    """Map ``notehead_aspect`` in [0.25, 2.20] to ``(rx_mult, ry_mult, stroke_mult)``.

    Values < 1.0 thin the notehead (needle-slim at < 0.55); values > 1.0 stretch it wide.
    At 1.0 returns (1.0, 1.0, 1.0).
    """
    a = float(np.clip(aspect, 0.25, 2.20))
    if a < 1.0:
        rx_mult = 1.0 + 0.16 * (1.0 - a)
        ry_mult = a
        stroke_mult = max(0.35, a ** 0.72)
    else:
        rx_mult = 1.0 + 0.70 * (a - 1.0)
        ry_mult = 1.0 / (1.0 + 0.16 * (a - 1.0))
        stroke_mult = 1.0 + 0.18 * (a - 1.0)
    return rx_mult, ry_mult, stroke_mult


def _draw_blob_with_fill_density(
    canvas: StrokeCanvas,
    nx: float,
    ny: float,
    rx: float,
    ry: float,
    angle_deg: float,
    color_rgb: RGBLike,
    style: dict,
    rng: np.random.Generator,
    fill_density: float,
    sp: float,
) -> None:
    """Draw a filled blob notehead modulated by ``notehead_fill_density``."""
    fd = float(np.clip(fill_density, 0.25, 1.50))
    if fd < 0.72:
        # Hurried sparse fill: crisp calligraphic outline + quick diagonal scribble stroke(s).
        local_style = dict(style)
        local_style["hollow_stroke_ratio"] = max(0.24, 0.34 * (0.65 + 0.5 * fd))
        gap = float(rng.uniform(0.05, 0.18)) if rng.random() < 0.45 else 0.0
        draw_filled_blob(canvas, nx, ny, rx, ry, angle_deg, color_rgb, local_style, hollow=True, open_gap=gap)
        rad = math.radians(angle_deg)
        n_strokes = 1 if fd < 0.50 else 2
        stroke_w = max(0.75, min(rx, ry) * (0.45 + 0.55 * fd))
        for k in range(n_strokes):
            off = (k - 0.5 * (n_strokes - 1)) * ry * 0.52
            seg = np.array(
                [[-rx * 0.68, off + ry * 0.12], [rx * 0.68, off - ry * 0.12]],
                dtype=np.float64,
            )
            seg += rng.normal(0.0, 0.03 * sp, size=seg.shape)
            draw_calligraphic_polyline(
                canvas,
                _rot_pts(seg, nx, ny, rad),
                color_rgb,
                stroke_w,
                local_style,
                kind="notehead_scribble",
            )
        return

    local_style = dict(style)
    base_op = float(style.get("stroke_opacity", style.get("opacity", 0.92)))
    local_style["stroke_opacity"] = float(np.clip(base_op * (0.82 + 0.18 * fd), 0.25, 1.0))
    draw_filled_blob(canvas, nx, ny, rx, ry, angle_deg, color_rgb, local_style, hollow=False)

    nib_aniso = float(np.clip(style.get("nib_anisotropy", 0.0), 0.0, 1.0))
    if nib_aniso >= 0.25 and fd >= 0.72:
        # Organic broad-edge quill-dab landing stroke along the nib angle.
        nib_rad = math.radians(float(style.get("nib_angle_deg", 35.0)))
        dab_len = min(rx, ry) * (0.45 + 0.30 * nib_aniso)
        dab_seg = np.array(
            [[-dab_len, 0.0], [dab_len, 0.0]],
            dtype=np.float64,
        )
        dab_style = dict(local_style)
        dab_style["taper_start"] = 0.08
        dab_style["taper_end"] = 0.12
        dab_style["wobble_px"] = min(float(style.get("wobble_px", 0.7)), 0.45)
        draw_calligraphic_polyline(
            canvas,
            _rot_pts(dab_seg, nx, ny, -nib_rad),
            color_rgb,
            max(0.85, min(rx, ry) * (0.72 + 0.25 * nib_aniso)),
            dab_style,
            kind="notehead_quill_dab",
        )

    if fd > 1.12:
        # Heavy/saturated ink pass across the body of the notehead.
        rad = math.radians(angle_deg)
        extra_scale = min(1.12, 0.85 + 0.18 * (fd - 1.0))
        seg = np.array([[-rx * 0.65, 0.0], [rx * 0.65, 0.0]], dtype=np.float64)
        draw_calligraphic_polyline(
            canvas,
            _rot_pts(seg, nx, ny, rad),
            color_rgb,
            max(1.0, ry * 1.15 * extra_scale),
            local_style,
            kind="notehead_dense",
        )


def draw_notehead(
    canvas: StrokeCanvas,
    duration_kind: str,
    cx: float,
    cy: float,
    sp: float,
    color_rgb: RGBLike,
    style: dict,
    rng: np.random.Generator,
) -> None:
    """Draw a handwritten notehead ('filled', 'half', or 'whole').

    Supports all 18 morphological archetypes in ``NOTEHEAD_ARCHETYPES``,
    ``notehead_aspect`` in [0.25, 2.20], and ``notehead_fill_density`` in [0.25, 1.50].
    """
    raw_dk = str(duration_kind).strip().lower()
    dk = raw_dk
    if dk in ("quarter", "eighth", "16th", "32nd", "black", "solid"):
        dk = "filled"
    elif dk in ("minim", "2"):
        dk = "half"
    elif dk in ("semibreve", "1"):
        dk = "whole"

    forced = style.get("notehead_archetype")
    if forced in NOTEHEAD_ARCHETYPES:
        arch = str(forced)
    else:
        arch = _choose_weighted(
            NOTEHEAD_ARCHETYPES,
            style.get("notehead_archetype_weights"),
            rng,
            default_weights=_DEFAULT_NOTEHEAD_WEIGHTS,
        )

    from .styles import resolve_ductus_param
    _raw_dur_beats = {
        "whole": 4.0, "semibreve": 4.0, "1": 4.0,
        "half": 2.0, "minim": 2.0, "2": 2.0,
        "quarter": 1.0, "eighth": 0.5, "16th": 0.25, "32nd": 0.125,
    }
    raw_beats = _raw_dur_beats.get(raw_dk, 1.0)
    dur_slope = float(np.clip(resolve_ductus_param(style, "duration_head_scale_slope", 0.0), -0.22, 0.25))
    dur_scale = float(np.clip(1.0 + dur_slope * math.log2(raw_beats), 0.72, 1.28))

    size_j = float(rng.uniform(0.90, 1.12)) * float(style.get("notehead_scale", 1.0)) * dur_scale
    aspect = float(np.clip(style.get("notehead_aspect", 1.0), 0.25, 2.20))
    fill_density = float(np.clip(style.get("notehead_fill_density", 1.0), 0.25, 1.50))
    rx_mult, ry_mult, stroke_mult = _notehead_aspect_scales(aspect)
    fill_w = 0.50 + 0.50 * fill_density  # 0.625 .. 1.25

    base_angle = float(style.get("notehead_angle_deg", -30.0))
    ox = float(rng.normal(0.0, 0.03 * sp))
    # Hard-clamp internal oy to ±0.025*sp so combined pitch-slop <= ±0.17*sp
    oy = float(np.clip(rng.normal(0.0, 0.020 * sp), -0.025 * sp, 0.025 * sp))
    nx, ny = cx + ox, cy + oy

    # ------------------------------------------------------------------
    # Hollow noteheads ("whole" and "half")
    # ------------------------------------------------------------------
    if dk in ("whole", "half"):
        is_whole = dk == "whole"
        w_boost = 1.12 if is_whole else 1.0
        open_prob = float(style.get("hollow_open_prob", 0.30 if is_whole else 0.38))
        gap = float(rng.uniform(0.05, 0.16)) if rng.random() < open_prob else 0.0

        if arch == "hollow_diamond":
            rx = sp * (0.72 if is_whole else 0.66) * size_j * rx_mult * float(rng.uniform(0.94, 1.08))
            ry = sp * (0.50 if is_whole else 0.46) * size_j * ry_mult * float(rng.uniform(0.92, 1.06))
            rad = math.radians(float(rng.normal(base_angle * (0.3 if is_whole else 0.4), 5.5)))
            end_pt = [rx, 0.0] if is_whole else [rx * 0.92, -ry * 0.08]
            verts = np.array([[rx, 0.0], [0.0, -ry], [-rx, 0.0], [0.0, ry], end_pt], dtype=np.float64)
            verts += rng.normal(0.0, 0.035 * sp, size=verts.shape)
            pts = _rot_pts(verts, nx, ny, rad)
            draw_calligraphic_polyline(
                canvas, pts, color_rgb, max(0.85, sp * (0.22 if is_whole else 0.20) * size_j * stroke_mult), style, kind="hollow_notehead"
            )
            return

        if arch == "slim_oval":
            rx = sp * 0.72 * w_boost * size_j * rx_mult * float(rng.uniform(0.94, 1.08))
            ry = sp * 0.23 * size_j * ry_mult * float(rng.uniform(0.92, 1.08))
            angle = float(rng.normal(base_angle - 4.0, 5.5))
            local_style = dict(style)
            local_style["hollow_stroke_ratio"] = float(style.get("hollow_stroke_ratio", 0.44)) * 0.68
            draw_filled_blob(canvas, nx, ny, rx, ry, angle, color_rgb, local_style, hollow=True, open_gap=gap)
            return

        if arch == "needle_slash":
            # Razor-thin double-hairline needle loop enclosing a narrow slit.
            rx = sp * 0.76 * w_boost * size_j * rx_mult
            ry = sp * 0.16 * size_j * ry_mult
            rad = math.radians(float(rng.normal(base_angle - 14.0, 6.0)))
            top_arc = _eval_quad(
                np.array([-rx, 0.0]),
                np.array([0.0, -ry * 1.45]),
                np.array([rx, 0.0]),
                n=10,
            )
            bot_arc = _eval_quad(
                np.array([rx, 0.0]),
                np.array([0.0, ry * 1.45]),
                np.array([-rx * (0.85 if gap > 0 else 1.0), ry * (0.25 if gap > 0 else 0.0)]),
                n=10,
            )
            pts = _rot_pts(np.vstack((top_arc, bot_arc[1:])), nx, ny, rad)
            local_style = dict(style)
            local_style["wobble_px"] = min(float(style.get("wobble_px", 0.7)), 0.45)
            draw_calligraphic_polyline(
                canvas, pts, color_rgb, max(0.55, sp * 0.11 * size_j * stroke_mult), local_style, kind="hollow_notehead"
            )
            return

        if arch == "hairline_dash":
            # Flat horizontal hairline capsule / hollow bracket slit.
            rx = sp * 0.72 * w_boost * size_j * rx_mult
            ry = sp * 0.18 * size_j * ry_mult
            rad = math.radians(float(rng.normal(0.0, 5.0)))
            upper = _eval_quad(np.array([-rx, -ry * 0.3]), np.array([0.0, -ry * 1.2]), np.array([rx, -ry * 0.2]), n=9)
            lower = _eval_quad(np.array([rx, ry * 0.2]), np.array([0.0, ry * 1.2]), np.array([-rx, ry * 0.3]), n=9)
            pts = _rot_pts(np.vstack((upper, lower)), nx, ny, rad)
            local_style = dict(style)
            local_style["wobble_px"] = min(float(style.get("wobble_px", 0.7)), 0.45)
            draw_calligraphic_polyline(
                canvas, pts, color_rgb, max(0.55, sp * 0.11 * size_j * stroke_mult), local_style, kind="hollow_notehead"
            )
            return

        if arch == "vertical_tick":
            # Tall narrow vertical hollow slit.
            rx = sp * 0.22 * w_boost * size_j * ry_mult
            ry = sp * 0.56 * size_j * rx_mult
            angle = float(rng.normal(0.0, 7.0))
            local_style = dict(style)
            local_style["hollow_stroke_ratio"] = float(style.get("hollow_stroke_ratio", 0.44)) * 0.65
            draw_filled_blob(canvas, nx, ny, rx, ry, angle, color_rgb, local_style, hollow=True, open_gap=gap)
            return

        if arch == "pinprick_dot":
            # Tiny delicate hollow eyelet ring.
            rx = sp * 0.32 * w_boost * size_j * rx_mult
            ry = sp * 0.26 * size_j * ry_mult
            angle = float(rng.normal(base_angle * 0.5, 8.0))
            local_style = dict(style)
            local_style["hollow_stroke_ratio"] = 0.34
            draw_filled_blob(canvas, nx, ny, rx, ry, angle, color_rgb, local_style, hollow=True, open_gap=gap)
            return

        if arch == "slim_diamond":
            # Razor-sharp elongated narrow hollow diamond.
            rx = sp * 0.76 * w_boost * size_j * rx_mult
            ry = sp * 0.24 * size_j * ry_mult
            rad = math.radians(float(rng.normal(base_angle * 0.6, 5.5)))
            verts = np.array([[rx, 0.0], [0.0, -ry], [-rx, 0.0], [0.0, ry], [rx, 0.0]], dtype=np.float64)
            verts += rng.normal(0.0, 0.025 * sp, size=verts.shape)
            pts = _rot_pts(verts, nx, ny, rad)
            draw_calligraphic_polyline(
                canvas, pts, color_rgb, max(0.60, sp * 0.13 * size_j * stroke_mult), style, kind="hollow_notehead"
            )
            return

        if arch == "boomerang_hook":
            # Crescent / curved boomerang hollow loop.
            rx = sp * 0.68 * w_boost * size_j * rx_mult
            ry = sp * 0.42 * size_j * ry_mult
            rad = math.radians(float(rng.normal(base_angle * 0.5, 7.0)))
            outer = _eval_quad(np.array([-rx, ry * 0.45]), np.array([0.0, -ry * 1.15]), np.array([rx, ry * 0.45]), n=12)
            inner = _eval_quad(np.array([rx, ry * 0.45]), np.array([0.0, -ry * 0.05]), np.array([-rx * 0.88, ry * 0.38]), n=10)
            pts = _rot_pts(np.vstack((outer, inner[1:])), nx, ny, rad)
            draw_calligraphic_polyline(
                canvas, pts, color_rgb, max(0.75, sp * 0.16 * size_j * stroke_mult), style, kind="hollow_notehead"
            )
            return

        if arch == "open_loop":
            # Deliberately unclosed cursive C/e loop with trailing tail.
            rx = sp * 0.64 * w_boost * size_j * rx_mult
            ry = sp * 0.44 * size_j * ry_mult
            angle = float(rng.normal(base_angle, 7.0))
            forced_gap = float(rng.uniform(0.22, 0.38))
            draw_filled_blob(canvas, nx, ny, rx, ry, angle, color_rgb, style, hollow=True, open_gap=forced_gap)
            return

        if arch == "zigzag_fill":
            # Angular sketchy polygonal hollow loop with brisk overshoots.
            rx = sp * 0.66 * w_boost * size_j * rx_mult
            ry = sp * 0.44 * size_j * ry_mult
            rad = math.radians(float(rng.normal(base_angle * 0.6, 7.0)))
            verts = np.array([
                [-rx * 1.05, -ry * 0.25],
                [rx * 0.25, -ry * 0.95],
                [rx * 1.05, ry * 0.10],
                [-rx * 0.15, ry * 0.95],
                [-rx * 0.95, -ry * 0.05],
            ], dtype=np.float64)
            verts += rng.normal(0.0, 0.05 * sp, size=verts.shape)
            pts = _rot_pts(verts, nx, ny, rad)
            draw_calligraphic_polyline(
                canvas, pts, color_rgb, max(0.80, sp * 0.18 * size_j * stroke_mult), style, kind="hollow_notehead"
            )
            return

        if arch == "broad_nib_lozenge":
            # Sheared calligraphic parallelogram hollow lozenge.
            rx = sp * 0.66 * w_boost * size_j * rx_mult
            ry = sp * 0.40 * size_j * ry_mult
            shear = rx * 0.38
            rad = math.radians(float(rng.normal(base_angle * 0.45, 5.0)))
            verts = np.array([
                [-rx + shear, -ry],
                [rx + shear, -ry],
                [rx - shear, ry],
                [-rx - shear, ry],
                [-rx + shear * 0.9, -ry * 0.92],
            ], dtype=np.float64)
            verts += rng.normal(0.0, 0.03 * sp, size=verts.shape)
            pts = _rot_pts(verts, nx, ny, rad)
            draw_calligraphic_polyline(
                canvas, pts, color_rgb, max(0.85, sp * 0.20 * size_j * stroke_mult), style, kind="hollow_notehead"
            )
            return

        if arch == "ink_blot":
            # Heavy, uneven, heavily-pooled hollow ink ring.
            rx = sp * 0.70 * w_boost * size_j * rx_mult
            ry = sp * 0.52 * size_j * ry_mult
            angle = float(rng.normal(base_angle * 0.6, 8.0))
            local_style = dict(style)
            local_style["hollow_stroke_ratio"] = float(style.get("hollow_stroke_ratio", 0.44)) * 1.45
            local_style["ink_pooling"] = min(0.85, float(style.get("ink_pooling", 0.2)) + 0.30)
            draw_filled_blob(canvas, nx, ny, rx, ry, angle, color_rgb, local_style, hollow=True, open_gap=gap * 0.5)
            return

        if arch == "chevron_wedge":
            # Crisp hollow triangular/chevron arrowhead wedge.
            rx = sp * 0.70 * w_boost * size_j * rx_mult
            ry = sp * 0.46 * size_j * ry_mult
            rad = math.radians(float(rng.normal(base_angle * 0.35, 6.0)))
            verts = np.array([
                [-rx, -ry],
                [rx, 0.0],
                [-rx, ry],
                [-rx * 0.82, -ry * 0.90],
            ], dtype=np.float64)
            verts += rng.normal(0.0, 0.035 * sp, size=verts.shape)
            pts = _rot_pts(verts, nx, ny, rad)
            draw_calligraphic_polyline(
                canvas, pts, color_rgb, max(0.80, sp * 0.19 * size_j * stroke_mult), style, kind="hollow_notehead"
            )
            return

        if arch == "heavy_wedge":
            rx = sp * (0.72 if is_whole else 0.68) * size_j * rx_mult
            ry = sp * (0.48 if is_whole else 0.44) * size_j * ry_mult
            angle = float(rng.normal(base_angle - 6.0, 5.5))
            local_style = dict(style)
            local_style["hollow_stroke_ratio"] = float(style.get("hollow_stroke_ratio", 0.44)) * 1.28
            draw_filled_blob(canvas, nx, ny, rx, ry, angle, color_rgb, local_style, hollow=True, open_gap=gap)
            return

        if arch == "slash":
            rx = sp * (0.72 if is_whole else 0.66) * size_j * rx_mult
            ry = sp * (0.40 if is_whole else 0.36) * size_j * ry_mult
            angle = float(rng.normal(base_angle - 8.0, 6.0))
        elif arch == "compact":
            rx = sp * (0.62 if is_whole else 0.56) * size_j * rx_mult
            ry = sp * (0.46 if is_whole else 0.44) * size_j * ry_mult
            angle = float(rng.normal(base_angle + 6.0, 7.0))
        elif arch == "teardrop":
            rx = sp * (0.70 if is_whole else 0.64) * size_j * rx_mult
            ry = sp * (0.46 if is_whole else 0.44) * size_j * ry_mult
            angle = float(rng.normal(base_angle - 3.0, 6.0))
            draw_filled_blob(canvas, nx, ny, rx, ry, angle, color_rgb, style, hollow=True, open_gap=gap)
            rad = math.radians(angle)
            sign = 1.0 if rng.random() < 0.5 else -1.0
            flick = _rot_pts(np.array([[sign * rx * 0.85, 0.0], [sign * rx * 1.22, -ry * 0.12]], dtype=np.float64), nx, ny, rad)
            draw_calligraphic_polyline(canvas, flick, color_rgb, max(0.7, sp * 0.16 * size_j * stroke_mult), style, kind="notehead_flick")
            return
        else:  # "oval"
            rx = sp * (0.72 if is_whole else 0.64) * size_j * rx_mult * float(rng.uniform(0.94, 1.08))
            ry = sp * (0.48 if is_whole else 0.46) * size_j * ry_mult * float(rng.uniform(0.92, 1.06))
            angle = float(rng.normal(base_angle * (0.4 if is_whole else 1.0), 6.5))

        draw_filled_blob(canvas, nx, ny, rx, ry, angle, color_rgb, style, hollow=True, open_gap=gap)
        return

    # ------------------------------------------------------------------
    # Filled noteheads ("filled")
    # ------------------------------------------------------------------
    if arch == "heavy_wedge":
        angle = math.radians(float(rng.normal(base_angle - 5.0, 6.0)))
        half_len = sp * 0.64 * size_j * rx_mult
        dx = half_len * math.cos(angle)
        dy = half_len * math.sin(angle)
        px = -0.14 * sp * ry_mult * math.sin(angle)
        py = 0.14 * sp * ry_mult * math.cos(angle)
        pts1 = np.array([[nx - dx, ny - dy], [nx + dx, ny + dy]], dtype=np.float64)
        pts2 = np.array([[nx - dx * 0.78 + px, ny - dy * 0.78 + py], [nx + dx * 0.85 - px, ny + dy * 0.85 - py]], dtype=np.float64)
        local_style = dict(style)
        local_style["taper_start"] = 0.04
        local_style["taper_end"] = 0.06
        local_style["nib_min_ratio"] = max(_get_style_min_ratio(style), 0.72)
        draw_calligraphic_polyline(canvas, pts1, color_rgb, max(0.8, sp * 0.72 * size_j * stroke_mult * fill_w), local_style, kind="notehead_wedge")
        if fill_density >= 0.55:
            draw_calligraphic_polyline(canvas, pts2, color_rgb, max(0.7, sp * 0.58 * size_j * stroke_mult * fill_w), local_style, kind="notehead_wedge")
        return

    if arch == "hollow_diamond":
        rx = sp * 0.62 * size_j * rx_mult * float(rng.uniform(0.92, 1.08))
        ry = sp * 0.45 * size_j * ry_mult * float(rng.uniform(0.92, 1.08))
        rad = math.radians(float(rng.normal(base_angle * 0.5, 6.5)))
        verts = np.array([[rx, 0.0], [0.0, -ry], [-rx, 0.0], [0.0, ry], [rx, 0.0]], dtype=np.float64)
        verts += rng.normal(0.0, 0.04 * sp, size=verts.shape)
        pts = _rot_pts(verts, nx, ny, rad)
        draw_calligraphic_polyline(canvas, pts, color_rgb, max(0.85, sp * 0.24 * size_j * stroke_mult), style, kind="notehead_diamond")
        slash = np.array([[-rx * 0.55, ry * 0.12], [rx * 0.55, -ry * 0.12]], dtype=np.float64)
        draw_calligraphic_polyline(
            canvas, _rot_pts(slash, nx, ny, rad), color_rgb, max(0.85, sp * 0.38 * size_j * stroke_mult * fill_w), style, kind="notehead_diamond"
        )
        if fill_density > 1.15:
            slash2 = np.array([[-rx * 0.35, -ry * 0.28], [rx * 0.35, ry * 0.28]], dtype=np.float64)
            draw_calligraphic_polyline(
                canvas, _rot_pts(slash2, nx, ny, rad), color_rgb, max(0.85, sp * 0.32 * size_j * stroke_mult), style, kind="notehead_diamond"
            )
        return

    if arch == "slash":
        angle = math.radians(float(rng.normal(base_angle - 8.0, 8.0)))
        half_len = sp * 0.60 * size_j * rx_mult
        dx = half_len * math.cos(angle)
        dy = half_len * math.sin(angle)
        mid_off = float(rng.normal(0.0, 0.06 * sp))
        pts = np.array([
            [nx - dx, ny - dy],
            [nx - mid_off * math.sin(angle), ny + mid_off * math.cos(angle)],
            [nx + dx, ny + dy],
        ], dtype=np.float64)
        local_style = dict(style)
        local_style["taper_start"] = 0.16
        local_style["taper_end"] = 0.20
        local_style["nib_min_ratio"] = max(_get_style_min_ratio(style), 0.65)
        draw_calligraphic_polyline(canvas, pts, color_rgb, max(0.75, sp * 0.56 * size_j * stroke_mult * fill_w), local_style, kind="notehead_slash")
        return

    if arch == "slim_oval":
        rx = sp * 0.68 * size_j * rx_mult * float(rng.uniform(0.93, 1.08))
        ry = sp * 0.21 * size_j * ry_mult * float(rng.uniform(0.90, 1.10))
        angle = float(rng.normal(base_angle - 4.0, 6.0))
        _draw_blob_with_fill_density(canvas, nx, ny, rx, ry, angle, color_rgb, style, rng, fill_density, sp)
        return

    if arch == "needle_slash":
        # Razor-thin, elongated diagonal needle slash.
        angle = math.radians(float(rng.normal(base_angle - 14.0, 7.0)))
        half_len = sp * 0.78 * size_j * rx_mult
        dx = half_len * math.cos(angle)
        dy = half_len * math.sin(angle)
        mid_off = float(rng.normal(0.0, 0.03 * sp))
        pts = np.array([
            [nx - dx, ny - dy],
            [nx - mid_off * math.sin(angle), ny + mid_off * math.cos(angle)],
            [nx + dx, ny + dy],
        ], dtype=np.float64)
        local_style = dict(style)
        local_style["taper_start"] = 0.25
        local_style["taper_end"] = 0.28
        local_style["wobble_px"] = min(float(style.get("wobble_px", 0.7)), 0.55)
        local_style["nib_min_ratio"] = max(_get_style_min_ratio(style), 0.55)
        draw_calligraphic_polyline(canvas, pts, color_rgb, max(0.50, sp * 0.16 * size_j * stroke_mult * fill_w), local_style, kind="notehead_needle")
        return

    if arch == "hairline_dash":
        # Ultra-thin nearly horizontal hairline dash.
        angle = math.radians(float(rng.normal(0.0, 5.5)))
        half_len = sp * 0.70 * size_j * rx_mult
        dx = half_len * math.cos(angle)
        dy = half_len * math.sin(angle)
        bow = float(rng.normal(0.0, 0.035 * sp))
        pts = np.array([
            [nx - dx, ny - dy],
            [nx, ny + bow],
            [nx + dx, ny + dy],
        ], dtype=np.float64)
        local_style = dict(style)
        local_style["taper_start"] = 0.18
        local_style["taper_end"] = 0.18
        local_style["wobble_px"] = min(float(style.get("wobble_px", 0.7)), 0.50)
        local_style["nib_min_ratio"] = max(_get_style_min_ratio(style), 0.60)
        draw_calligraphic_polyline(canvas, pts, color_rgb, max(0.50, sp * 0.15 * size_j * stroke_mult * fill_w), local_style, kind="notehead_hairline")
        if fill_density > 1.15:
            pts2 = pts + np.array([0.0, max(0.6, sp * 0.08 * ry_mult)])
            draw_calligraphic_polyline(canvas, pts2, color_rgb, max(0.45, sp * 0.11 * size_j * stroke_mult), local_style, kind="notehead_hairline")
        return

    if arch == "vertical_tick":
        # Sharp, slim near-vertical pen stab / tick mark.
        angle = math.radians(float(rng.normal(86.0, 8.0)))
        half_len = sp * 0.50 * size_j * rx_mult
        dx = half_len * math.cos(angle)
        dy = half_len * math.sin(angle)
        hook_x = float(rng.uniform(-0.14, 0.14)) * sp * size_j
        pts = np.array([
            [nx - dx + hook_x, ny - dy],
            [nx, ny],
            [nx + dx, ny + dy],
        ], dtype=np.float64)
        local_style = dict(style)
        local_style["taper_start"] = 0.15
        local_style["taper_end"] = 0.24
        local_style["wobble_px"] = min(float(style.get("wobble_px", 0.7)), 0.55)
        draw_calligraphic_polyline(canvas, pts, color_rgb, max(0.55, sp * 0.19 * size_j * stroke_mult * fill_w), local_style, kind="notehead_tick")
        return

    if arch == "pinprick_dot":
        # Tiny sharp pinprick / spider-leg dot head.
        rx = sp * 0.24 * size_j * rx_mult * float(rng.uniform(0.90, 1.10))
        ry = sp * 0.21 * size_j * ry_mult * float(rng.uniform(0.90, 1.10))
        angle = float(rng.normal(base_angle * 0.5, 10.0))
        _draw_blob_with_fill_density(canvas, nx, ny, rx, ry, angle, color_rgb, style, rng, fill_density, sp)
        return

    if arch == "slim_diamond":
        # Razor-sharp narrow elongated diamond / spearhead.
        rx = sp * 0.74 * size_j * rx_mult * float(rng.uniform(0.93, 1.07))
        ry = sp * 0.22 * size_j * ry_mult * float(rng.uniform(0.90, 1.10))
        rad = math.radians(float(rng.normal(base_angle * 0.65, 6.0)))
        verts = np.array([[rx, 0.0], [0.0, -ry], [-rx, 0.0], [0.0, ry], [rx, 0.0]], dtype=np.float64)
        verts += rng.normal(0.0, 0.025 * sp, size=verts.shape)
        draw_calligraphic_polyline(
            canvas, _rot_pts(verts, nx, ny, rad), color_rgb, max(0.60, sp * 0.14 * size_j * stroke_mult), style, kind="notehead_slim_diamond"
        )
        spine = np.array([[-rx * 0.72, 0.0], [rx * 0.72, 0.0]], dtype=np.float64)
        draw_calligraphic_polyline(
            canvas, _rot_pts(spine, nx, ny, rad), color_rgb, max(0.60, sp * 0.22 * size_j * stroke_mult * fill_w), style, kind="notehead_slim_diamond"
        )
        return

    if arch == "boomerang_hook":
        # Erratic curved boomerang / check-hook flick notehead.
        rx = sp * 0.68 * size_j * rx_mult
        ry = sp * 0.42 * size_j * ry_mult
        rad = math.radians(float(rng.normal(base_angle * 0.55, 8.0)))
        ctrl = np.array([
            [-rx, ry * 0.42],
            [-rx * 0.08, -ry * 0.95],
            [rx, ry * 0.48],
        ], dtype=np.float64)
        ctrl += rng.normal(0.0, 0.04 * sp, size=ctrl.shape)
        curve = _eval_quad(ctrl[0], ctrl[1], ctrl[2], n=14)
        draw_calligraphic_polyline(
            canvas, _rot_pts(curve, nx, ny, rad), color_rgb, max(0.80, sp * 0.34 * size_j * stroke_mult * fill_w), style, kind="notehead_boomerang"
        )
        if fill_density > 0.90:
            inner = _eval_quad(ctrl[0] * 0.65, ctrl[1] * 0.55, ctrl[2] * 0.65, n=10)
            draw_calligraphic_polyline(
                canvas, _rot_pts(inner, nx, ny, rad), color_rgb, max(0.70, sp * 0.24 * size_j * stroke_mult * fill_w), style, kind="notehead_boomerang"
            )
        return

    if arch == "open_loop":
        # Hurried unclosed cursive spiral / 'e' loop with optional interior slash.
        rx = sp * 0.62 * size_j * rx_mult
        ry = sp * 0.42 * size_j * ry_mult
        rad = math.radians(float(rng.normal(base_angle, 7.5)))
        n_pts = 22
        start_phi = float(rng.uniform(-0.4, 0.4))
        turns = 1.10 + 0.30 * min(1.3, fill_density)
        phis = np.linspace(start_phi, start_phi + turns * 2.0 * math.pi, n_pts)
        shrink = np.linspace(1.05, 0.55, n_pts)
        spiral = np.column_stack((rx * shrink * np.cos(phis), ry * shrink * np.sin(phis)))
        draw_calligraphic_polyline(
            canvas, _rot_pts(spiral, nx, ny, rad), color_rgb, max(0.75, sp * 0.26 * size_j * stroke_mult * fill_w), style, kind="notehead_open_loop"
        )
        if fill_density >= 0.65:
            dash = np.array([[-rx * 0.55, ry * 0.15], [rx * 0.60, -ry * 0.18]], dtype=np.float64)
            draw_calligraphic_polyline(
                canvas, _rot_pts(dash, nx, ny, rad), color_rgb, max(0.70, sp * 0.28 * size_j * stroke_mult * fill_w), style, kind="notehead_open_loop"
            )
        return

    if arch == "zigzag_fill":
        # Wild back-and-forth scribbly Z / M zigzag fill notehead.
        rx = sp * 0.66 * size_j * rx_mult
        ry = sp * 0.44 * size_j * ry_mult
        rad = math.radians(float(rng.normal(base_angle * 0.65, 8.0)))
        n_legs = 2 if fill_density < 0.55 else (3 if fill_density < 1.10 else 4)
        xs_zig = np.linspace(-rx, rx, n_legs + 2)
        ys_zig = np.array([(-ry if i % 2 == 0 else ry) * float(rng.uniform(0.78, 1.08)) for i in range(n_legs + 2)])
        zig = np.column_stack((xs_zig, ys_zig))
        zig += rng.normal(0.0, 0.04 * sp, size=zig.shape)
        draw_calligraphic_polyline(
            canvas, _rot_pts(zig, nx, ny, rad), color_rgb, max(0.75, sp * 0.28 * size_j * stroke_mult * fill_w), style, kind="notehead_zigzag"
        )
        return

    if arch == "broad_nib_lozenge":
        # Calligraphic broad-nib sheared parallelogram lozenge.
        rx = sp * 0.64 * size_j * rx_mult
        ry = sp * 0.38 * size_j * ry_mult
        shear = rx * 0.40
        rad = math.radians(float(rng.normal(base_angle * 0.45, 5.5)))
        verts = np.array([
            [-rx + shear, -ry],
            [rx + shear, -ry],
            [rx - shear, ry],
            [-rx - shear, ry],
            [-rx + shear, -ry],
        ], dtype=np.float64)
        verts += rng.normal(0.0, 0.025 * sp, size=verts.shape)
        local_style = dict(style)
        local_style["taper_start"] = 0.04
        local_style["taper_end"] = 0.05
        local_style["nib_min_ratio"] = max(_get_style_min_ratio(style), 0.70)
        draw_calligraphic_polyline(
            canvas, _rot_pts(verts, nx, ny, rad), color_rgb, max(0.80, sp * 0.22 * size_j * stroke_mult), local_style, kind="notehead_lozenge"
        )
        n_bars = 1 if fill_density < 0.60 else (2 if fill_density < 1.15 else 3)
        for k in range(n_bars):
            frac = 0.0 if n_bars == 1 else (k / (n_bars - 1) - 0.5) * 1.1
            y_bar = frac * ry
            x_shift = -frac * shear
            bar = np.array([[-rx * 0.85 + x_shift, y_bar], [rx * 0.85 + x_shift, y_bar]], dtype=np.float64)
            draw_calligraphic_polyline(
                canvas, _rot_pts(bar, nx, ny, rad), color_rgb, max(0.80, sp * 0.36 * size_j * stroke_mult * fill_w), local_style, kind="notehead_lozenge"
            )
        return

    if arch == "ink_blot":
        # Heavy, irregular, pooled ink-blot notehead with optional satellite lobe.
        rx = sp * 0.68 * size_j * rx_mult * float(rng.uniform(0.95, 1.12))
        ry = sp * 0.54 * size_j * ry_mult * float(rng.uniform(0.95, 1.12))
        angle = float(rng.normal(base_angle * 0.7, 10.0))
        local_style = dict(style)
        local_style["wobble_px"] = float(style.get("wobble_px", 0.7)) + 0.8
        local_style["ink_pooling"] = min(0.85, float(style.get("ink_pooling", 0.2)) + 0.35)
        _draw_blob_with_fill_density(canvas, nx, ny, rx, ry, angle, color_rgb, local_style, rng, fill_density, sp)
        if fill_density >= 0.60:
            lobe_ang = float(rng.uniform(0.0, 2.0 * math.pi))
            lx = nx + rx * 0.35 * math.cos(lobe_ang)
            ly = ny + ry * 0.35 * math.sin(lobe_ang)
            draw_filled_blob(
                canvas,
                lx,
                ly,
                rx * 0.42 * min(1.2, fill_density),
                ry * 0.38 * min(1.2, fill_density),
                angle + float(rng.normal(0.0, 25.0)),
                color_rgb,
                local_style,
                hollow=False,
            )
        return

    if arch == "chevron_wedge":
        # Sharp angular chevron / arrowhead wedge notehead.
        rx = sp * 0.68 * size_j * rx_mult
        ry = sp * 0.44 * size_j * ry_mult
        rad = math.radians(float(rng.normal(base_angle * 0.4, 6.5)))
        outer = np.array([[-rx, -ry], [rx, 0.0], [-rx, ry]], dtype=np.float64)
        outer += rng.normal(0.0, 0.03 * sp, size=outer.shape)
        local_style = dict(style)
        local_style["taper_start"] = 0.06
        local_style["taper_end"] = 0.08
        draw_calligraphic_polyline(
            canvas, _rot_pts(outer, nx, ny, rad), color_rgb, max(0.80, sp * 0.32 * size_j * stroke_mult * fill_w), local_style, kind="notehead_chevron"
        )
        if fill_density >= 0.50:
            inner = np.array([[-rx * 0.78, -ry * 0.45], [rx * 0.55, 0.0], [-rx * 0.78, ry * 0.45]], dtype=np.float64)
            draw_calligraphic_polyline(
                canvas, _rot_pts(inner, nx, ny, rad), color_rgb, max(0.75, sp * 0.30 * size_j * stroke_mult * fill_w), local_style, kind="notehead_chevron"
            )
        if fill_density > 1.10:
            spine = np.array([[-rx * 0.85, 0.0], [rx * 0.65, 0.0]], dtype=np.float64)
            draw_calligraphic_polyline(
                canvas, _rot_pts(spine, nx, ny, rad), color_rgb, max(0.75, sp * 0.34 * size_j * stroke_mult), local_style, kind="notehead_chevron"
            )
        return

    if arch == "teardrop":
        rx = sp * 0.60 * size_j * rx_mult
        ry = sp * 0.43 * size_j * ry_mult
        angle = float(rng.normal(base_angle - 3.0, 6.0))
        _draw_blob_with_fill_density(canvas, nx, ny, rx, ry, angle, color_rgb, style, rng, fill_density, sp)
        rad = math.radians(angle)
        sign = 1.0 if rng.random() < 0.5 else -1.0
        tip = np.array([nx + sign * rx * 1.18 * math.cos(rad), ny + sign * rx * 1.18 * math.sin(rad)])
        base = np.array([nx + sign * rx * 0.4 * math.cos(rad), ny + sign * rx * 0.4 * math.sin(rad)])
        draw_calligraphic_polyline(canvas, np.vstack((base, tip)), color_rgb, max(0.65, sp * 0.24 * size_j * stroke_mult * fill_w), style, kind="notehead_flick")
        return

    if arch == "compact":
        rx = sp * 0.50 * size_j * rx_mult * float(rng.uniform(0.92, 1.08))
        ry = sp * 0.43 * size_j * ry_mult * float(rng.uniform(0.92, 1.08))
        angle = float(rng.normal(base_angle + 8.0, 9.0))
        _draw_blob_with_fill_density(canvas, nx, ny, rx, ry, angle, color_rgb, style, rng, fill_density, sp)
        return

    # Default "oval"
    rx = sp * 0.62 * size_j * rx_mult * float(rng.uniform(0.93, 1.08))
    ry = sp * 0.45 * size_j * ry_mult * float(rng.uniform(0.92, 1.08))
    angle = float(rng.normal(base_angle, 6.5))
    _draw_blob_with_fill_density(canvas, nx, ny, rx, ry, angle, color_rgb, style, rng, fill_density, sp)


# ---------------------------------------------------------------------------
# Stems, Flags, Beams & Ledger Lines (52+5 Morphological Archetypes)
# ---------------------------------------------------------------------------


def get_stem_length_multiplier(style: dict) -> float:
    """Return effective stem length multiplier from stem_length_scale & stem_archetype."""
    scale = float(np.clip(style.get("stem_length_scale", 1.0), 0.65, 1.55))
    arch = resolve_stem_archetype(style)
    if arch == "short_stubby":
        scale *= 0.66
    elif arch == "tall_spindly":
        scale *= 1.42
    return float(np.clip(scale, 0.50, 1.85))


def get_beam_spacing_sp(style: dict) -> float:
    """Return inter-beam vertical spacing in staff spaces for multi-beam groups."""
    arch = resolve_beam_archetype(style)
    if arch == "wide_spaced_bars":
        return 0.92
    if arch == "tight_crammed_bars":
        return 0.42
    return 0.64


def _sample_point_along_stem(stem_pts: np.ndarray, frac_from_tip: float) -> np.ndarray:
    """Return (x, y) along stem_pts at fraction `frac_from_tip` in [0, 1] measured from tip (end)."""
    if len(stem_pts) == 0:
        return np.zeros(2, dtype=np.float64)
    if len(stem_pts) == 1:
        return stem_pts[0].copy()
    # stem_pts runs from base (index 0) to tip (index -1)
    t_from_base = float(np.clip(1.0 - frac_from_tip, 0.0, 1.0))
    idx_f = t_from_base * (len(stem_pts) - 1)
    i0 = int(math.floor(idx_f))
    i1 = min(len(stem_pts) - 1, i0 + 1)
    a = idx_f - i0
    return (1.0 - a) * stem_pts[i0] + a * stem_pts[i1]


def _draw_single_flag_archetype(
    canvas: StrokeCanvas,
    arch: str,
    fx: float,
    fy: float,
    sp: float,
    toward_head: float,
    flag_w: float,
    color_rgb: RGBLike,
    style: dict,
    rng: np.random.Generator,
) -> None:
    """Draw one of 20 visually distinct handwritten flag archetypes anchored at (fx, fy)."""
    th = float(toward_head)  # +1.0 for stem_up (downward), -1.0 for stem_down (upward)
    j = lambda s=0.04: float(rng.normal(0.0, s * sp))

    def _p(dx_sp: float, dy_sp: float) -> np.ndarray:
        return np.array([fx + dx_sp * sp + j(0.025), fy + dy_sp * sp * th + j(0.025)], dtype=np.float64)

    if arch == "classic_droop":
        # Traditional graceful S-curve droop tapering gently inward.
        fpts = _eval_cubic(_p(0.0, 0.0), _p(0.82, 0.30), _p(1.12, 0.85), _p(0.36, 1.48), n=16)
        draw_calligraphic_polyline(canvas, fpts, color_rgb, flag_w, style, kind="flag")

    elif arch == "baroque_curl":
        # Ornate 18th-century coiled scroll curling back inward at the tail.
        ctrl = np.vstack([
            _p(0.0, 0.0),
            _p(0.75, 0.14),
            _p(1.18, 0.65),
            _p(0.88, 1.28),
            _p(0.28, 1.18),
            _p(0.46, 0.78),
        ])
        fpts = _eval_catmull_rom(ctrl, samples_per_seg=7)
        draw_calligraphic_polyline(canvas, fpts, color_rgb, flag_w * 1.05, style, kind="flag")

    elif arch == "sharp_angular_7":
        # Crisp straight horizontal bar with sharp 7-corner bend dropping diagonally.
        fpts = np.vstack([
            _p(0.0, 0.0),
            _p(0.98, 0.04),
            _p(0.28, 1.40),
        ])
        local_style = dict(style)
        local_style["wobble_px"] = min(float(style.get("wobble_px", 0.7)), 0.45)
        draw_calligraphic_polyline(canvas, fpts, color_rgb, flag_w * 1.05, local_style, kind="flag")

    elif arch == "hurried_slash":
        # Fast sweeping diagonal pen slash shooting boldly out-and-down.
        fpts = _eval_quad(_p(0.0, 0.0), _p(0.84, 0.56), _p(1.56, 1.50), n=14)
        local_style = dict(style)
        local_style["taper_end"] = 0.28
        draw_calligraphic_polyline(canvas, fpts, color_rgb, flag_w * 1.12, local_style, kind="flag")

    elif arch == "tiny_flick":
        # Minimalist short stubby flick.
        fpts = _eval_quad(_p(0.0, 0.0), _p(0.28, 0.14), _p(0.46, 0.44), n=9)
        draw_calligraphic_polyline(canvas, fpts, color_rgb, max(0.80, flag_w * 0.74), style, kind="flag")

    elif arch == "long_dramatic_s":
        # Tall sweeping flamboyant S-curve descending far down the stem.
        ctrl = np.vstack([
            _p(0.0, 0.0),
            _p(1.05, 0.35),
            _p(1.38, 1.05),
            _p(0.45, 1.65),
            _p(0.92, 2.20),
        ])
        fpts = _eval_catmull_rom(ctrl, samples_per_seg=8)
        draw_calligraphic_polyline(canvas, fpts, color_rgb, flag_w * 1.05, style, kind="flag")

    elif arch == "closed_loop_flourish":
        # Pen loops completely around forming a visible closed eyelet before flicking out.
        ctrl = np.vstack([
            _p(0.0, 0.0),
            _p(0.95, 0.12),
            _p(1.02, 0.88),
            _p(0.28, 1.12),
            _p(0.06, 0.56),
            _p(0.55, 0.22),
            _p(1.10, 0.96),
        ])
        fpts = _eval_catmull_rom(ctrl, samples_per_seg=7)
        draw_calligraphic_polyline(canvas, fpts, color_rgb, flag_w * 0.92, style, kind="flag")

    elif arch == "broad_nib_pennant":
        # Bold triangular pennant-shaped calligraphic flag (thick upper + angled lower fill).
        local_style = dict(style)
        local_style["taper_start"] = 0.03
        local_style["taper_end"] = 0.24
        local_style["nib_min_ratio"] = max(_get_style_min_ratio(style), 0.72)
        upper = _eval_quad(_p(0.0, 0.0), _p(0.62, 0.22), _p(1.22, 0.92), n=12)
        lower = _eval_quad(_p(0.0, 0.42), _p(0.55, 0.46), _p(1.18, 0.92), n=12)
        draw_calligraphic_polyline(canvas, upper, color_rgb, flag_w * 1.72, local_style, kind="flag")
        draw_calligraphic_polyline(canvas, lower, color_rgb, flag_w * 1.35, local_style, kind="flag")

    elif arch == "detached_tick":
        # Floating tick mark visibly separated from the stem by a clear gap.
        fpts = _eval_quad(_p(0.30, 0.06), _p(0.76, 0.34), _p(1.16, 0.96), n=11)
        draw_calligraphic_polyline(canvas, fpts, color_rgb, flag_w * 1.02, style, kind="flag")

    elif arch == "wide_horizontal_wing":
        # Long, nearly horizontal wing stretching far right with gentle camber.
        fpts = _eval_cubic(_p(0.0, 0.0), _p(0.65, -0.14), _p(1.30, 0.16), _p(1.82, 0.46), n=16)
        draw_calligraphic_polyline(canvas, fpts, color_rgb, flag_w * 1.08, style, kind="flag")

    elif arch == "steep_vertical_drop":
        # Hugs tightly close to the stem and plunges almost vertically downward.
        fpts = _eval_cubic(_p(0.0, 0.0), _p(0.50, 0.18), _p(0.48, 1.10), _p(0.28, 1.85), n=16)
        draw_calligraphic_polyline(canvas, fpts, color_rgb, flag_w * 0.95, style, kind="flag")

    elif arch == "zigzag_frantic":
        # Nervous 3-corner zigzag lightning-bolt flag.
        fpts = np.vstack([
            _p(0.0, 0.0),
            _p(0.80, 0.22),
            _p(0.32, 0.72),
            _p(1.18, 1.18),
        ])
        draw_calligraphic_polyline(canvas, fpts, color_rgb, flag_w * 0.96, style, kind="flag")

    elif arch == "hooked_j_tail":
        # Sweeps down-right then hooks sharply back UPWARD at the outer tip like a fishhook/J.
        ctrl = np.vstack([
            _p(0.0, 0.0),
            _p(0.52, 0.48),
            _p(0.92, 1.24),
            _p(1.26, 1.06),
            _p(1.32, 0.58),
        ])
        fpts = _eval_catmull_rom(ctrl, samples_per_seg=7)
        draw_calligraphic_polyline(canvas, fpts, color_rgb, flag_w * 0.98, style, kind="flag")

    elif arch == "calligraphic_ribbon":
        # High-contrast italic ribbon: thin hairline entry, ultra-thick diagonal body, fine exit.
        entry = _eval_quad(_p(0.0, 0.0), _p(0.32, 0.04), _p(0.58, 0.24), n=8)
        body = _eval_quad(_p(0.52, 0.20), _p(0.92, 0.58), _p(0.84, 1.02), n=10)
        exit_tail = _eval_quad(_p(0.84, 1.00), _p(0.62, 1.32), _p(0.26, 1.48), n=8)
        local_style = dict(style)
        local_style["nib_angle_deg"] = 46.0
        local_style["nib_min_ratio"] = 0.20
        draw_calligraphic_polyline(canvas, entry, color_rgb, max(0.75, flag_w * 0.58), local_style, kind="flag")
        draw_calligraphic_polyline(canvas, body, color_rgb, flag_w * 1.90, local_style, kind="flag")
        draw_calligraphic_polyline(canvas, exit_tail, color_rgb, max(0.75, flag_w * 0.55), local_style, kind="flag")

    elif arch == "stubby_wedge":
        # Short, very thick blunt chisel wedge block.
        fpts = np.vstack([_p(0.0, 0.04), _p(0.36, 0.28), _p(0.70, 0.62)])
        local_style = dict(style)
        local_style["taper_start"] = 0.02
        local_style["taper_end"] = 0.02
        local_style["nib_min_ratio"] = max(_get_style_min_ratio(style), 0.86)
        draw_calligraphic_polyline(canvas, fpts, color_rgb, flag_w * 1.88, local_style, kind="flag")

    elif arch == "double_back_loop":
        # Figure-8 / double-reverse loop sweeping out, back toward stem, then out again.
        ctrl = np.vstack([
            _p(0.0, 0.0),
            _p(0.92, 0.14),
            _p(0.32, 0.66),
            _p(1.08, 0.96),
            _p(0.58, 1.52),
        ])
        fpts = _eval_catmull_rom(ctrl, samples_per_seg=7)
        draw_calligraphic_polyline(canvas, fpts, color_rgb, flag_w * 0.94, style, kind="flag")

    elif arch == "feathered_sweep":
        # Multi-strand sketchy feathered tail: main sweep plus 2 splayed feather wisps.
        main_c = _eval_cubic(_p(0.0, 0.0), _p(0.75, 0.25), _p(1.12, 0.72), _p(0.88, 1.38), n=14)
        wisp1 = _eval_quad(_p(0.12, 0.12), _p(0.88, 0.34), _p(1.36, 0.96), n=10)
        wisp2 = _eval_quad(_p(0.06, 0.24), _p(0.64, 0.66), _p(0.56, 1.54), n=10)
        draw_calligraphic_polyline(canvas, main_c, color_rgb, flag_w * 1.02, style, kind="flag")
        draw_calligraphic_polyline(canvas, wisp1, color_rgb, max(0.7, flag_w * 0.48), style, kind="flag")
        draw_calligraphic_polyline(canvas, wisp2, color_rgb, max(0.7, flag_w * 0.48), style, kind="flag")

    elif arch == "flat_bar_flag":
        # Dead-flat horizontal mini-beam stub.
        fpts = np.vstack([_p(0.0, 0.0), _p(0.55, 0.03), _p(1.08, 0.08)])
        local_style = dict(style)
        local_style["taper_start"] = 0.03
        local_style["taper_end"] = 0.04
        local_style["nib_min_ratio"] = max(_get_style_min_ratio(style), 0.80)
        draw_calligraphic_polyline(canvas, fpts, color_rgb, flag_w * 1.58, local_style, kind="flag")

    elif arch == "reverse_kick":
        # Shoots upward/away from notehead first, then kicks sharply back down-right.
        ctrl = np.vstack([
            _p(0.0, 0.0),
            _p(0.52, -0.34),
            _p(0.92, 0.10),
            _p(1.14, 1.08),
        ])
        fpts = _eval_catmull_rom(ctrl, samples_per_seg=7)
        draw_calligraphic_polyline(canvas, fpts, color_rgb, flag_w * 1.04, style, kind="flag")

    else:  # "minimal_dot_tail"
        # Delicate hairline curve terminating in a bold filled ink bead/dot at the tip.
        end_pt = _p(0.94, 0.88)
        fpts = _eval_quad(_p(0.0, 0.0), _p(0.68, 0.14), end_pt, n=12)
        draw_calligraphic_polyline(canvas, fpts, color_rgb, max(0.65, flag_w * 0.56), style, kind="flag")
        dot_r = max(1.5, sp * 0.25)
        draw_filled_blob(
            canvas,
            float(end_pt[0]),
            float(end_pt[1]),
            dot_r,
            dot_r * float(rng.uniform(0.88, 1.12)),
            float(rng.uniform(-25.0, 25.0)),
            color_rgb,
            style,
            hollow=False,
        )


def draw_stem_and_flag(
    canvas: StrokeCanvas,
    x: float,
    y0: float,
    y1: float,
    sp: float,
    flags: int,
    stem_up: bool,
    color_rgb: RGBLike,
    style: dict,
    rng: np.random.Generator,
    beamed: bool = False,
    scale_length: bool = True,
) -> None:
    """Draw a polymorphic handwritten stem (18 archetypes) and flags (20 archetypes)."""
    stem_arch = resolve_stem_archetype(style)
    flag_arch = resolve_flag_archetype(style)

    base_w = max(0.85, sp * float(style.get("stem_width_ratio", 0.135)))
    head_dir = 1.0 if stem_up else -1.0  # +1 down toward head, -1 up toward head
    tip_dir = -head_dir  # -1 up toward tip, +1 down toward tip

    from .styles import resolve_ductus_param

    dy_raw = y1 - y0
    if scale_length and not beamed:
        len_mult = get_stem_length_multiplier(style)
        y1_eff = y0 + dy_raw * len_mult
    else:
        y1_eff = y1

    len_jit_frac = float(np.clip(resolve_ductus_param(style, "stem_len_jitter_frac", 0.03), 0.0, 0.28))
    if not beamed and len_jit_frac > 0.035:
        y1_eff = y0 + (y1_eff - y0) * float(np.clip(1.0 + rng.normal(0.0, len_jit_frac), 0.68, 1.38))

    dy = y1_eff - y0
    L = max(sp * 0.8, abs(dy))

    # Base attachment & tip coordinates
    ax = x + float(rng.normal(0.0, 0.04 * sp))
    ay = y0 + float(rng.normal(0.0, 0.04 * sp))
    slant_deg = float(style.get("slant_deg", 0.0)) + float(rng.normal(0.0, 1.8))
    base_lean = -dy * math.tan(math.radians(float(np.clip(slant_deg, -10.0, 10.0)))) * 0.28
    tip_x = ax + (0.0 if beamed else base_lean) + float(rng.normal(0.0, 0.04 * sp))
    tip_y = y1_eff + (0.0 if beamed else tip_dir * float(rng.uniform(-0.03, 0.09)) * sp)

    if beamed:
        over_sp = float(np.clip(resolve_ductus_param(style, "beam_junction_overshoot_sp", 0.0), 0.0, 0.42))
        gap_prob = float(np.clip(resolve_ductus_param(style, "beam_junction_gap_prob", 0.0), 0.0, 0.45))
        if gap_prob > 0.0 and float(rng.random()) < gap_prob:
            tip_y = y1_eff - tip_dir * float(rng.uniform(0.08, 0.28)) * sp
        elif over_sp > 0.0 and float(rng.random()) < 0.55:
            tip_y = y1_eff + tip_dir * float(rng.uniform(0.25, 1.0)) * over_sp * sp

    local_style = dict(style)
    stem_w = base_w

    # ------------------------------------------------------------------
    # 18 Stem Archetypes
    # ------------------------------------------------------------------
    if stem_arch == "straight_classic":
        local_style["wobble_px"] = min(float(style.get("wobble_px", 0.7)), 0.45)
        mid = np.array([0.5 * (ax + tip_x) + float(rng.normal(0.0, 0.02 * sp)), 0.5 * (ay + tip_y)])
        stem_pts = _eval_quad(np.array([ax, ay]), mid, np.array([tip_x, tip_y]), n=12)
        draw_calligraphic_polyline(canvas, stem_pts, color_rgb, stem_w, local_style, kind="stem")

    elif stem_arch == "hairline_needle":
        local_style["wobble_px"] = min(float(style.get("wobble_px", 0.7)), 0.35)
        local_style["taper_end"] = 0.26
        stem_w = max(0.55, base_w * 0.42)
        mid = np.array([0.5 * (ax + tip_x), 0.5 * (ay + tip_y)])
        stem_pts = _eval_quad(np.array([ax, ay]), mid, np.array([tip_x, tip_y]), n=12)
        draw_calligraphic_polyline(canvas, stem_pts, color_rgb, stem_w, local_style, kind="stem")

    elif stem_arch == "heavy_thick":
        local_style["taper_start"] = 0.02
        local_style["taper_end"] = 0.02
        local_style["nib_min_ratio"] = max(_get_style_min_ratio(style), 0.80)
        stem_w = max(1.6, base_w * 2.05)
        mid = np.array([0.5 * (ax + tip_x) + float(rng.normal(0.0, 0.04 * sp)), 0.5 * (ay + tip_y)])
        stem_pts = _eval_quad(np.array([ax, ay]), mid, np.array([tip_x, tip_y]), n=12)
        draw_calligraphic_polyline(canvas, stem_pts, color_rgb, stem_w, local_style, kind="stem")

    elif stem_arch == "tapered_quill":
        # Thick at notehead base, tapering progressively to a needle-sharp tip.
        mid = np.array([0.5 * (ax + tip_x) + float(rng.normal(0.0, 0.04 * sp)), 0.5 * (ay + tip_y)])
        stem_pts = _eval_quad(np.array([ax, ay]), mid, np.array([tip_x, tip_y]), n=18)
        s_thin = dict(local_style)
        s_thin["taper_end"] = 0.30
        draw_calligraphic_polyline(canvas, stem_pts, color_rgb, max(0.60, base_w * 0.45), s_thin, kind="stem")
        draw_calligraphic_polyline(canvas, stem_pts[:13], color_rgb, base_w * 1.15, local_style, kind="stem")
        draw_calligraphic_polyline(canvas, stem_pts[:7], color_rgb, base_w * 1.85, local_style, kind="stem")

    elif stem_arch == "bowed_right":
        mid = np.array([0.5 * (ax + tip_x) + 0.50 * sp, 0.5 * (ay + tip_y)])
        stem_pts = _eval_quad(np.array([ax, ay]), mid, np.array([tip_x, tip_y]), n=16)
        draw_calligraphic_polyline(canvas, stem_pts, color_rgb, stem_w * 1.05, local_style, kind="stem")

    elif stem_arch == "bowed_left":
        mid = np.array([0.5 * (ax + tip_x) - 0.50 * sp, 0.5 * (ay + tip_y)])
        stem_pts = _eval_quad(np.array([ax, ay]), mid, np.array([tip_x, tip_y]), n=16)
        draw_calligraphic_polyline(canvas, stem_pts, color_rgb, stem_w * 1.05, local_style, kind="stem")

    elif stem_arch == "s_curve_flex":
        p0 = np.array([ax, ay])
        p3 = np.array([tip_x, tip_y])
        p1 = np.array([ax + 0.44 * sp, ay + 0.33 * (tip_y - ay)])
        p2 = np.array([tip_x - 0.44 * sp, ay + 0.67 * (tip_y - ay)])
        stem_pts = _eval_cubic(p0, p1, p2, p3, n=18)
        draw_calligraphic_polyline(canvas, stem_pts, color_rgb, stem_w * 1.02, local_style, kind="stem")

    elif stem_arch == "short_stubby":
        local_style["taper_start"] = 0.04
        local_style["taper_end"] = 0.04
        stem_w = base_w * 1.32
        mid = np.array([0.5 * (ax + tip_x), 0.5 * (ay + tip_y)])
        stem_pts = _eval_quad(np.array([ax, ay]), mid, np.array([tip_x, tip_y]), n=12)
        draw_calligraphic_polyline(canvas, stem_pts, color_rgb, stem_w, local_style, kind="stem")

    elif stem_arch == "tall_spindly":
        stem_w = max(0.65, base_w * 0.72)
        sway = float(rng.uniform(-0.14, 0.14)) * sp
        mid = np.array([0.5 * (ax + tip_x) + sway, 0.5 * (ay + tip_y)])
        stem_pts = _eval_quad(np.array([ax, ay]), mid, np.array([tip_x, tip_y]), n=16)
        draw_calligraphic_polyline(canvas, stem_pts, color_rgb, stem_w, local_style, kind="stem")

    elif stem_arch == "steep_forward_slant":
        # Strong forward italic lean (top right, bottom left).
        shift = 0.36 * L * (1.0 if stem_up else -1.0)
        if beamed:
            ax_s = tip_x - shift
            tip_x_s = tip_x
        else:
            ax_s = ax
            tip_x_s = ax + shift
        mid = np.array([0.5 * (ax_s + tip_x_s), 0.5 * (ay + tip_y)])
        stem_pts = _eval_quad(np.array([ax_s, ay]), mid, np.array([tip_x_s, tip_y]), n=14)
        draw_calligraphic_polyline(canvas, stem_pts, color_rgb, stem_w, local_style, kind="stem")

    elif stem_arch == "back_slant_lefty":
        # Strong backward left-handed lean (top left, bottom right).
        shift = -0.36 * L * (1.0 if stem_up else -1.0)
        if beamed:
            ax_s = tip_x - shift
            tip_x_s = tip_x
        else:
            ax_s = ax
            tip_x_s = ax + shift
        mid = np.array([0.5 * (ax_s + tip_x_s), 0.5 * (ay + tip_y)])
        stem_pts = _eval_quad(np.array([ax_s, ay]), mid, np.array([tip_x_s, tip_y]), n=14)
        draw_calligraphic_polyline(canvas, stem_pts, color_rgb, stem_w, local_style, kind="stem")

    elif stem_arch == "detached_gap":
        # Visible pen-lift gap between notehead and stem base.
        ay_gap = ay + tip_dir * (0.64 * sp)
        mid = np.array([0.5 * (ax + tip_x), 0.5 * (ay_gap + tip_y)])
        stem_pts = _eval_quad(np.array([ax, ay_gap]), mid, np.array([tip_x, tip_y]), n=12)
        draw_calligraphic_polyline(canvas, stem_pts, color_rgb, stem_w, local_style, kind="stem")

    elif stem_arch == "center_piercing":
        # Stem shifted horizontally into the center of the notehead, piercing through it.
        cx_shift = (-0.48 if stem_up else 0.48) * sp
        ax_c = ax + cx_shift
        tip_x_c = tip_x if beamed else (tip_x + cx_shift)
        ay_c = ay + head_dir * (0.24 * sp)
        mid = np.array([0.5 * (ax_c + tip_x_c), 0.5 * (ay_c + tip_y)])
        stem_pts = _eval_quad(np.array([ax_c, ay_c]), mid, np.array([tip_x_c, tip_y]), n=12)
        draw_calligraphic_polyline(canvas, stem_pts, color_rgb, stem_w * 1.05, local_style, kind="stem")

    elif stem_arch == "overshoot_notehead":
        # Stem hastily overshoots past the notehead by ~0.82 sp.
        ay_over = ay + head_dir * (0.82 * sp)
        mid = np.array([0.5 * (ax + tip_x), 0.5 * (ay_over + tip_y)])
        stem_pts = _eval_quad(np.array([ax, ay_over]), mid, np.array([tip_x, tip_y]), n=14)
        draw_calligraphic_polyline(canvas, stem_pts, color_rgb, stem_w, local_style, kind="stem")

    elif stem_arch == "tremor_shaky":
        # High-frequency jagged/wavering hand tremor along the entire stem.
        local_style["tremor_px"] = max(2.8, float(style.get("tremor_px", 0.2)))
        local_style["wobble_px"] = max(2.2, float(style.get("wobble_px", 0.7)))
        n_seg = 16
        ts = np.linspace(0.0, 1.0, n_seg)
        xs = ax + (tip_x - ax) * ts
        ys = ay + (tip_y - ay) * ts
        # Alternating high-frequency hand shake + random perturbation
        jitter_x = np.array([((-1.0) ** i) * 0.16 * sp for i in range(n_seg)]) + rng.normal(0.0, 0.10 * sp, size=n_seg)
        jitter_x[0] *= 0.25
        jitter_x[-1] *= 0.25 if beamed else 0.65
        stem_pts = np.column_stack((xs + jitter_x, ys))
        draw_calligraphic_polyline(canvas, stem_pts, color_rgb, stem_w * 0.96, local_style, kind="stem")

    elif stem_arch == "double_stroke_sketch":
        # Sketchy composer double-pass stem: two slightly offset, lightly diverging strokes.
        sep = 0.15 * sp
        mid1 = np.array([0.5 * (ax + tip_x) - sep * 0.8, 0.5 * (ay + tip_y)])
        mid2 = np.array([0.5 * (ax + tip_x) + sep * 1.1, 0.5 * (ay + tip_y)])
        pts1 = _eval_quad(np.array([ax - sep * 0.6, ay]), mid1, np.array([tip_x - sep * 0.3, tip_y]), n=12)
        pts2 = _eval_quad(
            np.array([ax + sep * 0.7, ay + tip_dir * 0.12 * sp]),
            mid2,
            np.array([tip_x + sep * 0.6, tip_y - tip_dir * (0.0 if beamed else 0.16 * sp)]),
            n=12,
        )
        stem_pts = pts1
        draw_calligraphic_polyline(canvas, pts1, color_rgb, max(0.7, stem_w * 0.80), local_style, kind="stem")
        draw_calligraphic_polyline(canvas, pts2, color_rgb, max(0.7, stem_w * 0.74), local_style, kind="stem")

    elif stem_arch == "flared_base":
        # Stem widens into a distinct flared triangular foot / serif at the notehead attachment.
        mid = np.array([0.5 * (ax + tip_x), 0.5 * (ay + tip_y)])
        stem_pts = _eval_quad(np.array([ax, ay]), mid, np.array([tip_x, tip_y]), n=14)
        draw_calligraphic_polyline(canvas, stem_pts, color_rgb, stem_w * 0.95, local_style, kind="stem")
        foot_y = ay + tip_dir * (0.78 * sp)
        left_flare = np.array([[ax - 0.28 * sp, ay], [ax, foot_y]], dtype=np.float64)
        right_flare = np.array([[ax + 0.28 * sp, ay], [ax, foot_y]], dtype=np.float64)
        base_bar = np.array([[ax - 0.26 * sp, ay], [ax + 0.26 * sp, ay]], dtype=np.float64)
        draw_calligraphic_polyline(canvas, left_flare, color_rgb, stem_w * 1.25, local_style, kind="stem")
        draw_calligraphic_polyline(canvas, right_flare, color_rgb, stem_w * 1.25, local_style, kind="stem")
        draw_calligraphic_polyline(canvas, base_bar, color_rgb, stem_w * 1.15, local_style, kind="stem")

    else:  # "whip_taper"
        # Firm lower stem that whips outward at the tip with a curved flick and razor taper.
        whip_dx = (0.38 if not beamed else 0.16) * sp
        p0 = np.array([ax, ay])
        p1 = np.array([ax - 0.08 * sp, ay + 0.45 * (tip_y - ay)])
        p2 = np.array([tip_x - 0.14 * sp, ay + 0.78 * (tip_y - ay)])
        p3 = np.array([tip_x + whip_dx, tip_y])
        stem_pts = _eval_cubic(p0, p1, p2, p3, n=18)
        s_tip = dict(local_style)
        s_tip["taper_end"] = 0.35
        draw_calligraphic_polyline(canvas, stem_pts, color_rgb, max(0.60, stem_w * 0.52), s_tip, kind="stem")
        draw_calligraphic_polyline(canvas, stem_pts[:13], color_rgb, stem_w * 1.18, local_style, kind="stem")

    # ------------------------------------------------------------------
    # 20 Flag Archetypes (for unbeamed 8th / 16th / 32nd+ notes)
    # ------------------------------------------------------------------
    n_flags = max(0, int(flags))
    if n_flags == 0:
        return

    flag_w = max(1.0, sp * float(style.get("flag_width_ratio", 0.22)))
    if flag_arch in ("tiny_flick", "stubby_wedge", "flat_bar_flag", "minimal_dot_tail"):
        flag_spacing = sp * float(rng.uniform(0.56, 0.66))
    elif flag_arch in ("long_dramatic_s", "closed_loop_flourish", "baroque_curl"):
        flag_spacing = sp * float(rng.uniform(0.68, 0.80))
    else:
        flag_spacing = sp * float(rng.uniform(0.60, 0.74))

    for i in range(n_flags):
        # Anchor each flag cleanly onto the actual stem polyline (so slanted/bowed stems stay attached)
        frac = min(0.75, (i * flag_spacing) / max(sp * 1.2, L))
        anchor = _sample_point_along_stem(stem_pts, frac)
        fx = float(anchor[0]) + float(rng.normal(0.0, 0.025 * sp))
        fy = float(anchor[1]) + float(rng.normal(0.0, 0.025 * sp))
        _draw_single_flag_archetype(
            canvas=canvas,
            arch=flag_arch,
            fx=fx,
            fy=fy,
            sp=sp,
            toward_head=head_dir,
            flag_w=flag_w,
            color_rgb=color_rgb,
            style=style,
            rng=rng,
        )


def compute_beam_geometry(
    grp_items: Sequence[Tuple[float, float, float, int, str]],
    sp: float,
    style: dict,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:
    """Compute beam endpoint (y0, y1), parabolic curve sag, and secondary-beam tilt.

    Supports all 10 ``BEAM_SLOPE_STYLES`` so beams vary from flat horizontal to steep
    contours, exaggerated swoops, reverse quirks, concave valleys, convex arches,
    stepped secondary tilts, high floating beams, and tight low-hugging beams.

    ``grp_items`` is a sequence of ``(stem_x, far_y, tip_y, stem_dir, duration_kind)``.
    """
    x0, _, y0_raw, sdir, _ = grp_items[0]
    x1, _, y1_raw, _, _ = grp_items[-1]
    x0 = float(x0)
    x1 = float(x1)
    y0_raw = float(y0_raw)
    y1_raw = float(y1_raw)
    sdir = int(sdir)
    dx_span = max(1e-3, x1 - x0)
    raw_dy = y1_raw - y0_raw

    slope_style = resolve_beam_slope_style(style)
    curve_sag_px = 0.0
    secondary_tilt_px = 0.0

    if slope_style == "horizontal_flat":
        dy = 0.0
        y0 = y0_raw
        y1 = y0_raw
    elif slope_style == "engraver_gentle":
        max_dy = min(sp * 1.15, dx_span * 0.18)
        dy = max(-max_dy, min(max_dy, raw_dy * 0.65))
        y0 = y0_raw
        y1 = y0 + dy
    elif slope_style == "steep_contour":
        max_dy = min(sp * 3.4, dx_span * 0.58)
        dy = max(-max_dy, min(max_dy, raw_dy * 1.05))
        y0 = y0_raw
        y1 = y0 + dy
    elif slope_style == "exaggerated_swoop":
        max_dy = min(sp * 4.4, dx_span * 0.75)
        if abs(raw_dy) < 0.25 * sp:
            dy = -0.75 * sp * sdir
        else:
            dy = max(-max_dy, min(max_dy, raw_dy * 1.50))
        curve_sag_px = -0.34 * sp * sdir
        y0 = y0_raw
        y1 = y0 + dy
    elif slope_style == "reverse_quirk":
        max_dy = min(sp * 2.2, dx_span * 0.38)
        if abs(raw_dy) < 0.25 * sp:
            dy = 0.65 * sp * sdir
        else:
            dy = max(-max_dy, min(max_dy, -raw_dy * 0.72))
        y0 = y0_raw
        y1 = y0 + dy
    elif slope_style == "concave_valley":
        max_dy = min(sp * 1.4, dx_span * 0.22)
        dy = max(-max_dy, min(max_dy, raw_dy * 0.55))
        curve_sag_px = 0.62 * sp * sdir
        y0 = y0_raw
        y1 = y0 + dy
    elif slope_style == "convex_arch":
        max_dy = min(sp * 1.4, dx_span * 0.22)
        dy = max(-max_dy, min(max_dy, raw_dy * 0.55))
        curve_sag_px = -0.68 * sp * sdir
        y0 = y0_raw
        y1 = y0 + dy
    elif slope_style == "stepped_secondary_tilt":
        max_dy = min(sp * 1.1, dx_span * 0.16)
        dy = max(-max_dy, min(max_dy, raw_dy * 0.45))
        tilt_sign = 1.0 if raw_dy >= 0.0 else -1.0
        secondary_tilt_px = 0.52 * sp * tilt_sign
        y0 = y0_raw
        y1 = y0 + dy
    elif slope_style == "high_floating_beam":
        max_dy = min(sp * 1.4, dx_span * 0.22)
        dy = max(-max_dy, min(max_dy, raw_dy * 0.65))
        y0 = y0_raw
        y1 = y0 + dy
    else:  # "tight_hug_beam"
        max_dy = min(sp * 1.35, dx_span * 0.22)
        dy = max(-max_dy, min(max_dy, raw_dy * 0.55))
        fy0 = float(grp_items[0][1])
        y0 = fy0 - sdir * max(1.75 * sp, abs(y0_raw - fy0) - 1.15 * sp)
        y1 = y0 + dy

    # Ensure every note in the group has clean stem clearance to the (possibly curved) beam.
    for sx, fy, ty, _, _ in grp_items:
        t = (float(sx) - x0) / dx_span
        by = y0 + t * (y1 - y0) + 4.0 * t * (1.0 - t) * curve_sag_px
        if slope_style == "tight_hug_beam":
            req_ty = float(fy) - sdir * max(1.75 * sp, abs(float(ty) - float(fy)) - 1.15 * sp)
        else:
            req_ty = float(ty)
        if sdir == 1 and by > req_ty:
            shift = req_ty - by
            y0 += shift
            y1 += shift
        elif sdir == -1 and by < req_ty:
            shift = req_ty - by
            y0 += shift
            y1 += shift

    if slope_style == "high_floating_beam":
        elev = -sdir * 1.45 * sp
        y0 += elev
        y1 += elev

    return {
        "y0": float(y0),
        "y1": float(y1),
        "curve_sag_px": float(curve_sag_px),
        "secondary_tilt_px": float(secondary_tilt_px),
        "slope_style": slope_style,
    }


def draw_beam(
    canvas: StrokeCanvas,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    sp: float,
    color_rgb: RGBLike,
    style: dict,
    rng: np.random.Generator,
    extra_sag_px: float = 0.0,
) -> None:
    """Draw a polymorphic handwritten beam across 14 distinct morphological archetypes."""
    dx = x1 - x0
    dy = y1 - y0
    length = math.hypot(dx, dy)
    if length < 1e-3:
        return
    ux, uy = dx / length, dy / length
    # Unit normal pointing downward when x1 > x0
    nx, ny = -uy, ux
    if ny < 0:
        nx, ny = -nx, -ny

    from .styles import resolve_ductus_param
    extra_over = float(np.clip(resolve_ductus_param(style, "beam_edge_overhang_sp", 0.0), -0.12, 0.55))
    if abs(extra_over) > 1e-4:
        over_l = extra_over * float(rng.uniform(0.55, 1.25)) * sp
        over_r = extra_over * float(rng.uniform(0.55, 1.25)) * sp
        x0, y0 = x0 - ux * over_l, y0 - uy * over_l
        x1, y1 = x1 + ux * over_r, y1 + uy * over_r
        dx, dy = x1 - x0, y1 - y0
        length = max(1e-3, math.hypot(dx, dy))

    arch = resolve_beam_archetype(style)
    thick_scale = float(np.clip(style.get("beam_thickness_scale", 1.0), 0.45, 2.10))
    base_w = max(1.1, sp * float(style.get("beam_width_ratio", 0.46)) * thick_scale * float(rng.uniform(0.94, 1.06)))

    local_style = dict(style)
    local_style["taper_start"] = 0.05
    local_style["taper_end"] = 0.05
    n_pts = max(12, int(length / 7.0))
    sag_ctrl = np.array([0.0, 2.0 * float(extra_sag_px)], dtype=np.float64)

    def _apply_sag_pts(pts_arr: np.ndarray) -> np.ndarray:
        if abs(extra_sag_px) <= 1e-3:
            return pts_arr
        out_p = pts_arr.copy()
        ts = np.clip((out_p[:, 0] - x0) / max(1e-3, dx), 0.0, 1.0)
        out_p[:, 1] += 4.0 * ts * (1.0 - ts) * float(extra_sag_px)
        return out_p

    if arch == "straight_classic":
        over = 0.06 * sp
        p0 = np.array([x0 - ux * over, y0 - uy * over])
        p2 = np.array([x1 + ux * over, y1 + uy * over])
        bow = float(rng.normal(0.0, 0.03 * sp))
        p1 = 0.5 * (p0 + p2) + np.array([nx * bow, ny * bow]) + sag_ctrl
        local_style["wobble_px"] = min(float(style.get("wobble_px", 0.7)), 0.45)
        draw_calligraphic_polyline(canvas, _eval_quad(p0, p1, p2, n=n_pts), color_rgb, base_w, local_style, kind="beam")

    elif arch == "thick_marker":
        over = 0.08 * sp
        p0 = np.array([x0 - ux * over, y0 - uy * over])
        p2 = np.array([x1 + ux * over, y1 + uy * over])
        p1 = 0.5 * (p0 + p2) + np.array([nx * float(rng.normal(0.0, 0.04 * sp)), ny * float(rng.normal(0.0, 0.04 * sp))]) + sag_ctrl
        local_style["taper_start"] = 0.01
        local_style["taper_end"] = 0.01
        local_style["nib_min_ratio"] = max(_get_style_min_ratio(style), 0.88)
        draw_calligraphic_polyline(canvas, _eval_quad(p0, p1, p2, n=n_pts), color_rgb, base_w * 1.95, local_style, kind="beam")

    elif arch == "hairline_thin":
        over = 0.04 * sp
        p0 = np.array([x0 - ux * over, y0 - uy * over])
        p2 = np.array([x1 + ux * over, y1 + uy * over])
        p1 = 0.5 * (p0 + p2) + sag_ctrl
        local_style["wobble_px"] = min(float(style.get("wobble_px", 0.7)), 0.35)
        draw_calligraphic_polyline(canvas, _eval_quad(p0, p1, p2, n=n_pts), color_rgb, max(0.85, base_w * 0.34), local_style, kind="beam")

    elif arch == "arched_convex":
        # Pronounced upward rainbow arch (-y normal direction).
        over = 0.08 * sp
        p0 = np.array([x0 - ux * over, y0 - uy * over])
        p2 = np.array([x1 + ux * over, y1 + uy * over])
        arch_h = max(0.46 * sp, min(0.85 * sp, length * 0.14))
        p1 = 0.5 * (p0 + p2) - np.array([nx * arch_h, ny * arch_h]) + sag_ctrl
        draw_calligraphic_polyline(canvas, _eval_quad(p0, p1, p2, n=max(16, n_pts)), color_rgb, base_w * 1.02, local_style, kind="beam")

    elif arch == "sagged_concave":
        # Pronounced downward hammock sag (+y normal direction).
        over = 0.08 * sp
        p0 = np.array([x0 - ux * over, y0 - uy * over])
        p2 = np.array([x1 + ux * over, y1 + uy * over])
        sag_h = max(0.46 * sp, min(0.85 * sp, length * 0.14))
        p1 = 0.5 * (p0 + p2) + np.array([nx * sag_h, ny * sag_h]) + sag_ctrl
        draw_calligraphic_polyline(canvas, _eval_quad(p0, p1, p2, n=max(16, n_pts)), color_rgb, base_w * 1.02, local_style, kind="beam")

    elif arch == "tapered_wedge":
        # Thick at left end, progressively tapering to a fine tip at right end.
        p0 = np.array([x0 - ux * 0.06 * sp, y0 - uy * 0.06 * sp])
        p2 = np.array([x1 + ux * 0.10 * sp, y1 + uy * 0.10 * sp])
        n_segs = 6
        local_style["taper_start"] = 0.02
        local_style["taper_end"] = 0.08
        for s_i in range(n_segs):
            t0 = s_i / n_segs
            t1 = min(1.0, (s_i + 1.15) / n_segs)
            pa = p0 * (1.0 - t0) + p2 * t0
            pb = p0 * (1.0 - t1) + p2 * t1
            w_seg = base_w * (1.70 - 1.32 * (s_i / max(1, n_segs - 1)))
            draw_calligraphic_polyline(canvas, _apply_sag_pts(np.vstack((pa, pb))), color_rgb, max(0.85, w_seg), local_style, kind="beam")

    elif arch == "overhanging_ends":
        # Wild dramatic overshoots extending ~0.62 sp past both ends with tapered flicks.
        over0 = 0.62 * sp
        over1 = 0.62 * sp
        p0 = np.array([x0 - ux * over0, y0 - uy * over0])
        p2 = np.array([x1 + ux * over1, y1 + uy * over1])
        p1 = 0.5 * (p0 + p2) + np.array([nx * float(rng.normal(0.0, 0.06 * sp)), ny * float(rng.normal(0.0, 0.06 * sp))]) + sag_ctrl
        local_style["taper_start"] = 0.24
        local_style["taper_end"] = 0.24
        draw_calligraphic_polyline(canvas, _eval_quad(p0, p1, p2, n=max(16, n_pts)), color_rgb, base_w * 1.05, local_style, kind="beam")

    elif arch == "short_clipped":
        # Clipped/recessed beam stopping slightly short of outer stem edges.
        inset = min(0.22 * sp, length * 0.18)
        p0 = np.array([x0 + ux * inset, y0 + uy * inset])
        p2 = np.array([x1 - ux * inset, y1 - uy * inset])
        local_style["taper_start"] = 0.01
        local_style["taper_end"] = 0.01
        if abs(extra_sag_px) > 1e-3:
            p1 = 0.5 * (p0 + p2) + sag_ctrl
            draw_calligraphic_polyline(canvas, _eval_quad(p0, p1, p2, n=n_pts), color_rgb, base_w * 0.92, local_style, kind="beam")
        else:
            draw_calligraphic_polyline(canvas, np.vstack((p0, p2)), color_rgb, base_w * 0.92, local_style, kind="beam")

    elif arch == "scribble_double_pass":
        # Hurried composer double-stroke beam: two slightly diverging/crossing passes.
        sep = 0.18 * sp
        p0a = np.array([x0 - ux * 0.14 * sp - nx * sep, y0 - uy * 0.14 * sp - ny * sep])
        p2a = np.array([x1 + ux * 0.18 * sp + nx * sep * 0.6, y1 + uy * 0.18 * sp + ny * sep * 0.6])
        p1a = 0.5 * (p0a + p2a) - np.array([nx * 0.10 * sp, ny * 0.10 * sp]) + sag_ctrl

        p0b = np.array([x0 - ux * 0.06 * sp + nx * sep, y0 - uy * 0.06 * sp + ny * sep])
        p2b = np.array([x1 + ux * 0.10 * sp - nx * sep * 0.8, y1 + uy * 0.10 * sp - ny * sep * 0.8])
        p1b = 0.5 * (p0b + p2b) + np.array([nx * 0.12 * sp, ny * 0.12 * sp]) + sag_ctrl

        draw_calligraphic_polyline(canvas, _eval_quad(p0a, p1a, p2a, n=n_pts), color_rgb, max(1.0, base_w * 0.62), local_style, kind="beam")
        draw_calligraphic_polyline(canvas, _eval_quad(p0b, p1b, p2b, n=n_pts), color_rgb, max(1.0, base_w * 0.58), local_style, kind="beam")

    elif arch == "calligraphic_ribbon":
        # S-curved calligraphic ribbon beam with dramatic central swell and nib contrast.
        p0 = np.array([x0 - ux * 0.10 * sp, y0 - uy * 0.10 * sp])
        p3 = np.array([x1 + ux * 0.10 * sp, y1 + uy * 0.10 * sp])
        wave = 0.24 * sp
        p1 = p0 + (p3 - p0) * 0.30 - np.array([nx * wave, ny * wave])
        p2 = p0 + (p3 - p0) * 0.70 + np.array([nx * wave, ny * wave])
        curve = _apply_sag_pts(_eval_cubic(p0, p1, p2, p3, n=max(20, n_pts)))
        local_style["nib_angle_deg"] = 46.0
        local_style["nib_min_ratio"] = 0.20
        local_style["taper_start"] = 0.22
        local_style["taper_end"] = 0.22
        draw_calligraphic_polyline(canvas, curve, color_rgb, base_w * 0.72, local_style, kind="beam")
        m0 = max(1, len(curve) // 5)
        m1 = max(m0 + 2, len(curve) - len(curve) // 5)
        draw_calligraphic_polyline(canvas, curve[m0:m1], color_rgb, base_w * 1.48, local_style, kind="beam")

    elif arch == "stepped_wobbly":
        # Uneven hand-drawn stepped/undulating beam with pronounced vertical steps.
        n_k = max(6, min(10, int(length / (0.8 * sp)) + 3))
        ts = np.linspace(0.0, 1.0, n_k)
        xs = x0 + dx * ts
        ys = y0 + dy * ts
        steps = np.array([((-1.0) ** i) * 0.18 * sp for i in range(n_k)]) + rng.normal(0.0, 0.08 * sp, size=n_k)
        steps[0] *= 0.2
        steps[-1] *= 0.2
        pts = _apply_sag_pts(np.column_stack((xs + nx * steps, ys + ny * steps)))
        local_style["wobble_px"] = max(2.2, float(style.get("wobble_px", 0.7)))
        draw_calligraphic_polyline(canvas, pts, color_rgb, base_w * 0.96, local_style, kind="beam")

    elif arch == "gapped_broken":
        # Pen-skip broken beam with 1-2 crisp ink-lift gaps in the middle.
        p0 = np.array([x0 - ux * 0.06 * sp, y0 - uy * 0.06 * sp])
        p2 = np.array([x1 + ux * 0.06 * sp, y1 + uy * 0.06 * sp])
        if length > 1.6 * sp:
            gap_frac = min(0.24, max(0.10, (0.34 * sp) / length))
            if length > 4.2 * sp:
                spans = [(0.0, 0.30), (0.30 + gap_frac, 0.64), (0.64 + gap_frac, 1.0)]
            else:
                spans = [(0.0, 0.50 - 0.5 * gap_frac), (0.50 + 0.5 * gap_frac, 1.0)]
            for ta, tb in spans:
                pa = p0 * (1.0 - ta) + p2 * ta
                pb = p0 * (1.0 - tb) + p2 * tb
                pm = 0.5 * (pa + pb)
                draw_calligraphic_polyline(canvas, _apply_sag_pts(np.vstack((pa, pm, pb))), color_rgb, base_w, local_style, kind="beam")
        else:
            draw_calligraphic_polyline(canvas, _apply_sag_pts(np.vstack((p0, 0.5 * (p0 + p2), p2))), color_rgb, base_w, local_style, kind="beam")

    elif arch == "wide_spaced_bars":
        # Lean crisp beam bar (paired with 0.92*sp wide inter-beam spacing in multi-beam groups).
        over = 0.10 * sp
        p0 = np.array([x0 - ux * over, y0 - uy * over])
        p2 = np.array([x1 + ux * over, y1 + uy * over])
        p1 = 0.5 * (p0 + p2) + sag_ctrl
        local_style["wobble_px"] = min(float(style.get("wobble_px", 0.7)), 0.50)
        draw_calligraphic_polyline(canvas, _eval_quad(p0, p1, p2, n=n_pts), color_rgb, max(1.0, base_w * 0.76), local_style, kind="beam")

    else:  # "tight_crammed_bars"
        # Compact dense bar (paired with 0.42*sp tight inter-beam spacing in multi-beam groups).
        over = 0.04 * sp
        p0 = np.array([x0 - ux * over, y0 - uy * over])
        p2 = np.array([x1 + ux * over, y1 + uy * over])
        p1 = 0.5 * (p0 + p2) + np.array([nx * float(rng.normal(0.0, 0.04 * sp)), ny * float(rng.normal(0.0, 0.04 * sp))]) + sag_ctrl
        local_style["taper_start"] = 0.02
        local_style["taper_end"] = 0.02
        draw_calligraphic_polyline(canvas, _eval_quad(p0, p1, p2, n=n_pts), color_rgb, base_w * 1.12, local_style, kind="beam")


def draw_ledger_line(
    canvas: StrokeCanvas,
    x: float,
    y: float,
    sp: float,
    color_rgb: RGBLike,
    style: dict,
    rng: np.random.Generator,
) -> None:
    """Draw a single handwritten ledger line in one of 5 morphological ledger_style variants."""
    lstyle = resolve_ledger_style(style)
    w_base = max(0.95, sp * 0.095 * float(style.get("stroke_scale", 1.0)))
    local_style = dict(style)

    if lstyle == "crisp":
        half_w = sp * 0.92
        local_style["wobble_px"] = min(float(style.get("wobble_px", 0.7)), 0.40)
        pts = np.array(
            [
                [x - half_w, y + float(rng.uniform(-0.25, 0.25))],
                [x, y + float(rng.uniform(-0.20, 0.20))],
                [x + half_w, y + float(rng.uniform(-0.25, 0.25))],
            ],
            dtype=np.float64,
        )
        draw_calligraphic_polyline(canvas, pts, color_rgb, w_base, local_style, kind="ledger")

    elif lstyle == "wide_sweep":
        half_w = sp * 1.48
        bow = float(rng.uniform(-0.10, 0.10)) * sp
        local_style["taper_start"] = 0.20
        local_style["taper_end"] = 0.20
        pts = _eval_quad(
            np.array([x - half_w, y + bow * 0.4]),
            np.array([x, y - bow]),
            np.array([x + half_w, y + bow * 0.4]),
            n=12,
        )
        draw_calligraphic_polyline(canvas, pts, color_rgb, w_base * 1.05, local_style, kind="ledger")

    elif lstyle == "slanted":
        half_w = sp * 1.05
        tilt_rad = math.radians(float(rng.uniform(-15.0, -10.0)))
        dx = half_w * math.cos(tilt_rad)
        dy = half_w * math.sin(tilt_rad)
        pts = np.array([[x - dx, y - dy], [x, y], [x + dx, y + dy]], dtype=np.float64)
        draw_calligraphic_polyline(canvas, pts, color_rgb, w_base * 1.05, local_style, kind="ledger")

    elif lstyle == "short_tick":
        half_w = sp * 0.60
        local_style["taper_start"] = 0.04
        local_style["taper_end"] = 0.04
        pts = np.array(
            [
                [x - half_w, y + float(rng.uniform(-0.35, 0.35))],
                [x + half_w, y + float(rng.uniform(-0.35, 0.35))],
            ],
            dtype=np.float64,
        )
        draw_calligraphic_polyline(canvas, pts, color_rgb, w_base * 1.32, local_style, kind="ledger")

    else:  # "wavy"
        half_w = sp * 1.14
        amp = sp * 0.12
        ctrl = np.array(
            [
                [x - half_w, y + amp * 0.6],
                [x - half_w * 0.45, y - amp],
                [x, y + amp * 0.2],
                [x + half_w * 0.45, y + amp],
                [x + half_w, y - amp * 0.6],
            ],
            dtype=np.float64,
        )
        pts = _eval_catmull_rom(ctrl, samples_per_seg=6)
        draw_calligraphic_polyline(canvas, pts, color_rgb, w_base * 1.02, local_style, kind="ledger")


def draw_ledger_lines(
    canvas: StrokeCanvas,
    x: float,
    y_center: float,
    sp: float,
    steps: Sequence[int],
    color_rgb: RGBLike,
    style: dict,
    rng: np.random.Generator,
) -> None:
    """Draw all required ledger lines above/below the staff for a note/chord's staff_steps."""
    if not steps:
        return
    min_s = min(steps)
    max_s = max(steps)
    ledger_steps: List[int] = []
    if min_s <= -2:
        ledger_steps.extend(range(-2, min_s - 1, -2))
    if max_s >= 10:
        ledger_steps.extend(range(10, max_s + 1, 2))
    for ls in ledger_steps:
        ly = y_center - (ls - 4) * (sp * 0.5)
        draw_ledger_line(canvas, x, ly, sp, color_rgb, style, rng)


# ---------------------------------------------------------------------------
# Clefs
# ---------------------------------------------------------------------------


def draw_clef(
    canvas: StrokeCanvas,
    clef_name: str,
    cx: float,
    cy: float,
    sp: float,
    color_rgb: RGBLike,
    style: dict,
    rng: np.random.Generator,
) -> None:
    """Draw a handwritten clef ('G-2', 'F-4', 'C-3', 'C-4', 'G8vb', 'G8va') across 18 archetypes."""
    st = style or {}
    arch: Optional[str] = None
    explicit = st.get("clef_archetype")
    if isinstance(explicit, str) and explicit in CLEF_ARCHETYPES:
        arch = explicit
    elif isinstance(explicit, str) and explicit in _LEGACY_CLEF_ARCHETYPE_MAP:
        arch = _LEGACY_CLEF_ARCHETYPE_MAP[explicit]
    else:
        weights = st.get("clef_archetype_weights")
        if isinstance(weights, dict) and weights:
            if any(k in CLEF_ARCHETYPES for k in weights):
                arch = _choose_weighted(CLEF_ARCHETYPES, weights, rng)
            elif any(k in _LEGACY_CLEF_ARCHETYPE_MAP for k in weights):
                legacy_keys = tuple(_LEGACY_CLEF_ARCHETYPE_MAP.keys())
                picked = _choose_weighted(legacy_keys, weights, rng)
                arch = _LEGACY_CLEF_ARCHETYPE_MAP[picked]
    if arch is None:
        arch = resolve_structural_archetypes(st)[0]

    draw_clef_variant(
        canvas,
        cx,
        cy,
        sp,
        clef_name,
        arch,
        st,
        rng,
        color_rgb=color_rgb,
    )


# ---------------------------------------------------------------------------
# Accidentals & Key Signatures
# ---------------------------------------------------------------------------


def draw_accidental(
    canvas: StrokeCanvas,
    acc: str,
    cx: float,
    cy: float,
    sp: float,
    color_rgb: RGBLike,
    style: dict,
    rng: np.random.Generator,
) -> None:
    """Draw a handwritten accidental ('sharp', 'flat', 'natural', 'double_sharp', 'double_flat')."""
    st = style or {}
    _, keysig_arch, _ = resolve_structural_archetypes(st)
    draw_accidental_variant(
        canvas,
        cx,
        cy,
        sp,
        acc,
        keysig_arch,
        st,
        rng,
        color_rgb=color_rgb,
    )


def draw_key_sig(
    canvas: StrokeCanvas,
    x_start: float,
    y_lines: object,
    sp: float,
    fifths: int,
    clef_kind: str,
    color_rgb: RGBLike,
    style: dict,
    rng: np.random.Generator,
) -> float:
    """Draw a handwritten key signature (fifths in -7..+7) across 18 structural archetypes."""
    st = style or {}
    _, keysig_arch, _ = resolve_structural_archetypes(st)
    return draw_keysig_variant(
        canvas,
        x_start,
        y_lines,
        sp,
        fifths,
        clef_kind,
        keysig_arch,
        st,
        rng,
        color_rgb=color_rgb,
    )


# ---------------------------------------------------------------------------
# Time Signatures & Handwritten Numerals
# ---------------------------------------------------------------------------


def draw_time_sig(
    canvas: StrokeCanvas,
    num: Union[int, str],
    den: Union[int, str],
    cx: float,
    cy: float,
    sp: float,
    color_rgb: RGBLike,
    style: dict,
    rng: np.random.Generator,
) -> None:
    """Draw a hand-drawn time signature (10 archetypes, incl. C / cut-time)."""
    from .expressive_glyphs import draw_time_sig_variant

    draw_time_sig_variant(canvas, num, den, cx, cy, sp, color_rgb, style, rng)


# ---------------------------------------------------------------------------
# Barlines & Braces
# ---------------------------------------------------------------------------


def draw_barline(
    canvas: StrokeCanvas,
    x: float,
    y_top: float,
    y_bot: float,
    sp: float,
    kind: str,
    color_rgb: RGBLike,
    style: dict,
    rng: np.random.Generator,
) -> None:
    """Draw a handwritten barline ('single', 'double', 'final', 'repeat-right', 'repeat-left', 'repeat-both')."""
    st = style or {}
    _, _, barline_arch = resolve_structural_archetypes(st)
    draw_barline_variant(
        canvas,
        x,
        y_top,
        y_bot,
        sp,
        kind,
        barline_arch,
        st,
        rng,
        color_rgb=color_rgb,
    )


def draw_brace(
    canvas: StrokeCanvas,
    x: float,
    y_top: float,
    y_bot: float,
    sp: float,
    color_rgb: RGBLike,
    style: dict,
    rng: np.random.Generator,
) -> None:
    """Draw a hand-drawn curly brace / bracket (5 archetypes via expressive_glyphs)."""
    from .expressive_glyphs import draw_brace_variant

    draw_brace_variant(canvas, x, y_top, y_bot, sp, color_rgb, style, rng)


def draw_cursive_ligature(
    canvas: StrokeCanvas,
    pattern: str,
    p_start: Sequence[float],
    p_end: Sequence[float],
    sp: float,
    color_rgb: RGBLike,
    style: dict,
    rng: np.random.Generator,
    stem_dir: int = 1,
) -> None:
    """Draw one of 4 pen-lift-laziness cursive hairline drags."""
    from .styles import resolve_ductus_param

    p0 = np.asarray(p_start, dtype=np.float64)
    p3 = np.asarray(p_end, dtype=np.float64)
    dist = float(np.hypot(*(p3 - p0)))
    if dist < 0.25 * sp or dist > 6.0 * sp:
        return

    lig_op = float(np.clip(resolve_ductus_param(style, "ligature_opacity", 0.34), 0.10, 0.70))
    local_style = dict(style)
    base_op = float(style.get("stroke_opacity", style.get("opacity", 0.92)))
    local_style["stroke_opacity"] = float(np.clip(base_op * lig_op * float(rng.uniform(0.75, 1.15)), 0.08, 0.65))
    local_style["ink_dropout"] = float(np.clip(float(style.get("ink_dropout", 0.08)) + 0.26, 0.0, 0.78))
    local_style["taper_start"] = 0.28
    local_style["taper_end"] = 0.34
    hair_w = max(0.45, sp * float(rng.uniform(0.045, 0.082)))

    dx, dy = float(p3[0] - p0[0]), float(p3[1] - p0[1])
    if pattern == "notehead_to_notehead_swoop":
        sag = float(rng.uniform(0.28, 0.78)) * sp * (1.0 if stem_dir >= 0 else -1.0)
        p1 = np.array([p0[0] + dx * 0.30, p0[1] + dy * 0.20 + sag])
        p2 = np.array([p0[0] + dx * 0.72, p0[1] + dy * 0.85 + sag * 0.75])
        pts = _eval_cubic(p0, p1, p2, p3, n=14)
        draw_calligraphic_polyline(canvas, pts, color_rgb, hair_w, local_style, kind="ligature")
    elif pattern == "notehead_to_stem_base_hook":
        elbow = 0.5 * (p0 + p3) + np.array([
            float(rng.uniform(-0.22, 0.28)) * sp,
            float(rng.uniform(0.18, 0.42)) * sp * stem_dir,
        ])
        pts = _eval_quad(p0, elbow, p3, n=12)
        draw_calligraphic_polyline(canvas, pts, color_rgb, hair_w * 1.05, local_style, kind="ligature")
    elif pattern == "flag_to_flag_return_loop":
        loop_y = -stem_dir * float(rng.uniform(0.35, 0.85)) * sp
        p1 = np.array([p0[0] + max(0.35 * sp, dx * 0.38), p0[1] + loop_y])
        p2 = np.array([p3[0] - max(0.25 * sp, dx * 0.28), p3[1] - loop_y * 0.5])
        pts = _eval_cubic(p0, p1, p2, p3, n=18)
        if dist > 2.2 * sp:
            draw_calligraphic_polyline(canvas, pts[:8], color_rgb, hair_w, local_style, kind="ligature")
            draw_calligraphic_polyline(canvas, pts[11:], color_rgb, hair_w * 0.9, local_style, kind="ligature")
        else:
            draw_calligraphic_polyline(canvas, pts, color_rgb, hair_w, local_style, kind="ligature")
    else:  # "chord_roll_drag"
        bow = float(rng.uniform(-0.24, 0.24)) * sp
        mid = 0.5 * (p0 + p3) + np.array([bow, float(rng.normal(0.0, 0.04 * sp))])
        pts = _eval_quad(p0, mid, p3, n=10)
        draw_calligraphic_polyline(canvas, pts, color_rgb, hair_w * 0.95, local_style, kind="ligature")


# ---------------------------------------------------------------------------
# Expressive Primitives: Slurs/Ties, Fermatas, Dynamics, Hairpins, Corrections
# ---------------------------------------------------------------------------


def draw_slur_or_tie(
    canvas: StrokeCanvas,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    sp: float,
    above: bool,
    color_rgb: RGBLike,
    style: dict,
    rng: np.random.Generator,
) -> None:
    """Draw a tapered organic handwritten slur or tie between (x0, y0) and (x1, y1)."""
    dx = x1 - x0
    dy = y1 - y0
    span = math.hypot(dx, dy)
    if span < 1.0:
        return

    sign = -1.0 if above else 1.0
    # Natural arch height scales sublinearly with span.
    arch_h = sign * max(0.55 * sp, min(2.4 * sp, 0.18 * span + 0.42 * sp)) * float(rng.uniform(0.85, 1.18))
    asym = float(rng.uniform(-0.08, 0.08))

    p0 = np.array([x0 + rng.normal(0.0, 0.04 * sp), y0 + rng.normal(0.0, 0.04 * sp)])
    p3 = np.array([x1 + rng.normal(0.0, 0.04 * sp), y1 + rng.normal(0.0, 0.04 * sp)])
    p1 = p0 + np.array([dx * (0.28 + asym), dy * 0.28 + arch_h])
    p2 = p0 + np.array([dx * (0.72 + asym), dy * 0.72 + arch_h])

    n_pts = max(16, int(span / 4.5))
    curve = _eval_cubic(p0, p1, p2, p3, n=n_pts)

    thin_w = max(0.85, sp * float(style.get("slur_stroke_ratio", 0.13)))
    local_style = dict(style)
    local_style["taper_start"] = 0.24
    local_style["taper_end"] = 0.24
    draw_calligraphic_polyline(canvas, curve, color_rgb, thin_w, local_style, kind="slur")

    # Subtle calligraphic thickening at the apex of the arch.
    mid_p1 = p1 + np.array([0.0, sign * 0.12 * sp])
    mid_p2 = p2 + np.array([0.0, sign * 0.12 * sp])
    inner = _eval_cubic(p0, mid_p1, mid_p2, p3, n=n_pts)
    s0 = max(1, n_pts // 6)
    s1 = min(n_pts - 1, n_pts - n_pts // 6)
    if s1 > s0 + 1:
        draw_calligraphic_polyline(canvas, inner[s0:s1], color_rgb, thin_w * 1.15, local_style, kind="slur")


def draw_fermata(
    canvas: StrokeCanvas,
    cx: float,
    cy: float,
    sp: float,
    above: bool,
    color_rgb: RGBLike,
    style: dict,
    rng: np.random.Generator,
) -> None:
    """Draw a hand-drawn fermata (semicircular arch + center ink dot)."""
    sign = -1.0 if above else 1.0
    rx = sp * float(rng.uniform(0.95, 1.22))
    ry = sp * float(rng.uniform(0.68, 0.92))
    phis = np.linspace(math.pi * 0.08, math.pi * 0.92, 18)
    arch = np.column_stack((
        cx + rx * np.cos(phis) + rng.normal(0.0, 0.03 * sp, size=len(phis)),
        cy + sign * ry * np.sin(phis) + rng.normal(0.0, 0.03 * sp, size=len(phis)),
    ))
    base_w = max(1.0, sp * float(style.get("glyph_stroke_ratio", 0.18)))
    local_style = dict(style)
    local_style["taper_start"] = 0.22
    local_style["taper_end"] = 0.22
    draw_calligraphic_polyline(canvas, arch, color_rgb, base_w * 1.12, local_style, kind="fermata")

    dot_r = max(1.1, sp * float(rng.uniform(0.16, 0.24)))
    dot_y = cy + sign * ry * float(rng.uniform(0.22, 0.36))
    draw_filled_blob(
        canvas,
        cx + float(rng.normal(0.0, 0.04 * sp)),
        dot_y,
        dot_r,
        dot_r * float(rng.uniform(0.88, 1.12)),
        float(rng.uniform(-25.0, 25.0)),
        color_rgb,
        style,
        hollow=False,
    )


def _draw_dynamic_letter(
    canvas: StrokeCanvas,
    ch: str,
    cx: float,
    cy: float,
    sp: float,
    color_rgb: RGBLike,
    style: dict,
    rng: np.random.Generator,
) -> None:
    """Draw a single cursive handwritten dynamic letter ('p', 'f', 'm', 's', 'z', 'r')."""
    w = max(1.0, sp * float(style.get("dynamic_stroke_ratio", 0.19)))

    def _stroke(ctrl: np.ndarray, smooth: bool = True, width_mult: float = 1.0) -> None:
        pts = _transform_local(
            ctrl,
            cx,
            cy,
            sp,
            rng,
            style,
            jitter_sp=0.06,
            scale_x_range=(0.88, 1.14),
            scale_y_range=(0.88, 1.14),
            extra_slant_deg=10.0,
        )
        path = _eval_catmull_rom(pts, 7) if (smooth and len(pts) >= 3) else pts
        draw_calligraphic_polyline(canvas, path, color_rgb, w * width_mult, style, kind="dynamic")

    c = ch.lower()
    if c == "p":
        # Descending stem + cursive open belly loop.
        _stroke(np.array([[-0.22, -0.52], [-0.38, 1.05]]), smooth=False, width_mult=1.12)
        _stroke(np.array([[-0.24, -0.28], [0.22, -0.56], [0.46, -0.10], [0.16, 0.38], [-0.28, 0.24]]), smooth=True)
    elif c == "f":
        # Sweeping cursive forte S-stem + angled crossbar.
        _stroke(np.array([[0.48, -1.05], [0.12, -1.16], [-0.04, -0.20], [-0.22, 0.92], [-0.56, 1.08]]), smooth=True, width_mult=1.18)
        _stroke(np.array([[-0.38, -0.18], [0.36, -0.24]]), smooth=False, width_mult=0.92)
    elif c == "m":
        # Three-arch cursive mezzo 'm'.
        _stroke(np.array([
            [-0.64, 0.34], [-0.50, -0.34], [-0.22, -0.30], [-0.20, 0.34],
            [-0.08, -0.34], [0.18, -0.30], [0.20, 0.34],
            [0.34, -0.34], [0.58, -0.26], [0.64, 0.36],
        ]), smooth=True, width_mult=0.95)
    elif c == "s":
        _stroke(np.array([[0.32, -0.44], [-0.14, -0.52], [-0.26, -0.10], [0.26, 0.10], [0.16, 0.48], [-0.34, 0.42]]), smooth=True)
    elif c == "z":
        _stroke(np.array([[-0.34, -0.42], [0.36, -0.44], [-0.36, 0.42], [0.38, 0.40]]), smooth=False)
    elif c == "r":
        _stroke(np.array([[-0.26, 0.38], [-0.20, -0.38], [0.10, -0.42], [0.34, -0.18]]), smooth=True)
    else:
        # Generic small cursive stroke fallback.
        _stroke(np.array([[-0.24, -0.32], [0.24, 0.32]]), smooth=False)


def draw_dynamic_text(
    canvas: StrokeCanvas,
    text: str,
    cx: float,
    cy: float,
    sp: float,
    color_rgb: RGBLike,
    style: dict,
    rng: np.random.Generator,
) -> None:
    """Draw handwritten cursive dynamic markings (e.g. 'p', 'mf', 'ff', 'sfz', 'fp')."""
    chars = [ch for ch in str(text).strip() if not ch.isspace()]
    if not chars:
        return
    step = sp * float(rng.uniform(0.84, 1.02))
    start_x = cx - 0.5 * step * (len(chars) - 1)
    for i, ch in enumerate(chars):
        lx = start_x + i * step + float(rng.normal(0.0, 0.04 * sp))
        ly = cy + float(rng.normal(0.0, 0.04 * sp))
        _draw_dynamic_letter(canvas, ch, lx, ly, sp, color_rgb, style, rng)


def draw_hairpin(
    canvas: StrokeCanvas,
    x0: float,
    x1: float,
    cy: float,
    sp: float,
    kind: str,
    color_rgb: RGBLike,
    style: dict,
    rng: np.random.Generator,
) -> None:
    """Draw a handwritten crescendo ('cresc'/'<') or diminuendo ('dim'/'decresc'/'>') hairpin."""
    if x1 < x0:
        x0, x1 = x1, x0
    span = x1 - x0
    if span < 1.0:
        return

    k = str(kind).strip().lower()
    is_cresc = k in ("cresc", "crescendo", "<", "open_right")
    half_open = sp * float(rng.uniform(0.45, 0.72))
    base_w = max(0.9, sp * float(style.get("hairpin_stroke_ratio", 0.135)))

    if is_cresc:
        vx, ox = x0, x1
    else:
        vx, ox = x1, x0

    # Slight organic hand-drawn bow on each arm.
    for sign in (-1.0, 1.0):
        vy = cy + float(rng.normal(0.0, 0.04 * sp))
        oy = cy + sign * half_open + float(rng.normal(0.0, 0.05 * sp))
        mid_x = 0.5 * (vx + ox)
        mid_y = 0.5 * (vy + oy) + float(rng.normal(0.0, 0.07 * sp))
        pts = _eval_quad(np.array([vx, vy]), np.array([mid_x, mid_y]), np.array([ox, oy]), n=max(10, int(span / 8.0)))
        draw_calligraphic_polyline(canvas, pts, color_rgb, base_w, style, kind="hairpin")


def draw_correction_scratch(
    canvas: StrokeCanvas,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    sp: float,
    color_rgb: RGBLike,
    style: dict,
    rng: np.random.Generator,
    kind: str = "auto",
) -> None:
    """Draw a composer/student cross-out scratch over a mistaken note or bar region."""
    lx0, lx1 = (min(x0, x1), max(x0, x1))
    ly0, ly1 = (min(y0, y1), max(y0, y1))
    w = max(sp * 0.8, lx1 - lx0)
    h = max(sp * 0.8, ly1 - ly0)

    k = str(kind).strip().lower()
    if k == "auto":
        k = str(rng.choice(["scribble", "cross_x", "hatch"]))

    stroke_w = max(1.0, sp * float(style.get("glyph_stroke_ratio", 0.18)) * float(rng.uniform(0.95, 1.35)))

    if k == "cross_x":
        # Bold X plus optional extra horizontal slash.
        p1 = np.array([[lx0, ly0], [lx1, ly1]], dtype=np.float64) + rng.normal(0.0, 0.08 * sp, size=(2, 2))
        p2 = np.array([[lx0, ly1], [lx1, ly0]], dtype=np.float64) + rng.normal(0.0, 0.08 * sp, size=(2, 2))
        draw_calligraphic_polyline(canvas, p1, color_rgb, stroke_w * 1.15, style, kind="correction")
        draw_calligraphic_polyline(canvas, p2, color_rgb, stroke_w * 1.15, style, kind="correction")
        if rng.random() < 0.5:
            ym = 0.5 * (ly0 + ly1)
            p3 = np.array([[lx0 - 0.15 * sp, ym], [lx1 + 0.15 * sp, ym + rng.normal(0.0, 0.15 * sp)]], dtype=np.float64)
            draw_calligraphic_polyline(canvas, p3, color_rgb, stroke_w * 0.95, style, kind="correction")
    elif k == "hatch":
        # Multiple rapid parallel diagonal cross-out strokes.
        n_lines = int(rng.integers(3, 7))
        for i in range(n_lines):
            frac = (i + 0.5) / n_lines
            cx = lx0 + frac * w
            dx = 0.25 * h * float(rng.uniform(0.6, 1.4))
            pts = np.array([
                [cx - dx + rng.normal(0.0, 0.06 * sp), ly1 + rng.normal(0.0, 0.08 * sp)],
                [cx + dx + rng.normal(0.0, 0.06 * sp), ly0 + rng.normal(0.0, 0.08 * sp)],
            ], dtype=np.float64)
            draw_calligraphic_polyline(canvas, pts, color_rgb, stroke_w * 0.95, style, kind="correction")
    else:  # "scribble"
        # Dense back-and-forth zigzag scribble crossing out the mistake.
        n_zigs = int(rng.integers(5, 11))
        xs = np.linspace(lx0, lx1, n_zigs)
        ys = np.array([ly0 if i % 2 == 0 else ly1 for i in range(n_zigs)], dtype=np.float64)
        ctrl = np.column_stack((xs, ys)) + rng.normal(0.0, 0.12 * sp, size=(n_zigs, 2))
        curve = _eval_catmull_rom(ctrl, samples_per_seg=6)
        draw_calligraphic_polyline(canvas, curve, color_rgb, stroke_w * 1.08, style, kind="correction")
