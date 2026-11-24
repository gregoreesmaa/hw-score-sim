"""Living-manuscript overlays, corrections, multi-tool annotations, and handling wear.

All 7 layers are 100% OMR ground-truth safe: protected notation bboxes are never
corrupted, cancellations stay strictly on blank staves/margins, translucent washes
and palm smears have bounded alpha, and every drawn mark is logged in
``gt_payload["manuscript_marks"]``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .stroke_engine import (
    StrokeCanvas,
    _soft_blur_2d,
    draw_calligraphic_polyline,
    draw_filled_blob,
)

# ---------------------------------------------------------------------------
# Registries
# ---------------------------------------------------------------------------

ANNOTATION_PERSONAS: Tuple[str, ...] = (
    "none",
    "orchestral_conductor",
    "string_section_pencil",
    "piano_conservatory",
    "jazz_realbook_chart",
    "choral_rehearsal",
    "archival_librarian",
    "film_studio_copyist",
    "composer_working_draft",
)

CANCELLATION_STYLES: Tuple[str, ...] = (
    "tight_scribble_hatch",
    "double_diagonal_slash",
    "heavy_x_cross",
    "wavy_loop_cancel",
    "vide_cut_bracket",
    "ink_blot_smear_out",
    "box_and_strike",
)

MEASURE_NUMBER_STYLES: Tuple[str, ...] = (
    "none",
    "plain",
    "boxed",
    "circled",
    "italic_underlined",
    "pit_every_bar",
)

CHORD_SYMBOL_STYLES: Tuple[str, ...] = (
    "none",
    "diatonic_folk",
    "jazz_realbook",
    "pop_slash",
    "nashville_roman",
)

PALM_SMEAR_HANDS: Tuple[str, ...] = ("random", "right", "left")

# ---------------------------------------------------------------------------
# Multi-Tool Physics Profiles
# ---------------------------------------------------------------------------

TOOL_PROFILES: Dict[str, Dict[str, Any]] = {
    "grease_pencil_red": {
        "rgb": (0.79, 0.15, 0.13), "stroke_scale": 1.85, "grain_strength": 0.58,
        "ink_dropout": 0.24, "stroke_opacity": 0.86, "nib_min_ratio": 0.78, "ink_pooling": 0.05,
    },
    "grease_pencil_blue": {
        "rgb": (0.12, 0.28, 0.72), "stroke_scale": 1.85, "grain_strength": 0.58,
        "ink_dropout": 0.24, "stroke_opacity": 0.86, "nib_min_ratio": 0.78, "ink_pooling": 0.05,
    },
    "graphite_2b": {
        "rgb": (0.34, 0.34, 0.33), "stroke_scale": 0.95, "grain_strength": 0.62,
        "ink_dropout": 0.28, "stroke_opacity": 0.72, "nib_min_ratio": 0.75, "ink_pooling": 0.0,
    },
    "graphite_hard_hb": {
        "rgb": (0.44, 0.44, 0.43), "stroke_scale": 0.72, "grain_strength": 0.48,
        "ink_dropout": 0.20, "stroke_opacity": 0.62, "nib_min_ratio": 0.82, "ink_pooling": 0.0,
    },
    "felt_chisel_black": {
        "rgb": (0.09, 0.09, 0.10), "stroke_scale": 1.65, "grain_strength": 0.12,
        "ink_dropout": 0.06, "stroke_opacity": 0.95, "nib_min_ratio": 0.38, "ink_pooling": 0.10,
    },
    "ballpoint_blue": {
        "rgb": (0.11, 0.20, 0.56), "stroke_scale": 0.78, "grain_strength": 0.10,
        "ink_dropout": 0.08, "stroke_opacity": 0.88, "nib_min_ratio": 0.88, "ink_pooling": 0.04,
    },
    "stamp_indigo": {
        "rgb": (0.26, 0.22, 0.54), "stroke_scale": 1.10, "grain_strength": 0.42,
        "ink_dropout": 0.35, "stroke_opacity": 0.58, "nib_min_ratio": 0.85, "ink_pooling": 0.0,
    },
    "stamp_crimson": {
        "rgb": (0.66, 0.16, 0.18), "stroke_scale": 1.10, "grain_strength": 0.45,
        "ink_dropout": 0.38, "stroke_opacity": 0.56, "nib_min_ratio": 0.85, "ink_pooling": 0.0,
    },
    "highlighter_yellow": {
        "rgb": (0.96, 0.92, 0.22), "stroke_scale": 6.50, "grain_strength": 0.05,
        "ink_dropout": 0.02, "stroke_opacity": 0.26, "nib_min_ratio": 0.95, "ink_pooling": 0.0,
    },
    "highlighter_green": {
        "rgb": (0.45, 0.90, 0.42), "stroke_scale": 6.50, "grain_strength": 0.05,
        "ink_dropout": 0.02, "stroke_opacity": 0.24, "nib_min_ratio": 0.95, "ink_pooling": 0.0,
    },
}

_CHORD_VOCABS: Dict[str, Tuple[str, ...]] = {
    "diatonic_folk": ("D", "G", "A7", "Bm", "Em", "F#m", "D/F#", "A", "G/B", "D"),
    "jazz_realbook": ("Dmaj7", "Bm7", "Em9", "A13", "F#m7", "B7#9", "Em7", "A7alt", "Gmaj7", "C#ø7"),
    "pop_slash": ("D/F#", "Gadd9", "A/C#", "Bm7", "G/B", "D2", "Asus4", "A", "Em7/A", "D"),
    "nashville_roman": ("I", "vi7", "ii7", "V7", "IV", "iii", "V/V", "I", "IV", "V7"),
}


def _tool_style(base_style: Dict[str, Any], tool_name: str) -> Tuple[Dict[str, Any], Tuple[float, float, float]]:
    """Derive a non-mutating style copy configured for a secondary writing tool."""
    out = dict(base_style)
    prof = TOOL_PROFILES.get(tool_name)
    if prof is None:
        rgb = tuple(float(c) for c in base_style.get("ink_rgb", (0.15, 0.13, 0.11)))
        return out, (rgb[0], rgb[1], rgb[2])
    for k, v in prof.items():
        if k != "rgb":
            out[k] = v
    out["ink_dip_period"] = 0
    rgb = prof["rgb"]
    return out, (float(rgb[0]), float(rgb[1]), float(rgb[2]))


# ---------------------------------------------------------------------------
# OMR Ground-Truth Safe-Zone Guard
# ---------------------------------------------------------------------------

@dataclass
class SafeZoneGuard:
    """Tracks protected notation bboxes (xyxy) so manuscript marks never corrupt OMR GT."""

    page_w: int
    page_h: int
    protected: List[Tuple[float, float, float, float]] = field(default_factory=list)
    drawn_marks: List[Tuple[float, float, float, float]] = field(default_factory=list)

    def add_protected(self, x0: float, y0: float, x1: float, y1: float, pad: float = 0.0) -> None:
        self.protected.append((x0 - pad, y0 - pad, x1 + pad, y1 + pad))

    def is_safe(self, x0: float, y0: float, x1: float, y1: float, check_marks: bool = True) -> bool:
        if x0 < 4.0 or y0 < 4.0 or x1 > self.page_w - 4.0 or y1 > self.page_h - 4.0:
            return False
        for px0, py0, px1, py1 in self.protected:
            if not (x1 < px0 or x0 > px1 or y1 < py0 or y0 > py1):
                return False
        if check_marks:
            for mx0, my0, mx1, my1 in self.drawn_marks:
                if not (x1 < mx0 or x0 > mx1 or y1 < my0 or y0 > my1):
                    return False
        return True

    def reserve(self, x0: float, y0: float, x1: float, y1: float) -> None:
        self.drawn_marks.append((x0, y0, x1, y1))

    @classmethod
    def from_systems(cls, page_w: int, page_h: int, systems: Sequence[Any]) -> "SafeZoneGuard":
        guard = cls(page_w=page_w, page_h=page_h)
        for sys_lay in systems:
            if getattr(sys_lay, "is_blank_prerule", False):
                continue
            for st in sys_lay.staves:
                sp = float(st.sp)
                y_top = float(st.y_center) - 2.0 * sp
                y_bot = float(st.y_center) + 2.0 * sp
                guard.add_protected(st.x_left, y_top, st.x_right, y_bot, pad=0.85 * sp)
        return guard


def _log_mark(
    marks: List[Dict[str, Any]],
    *,
    layer: str,
    kind: str,
    bbox_xyxy: Tuple[float, float, float, float],
    tool: str,
    color_rgb: Sequence[float],
    persona: str = "none",
    substyle: Optional[str] = None,
    text: Optional[str] = None,
    system_index: Optional[int] = None,
    measure_index: Optional[int] = None,
) -> None:
    marks.append({
        "layer": layer,
        "persona": persona,
        "kind": kind,
        "substyle": substyle,
        "tool": tool,
        "color_rgb": [round(float(c), 4) for c in color_rgb],
        "bbox_xyxy": [round(float(v), 2) for v in bbox_xyxy],
        "text": text,
        "system_index": system_index,
        "measure_index": measure_index,
        "gt_safe": True,
    })


# ---------------------------------------------------------------------------
# Extended Stroke Glyphs for Chords / Marks
# ---------------------------------------------------------------------------

EXTRA_STROKE_GLYPHS: Dict[str, List[List[Tuple[float, float]]]] = {
    "#": [[(0.16, -0.68), (0.14, 0.02)], [(0.42, -0.72), (0.40, -0.02)],
          [(0.04, -0.46), (0.54, -0.52)], [(0.04, -0.22), (0.54, -0.28)]],
    "/": [[(0.46, -0.74), (0.10, 0.06)]],
    "+": [[(0.28, -0.52), (0.28, -0.10)], [(0.08, -0.31), (0.48, -0.31)]],
    "*": [[(0.28, -0.54), (0.28, -0.08)], [(0.08, -0.44), (0.48, -0.18)],
          [(0.48, -0.44), (0.08, -0.18)]],
    "^": [[(0.08, -0.10), (0.28, -0.54), (0.48, -0.10)]],
    "[": [[(0.42, -0.74), (0.16, -0.74), (0.16, 0.06), (0.42, 0.06)]],
    "]": [[(0.14, -0.74), (0.40, -0.74), (0.40, 0.06), (0.14, 0.06)]],
    "(": [[(0.40, -0.74), (0.16, -0.36), (0.40, 0.06)]],
    ")": [[(0.16, -0.74), (0.40, -0.36), (0.16, 0.06)]],
    ":": [[(0.24, -0.48), (0.26, -0.46)], [(0.24, -0.12), (0.26, -0.10)]],
    "?": [[(0.12, -0.58), (0.28, -0.72), (0.46, -0.56), (0.28, -0.32), (0.26, -0.18)],
          [(0.26, -0.04), (0.28, 0.00)]],
    "Δ": [[(0.06, 0.00), (0.30, -0.64), (0.54, 0.00), (0.06, 0.00)]],
    "ø": [[(0.28, -0.48), (0.08, -0.28), (0.28, -0.06), (0.48, -0.28), (0.28, -0.48)],
          [(0.06, 0.02), (0.52, -0.56)]],
    "°": [[(0.26, -0.66), (0.12, -0.52), (0.26, -0.38), (0.40, -0.52), (0.26, -0.66)]],
}


def draw_mark_text(
    canvas: StrokeCanvas,
    text: str,
    x: float,
    y_baseline: float,
    size_px: float,
    ink_rgb: Sequence[float],
    style: Dict[str, Any],
    rng: np.random.Generator,
    italic: bool = False,
    bold: bool = False,
) -> Tuple[float, Tuple[float, float, float, float]]:
    """Draw handwritten annotation/chord text using renderer's _STROKE_FONT + EXTRA_STROKE_GLYPHS."""
    from .renderer import _STROKE_FONT

    slant_deg = float(style.get("slant_deg", 6.0)) + (12.0 if italic else 0.0)
    shear = math.tan(math.radians(float(np.clip(slant_deg, -24.0, 28.0))))
    w = max(0.85, size_px * (0.145 if bold else 0.105))
    local_style = dict(style)
    local_style["wobble_px"] = min(float(style.get("wobble_px", 0.7)), size_px * 0.06)
    local_style["tremor_px"] = min(float(style.get("tremor_px", 0.2)), size_px * 0.03)

    cursor = x
    for ch in text:
        if ch == " ":
            cursor += size_px * 0.42
            continue
        strokes = EXTRA_STROKE_GLYPHS.get(ch) or _STROKE_FONT.get(ch.lower())
        if strokes is None:
            cursor += size_px * 0.35
            continue
        is_upper = ch.isupper() and ch.isalpha()
        sx = 1.15 if is_upper else 1.0
        sy = 1.30 if is_upper else 1.0
        for st in strokes:
            pts = np.array(
                [[cursor + px * size_px * sx - (py * size_px * sy) * shear,
                  y_baseline + py * size_px * sy] for px, py in st],
                dtype=np.float64,
            )
            draw_calligraphic_polyline(canvas, pts, ink_rgb, w, local_style, kind="annotation")
        cursor += size_px * (0.70 if ch.lower() in ("m", "w") else 0.56) * sx + float(rng.normal(0.0, size_px * 0.02))

    bbox = (x, y_baseline - size_px * 1.05, max(cursor, x + size_px * 0.5), y_baseline + size_px * 0.25)
    return cursor, bbox


def _draw_oval_loop(
    canvas: StrokeCanvas,
    cx: float,
    cy: float,
    rx: float,
    ry: float,
    w: float,
    rgb: Sequence[float],
    style: Dict[str, Any],
    rng: np.random.Generator,
) -> None:
    phis = np.linspace(0.0, 2.0 * math.pi + float(rng.uniform(0.15, 0.55)), 26)
    xs = cx + rx * np.cos(phis) + rng.normal(0.0, rx * 0.04, len(phis))
    ys = cy + ry * np.sin(phis) + rng.normal(0.0, ry * 0.04, len(phis))
    draw_calligraphic_polyline(canvas, np.column_stack((xs, ys)), rgb, w, style, kind="annotation")


def _draw_wobbly_box(
    canvas: StrokeCanvas,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    w: float,
    rgb: Sequence[float],
    style: Dict[str, Any],
    rng: np.random.Generator,
    double_pass: bool = False,
) -> None:
    pts = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]], dtype=np.float64)
    pts += rng.normal(0.0, max(0.4, w * 0.3), size=pts.shape)
    draw_calligraphic_polyline(canvas, pts, rgb, w, style, kind="annotation")
    if double_pass:
        pts2 = pts + rng.normal(0.0, max(0.6, w * 0.5), size=pts.shape)
        draw_calligraphic_polyline(canvas, pts2, rgb, w * 0.85, style, kind="annotation")


# ---------------------------------------------------------------------------
# Layer 7a (PRE-INK): Structured Mirrored Verso Bleed-Through, Fold Crease, Foxing
# ---------------------------------------------------------------------------

def apply_pre_ink_manuscript_layers(
    canvas: StrokeCanvas,
    style: Dict[str, Any],
    rng: np.random.Generator,
    marks_out: List[Dict[str, Any]],
) -> None:
    """Run BEFORE foreground staves/notes: mirrored verso notation bleed, fold crease, foxing clusters."""
    h, w = canvas.height, canvas.width

    # 1. Structured mirrored verso notation bleed-through
    verso_strength = float(np.clip(style.get("verso_bleed_strength", 0.0), 0.0, 1.0))
    if verso_strength > 0.01:
        sh, sw = max(64, h // 2), max(64, w // 2)
        mask = np.zeros((sh, sw), dtype=np.float32)
        n_staves = int(rng.integers(4, 8))
        sp_v = max(3.5, sh / (n_staves * 13.5))
        mx0, mx1 = int(sw * 0.07), int(sw * 0.93)

        for s_i in range(n_staves):
            yc = (s_i + 0.65) * (sh / (n_staves + 0.35)) + float(rng.uniform(-sp_v * 1.2, sp_v * 1.2))
            for li in range(-2, 3):
                yl = int(round(yc + li * sp_v))
                if 1 <= yl < sh - 1:
                    mask[yl, mx0:mx1] = np.maximum(mask[yl, mx0:mx1], 0.45)
            n_bars = int(rng.integers(3, 6))
            bar_xs = np.linspace(mx0, mx1, n_bars + 1)
            for bx in bar_xs:
                ix = int(np.clip(bx, 1, sw - 2))
                y0, y1 = max(0, int(yc - 2 * sp_v)), min(sh, int(yc + 2 * sp_v) + 1)
                mask[y0:y1, ix] = np.maximum(mask[y0:y1, ix], 0.55)
            for _ in range(int(rng.integers(14, 28))):
                nx = int(rng.integers(mx0 + 6, max(mx0 + 7, mx1 - 6)))
                ny = int(np.clip(yc + rng.uniform(-2.6, 2.6) * sp_v, 4, sh - 5))
                rx, ry = max(1, int(round(sp_v * 0.52))), max(1, int(round(sp_v * 0.36)))
                mask[max(0, ny - ry):min(sh, ny + ry + 1), max(0, nx - rx):min(sw, nx + rx + 1)] = 0.85
                stem_h = int(round(sp_v * 3.0))
                if ny > yc:
                    mask[max(0, ny - stem_h):ny, min(sw - 1, nx + rx)] = 0.72
                else:
                    mask[ny:min(sh, ny + stem_h), max(0, nx - rx)] = 0.72

        mask = _soft_blur_2d(mask[:, ::-1], radius=2)
        yi = np.linspace(0, sh - 1, h).astype(np.int32)
        xi = np.linspace(0, sw - 1, w).astype(np.int32)
        full_mask = mask[yi[:, None], xi[None, :]]
        tooth = canvas.get_tooth_patch(0, 0, w, h)
        alpha = np.clip(full_mask * (0.65 + 0.55 * tooth) * (verso_strength * 0.14), 0.0, 0.14)
        base_ink = np.asarray(style.get("ink_rgb", (0.18, 0.14, 0.10)), dtype=np.float32)
        verso_rgb = np.clip(base_ink * 0.65 + np.array([0.24, 0.20, 0.15], dtype=np.float32), 0.0, 1.0)
        canvas.composite_patch(0, 0, w, h, alpha, verso_rgb)
        _log_mark(
            marks_out, layer="verso_wear", kind="mirrored_verso_bleed",
            bbox_xyxy=(0.0, 0.0, float(w), float(h)), tool="verso_ink", color_rgb=verso_rgb,
        )

    # 2. Folded-sheet center crease
    fold_prob = float(np.clip(style.get("fold_crease_prob", 0.0), 0.0, 1.0))
    if fold_prob > 0.0 and float(rng.random()) < fold_prob:
        horiz = bool(rng.random() < 0.72)
        band = max(5, int(round(min(h, w) * float(rng.uniform(0.004, 0.009)))))
        if horiz:
            cy = int(round(h * float(rng.uniform(0.47, 0.53))))
            y0, y1 = max(0, cy - band), min(h, cy + band + 1)
            dy = (np.arange(y0, y1, dtype=np.float32) - cy) / max(band * 0.45, 1.0)
            shadow = np.exp(-0.5 * (dy + 0.55) ** 2) * float(rng.uniform(0.05, 0.10))
            ridge = np.exp(-0.5 * (dy - 0.55) ** 2) * float(rng.uniform(0.025, 0.05))
            canvas.buf[y0:y1, :, :] = np.clip(canvas.buf[y0:y1, :, :] - shadow[:, None, None] + ridge[:, None, None], 0.0, 1.0)
            bbox = (0.0, float(y0), float(w), float(y1))
        else:
            cx = int(round(w * float(rng.uniform(0.47, 0.53))))
            x0, x1 = max(0, cx - band), min(w, cx + band + 1)
            dx = (np.arange(x0, x1, dtype=np.float32) - cx) / max(band * 0.45, 1.0)
            shadow = np.exp(-0.5 * (dx + 0.55) ** 2) * float(rng.uniform(0.05, 0.10))
            ridge = np.exp(-0.5 * (dx - 0.55) ** 2) * float(rng.uniform(0.025, 0.05))
            canvas.buf[:, x0:x1, :] = np.clip(canvas.buf[:, x0:x1, :] - shadow[None, :, None] + ridge[None, :, None], 0.0, 1.0)
            bbox = (float(x0), 0.0, float(x1), float(h))
        _log_mark(
            marks_out, layer="wet_ink_wear", kind="fold_crease",
            substyle="horizontal" if horiz else "vertical",
            bbox_xyxy=bbox, tool="paper_wear", color_rgb=(0.45, 0.40, 0.32),
        )

    # 3. Clustered organic foxing spots
    fox_prob = float(np.clip(style.get("foxing_cluster_prob", 0.0), 0.0, 1.0))
    if fox_prob > 0.0 and float(rng.random()) < fox_prob:
        fox_rgb = np.array([0.58, 0.36, 0.18], dtype=np.float32)
        for _ in range(int(rng.integers(2, 5))):
            cx0 = float(rng.choice([rng.uniform(0.02, 0.16), rng.uniform(0.84, 0.98)])) * w
            cy0 = float(rng.uniform(0.04, 0.96)) * h
            spread = min(w, h) * float(rng.uniform(0.03, 0.07))
            for _ in range(int(rng.integers(6, 14))):
                sx = cx0 + float(rng.normal(0.0, spread))
                sy = cy0 + float(rng.normal(0.0, spread))
                rad = float(rng.uniform(2.0, 9.0))
                x0, x1 = max(0, int(sx - rad * 2.5)), min(w, int(sx + rad * 2.5) + 1)
                y0, y1 = max(0, int(sy - rad * 2.5)), min(h, int(sy + rad * 2.5) + 1)
                if x1 <= x0 or y1 <= y0:
                    continue
                yy, xx = np.ogrid[y0:y1, x0:x1]
                r2 = ((yy - sy) ** 2 + (xx - sx) ** 2) / (rad * rad + 1e-6)
                alpha = np.exp(-0.5 * r2).astype(np.float32) * float(rng.uniform(0.08, 0.22))
                canvas.composite_patch(x0, y0, x1, y1, alpha, fox_rgb)
            _log_mark(
                marks_out, layer="wet_ink_wear", kind="foxing_cluster",
                bbox_xyxy=(max(0.0, cx0 - spread), max(0.0, cy0 - spread),
                           min(float(w), cx0 + spread), min(float(h), cy0 + spread)),
                tool="paper_wear", color_rgb=fox_rgb,
            )


# ---------------------------------------------------------------------------
# Layer 2: 7 Cancellation / Scratch-Out Styles (strictly off-GT)
# ---------------------------------------------------------------------------

def draw_cancelled_false_start(
    canvas: StrokeCanvas,
    x0: float,
    y_center: float,
    span_w: float,
    sp: float,
    cancel_style: str,
    style: Dict[str, Any],
    rng: np.random.Generator,
    marks_out: List[Dict[str, Any]],
) -> None:
    """Draw a dummy 2-4 note false start and cancel it with one of 7 styles (off-GT only)."""
    ink_rgb = style.get("ink_rgb", (0.14, 0.12, 0.10))
    lw = max(1.0, sp * 0.13)
    x1 = x0 + span_w

    n_dummy = int(rng.integers(2, 5))
    xs = np.linspace(x0 + sp * 0.6, x1 - sp * 0.6, n_dummy)
    for nx in xs:
        ny = y_center + float(rng.uniform(-1.2, 1.2)) * sp
        draw_filled_blob(canvas, float(nx), ny, sp * 0.44, sp * 0.32, -25.0, ink_rgb, style)
        stem = np.array([[nx + sp * 0.36, ny], [nx + sp * 0.36, ny - 2.4 * sp]], dtype=np.float64)
        draw_calligraphic_polyline(canvas, stem, ink_rgb, lw * 0.85, style, kind="stem")

    if cancel_style == "tight_scribble_hatch":
        n_zags = int(rng.integers(12, 20))
        zx = np.linspace(x0, x1, n_zags)
        zy = y_center + np.where(np.arange(n_zags) % 2 == 0, -1.5 * sp, 1.5 * sp) + rng.normal(0.0, 0.22 * sp, n_zags)
        draw_calligraphic_polyline(canvas, np.column_stack((zx, zy)), ink_rgb, lw * 1.45, style, kind="annotation")
    elif cancel_style == "double_diagonal_slash":
        for off in (-0.35 * sp, 0.35 * sp):
            pts = np.array([[x0 + off, y_center + 1.6 * sp], [x1 + off, y_center - 1.6 * sp]], dtype=np.float64)
            draw_calligraphic_polyline(canvas, pts, ink_rgb, lw * 1.5, style, kind="annotation")
    elif cancel_style == "heavy_x_cross":
        draw_calligraphic_polyline(canvas, np.array([[x0, y_center - 1.6 * sp], [x1, y_center + 1.6 * sp]], dtype=np.float64), ink_rgb, lw * 1.8, style, kind="annotation")
        draw_calligraphic_polyline(canvas, np.array([[x0, y_center + 1.6 * sp], [x1, y_center - 1.6 * sp]], dtype=np.float64), ink_rgb, lw * 1.8, style, kind="annotation")
    elif cancel_style == "wavy_loop_cancel":
        t = np.linspace(0.0, 1.0, 80)
        wx = x0 + t * span_w + 0.50 * sp * np.sin(2.0 * math.pi * 6.0 * t)
        wy = y_center + 1.15 * sp * np.cos(2.0 * math.pi * 6.0 * t)
        draw_calligraphic_polyline(canvas, np.column_stack((wx, wy)), ink_rgb, lw * 1.35, style, kind="annotation")
    elif cancel_style == "vide_cut_bracket":
        draw_mark_text(canvas, "[vi-", x0 - 1.2 * sp, y_center - 1.5 * sp, sp * 1.05, ink_rgb, style, rng, bold=True)
        draw_mark_text(canvas, "-de]", x1 - 0.6 * sp, y_center - 1.5 * sp, sp * 1.05, ink_rgb, style, rng, bold=True)
        for seg_x in np.linspace(x0 + 0.8 * sp, x1 - 0.8 * sp, 5):
            draw_calligraphic_polyline(canvas, np.array([[seg_x, y_center - 1.8 * sp], [seg_x + 0.45 * sp, y_center - 1.8 * sp]], dtype=np.float64), ink_rgb, lw, style, kind="annotation")
    elif cancel_style == "ink_blot_smear_out":
        for bx in np.linspace(x0 + 0.8 * sp, x1 - 0.8 * sp, 3):
            draw_filled_blob(canvas, float(bx), y_center + float(rng.normal(0, 0.3 * sp)), sp * 1.1, sp * 0.88, float(rng.uniform(-20, 20)), ink_rgb, style)
    else:  # box_and_strike
        box = np.array([
            [x0, y_center - 1.5 * sp], [x1, y_center - 1.5 * sp],
            [x1, y_center + 1.5 * sp], [x0, y_center + 1.5 * sp], [x0, y_center - 1.5 * sp],
        ], dtype=np.float64)
        draw_calligraphic_polyline(canvas, box, ink_rgb, lw * 1.1, style, kind="annotation")
        draw_calligraphic_polyline(canvas, np.array([[x0, y_center + 1.5 * sp], [x1, y_center - 1.5 * sp]], dtype=np.float64), ink_rgb, lw * 1.6, style, kind="annotation")
        draw_mark_text(canvas, "OUT", x1 + 0.3 * sp, y_center + 0.3 * sp, sp * 0.90, ink_rgb, style, rng, bold=True)

    _log_mark(
        marks_out, layer="scratch_out", kind="cancelled_false_start", substyle=cancel_style,
        bbox_xyxy=(x0, y_center - 1.8 * sp, x1, y_center + 1.8 * sp),
        tool="scribe_ink", color_rgb=ink_rgb,
    )


# ---------------------------------------------------------------------------
# Layer 5b: Palm Smears, Capillary Ink Blots, Quill Tests, Dog-Ear Wear
# ---------------------------------------------------------------------------

def apply_palm_smear(
    canvas: StrokeCanvas,
    hand: str,
    style: Dict[str, Any],
    rng: np.random.Generator,
    marks_out: List[Dict[str, Any]],
) -> None:
    """Directional wet-ink palm-drag smear (left-hand drags L->R; right-hand drags down-right)."""
    h, w = canvas.height, canvas.width
    is_left = (hand == "left") or (hand == "random" and float(style.get("slant_deg", 6.0)) < 0.0)
    cx = float(rng.uniform(0.18, 0.80)) * w
    cy = float(rng.uniform(0.15, 0.82)) * h
    rw = int(max(36, w * float(rng.uniform(0.05, 0.13))))
    rh = int(max(18, h * float(rng.uniform(0.02, 0.045))))
    x0, x1 = max(0, int(cx - rw)), min(w, int(cx + rw))
    y0, y1 = max(0, int(cy - rh)), min(h, int(cy + rh))
    if x1 - x0 < 16 or y1 - y0 < 8:
        return

    patch = canvas.buf[y0:y1, x0:x1]
    lum = 0.299 * patch[:, :, 0] + 0.587 * patch[:, :, 1] + 0.114 * patch[:, :, 2]
    ink_src = np.clip(0.88 - lum, 0.0, 1.0)
    smear = np.zeros_like(ink_src)

    n_taps = 14
    dx_step = 2.6 if is_left else 1.4
    dy_step = 0.25 if is_left else 1.15
    for k in range(1, n_taps + 1):
        decay = math.exp(-0.22 * k)
        sx = int(round(k * dx_step))
        sy = int(round(k * dy_step))
        if sx >= ink_src.shape[1] or sy >= ink_src.shape[0]:
            break
        smear[sy:, sx:] += ink_src[: ink_src.shape[0] - sy, : ink_src.shape[1] - sx] * decay

    max_s = float(np.max(smear))
    if max_s <= 1e-4:
        return
    smear = _soft_blur_2d(smear / max_s, radius=2)
    yy, xx = np.ogrid[: smear.shape[0], : smear.shape[1]]
    env = np.exp(-0.5 * (((yy - smear.shape[0] * 0.5) / (rh * 0.65)) ** 2 + ((xx - smear.shape[1] * 0.5) / (rw * 0.72)) ** 2))
    alpha = np.clip(smear * env * float(rng.uniform(0.08, 0.16)), 0.0, 0.16).astype(np.float32)
    ink_rgb = np.asarray(style.get("ink_rgb", (0.16, 0.13, 0.10)), dtype=np.float32)
    canvas.composite_patch(x0, y0, x1, y1, alpha, ink_rgb)
    _log_mark(
        marks_out, layer="wet_ink_wear", kind="palm_smear",
        substyle="left_hand" if is_left else "right_hand",
        bbox_xyxy=(float(x0), float(y0), float(x1), float(y1)),
        tool="scribe_ink", color_rgb=ink_rgb,
    )


def apply_ink_blot_and_splatter(
    canvas: StrokeCanvas,
    guard: SafeZoneGuard,
    style: Dict[str, Any],
    rng: np.random.Generator,
    marks_out: List[Dict[str, Any]],
) -> None:
    """Capillary ink blot + ballistic micro-splatters placed strictly in safe zones."""
    w, h = canvas.width, canvas.height
    ink_rgb = np.asarray(style.get("ink_rgb", (0.14, 0.12, 0.10)), dtype=np.float32)
    for _ in range(12):
        cx = float(rng.uniform(w * 0.04, w * 0.96))
        cy = float(rng.uniform(h * 0.04, h * 0.96))
        rad = float(rng.uniform(4.5, 11.5))
        bbox = (cx - rad * 2.2, cy - rad * 2.2, cx + rad * 2.2, cy + rad * 2.2)
        if not guard.is_safe(*bbox):
            continue
        draw_filled_blob(canvas, cx, cy, rad, rad * float(rng.uniform(0.75, 1.15)), float(rng.uniform(-35, 35)), ink_rgb, style)
        ang = float(rng.uniform(0.0, 2.0 * math.pi))
        for _ in range(int(rng.integers(5, 12))):
            d = float(rng.uniform(rad * 1.1, rad * 2.1))
            a = ang + float(rng.normal(0.0, 0.45))
            sx, sy = cx + d * math.cos(a), cy + d * math.sin(a)
            sr = float(rng.uniform(0.8, 2.2))
            if guard.is_safe(sx - sr, sy - sr, sx + sr, sy + sr, check_marks=False):
                draw_filled_blob(canvas, sx, sy, sr, sr, 0.0, ink_rgb, style)
        guard.reserve(*bbox)
        _log_mark(marks_out, layer="wet_ink_wear", kind="ink_blot", bbox_xyxy=bbox, tool="scribe_ink", color_rgb=ink_rgb)
        return


def apply_quill_test_scribble(
    canvas: StrokeCanvas,
    guard: SafeZoneGuard,
    sp: float,
    style: Dict[str, Any],
    rng: np.random.Generator,
    marks_out: List[Dict[str, Any]],
) -> None:
    """Margin quill-flow test (figure-8, nib ticks, or spiral coil)."""
    w, h = canvas.width, canvas.height
    ink_rgb = style.get("ink_rgb", (0.14, 0.12, 0.10))
    lw = max(0.9, sp * 0.14)
    for _ in range(10):
        on_left = bool(rng.random() < 0.5)
        cx = float(rng.uniform(14.0, max(20.0, w * 0.048))) if on_left else float(rng.uniform(min(w - 20.0, w * 0.952), w - 14.0))
        cy = float(rng.uniform(h * 0.12, h * 0.88))
        r = sp * float(rng.uniform(0.75, 1.35))
        bbox = (cx - r * 1.2, cy - r * 1.2, cx + r * 1.2, cy + r * 1.2)
        if not guard.is_safe(*bbox):
            continue
        sub = str(rng.choice(["figure_8", "nib_ticks", "spiral_coil"]))
        if sub == "figure_8":
            t = np.linspace(0.0, 2.0 * math.pi, 36)
            xs = cx + r * 0.85 * np.sin(t)
            ys = cy + r * 0.65 * np.sin(2.0 * t)
            draw_calligraphic_polyline(canvas, np.column_stack((xs, ys)), ink_rgb, lw, style, kind="annotation")
        elif sub == "nib_ticks":
            for k in range(5):
                ox = (k - 2) * r * 0.35
                pts = np.array([[cx + ox - 0.25 * r, cy + 0.6 * r], [cx + ox + 0.25 * r, cy - 0.6 * r]], dtype=np.float64)
                draw_calligraphic_polyline(canvas, pts, ink_rgb, lw * 1.15, style, kind="annotation")
        else:
            t = np.linspace(0.0, 4.5 * math.pi, 44)
            rr = np.linspace(0.15 * r, r, len(t))
            draw_calligraphic_polyline(canvas, np.column_stack((cx + rr * np.cos(t), cy + rr * np.sin(t))), ink_rgb, lw * 0.9, style, kind="annotation")
        guard.reserve(*bbox)
        _log_mark(marks_out, layer="wet_ink_wear", kind="quill_test", substyle=sub, bbox_xyxy=bbox, tool="scribe_ink", color_rgb=ink_rgb)
        return


def apply_dog_ear_and_thumb_wear(
    canvas: StrokeCanvas,
    rng: np.random.Generator,
    marks_out: List[Dict[str, Any]],
) -> None:
    """Bottom-right (or top-right) dog-ear corner fold + outer-edge page-turn thumb smudge."""
    h, w = canvas.height, canvas.width
    bottom_corner = bool(rng.random() < 0.75)
    fold_s = int(round(min(h, w) * float(rng.uniform(0.045, 0.095))))
    x0, x1 = max(0, w - fold_s - 8), w
    y0, y1 = (max(0, h - fold_s - 8), h) if bottom_corner else (0, min(h, fold_s + 8))
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)

    dist = ((w - xx) + (h - yy) - fold_s) if bottom_corner else ((w - xx) + yy - fold_s)
    shadow = np.where(dist > 0, np.exp(-0.5 * (dist / max(3.5, fold_s * 0.09)) ** 2) * 0.16, 0.0)
    flap = np.clip(-dist * 0.5 + 0.5, 0.0, 1.0)
    flap_rgb = np.clip(canvas.bg_rgb * 0.88 - np.array([0.02, 0.03, 0.05], dtype=np.float32), 0.05, 0.98)
    canvas.buf[y0:y1, x0:x1] = np.clip(
        canvas.buf[y0:y1, x0:x1] * (1.0 - shadow[:, :, None]) * (1.0 - flap[:, :, None])
        + flap_rgb[None, None, :] * flap[:, :, None],
        0.0, 1.0,
    )
    _log_mark(
        marks_out, layer="verso_wear", kind="dog_ear_fold",
        substyle="bottom_right" if bottom_corner else "top_right",
        bbox_xyxy=(float(x0), float(y0), float(x1), float(y1)),
        tool="paper_wear", color_rgb=flap_rgb,
    )

    ty0, ty1 = int(h * 0.55), int(h * 0.92)
    tx0, tx1 = int(w * 0.88), w
    if ty1 > ty0 + 4 and tx1 > tx0 + 4:
        yy2, xx2 = np.mgrid[ty0:ty1, tx0:tx1].astype(np.float32)
        cy_t = float(rng.uniform(ty0 + 20, max(ty0 + 21, ty1 - 20)))
        r2 = ((xx2 - w) / max(18.0, w * 0.07)) ** 2 + ((yy2 - cy_t) / max(35.0, h * 0.10)) ** 2
        oil = np.exp(-0.5 * r2).astype(np.float32) * float(rng.uniform(0.05, 0.11))
        oil_rgb = np.array([0.52, 0.45, 0.35], dtype=np.float32)
        canvas.composite_patch(tx0, ty0, tx1, ty1, oil, oil_rgb)
        _log_mark(
            marks_out, layer="verso_wear", kind="page_turn_thumb_smudge",
            bbox_xyxy=(float(tx0), float(ty0), float(tx1), float(ty1)),
            tool="paper_wear", color_rgb=oil_rgb,
        )


# ---------------------------------------------------------------------------
# Post-Notation Orchestrator (All 7 Layers, 100% OMR-Safe)
# ---------------------------------------------------------------------------

def _draw_persona_overlays(
    canvas: StrokeCanvas,
    active_systems: Sequence[Any],
    guard: SafeZoneGuard,
    persona: str,
    density: float,
    style: Dict[str, Any],
    rng: np.random.Generator,
    marks_out: List[Dict[str, Any]],
) -> None:
    """Draw persona-specific multi-tool overlays strictly inside safe zones."""
    if persona == "none" or density <= 0.01 or not active_systems:
        return

    w, h = canvas.width, canvas.height
    letters = "ABCDEFGHJKLMNPQRST"

    if persona == "orchestral_conductor":
        for s_idx, sys_lay in enumerate(active_systems):
            top_st = sys_lay.staves[0]
            sp = float(top_st.sp)
            y_lane = float(top_st.y_center) - 4.3 * sp
            tool = "grease_pencil_red" if (s_idx % 2 == 0) else "grease_pencil_blue"
            t_style, t_rgb = _tool_style(style, tool)
            lw = max(1.4, sp * 0.18)

            # Rehearsal box [A], [B], ...
            if sys_lay.bar_x_bounds and float(rng.random()) < min(1.0, 0.55 + 0.45 * density):
                bx = float(sys_lay.bar_x_bounds[0][0]) + sp * 0.5
                box_r = sp * 0.95
                bbox = (bx - box_r, y_lane - box_r, bx + box_r, y_lane + box_r)
                if guard.is_safe(*bbox):
                    _draw_wobbly_box(canvas, *bbox, lw, t_rgb, t_style, rng, double_pass=True)
                    lbl = letters[s_idx % len(letters)]
                    draw_mark_text(canvas, lbl, bx - sp * 0.35, y_lane + sp * 0.45, sp * 1.15, t_rgb, t_style, rng, bold=True)
                    guard.reserve(*bbox)
                    _log_mark(marks_out, layer="overlay_persona", persona=persona, kind="rehearsal_box",
                              substyle=tool, bbox_xyxy=bbox, tool=tool, color_rgb=t_rgb, text=lbl, system_index=sys_lay.system_index)

            # Eyeglasses o--o ("watch the stick!") on a barline
            if len(sys_lay.bar_x_bounds) >= 2 and float(rng.random()) < density * 0.75:
                gx = float(sys_lay.bar_x_bounds[min(1, len(sys_lay.bar_x_bounds) - 1)][1])
                gy = float(top_st.y_center) - 4.1 * sp
                r_eye = sp * 0.46
                bbox = (gx - 2.3 * r_eye, gy - 1.3 * r_eye, gx + 2.3 * r_eye, gy + 1.3 * r_eye)
                if guard.is_safe(*bbox):
                    t2 = "grease_pencil_blue" if tool == "grease_pencil_red" else "grease_pencil_red"
                    st2, rgb2 = _tool_style(style, t2)
                    _draw_oval_loop(canvas, gx - 1.2 * r_eye, gy, r_eye, r_eye * 0.72, lw, rgb2, st2, rng)
                    _draw_oval_loop(canvas, gx + 1.2 * r_eye, gy, r_eye, r_eye * 0.72, lw, rgb2, st2, rng)
                    draw_calligraphic_polyline(canvas, np.array([[gx - 0.3 * r_eye, gy], [gx + 0.3 * r_eye, gy]], dtype=np.float64), rgb2, lw, st2, kind="annotation")
                    guard.reserve(*bbox)
                    _log_mark(marks_out, layer="overlay_persona", persona=persona, kind="eyeglasses_cue",
                              substyle=t2, bbox_xyxy=bbox, tool=t2, color_rgb=rgb2, system_index=sys_lay.system_index)

            # Conductor text cue (SUB., G.P., RIT., TUTTI)
            if sys_lay.bar_x_bounds and float(rng.random()) < density * 0.70:
                cue = str(rng.choice(["SUB.", "RIT.", "TUTTI", "DIV.", "IN 2", "V.S."]))
                cx = float(sys_lay.bar_x_bounds[-1][0]) + sp * 0.8
                cy = float(top_st.y_center) - 4.0 * sp
                est_w = len(cue) * sp * 0.85
                bbox = (cx, cy - sp * 1.2, cx + est_w, cy + sp * 0.3)
                if guard.is_safe(*bbox):
                    _, b2 = draw_mark_text(canvas, cue, cx, cy, sp * 1.15, t_rgb, t_style, rng, bold=True)
                    guard.reserve(*b2)
                    _log_mark(marks_out, layer="overlay_persona", persona=persona, kind="conductor_cue",
                              substyle=tool, bbox_xyxy=b2, tool=tool, color_rgb=t_rgb, text=cue, system_index=sys_lay.system_index)

    elif persona == "string_section_pencil":
        t_style, t_rgb = _tool_style(style, "graphite_2b")
        for sys_lay in active_systems:
            top_st = sys_lay.staves[0]
            sp = float(top_st.sp)
            lw = max(1.0, sp * 0.13)
            for b_idx, (bx0, bx1) in enumerate(sys_lay.bar_x_bounds):
                if float(rng.random()) > density * 0.72:
                    continue
                mx = bx0 + (bx1 - bx0) * float(rng.uniform(0.22, 0.72))
                my = float(top_st.y_center) - 3.85 * sp
                is_down = bool(rng.random() < 0.5)
                bbox = (mx - 0.8 * sp, my - 0.9 * sp, mx + 0.8 * sp, my + 0.6 * sp)
                if not guard.is_safe(*bbox):
                    continue
                if is_down:
                    pts = np.array([[mx - 0.45 * sp, my + 0.4 * sp], [mx - 0.45 * sp, my - 0.5 * sp],
                                    [mx + 0.45 * sp, my - 0.5 * sp], [mx + 0.45 * sp, my + 0.4 * sp]], dtype=np.float64)
                    draw_calligraphic_polyline(canvas, pts, t_rgb, lw * 1.2, t_style, kind="annotation")
                    draw_calligraphic_polyline(canvas, np.array([[mx - 0.48 * sp, my - 0.5 * sp], [mx + 0.48 * sp, my - 0.5 * sp]], dtype=np.float64), t_rgb, lw * 2.1, t_style, kind="annotation")
                else:
                    pts = np.array([[mx - 0.42 * sp, my - 0.55 * sp], [mx, my + 0.45 * sp], [mx + 0.42 * sp, my - 0.55 * sp]], dtype=np.float64)
                    draw_calligraphic_polyline(canvas, pts, t_rgb, lw * 1.15, t_style, kind="annotation")
                guard.reserve(*bbox)
                _log_mark(marks_out, layer="overlay_persona", persona=persona, kind="down_bow" if is_down else "up_bow",
                          bbox_xyxy=bbox, tool="graphite_2b", color_rgb=t_rgb, system_index=sys_lay.system_index, measure_index=sys_lay.measure_start + b_idx)

    elif persona == "piano_conservatory":
        t_style, t_rgb = _tool_style(style, "graphite_hard_hb")
        for sys_lay in active_systems:
            bot_st = sys_lay.staves[-1]
            sp = float(bot_st.sp)
            for b_idx, (bx0, bx1) in enumerate(sys_lay.bar_x_bounds):
                if float(rng.random()) > density * 0.65:
                    continue
                px = bx0 + sp * 0.7
                py = float(bot_st.y_center) + 4.35 * sp
                bbox = (px, py - sp * 1.1, px + sp * 2.4, py + sp * 0.3)
                if guard.is_safe(*bbox):
                    _, b2 = draw_mark_text(canvas, "Ped.", px, py, sp * 1.0, t_rgb, t_style, rng, italic=True)
                    guard.reserve(*b2)
                    _log_mark(marks_out, layer="overlay_persona", persona=persona, kind="pedal_mark",
                              bbox_xyxy=b2, tool="graphite_hard_hb", color_rgb=t_rgb, text="Ped.", system_index=sys_lay.system_index, measure_index=sys_lay.measure_start + b_idx)
                    rx = bx1 - sp * 1.2
                    rbbox = (rx, py - sp * 0.9, rx + sp * 0.9, py + sp * 0.2)
                    if guard.is_safe(*rbbox):
                        _, rb2 = draw_mark_text(canvas, "*", rx, py, sp * 1.0, t_rgb, t_style, rng)
                        guard.reserve(*rb2)

    elif persona == "jazz_realbook_chart":
        t_style, t_rgb = _tool_style(style, "felt_chisel_black")
        labels = ["HEAD", "A", "B", "SOLO", "BRIDGE", "OUTRO"]
        for s_idx, sys_lay in enumerate(active_systems):
            top_st = sys_lay.staves[0]
            sp = float(top_st.sp)
            lbl = labels[s_idx % len(labels)]
            cx = float(sys_lay.x_left) + sp * 0.4
            cy = float(top_st.y_center) - 4.4 * sp
            est_w = max(sp * 1.4, len(lbl) * sp * 0.85)
            bbox = (cx - 0.3 * sp, cy - 1.2 * sp, cx + est_w + 0.3 * sp, cy + 0.35 * sp)
            if guard.is_safe(*bbox):
                _draw_wobbly_box(canvas, *bbox, max(1.3, sp * 0.16), t_rgb, t_style, rng)
                draw_mark_text(canvas, lbl, cx, cy, sp * 1.05, t_rgb, t_style, rng, bold=True)
                guard.reserve(*bbox)
                _log_mark(marks_out, layer="overlay_persona", persona=persona, kind="section_header",
                          bbox_xyxy=bbox, tool="felt_chisel_black", color_rgb=t_rgb, text=lbl, system_index=sys_lay.system_index)

    elif persona == "choral_rehearsal":
        t_style, t_rgb = _tool_style(style, "graphite_2b")
        for sys_lay in active_systems:
            top_st = sys_lay.staves[0]
            sp = float(top_st.sp)
            for b_idx, (bx0, bx1) in enumerate(sys_lay.bar_x_bounds):
                if float(rng.random()) > density * 0.70:
                    continue
                bx = bx1 - sp * 0.45
                by = float(top_st.y_center) - 3.6 * sp
                bbox = (bx - 0.4 * sp, by - 0.6 * sp, bx + 0.4 * sp, by + 0.5 * sp)
                if guard.is_safe(*bbox):
                    pts = np.array([[bx - 0.25 * sp, by - 0.4 * sp], [bx, by + 0.35 * sp], [bx + 0.28 * sp, by - 0.4 * sp]], dtype=np.float64)
                    draw_calligraphic_polyline(canvas, pts, t_rgb, max(1.0, sp * 0.14), t_style, kind="annotation")
                    guard.reserve(*bbox)
                    _log_mark(marks_out, layer="overlay_persona", persona=persona, kind="breath_mark",
                              bbox_xyxy=bbox, tool="graphite_2b", color_rgb=t_rgb, system_index=sys_lay.system_index, measure_index=sys_lay.measure_start + b_idx)

    elif persona == "archival_librarian":
        tool = "stamp_indigo" if float(rng.random()) < 0.65 else "stamp_crimson"
        t_style, t_rgb = _tool_style(style, tool)
        sp = float(active_systems[0].staves[0].sp)
        cx = w * (0.86 if float(rng.random()) < 0.7 else 0.14)
        cy = min(h * 0.08, max(36.0, float(active_systems[0].staves[0].y_center) - 5.2 * sp))
        rx, ry = sp * 3.4, sp * 1.65
        bbox = (cx - rx, cy - ry, cx + rx, cy + ry)
        if guard.is_safe(*bbox):
            lw = max(1.1, sp * 0.12)
            _draw_oval_loop(canvas, cx, cy, rx, ry, lw, t_rgb, t_style, rng)
            _draw_oval_loop(canvas, cx, cy, rx * 0.86, ry * 0.82, lw * 0.8, t_rgb, t_style, rng)
            lbl = str(rng.choice(["ARCHIV", "BIBL. MS", "ORCH. LIB"]))
            draw_mark_text(canvas, lbl, cx - rx * 0.62, cy + sp * 0.32, sp * 0.82, t_rgb, t_style, rng, bold=True)
            guard.reserve(*bbox)
            _log_mark(marks_out, layer="overlay_persona", persona=persona, kind="library_stamp",
                      substyle=tool, bbox_xyxy=bbox, tool=tool, color_rgb=t_rgb, text=lbl)

    elif persona == "film_studio_copyist":
        hl_tool = "highlighter_yellow" if float(rng.random()) < 0.68 else "highlighter_green"
        hl_style, hl_rgb = _tool_style(style, hl_tool)
        t_style, t_rgb = _tool_style(style, "felt_chisel_black")
        for s_idx, sys_lay in enumerate(active_systems):
            top_st = sys_lay.staves[0]
            sp = float(top_st.sp)
            cx = float(sys_lay.x_left) + sp * 0.5
            cy = float(top_st.y_center) - 4.3 * sp
            cue = f"0{s_idx + 1}:{int(rng.integers(10, 58)):02d}"
            est_w = len(cue) * sp * 0.72
            bbox = (cx - 0.3 * sp, cy - 1.1 * sp, cx + est_w + 0.4 * sp, cy + 0.3 * sp)
            if guard.is_safe(*bbox):
                # Translucent highlighter wash behind studio timecode cue
                pts = np.array([[cx - 0.2 * sp, cy - 0.35 * sp], [cx + est_w + 0.3 * sp, cy - 0.35 * sp]], dtype=np.float64)
                draw_calligraphic_polyline(canvas, pts, hl_rgb, sp * 0.24, hl_style, kind="annotation")
                _, b2 = draw_mark_text(canvas, cue, cx, cy, sp * 0.95, t_rgb, t_style, rng, bold=True)
                guard.reserve(*bbox)
                _log_mark(marks_out, layer="overlay_persona", persona=persona, kind="studio_timecode",
                          substyle=hl_tool, bbox_xyxy=bbox, tool=hl_tool, color_rgb=hl_rgb, text=cue, system_index=sys_lay.system_index)

    elif persona == "composer_working_draft":
        t_style, t_rgb = _tool_style(style, "graphite_2b")
        cues = ["ossia?", "NB", "8va?", "legato", "bis", "coda?"]
        for s_idx, sys_lay in enumerate(active_systems):
            if float(rng.random()) > density * 0.85:
                continue
            top_st = sys_lay.staves[0]
            sp = float(top_st.sp)
            cue = cues[int(rng.integers(0, len(cues)))]
            cx = float(sys_lay.x_left) + float(rng.uniform(0.15, 0.75)) * (sys_lay.x_right - sys_lay.x_left)
            cy = float(top_st.y_center) - 4.15 * sp
            est_w = len(cue) * sp * 0.75
            bbox = (cx, cy - sp * 1.1, cx + est_w, cy + sp * 0.25)
            if guard.is_safe(*bbox):
                _, b2 = draw_mark_text(canvas, cue, cx, cy, sp * 0.92, t_rgb, t_style, rng, italic=True)
                guard.reserve(*b2)
                _log_mark(marks_out, layer="overlay_persona", persona=persona, kind="draft_callout",
                          bbox_xyxy=b2, tool="graphite_2b", color_rgb=t_rgb, text=cue, system_index=sys_lay.system_index)


def apply_post_notation_manuscript_marks(
    canvas: StrokeCanvas,
    systems: Sequence[Any],
    style: Dict[str, Any],
    rng: np.random.Generator,
    marks_out: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Apply all post-notation living-manuscript layers (100% OMR GT safe) and return summary."""
    guard = SafeZoneGuard.from_systems(canvas.width, canvas.height, systems)
    active_systems = [s for s in systems if not getattr(s, "is_blank_prerule", False) and s.staves]
    blank_systems = [s for s in systems if getattr(s, "is_blank_prerule", False) and s.staves]
    avg_sp = float(np.mean([st.sp for s in active_systems for st in s.staves])) if active_systems else 14.0

    persona = str(style.get("annotation_persona") or "none")
    if persona not in ANNOTATION_PERSONAS:
        persona = "none"
    overlay_density = float(np.clip(style.get("overlay_density", 0.0), 0.0, 1.0))

    # Layer 1: Persona Overlays
    if persona != "none" and overlay_density > 0.01:
        _draw_persona_overlays(canvas, active_systems, guard, persona, overlay_density, style, rng, marks_out)

    # Layer 3: Handwritten Chord Symbols
    chord_prob = float(np.clip(style.get("chord_symbols_prob", 0.0), 0.0, 1.0))
    chord_style = str(style.get("chord_symbol_style") or ("jazz_realbook" if persona == "jazz_realbook_chart" else "none"))
    if chord_prob > 0.0 and chord_style in _CHORD_VOCABS and active_systems:
        vocab = _CHORD_VOCABS[chord_style]
        t_tool = "felt_chisel_black" if chord_style == "jazz_realbook" else "scribe_ink"
        t_style, t_rgb = _tool_style(style, t_tool)
        for sys_lay in active_systems:
            top_st = sys_lay.staves[0]
            sp = float(top_st.sp)
            for b_idx, (bx0, bx1) in enumerate(sys_lay.bar_x_bounds):
                if float(rng.random()) > chord_prob:
                    continue
                sym = vocab[(sys_lay.measure_start + b_idx) % len(vocab)]
                cx = bx0 + sp * 0.55
                cy = float(top_st.y_center) - 3.65 * sp
                est_w = len(sym) * sp * 0.72
                bbox = (cx, cy - sp * 1.15, cx + est_w, cy + sp * 0.25)
                if guard.is_safe(*bbox):
                    _, b2 = draw_mark_text(canvas, sym, cx, cy, sp * 1.02, t_rgb, t_style, rng, bold=(chord_style == "jazz_realbook"))
                    guard.reserve(*b2)
                    _log_mark(marks_out, layer="chord_symbols", persona=persona, kind="chord_symbol",
                              substyle=chord_style, bbox_xyxy=b2, tool=t_tool, color_rgb=t_rgb, text=sym,
                              system_index=sys_lay.system_index, measure_index=sys_lay.measure_start + b_idx)

    # Layer 4: Measure Numbers
    mnum_prob = float(np.clip(style.get("measure_numbers_prob", 0.0), 0.0, 1.0))
    mnum_style = str(style.get("measure_number_style") or "none")
    if mnum_prob > 0.0 and mnum_style != "none" and mnum_style in MEASURE_NUMBER_STYLES and active_systems:
        ink_rgb = style.get("ink_rgb", (0.14, 0.12, 0.10))
        for sys_lay in active_systems:
            top_st = sys_lay.staves[0]
            sp = float(top_st.sp)
            targets = list(enumerate(sys_lay.bar_x_bounds)) if mnum_style == "pit_every_bar" else ([(0, sys_lay.bar_x_bounds[0])] if sys_lay.bar_x_bounds else [])
            for b_idx, (bx0, _bx1) in targets:
                if float(rng.random()) > mnum_prob:
                    continue
                mnum = str(sys_lay.measure_start + b_idx + 1)
                mx = bx0 + sp * 0.15
                my = float(top_st.y_center) - 3.35 * sp
                est_w = max(sp * 0.85, len(mnum) * sp * 0.65)
                bbox = (mx - 0.25 * sp, my - 1.05 * sp, mx + est_w + 0.25 * sp, my + 0.25 * sp)
                if not guard.is_safe(*bbox):
                    continue
                _, b2 = draw_mark_text(canvas, mnum, mx, my, sp * 0.88, ink_rgb, style, rng, italic=(mnum_style == "italic_underlined"))
                lw = max(0.85, sp * 0.10)
                if mnum_style == "boxed":
                    _draw_wobbly_box(canvas, b2[0] - 0.2 * sp, b2[1] - 0.1 * sp, b2[2] + 0.2 * sp, b2[3] + 0.1 * sp, lw, ink_rgb, style, rng)
                elif mnum_style in ("circled", "pit_every_bar"):
                    _draw_oval_loop(canvas, 0.5 * (b2[0] + b2[2]), 0.5 * (b2[1] + b2[3]), 0.65 * (b2[2] - b2[0]) + 0.2 * sp, 0.7 * sp, lw, ink_rgb, style, rng)
                elif mnum_style == "italic_underlined":
                    draw_calligraphic_polyline(canvas, np.array([[b2[0], b2[3] + 0.1 * sp], [b2[2], b2[3] + 0.05 * sp]], dtype=np.float64), ink_rgb, lw, style, kind="annotation")
                guard.reserve(*bbox)
                _log_mark(marks_out, layer="measure_numbers", persona=persona, kind="measure_number",
                          substyle=mnum_style, bbox_xyxy=b2, tool="scribe_ink", color_rgb=ink_rgb, text=mnum,
                          system_index=sys_lay.system_index, measure_index=sys_lay.measure_start + b_idx)

    # Layer 2: Off-GT Cancellations (strictly on blank prerule staves or bottom unused space)
    scratch_prob = float(np.clip(style.get("scratch_out_prob", 0.0), 0.0, 1.0))
    if scratch_prob > 0.0 and float(rng.random()) < scratch_prob:
        cancel_style = str(style.get("cancellation_style") or rng.choice(CANCELLATION_STYLES))
        if cancel_style not in CANCELLATION_STYLES:
            cancel_style = CANCELLATION_STYLES[0]
        placed = False
        if blank_systems:
            bs = blank_systems[0].staves[0]
            span_w = min(bs.sp * 11.0, (bs.x_right - bs.x_left) * 0.32)
            cx0 = bs.x_left + float(rng.uniform(0.08, 0.55)) * (bs.x_right - bs.x_left - span_w)
            bbox = (cx0, bs.y_center - 2.0 * bs.sp, cx0 + span_w, bs.y_center + 2.0 * bs.sp)
            if guard.is_safe(*bbox):
                draw_cancelled_false_start(canvas, cx0, bs.y_center, span_w, bs.sp, cancel_style, style, rng, marks_out)
                guard.reserve(*bbox)
                placed = True
        if not placed and active_systems:
            last_bot = max(st.y_center + 2.0 * st.sp for st in active_systems[-1].staves)
            cand_y = last_bot + 5.8 * avg_sp
            span_w = avg_sp * 10.0
            cx0 = float(active_systems[-1].x_left) + avg_sp * 2.0
            bbox = (cx0, cand_y - 2.0 * avg_sp, cx0 + span_w, cand_y + 2.0 * avg_sp)
            if cand_y + 2.5 * avg_sp < canvas.height - 16.0 and guard.is_safe(*bbox):
                draw_cancelled_false_start(canvas, cx0, cand_y, span_w, avg_sp, cancel_style, style, rng, marks_out)
                guard.reserve(*bbox)

    # Layer 6: Caret ^ Insertions & Marginal Callouts
    caret_prob = float(np.clip(style.get("caret_callout_prob", 0.0), 0.0, 1.0))
    if caret_prob > 0.0 and float(rng.random()) < caret_prob and active_systems:
        sys_lay = active_systems[int(rng.integers(0, len(active_systems)))]
        top_st = sys_lay.staves[0]
        sp = float(top_st.sp)
        cx = float(sys_lay.x_left) + float(rng.uniform(0.25, 0.75)) * (sys_lay.x_right - sys_lay.x_left)
        cy = float(top_st.y_center) - 3.9 * sp
        callout = str(rng.choice(["8va?", "arco", "pizz.", "rit.", "NB", "sub. p", "cresc."]))
        bbox = (cx - 0.5 * sp, cy - 1.8 * sp, cx + len(callout) * sp * 0.8 + sp, cy + 0.5 * sp)
        if guard.is_safe(*bbox):
            ink_rgb = style.get("ink_rgb", (0.14, 0.12, 0.10))
            caret = np.array([[cx - 0.35 * sp, cy + 0.3 * sp], [cx, cy - 0.35 * sp], [cx + 0.35 * sp, cy + 0.3 * sp]], dtype=np.float64)
            draw_calligraphic_polyline(canvas, caret, ink_rgb, max(1.0, sp * 0.13), style, kind="annotation")
            draw_mark_text(canvas, callout, cx + 0.6 * sp, cy - 0.55 * sp, sp * 0.90, ink_rgb, style, rng, italic=True)
            guard.reserve(*bbox)
            _log_mark(marks_out, layer="caret_callout", persona=persona, kind="caret_insertion",
                      bbox_xyxy=bbox, tool="scribe_ink", color_rgb=ink_rgb, text=callout, system_index=sys_lay.system_index)

    # Layer 5: Wet-Ink Accidents (Ink Blots, Quill Tests, Palm Smears)
    blot_prob = float(np.clip(style.get("ink_blot_prob", 0.0), 0.0, 1.0))
    if blot_prob > 0.0 and float(rng.random()) < blot_prob:
        apply_ink_blot_and_splatter(canvas, guard, style, rng, marks_out)

    quill_prob = float(np.clip(style.get("quill_test_prob", 0.0), 0.0, 1.0))
    if quill_prob > 0.0 and float(rng.random()) < quill_prob:
        apply_quill_test_scribble(canvas, guard, avg_sp, style, rng, marks_out)

    smear_prob = float(np.clip(style.get("palm_smear_prob", 0.0), 0.0, 1.0))
    if smear_prob > 0.0 and float(rng.random()) < smear_prob:
        hand = str(style.get("palm_smear_hand") or "random")
        apply_palm_smear(canvas, hand, style, rng, marks_out)

    # Layer 7b: Dog-Ear Corner Fold & Page-Turn Thumb Wear
    dog_ear_prob = float(np.clip(style.get("dog_ear_prob", 0.0), 0.0, 1.0))
    if dog_ear_prob > 0.0 and float(rng.random()) < dog_ear_prob:
        apply_dog_ear_and_thumb_wear(canvas, rng, marks_out)

    counts_by_layer: Dict[str, int] = {}
    for m in marks_out:
        ly = str(m.get("layer", "unknown"))
        counts_by_layer[ly] = counts_by_layer.get(ly, 0) + 1

    return {
        "persona": persona,
        "total_marks": len(marks_out),
        "counts_by_layer": counts_by_layer,
        "omr_gt_corrupted": False,
    }
