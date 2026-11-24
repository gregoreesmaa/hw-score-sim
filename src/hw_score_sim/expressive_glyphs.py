"""86 handwritten secondary & expressive glyph archetypes across 13 registries.

Covers slurs/ties (10), time signatures (10, activating C / cut-time), dynamics (8),
hairpins (8), articulations & ornaments (8, incl. Keil wedges, tr~~~, baroque +,
mordents, tent/horseshoe/flat fermatas), augmentation dots (6), tuplets (6),
grace slashes (5), arpeggio squiggles (5), courtesy brackets (5), braces (5),
voltas (5), and multi-bar rests (5).

Pure NumPy + StrokeCanvas primitives with fresh per-instance motor jitter.
"""

from __future__ import annotations

from functools import lru_cache
import hashlib
import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from .stroke_engine import (
    RGBLike,
    StrokeCanvas,
    draw_calligraphic_polyline,
    draw_filled_blob,
)

# ---------------------------------------------------------------------------
# 13 Expressive Archetype Registries (86 Archetypes Total)
# ---------------------------------------------------------------------------

SLUR_ARCHETYPES: Tuple[str, ...] = (
    "classic_tapered_arc",       # Balanced cubic + subtle inner belly thickening
    "high_romantic_dome",        # Tall 1.55x Chopin/Liszt arch, gossamer tapered ends
    "flat_hurried_bridge",       # Low, nearly flat straight-topped Beethoven sketch bridge
    "asymmetric_swoop_left",     # Apex pulled left (22%), long trailing right tail
    "asymmetric_swoop_right",    # Late-rising apex (78%), brisk right drop
    "hooked_serif_ends",         # Tiny inward vertical terminal hooks at both ends
    "angular_bracket_tie",       # 20th-c. Stravinsky trapezoidal 3-segment tie/slur
    "heavy_broad_nib_crescent",  # Thick calligraphic crescent with 2.2x belly double-pass
    "dashed_editorial_slur",     # 4-7 broken dash segments (editorial/modernist phrasing)
    "overshoot_whip_slur",       # Wild romantic/jazz ends whipping past noteheads
)

TIMESIG_ARCHETYPES: Tuple[str, ...] = (
    "classic_engraved",          # Balanced numerals centered on lines 2 & 4
    "common_cut_c_symbol",       # Activates 4/4 -> 'C', 2/2 -> 'C|' alla breve
    "giant_oversized_broadcast", # Stravinsky/Hollywood conductor 1.55x giant numerals
    "baroque_quill_florid",      # Swashed 18th-c. quill numerals + ornate C / C|
    "jazz_realbook_chisel",      # Bold slanted felt-tip numerals, open 4, crossed 7
    "spindly_tall_compressed",   # Narrow 0.62x width, tall 1.25x height Romantic numerals
    "slashed_fraction_bar",      # Horizontal/diagonal fraction bar between num & den
    "staggered_diagonal_meter",  # Hurried diagonal offset (num left-up, den right-down)
    "hairline_minimal_pencil",   # Ultra-thin 0.55x graphite sketch numerals
    "heavy_box_framed_meter",    # Bold double-stroke numerals with rehearsal corner ticks
)

DYNAMIC_ARCHETYPES: Tuple[str, ...] = (
    "classic_italic_engraved",   # Balanced cursive italic, medium nib contrast
    "romantic_flourish_swash",   # Tall sweeping S-curve 'f' with upper/lower spirals
    "beethoven_slashed_bold",    # Heavy violent thick stroke, stabbing 'f' crossbar, pooling
    "jazz_block_marker",         # Upright/backslant thick felt-tip block letters
    "hairline_delicate_salon",   # Whisper-thin 0.25mm needle cursive, compact 'pp'/'mp'
    "hurried_scribble_ligature", # Continuous cursive baseline-joined ligature ('mf','sfz')
    "baroque_spelled_dot",       # 18th-c. serifed 'p.' / 'f.' with trailing period dot
    "underlined_conductor_emph", # Bold dynamic with brisk slanted underline flourish
)

HAIRPIN_ARCHETYPES: Tuple[str, ...] = (
    "classic_straight_wedge",    # Clean straight tapered two-arm wedge
    "bowed_curved_arms",         # Organic outward/inward curved Romantic arcs
    "wide_flared_dramatic",      # Wide 1.15*sp aperture, heavy stroke toward open end
    "narrow_needle_slit",        # Razor-thin 0.26*sp aperture, whisper hairline
    "open_vertex_gap",           # Hurried arms leaving 0.30*sp gap at apex
    "crossed_overshoot_x",       # Fast Beethoven/studio strokes crossing past apex ('x')
    "niente_circle_vertex",      # Small hand-drawn 'o' (al/dal niente) at closed apex
    "asymmetric_tilted_wedge",   # One horizontal arm + one steeply angled arm
)

ARTICULATION_ARCHETYPES: Tuple[str, ...] = (
    "classic_engraved",          # Round dot, crisp '>', neat tenuto, semicircular fermata, 'tr'
    "keil_baroque_wedge",        # 18th-c. Mozart/Bach vertical Keil wedge staccato, baroque '+' trill
    "romantic_expressive_trill", # Full 'tr~~~' wavy extension, wide swooping fermata, curved '>'
    "tent_angular_modern",       # Triangular tent fermata ('^'+dot), razor geometric '>'/'^'
    "horseshoe_deep_fermata",    # Deep U-shaped horseshoe fermata, heavy ink-blot staccato
    "flat_squared_fermata",      # Flat-topped squared-arch fermata, short-dash staccato
    "hurried_slash_flick",       # Diagonal pen-flick staccato, quick V-hook accent, '~' trill
    "heavy_jazz_marker",         # Oversized felt-tip rooftop '^', bold wide '>', fat tenuto, 'tr~~~'
)

DOT_ARCHETYPES: Tuple[str, ...] = (
    "classic_round_dot",         # Clean circular ink dot
    "tiny_pinprick",             # Microscopic 0.08*sp needle prick
    "heavy_ink_blot",            # Oversized 0.24*sp wet-ink pooled blob
    "diagonal_pen_flick",        # Hurried 45-deg comma/tick stroke (Bach/Beethoven MSS)
    "horizontal_micro_dash",     # Short flat broad-nib horizontal dash
    "diamond_broad_nib",         # 45-deg lozenge/diamond nib impression
)

TUPLET_ARCHETYPES: Tuple[str, ...] = (
    "classic_square_hook",       # Split straight arms + 90-deg end hooks + numeral
    "slur_arc_tuplet",           # Curved Romantic slur-arc + nested numeral
    "numeral_only_bare",         # Bare handwritten numeral (no bracket)
    "ratio_colon_bracket",       # Modernist explicit ratio ('3:2', '5:4') in split bracket
    "slanted_diagonal_hooks",    # 45-deg inward diagonal end ticks (hurried copyist)
    "continuous_over_bar",       # Unbroken straight bar with numeral perched above/below
)

GRACE_SLASH_ARCHETYPES: Tuple[str, ...] = (
    "classic_diagonal_cut",      # Clean 40-deg slash across stem/flag
    "steep_whip_slash",          # Steep 62-deg curved whip cut
    "horizontal_crossbar",       # Nearly flat crossbar cutting stem
    "curved_hook_cut",           # C-curved calligraphic swoosh through flag
    "double_thick_jazz_cut",     # Heavy marker slash with slight double-pass
)

ARPEGGIO_ARCHETYPES: Tuple[str, ...] = (
    "classic_sinusoid_wave",     # Smooth vertical sinusoidal squiggle
    "sharp_zigzag_sawtooth",     # Crisp angular Z-zigzag
    "up_arrow_arpeggio",         # Wavy roll topped with upward arrowhead
    "down_arrow_arpeggio",       # Wavy roll with downward arrowhead at bottom
    "bracket_non_arp_roll",      # Curved vertical roll-arc / non-arpeggio bracket
)

COURTESY_ARCHETYPES: Tuple[str, ...] = (
    "classic_round_parens",      # Curved '(' ')'
    "square_editorial_brackets", # Crisp '[' ']' editorial brackets
    "tall_narrow_parens",        # Spindly tall thin arcs
    "hurried_c_ticks",           # Short stubby flanking ticks
    "wide_heavy_parens",         # Bold wide-set parentheses
)

BRACE_ARCHETYPES: Tuple[str, ...] = (
    "classic_curly_brace",       # Mirrored S-curves meeting at center cusp
    "ornate_double_swash",       # Baroque/Romantic heavy center belly + curled finials
    "straight_square_bracket",   # Orchestral vertical spine + angled wing tips
    "hurried_single_c_arc",      # Fast sketch single C-arc connecting staves
    "angular_chevron_brace",     # Modernist/jazz crisp 2-segment '<' chevron
)

VOLTA_ARCHETYPES: Tuple[str, ...] = (
    "classic_open_right",        # Horizontal roof + left hook + '1.'
    "closed_box_both_hooks",     # Both left & right hooks drawn
    "angled_hook_jazz",          # 45-deg diagonal entry/exit hooks, bold numeral
    "slanted_roof_hurried",      # Rising/falling hand-drawn roof, loose hook
    "double_roof_editorial",     # Double-stroked thick top bar
)

MULTIBAR_ARCHETYPES: Tuple[str, ...] = (
    "classic_h_bar",             # Thick horizontal bar + vertical end serifs
    "slanted_diagonal_bar",      # Old-style diagonal thick bar between serifs
    "church_ladder_bars",        # Baroque/19th-c. stacked double-bar rest
    "thick_marker_block",        # Heavy felt-tip solid slab + overshoot serifs
    "thin_split_line_numeral",   # Thin line broken around centered numeral inside staff
)

EXPRESSIVE_REGISTRIES: Dict[str, Tuple[str, ...]] = {
    "slur_archetype": SLUR_ARCHETYPES,
    "timesig_archetype": TIMESIG_ARCHETYPES,
    "dynamic_archetype": DYNAMIC_ARCHETYPES,
    "hairpin_archetype": HAIRPIN_ARCHETYPES,
    "articulation_archetype": ARTICULATION_ARCHETYPES,
    "dot_archetype": DOT_ARCHETYPES,
    "tuplet_archetype": TUPLET_ARCHETYPES,
    "grace_slash_archetype": GRACE_SLASH_ARCHETYPES,
    "arpeggio_archetype": ARPEGGIO_ARCHETYPES,
    "courtesy_archetype": COURTESY_ARCHETYPES,
    "brace_archetype": BRACE_ARCHETYPES,
    "volta_archetype": VOLTA_ARCHETYPES,
    "multibar_archetype": MULTIBAR_ARCHETYPES,
}


# ---------------------------------------------------------------------------
# Deterministic Resolution (Backward-Compatible for all 234 Existing Styles)
# ---------------------------------------------------------------------------

def _stable_pick(seed_key: str, choices: Tuple[str, ...]) -> str:
    h = hashlib.sha256(seed_key.encode("utf-8")).digest()
    return choices[int.from_bytes(h[:4], "big") % len(choices)]


_EXPRESSIVE_ERA_OVERRIDES: Dict[str, Dict[str, str]] = {
    "slur_archetype": {
        "chopin": "high_romantic_dome", "liszt": "high_romantic_dome",
        "beethoven": "flat_hurried_bridge", "stravinsky": "angular_bracket_tie",
        "modern": "angular_bracket_tie", "baroque": "hooked_serif_ends",
        "bach": "heavy_broad_nib_crescent", "jazz": "overshoot_whip_slur",
        "ellington": "overshoot_whip_slur",
    },
    "timesig_archetype": {
        "baroque": "baroque_quill_florid", "bach": "common_cut_c_symbol",
        "palestrina": "baroque_quill_florid", "renaissance": "baroque_quill_florid",
        "classical": "common_cut_c_symbol", "jazz": "jazz_realbook_chisel",
        "ellington": "jazz_realbook_chisel", "realbook": "jazz_realbook_chisel",
        "film": "giant_oversized_broadcast", "williams": "giant_oversized_broadcast",
        "studio": "giant_oversized_broadcast", "chopin": "spindly_tall_compressed",
        "ravel": "hairline_minimal_pencil", "student": "slashed_fraction_bar",
        "pencil": "hairline_minimal_pencil", "mahler": "heavy_box_framed_meter",
    },
    "dynamic_archetype": {
        "romantic": "romantic_flourish_swash", "chopin": "romantic_flourish_swash",
        "beethoven": "beethoven_slashed_bold", "mahler": "underlined_conductor_emph",
        "jazz": "jazz_block_marker", "ellington": "jazz_block_marker",
        "realbook": "jazz_block_marker", "baroque": "baroque_spelled_dot",
        "ravel": "hairline_delicate_salon", "salon": "hairline_delicate_salon",
        "conductor": "underlined_conductor_emph", "vivaldi": "hurried_scribble_ligature",
    },
    "hairpin_archetype": {
        "chopin": "bowed_curved_arms", "romantic": "bowed_curved_arms",
        "beethoven": "crossed_overshoot_x", "debussy": "narrow_needle_slit",
        "ravel": "narrow_needle_slit", "modern": "niente_circle_vertex",
        "prokofiev": "asymmetric_tilted_wedge", "jazz": "wide_flared_dramatic",
        "ellington": "wide_flared_dramatic",
    },
    "articulation_archetype": {
        "baroque": "keil_baroque_wedge", "mozart": "keil_baroque_wedge",
        "bach": "keil_baroque_wedge", "vivaldi": "keil_baroque_wedge",
        "romantic": "romantic_expressive_trill", "modern": "tent_angular_modern",
        "stravinsky": "tent_angular_modern", "prokofiev": "tent_angular_modern",
        "jazz": "heavy_jazz_marker", "realbook": "heavy_jazz_marker",
        "ellington": "heavy_jazz_marker",
    },
    "dot_archetype": {
        "bach": "diagonal_pen_flick", "vivaldi": "diagonal_pen_flick",
        "beethoven": "heavy_ink_blot", "mahler": "heavy_ink_blot",
        "baroque": "diamond_broad_nib", "palestrina": "diamond_broad_nib",
        "ravel": "tiny_pinprick", "hairline": "tiny_pinprick",
        "jazz": "horizontal_micro_dash", "ellington": "horizontal_micro_dash",
    },
    "tuplet_archetype": {
        "romantic": "slur_arc_tuplet", "chopin": "slur_arc_tuplet",
        "modern": "ratio_colon_bracket", "baroque": "numeral_only_bare",
        "jazz": "slanted_diagonal_hooks", "ellington": "slanted_diagonal_hooks",
    },
    "grace_slash_archetype": {
        "romantic": "steep_whip_slash", "baroque": "curved_hook_cut",
        "jazz": "double_thick_jazz_cut", "hurried": "horizontal_crossbar",
    },
    "arpeggio_archetype": {
        "modern": "sharp_zigzag_sawtooth", "romantic": "up_arrow_arpeggio",
        "baroque": "classic_sinusoid_wave",
    },
    "courtesy_archetype": {
        "modern": "square_editorial_brackets", "baroque": "square_editorial_brackets",
        "chopin": "tall_narrow_parens", "jazz": "wide_heavy_parens",
    },
    "brace_archetype": {
        "baroque": "ornate_double_swash", "palestrina": "ornate_double_swash",
        "modern": "straight_square_bracket", "mahler": "straight_square_bracket",
        "jazz": "angular_chevron_brace", "ellington": "angular_chevron_brace",
        "sketch": "hurried_single_c_arc", "vivaldi": "hurried_single_c_arc",
    },
    "volta_archetype": {
        "jazz": "angled_hook_jazz", "realbook": "angled_hook_jazz",
        "modern": "closed_box_both_hooks", "sketch": "slanted_roof_hurried",
    },
    "multibar_archetype": {
        "baroque": "church_ladder_bars", "romantic": "slanted_diagonal_bar",
        "jazz": "thick_marker_block", "modern": "thin_split_line_numeral",
    },
}


def resolve_expressive_archetype(style: Optional[dict], key: str) -> str:
    """Resolve a single expressive archetype key without hashing all 13 registries."""
    st = style or {}
    reg = EXPRESSIVE_REGISTRIES[key]
    val = st.get(key)
    if isinstance(val, str) and val in reg:
        return val
    # Physics overrides when no explicit key was supplied
    if key == "dot_archetype" and key not in st and float(st.get("stroke_scale", 1.0)) <= 0.65:
        return "tiny_pinprick"
    if key == "hairpin_archetype" and key not in st and float(st.get("wobble_px", 0.5)) >= 1.4:
        return "open_vertex_gap"
    name = str(st.get("name", "")).strip().lower()
    era = str(st.get("era", "")).strip().lower()
    for tok, arch in _EXPRESSIVE_ERA_OVERRIDES[key].items():
        if tok in name or tok in era:
            return arch
    if not name:
        return reg[0]
    return _stable_pick(f"{name}:{key}", reg)


def resolve_expressive_archetypes(style: Optional[dict]) -> Dict[str, str]:
    """Resolve all 13 expressive archetypes deterministically from style metadata.

    Explicit valid keys in ``style`` always win. Missing keys map via era/scribe
    cues + deterministic name hash so all 234 existing styles immediately gain
    diverse, era-coherent expressive glyphs with zero breaking changes.
    """
    return {k: resolve_expressive_archetype(style, k) for k in EXPRESSIVE_REGISTRIES}


# ---------------------------------------------------------------------------
# Shared Geometry Helpers
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


def _eval_cubic(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, n: int = 20) -> np.ndarray:
    b0, b1, b2, b3 = _cubic_basis(n)
    return b0 * p0 + b1 * p1 + b2 * p2 + b3 * p3


def _eval_quad(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, n: int = 12) -> np.ndarray:
    b0, b1, b2 = _quad_basis(n)
    return b0 * p0 + b1 * p1 + b2 * p2


def _eval_catmull_rom(ctrl: np.ndarray, samples_per_seg: int = 7) -> np.ndarray:
    pts = np.asarray(ctrl, dtype=np.float64)
    if len(pts) <= 2:
        return pts
    p_ext = np.vstack((2.0 * pts[0] - pts[1], pts, 2.0 * pts[-1] - pts[-2]))
    out: List[np.ndarray] = []
    t = np.linspace(0.0, 1.0, samples_per_seg, endpoint=False)[:, None]
    t2, t3 = t * t, t * t * t
    for i in range(len(pts) - 1):
        p0, p1, p2, p3 = p_ext[i], p_ext[i + 1], p_ext[i + 2], p_ext[i + 3]
        out.append(0.5 * ((2.0 * p1) + (-p0 + p2) * t
                          + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
                          + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3))
    out.append(pts[-1:])
    return np.vstack(out)


def _xform(
    pts_norm: np.ndarray, cx: float, cy: float, sp: float,
    rng: np.random.Generator, style: dict,
    sx: float = 1.0, sy: float = 1.0, extra_slant: float = 0.0, jitter: float = 0.045,
) -> np.ndarray:
    pts = np.asarray(pts_norm, dtype=np.float64).copy()
    if jitter > 0.0:
        pts += rng.normal(0.0, jitter, size=pts.shape)
    slant = float(style.get("slant_deg", 0.0)) + extra_slant + float(rng.normal(0.0, 2.5))
    shear = math.tan(math.radians(float(np.clip(slant, -28.0, 28.0))))
    x = pts[:, 0] * sx - pts[:, 1] * shear
    y = pts[:, 1] * sy
    return np.column_stack((cx + x * sp, cy + y * sp))


def _st(style: dict, **overrides: Any) -> dict:
    s = dict(style)
    s.update(overrides)
    return s


# ---------------------------------------------------------------------------
# 1. Slurs & Ties (10 Archetypes)
# ---------------------------------------------------------------------------

def draw_slur_variant(
    canvas: StrokeCanvas,
    x0: float, y0: float, x1: float, y1: float,
    arch_dir: int,
    sp: float, color_rgb: RGBLike, style: dict, rng: np.random.Generator,
    is_tie: bool = False, archetype: Optional[str] = None,
) -> None:
    arch = archetype or resolve_expressive_archetype(style, "slur_archetype")
    dx, dy = x1 - x0, y1 - y0
    span = max(sp * 0.8, math.hypot(dx, dy))
    h_base = (min(sp * 1.15, max(sp * 0.55, span * 0.16)) if is_tie
              else min(sp * 2.4, max(sp * 0.95, span * 0.18)))
    h = h_base * float(rng.uniform(0.90, 1.12))
    base_w = max(0.9, sp * (0.14 if is_tie else 0.16))
    lst = _st(style, taper_start=0.25, taper_end=0.25)

    p0 = np.array([x0 + float(rng.normal(0.0, 0.02 * sp)), y0 + float(rng.normal(0.0, 0.02 * sp))], dtype=np.float64)
    p3 = np.array([x1 + float(rng.normal(0.0, 0.02 * sp)), y1 + float(rng.normal(0.0, 0.02 * sp))], dtype=np.float64)
    n_pts = max(14, min(48, int(span / 5.5)))

    if arch == "angular_bracket_tie":
        bh = arch_dir * h * 0.72
        pts = np.array([
            p0,
            [x0 + dx * 0.16, y0 + dy * 0.16 + bh],
            [x0 + dx * 0.84, y0 + dy * 0.84 + bh],
            p3,
        ], dtype=np.float64)
        draw_calligraphic_polyline(canvas, pts, color_rgb, base_w * 0.95, style, kind="slur")
        return

    if arch == "overshoot_whip_slur":
        p0 = p0 + np.array([-sp * 0.35, -arch_dir * sp * 0.15])
        p3 = p3 + np.array([sp * 0.45, -arch_dir * sp * 0.18])
        h *= 1.15
    elif arch == "high_romantic_dome":
        h *= 1.55
        lst = _st(lst, taper_start=0.38, taper_end=0.38)
    elif arch == "flat_hurried_bridge":
        h *= 0.48
        lst = _st(lst, taper_start=0.10, taper_end=0.10)

    t1, t2 = 0.28, 0.72
    if arch == "asymmetric_swoop_left":
        t1, t2 = 0.14, 0.48
    elif arch == "asymmetric_swoop_right":
        t1, t2 = 0.52, 0.86

    p1 = np.array([x0 + dx * t1, y0 + dy * t1 + arch_dir * h], dtype=np.float64)
    p2 = np.array([x0 + dx * t2, y0 + dy * t2 + arch_dir * h], dtype=np.float64)
    curve = _eval_cubic(p0, p1, p2, p3, n=n_pts)

    if arch == "dashed_editorial_slur":
        n_dash = max(3, min(8, int(span / (sp * 1.1))))
        seg = max(2, len(curve) // (2 * n_dash))
        for i in range(0, len(curve) - 1, seg * 2):
            chunk = curve[i : min(len(curve), i + seg + 1)]
            if len(chunk) >= 2:
                draw_calligraphic_polyline(canvas, chunk, color_rgb, base_w * 0.85, style, kind="slur")
        return

    w_mult = 1.45 if arch == "heavy_broad_nib_crescent" else (0.78 if arch == "high_romantic_dome" else 1.0)
    draw_calligraphic_polyline(canvas, curve, color_rgb, base_w * w_mult, lst, kind="slur")

    if arch in ("classic_tapered_arc", "high_romantic_dome", "heavy_broad_nib_crescent") and span > sp * 1.3:
        off = sp * (0.24 if arch == "heavy_broad_nib_crescent" else 0.12)
        inner = _eval_cubic(p0, p1 + [0.0, -arch_dir * off], p2 + [0.0, -arch_dir * off], p3, n=n_pts)
        s0, s1 = max(1, n_pts // 6), max(2, n_pts - n_pts // 6)
        draw_calligraphic_polyline(
            canvas,
            inner[s0:s1],
            color_rgb,
            base_w * (1.35 if arch == "heavy_broad_nib_crescent" else 0.85),
            lst,
            kind="slur",
        )

    if arch == "hooked_serif_ends":
        hk = sp * 0.28
        for ep, sgn in ((p0, 1.0), (p3, -1.0)):
            hook = np.array([ep, ep + np.array([sgn * hk * 0.45, -arch_dir * hk])], dtype=np.float64)
            draw_calligraphic_polyline(canvas, hook, color_rgb, base_w * 0.82, style, kind="slur")


# ---------------------------------------------------------------------------
# 2. Time Signatures (10 Archetypes — Activates C / Cut-Time!)
# ---------------------------------------------------------------------------

_DIGIT_STROKES: Dict[str, List[Tuple[bool, List[Tuple[float, float]]]]] = {
    "0": [(True, [(0.0, -0.85), (-0.48, -0.35), (-0.44, 0.45), (0.0, 0.86), (0.48, 0.35), (0.44, -0.45), (0.0, -0.85)])],
    "1": [(False, [(-0.22, -0.58), (0.05, -0.88), (0.02, 0.88)])],
    "2": [(True, [(-0.42, -0.52), (-0.06, -0.88), (0.42, -0.64), (0.34, -0.08), (-0.46, 0.84), (0.48, 0.82)])],
    "3": [(True, [(-0.38, -0.68), (0.10, -0.90), (0.42, -0.45), (-0.05, -0.02), (0.46, 0.42), (0.12, 0.90), (-0.44, 0.68)])],
    "4": [(False, [(0.22, -0.88), (-0.46, 0.22), (0.48, 0.20)]), (False, [(0.22, -0.86), (0.18, 0.88)])],
    "5": [(False, [(0.42, -0.84), (-0.34, -0.82), (-0.38, -0.10)]), (True, [(-0.38, -0.10), (0.26, -0.16), (0.46, 0.40), (0.06, 0.88), (-0.42, 0.68)])],
    "6": [(True, [(0.32, -0.86), (-0.28, -0.45), (-0.46, 0.35), (-0.02, 0.88), (0.44, 0.46), (0.16, -0.06), (-0.38, 0.24)])],
    "7": [(False, [(-0.44, -0.82), (0.44, -0.86), (-0.14, 0.88)])],
    "8": [(True, [(0.02, -0.86), (-0.40, -0.48), (0.02, -0.04), (0.44, 0.44), (0.0, 0.88), (-0.44, 0.46), (-0.02, -0.02), (0.40, -0.48), (0.02, -0.86)])],
    "9": [(True, [(0.34, -0.22), (0.02, -0.88), (-0.44, -0.46), (-0.02, 0.04), (0.38, -0.28), (0.22, 0.48), (-0.22, 0.88)])],
}


def _draw_ts_digit(
    canvas: StrokeCanvas, digit: str, cx: float, cy: float, sp: float,
    color_rgb: RGBLike, style: dict, rng: np.random.Generator,
    sx: float, sy: float, w_mult: float, extra_slant: float, arch: str,
) -> None:
    w = max(0.9, sp * float(style.get("timesig_stroke_ratio", 0.21)) * w_mult)
    if digit in ("C", "CUT"):
        c_ctrl = np.array([[0.50, -0.68], [0.02, -0.96], [-0.54, -0.42], [-0.54, 0.42], [0.04, 0.96], [0.52, 0.66]])
        pts = _xform(c_ctrl, cx, cy, sp, rng, style, sx=sx, sy=sy, extra_slant=extra_slant)
        draw_calligraphic_polyline(canvas, _eval_catmull_rom(pts, 7), color_rgb, w * 1.18, style, kind="timesig")
        if arch == "baroque_quill_florid":
            draw_filled_blob(canvas, pts[0, 0], pts[0, 1], max(1.3, w * 0.75), max(1.3, w * 0.75), 0.0, color_rgb, style)
        if digit == "CUT":
            slash = _xform(np.array([[0.05, -1.38], [-0.05, 1.38]]), cx, cy, sp, rng, style, sx=sx, sy=sy, extra_slant=extra_slant)
            draw_calligraphic_polyline(canvas, slash, color_rgb, w * 0.92, style, kind="timesig")
        return

    strokes = _DIGIT_STROKES.get(digit, [])
    if digit == "4" and arch in ("jazz_realbook_chisel", "hairline_minimal_pencil"):
        strokes = [(False, [(-0.34, -0.84), (-0.40, 0.18), (0.46, 0.16)]), (False, [(0.26, -0.86), (0.20, 0.88)])]
    for smooth, raw in strokes:
        pts = _xform(np.asarray(raw, dtype=np.float64), cx, cy, sp, rng, style, sx=sx, sy=sy, extra_slant=extra_slant)
        path = _eval_catmull_rom(pts, 7) if (smooth and len(pts) >= 3) else pts
        draw_calligraphic_polyline(canvas, path, color_rgb, w, style, kind="timesig")
        if arch == "heavy_box_framed_meter":
            draw_calligraphic_polyline(canvas, path + [0.10 * sp, 0.04 * sp], color_rgb, w * 0.75, style, kind="timesig")
    if digit == "7" and arch in ("jazz_realbook_chisel", "baroque_quill_florid"):
        bar = _xform(np.array([[-0.26, 0.02], [0.36, -0.02]]), cx, cy, sp, rng, style, sx=sx, sy=sy, extra_slant=extra_slant)
        draw_calligraphic_polyline(canvas, bar, color_rgb, w * 0.82, style, kind="timesig")


def draw_time_sig_variant(
    canvas: StrokeCanvas, num: Union[int, str], den: Union[int, str],
    cx: float, cy: float, sp: float, color_rgb: RGBLike,
    style: dict, rng: np.random.Generator, archetype: Optional[str] = None,
) -> None:
    arch = archetype or resolve_expressive_archetype(style, "timesig_archetype")
    num_s, den_s = str(num).strip().upper(), str(den).strip().upper()

    use_c = num_s in ("C", "COMMON") or (
        arch in ("common_cut_c_symbol", "baroque_quill_florid") and num_s == "4" and den_s == "4"
    )
    use_cut = num_s in ("CUT", "C|", "ALLA_BREVE") or den_s in ("CUT", "0") or (
        arch in ("common_cut_c_symbol", "baroque_quill_florid") and num_s == "2" and den_s == "2"
    )

    cfg = {
        "classic_engraved":          (1.00, 1.00, 1.00, 0.0,  1.00),
        "common_cut_c_symbol":       (1.05, 1.05, 1.05, 3.0,  1.00),
        "giant_oversized_broadcast": (1.42, 1.55, 1.45, 0.0,  1.45),
        "baroque_quill_florid":      (1.05, 1.08, 1.12, 8.0,  1.02),
        "jazz_realbook_chisel":      (1.12, 1.05, 1.48, 12.0, 1.02),
        "spindly_tall_compressed":   (0.62, 1.26, 0.72, 6.0,  1.06),
        "slashed_fraction_bar":      (0.92, 0.88, 0.95, 4.0,  1.15),
        "staggered_diagonal_meter":  (0.95, 0.95, 1.02, 10.0, 1.02),
        "hairline_minimal_pencil":   (0.90, 0.94, 0.56, 3.0,  0.98),
        "heavy_box_framed_meter":    (1.08, 1.06, 1.32, 0.0,  1.04),
    }[arch]
    sx, sy, wm, slant, y_off_sp = cfg

    if use_c or use_cut:
        _draw_ts_digit(canvas, "CUT" if use_cut else "C", cx, cy, sp * 1.15, color_rgb, style, rng, sx, sy, wm, slant, arch)
        return

    def _draw_num(s: str, x_c: float, y_c: float) -> None:
        digs = [c for c in s if c.isdigit()]
        if not digs:
            return
        step = sp * 0.88 * sx
        x0 = x_c - 0.5 * step * (len(digs) - 1)
        for i, ch in enumerate(digs):
            _draw_ts_digit(canvas, ch, x0 + i * step, y_c, sp * (1.0 if len(digs) == 1 else 0.90),
                           color_rgb, style, rng, sx, sy, wm, slant, arch)

    x_shift = sp * 0.32 if arch == "staggered_diagonal_meter" else 0.0
    _draw_num(num_s, cx - x_shift, cy - y_off_sp * sp)
    _draw_num(den_s, cx + x_shift, cy + y_off_sp * sp)

    if arch == "slashed_fraction_bar":
        bar = np.array([[cx - sp * 0.78, cy + sp * 0.08], [cx + sp * 0.78, cy - sp * 0.08]], dtype=np.float64)
        draw_calligraphic_polyline(canvas, bar, color_rgb, max(1.0, sp * 0.14), style, kind="timesig")
    elif arch == "heavy_box_framed_meter":
        bw, bh = sp * 0.95, sp * 2.15
        for sx_c, sy_c in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
            c = np.array([
                [cx + sx_c * bw, cy + sy_c * (bh - 0.45 * sp)],
                [cx + sx_c * bw, cy + sy_c * bh],
                [cx + sx_c * (bw - 0.45 * sp), cy + sy_c * bh],
            ], dtype=np.float64)
            draw_calligraphic_polyline(canvas, c, color_rgb, max(0.9, sp * 0.11), style, kind="timesig")


# ---------------------------------------------------------------------------
# 3. Dynamics (8 Archetypes)
# ---------------------------------------------------------------------------

def draw_dynamic_variant(
    canvas: StrokeCanvas, text: str, cx: float, cy: float, sp: float,
    color_rgb: RGBLike, style: dict, rng: np.random.Generator,
    archetype: Optional[str] = None,
) -> None:
    arch = archetype or resolve_expressive_archetype(style, "dynamic_archetype")
    chars = [c.lower() for c in str(text).strip() if not c.isspace()]
    if not chars:
        return

    cfg = {
        "classic_italic_engraved":   (1.00, 1.00, 1.00, 14.0, 0.88),
        "romantic_flourish_swash":   (1.12, 1.24, 1.05, 20.0, 0.96),
        "beethoven_slashed_bold":    (1.18, 1.15, 1.65, 16.0, 0.92),
        "jazz_block_marker":         (1.08, 1.04, 1.55, -2.0, 0.95),
        "hairline_delicate_salon":   (0.84, 0.86, 0.52, 18.0, 0.78),
        "hurried_scribble_ligature": (0.96, 0.94, 0.95, 22.0, 0.74),
        "baroque_spelled_dot":       (1.02, 1.02, 1.08, 10.0, 0.90),
        "underlined_conductor_emph": (1.14, 1.10, 1.38, 12.0, 0.94),
    }[arch]
    sx, sy, wm, slant, step_sp = cfg
    w = max(0.85, sp * float(style.get("dynamic_stroke_ratio", 0.19)) * wm)
    lst = _st(style, ink_pooling=0.55) if arch == "beethoven_slashed_bold" else style
    step = sp * step_sp
    x0 = cx - 0.5 * step * (len(chars) - 1)

    for i, ch in enumerate(chars):
        lx = x0 + i * step + float(rng.normal(0.0, 0.03 * sp))
        ly = cy + float(rng.normal(0.0, 0.03 * sp))

        def _s(ctrl: List[Tuple[float, float]], smooth: bool = True, wmul: float = 1.0) -> None:
            pts = _xform(np.asarray(ctrl, dtype=np.float64), lx, ly, sp, rng, lst, sx=sx, sy=sy, extra_slant=slant)
            path = _eval_catmull_rom(pts, 7) if (smooth and len(pts) >= 3) else pts
            draw_calligraphic_polyline(canvas, path, color_rgb, w * wmul, lst, kind="dynamic")

        if ch == "p":
            _s([(-0.22, -0.55), (-0.36, 1.08)], smooth=False, wmul=1.12)
            _s([(-0.22, -0.28), (0.24, -0.56), (0.48, -0.08), (0.16, 0.38), (-0.28, 0.24)])
        elif ch == "f":
            if arch == "romantic_flourish_swash":
                _s([(0.68, -0.92), (0.42, -1.32), (0.04, -1.10), (-0.06, -0.15), (-0.26, 1.12), (-0.64, 1.24), (-0.72, 0.96)], wmul=1.15)
            else:
                _s([(0.48, -1.06), (0.12, -1.16), (-0.04, -0.20), (-0.22, 0.94), (-0.54, 1.08)], wmul=1.18)
            bar_x = 0.58 if arch == "beethoven_slashed_bold" else 0.36
            _s([(-bar_x, -0.18), (bar_x, -0.26)], smooth=False, wmul=1.15 if arch == "beethoven_slashed_bold" else 0.90)
        elif ch == "m":
            _s([(-0.62, 0.34), (-0.48, -0.34), (-0.22, -0.30), (-0.20, 0.34),
                (-0.08, -0.34), (0.18, -0.30), (0.20, 0.34), (0.34, -0.34), (0.58, -0.26), (0.64, 0.36)], wmul=0.92)
        elif ch == "s":
            _s([(0.34, -0.44), (-0.14, -0.52), (-0.26, -0.10), (0.28, 0.10), (0.16, 0.48), (-0.34, 0.42)])
        elif ch == "z":
            _s([(-0.34, -0.42), (0.36, -0.44), (-0.36, 0.42), (0.38, 0.40)], smooth=False)
        elif ch == "r":
            _s([(-0.26, 0.38), (-0.20, -0.38), (0.10, -0.42), (0.34, -0.18)])
        if arch == "hurried_scribble_ligature" and i < len(chars) - 1:
            lig = np.array([[lx + step * 0.25, ly + 0.32 * sp], [lx + step * 0.78, ly + 0.18 * sp]], dtype=np.float64)
            draw_calligraphic_polyline(canvas, lig, color_rgb, w * 0.55, lst, kind="dynamic")

    if arch == "baroque_spelled_dot":
        draw_filled_blob(canvas, x0 + len(chars) * step - 0.15 * sp, cy + 0.35 * sp, max(1.2, sp * 0.14), max(1.2, sp * 0.14), 0.0, color_rgb, lst)
    elif arch == "underlined_conductor_emph":
        hw = 0.5 * len(chars) * step + 0.35 * sp
        ul = np.array([[cx - hw, cy + 0.85 * sp], [cx + hw, cy + 0.72 * sp]], dtype=np.float64)
        draw_calligraphic_polyline(canvas, ul, color_rgb, w * 0.85, lst, kind="dynamic")


# ---------------------------------------------------------------------------
# 4. Hairpins (8 Archetypes)
# ---------------------------------------------------------------------------

def draw_hairpin_variant(
    canvas: StrokeCanvas, kind: str, x0: float, x1: float, yc: float,
    sp: float, color_rgb: RGBLike, style: dict, rng: np.random.Generator,
    archetype: Optional[str] = None,
) -> None:
    arch = archetype or resolve_expressive_archetype(style, "hairpin_archetype")
    if x1 <= x0 + sp * 0.4:
        x1 = x0 + sp * 2.5
    is_cresc = str(kind).strip().lower() in ("cresc", "crescendo", "<", "open_right")
    vx, ox = (x0, x1) if is_cresc else (x1, x0)
    sgn_dir = 1.0 if is_cresc else -1.0

    half_h = sp * {
        "wide_flared_dramatic": 0.88,
        "narrow_needle_slit": 0.26,
        "bowed_curved_arms": 0.58,
    }.get(arch, 0.50) * float(rng.uniform(0.92, 1.08))
    base_w = max(0.8, sp * float(style.get("hairpin_stroke_ratio", 0.13)) * (
        1.40 if arch == "wide_flared_dramatic" else 0.65 if arch == "narrow_needle_slit" else 1.0
    ))

    gap_x = sgn_dir * sp * 0.30 if arch == "open_vertex_gap" else (-sgn_dir * sp * 0.28 if arch == "crossed_overshoot_x" else 0.0)
    for arm_sign in (-1.0, 1.0):
        vy = yc + (arm_sign * sp * 0.12 if arch == "open_vertex_gap" else -arm_sign * sp * 0.14 if arch == "crossed_overshoot_x" else 0.0)
        oy = yc + (0.0 if (arch == "asymmetric_tilted_wedge" and arm_sign > 0) else arm_sign * half_h * (1.55 if arch == "asymmetric_tilted_wedge" else 1.0))
        p0 = np.array([vx + gap_x, vy + float(rng.normal(0.0, 0.03 * sp))], dtype=np.float64)
        p2 = np.array([ox, oy + float(rng.normal(0.0, 0.03 * sp))], dtype=np.float64)
        if arch == "bowed_curved_arms":
            mid = 0.5 * (p0 + p2) + np.array([0.0, arm_sign * sp * 0.26])
            pts = _eval_quad(p0, mid, p2, n=16)
        else:
            pts = np.vstack((p0, p2))
        draw_calligraphic_polyline(canvas, pts, color_rgb, base_w, style, kind="hairpin")

    if arch == "niente_circle_vertex":
        draw_filled_blob(canvas, vx - sgn_dir * sp * 0.28, yc, max(1.4, sp * 0.24), max(1.4, sp * 0.24), 0.0, color_rgb, style, hollow=True)


# ---------------------------------------------------------------------------
# 5. Articulations & Ornaments (8 Archetypes — Keil, tr~~~, +, mordents, fermatas)
# ---------------------------------------------------------------------------

def draw_articulation_variant(
    canvas: StrokeCanvas, art: str, cx: float, cy: float, above: bool,
    sp: float, color_rgb: RGBLike, style: dict, rng: np.random.Generator,
    archetype: Optional[str] = None,
) -> None:
    arch = archetype or resolve_expressive_archetype(style, "articulation_archetype")
    s = -1.0 if above else 1.0
    w = max(0.95, sp * 0.14 * (1.40 if arch == "heavy_jazz_marker" else 1.0))
    cx = cx + float(rng.normal(0.0, 0.02 * sp))
    cy = cy + float(rng.normal(0.0, 0.02 * sp))

    if art == "staccato":
        if arch == "keil_baroque_wedge":
            pts = np.array([
                [cx, cy - s * sp * 0.45],
                [cx - sp * 0.14, cy + s * sp * 0.18],
                [cx + sp * 0.14, cy + s * sp * 0.18],
                [cx, cy - s * sp * 0.45],
            ], dtype=np.float64)
            draw_calligraphic_polyline(canvas, pts, color_rgb, w * 1.15, style, kind="articulation")
        elif arch == "hurried_slash_flick":
            draw_calligraphic_polyline(
                canvas,
                np.array([[cx - sp * 0.14, cy + sp * 0.16], [cx + sp * 0.18, cy - sp * 0.18]], dtype=np.float64),
                color_rgb, w * 1.1, style, kind="articulation",
            )
        elif arch == "flat_squared_fermata":
            draw_calligraphic_polyline(
                canvas,
                np.array([[cx - sp * 0.18, cy], [cx + sp * 0.18, cy]], dtype=np.float64),
                color_rgb, w * 1.35, style, kind="articulation",
            )
        else:
            r = max(1.4, sp * (0.24 if arch == "horseshoe_deep_fermata" else 0.17))
            draw_filled_blob(canvas, cx, cy, r, r, 0.0, color_rgb, style)
    elif art == "staccatissimo":
        hh = sp * 0.46
        pts = np.array([[cx - sp * 0.14, cy + s * hh], [cx, cy - s * hh * 0.65], [cx + sp * 0.14, cy + s * hh]], dtype=np.float64)
        draw_calligraphic_polyline(canvas, pts, color_rgb, w * 1.25, style, kind="articulation")
    elif art == "tenuto":
        hw = sp * (0.64 if arch == "heavy_jazz_marker" else 0.46 if arch == "flat_squared_fermata" else 0.54)
        draw_calligraphic_polyline(
            canvas,
            np.array([[cx - hw, cy], [cx + hw, cy + float(rng.normal(0.0, 0.04 * sp))]], dtype=np.float64),
            color_rgb, w * 1.35, style, kind="articulation",
        )
    elif art == "accent":
        hw = sp * (0.72 if arch in ("romantic_expressive_trill", "heavy_jazz_marker") else 0.56)
        hh = sp * (0.42 if arch == "heavy_jazz_marker" else 0.34)
        if arch == "romantic_expressive_trill":
            top = _eval_quad(np.array([cx - hw, cy - hh]), np.array([cx, cy - hh * 0.35]), np.array([cx + hw, cy]), 10)
            bot = _eval_quad(np.array([cx - hw, cy + hh]), np.array([cx, cy + hh * 0.35]), np.array([cx + hw, cy]), 10)
            draw_calligraphic_polyline(canvas, top, color_rgb, w * 1.15, style, kind="articulation")
            draw_calligraphic_polyline(canvas, bot, color_rgb, w * 1.15, style, kind="articulation")
        else:
            pts = np.array([[cx - hw, cy - hh], [cx + hw, cy], [cx - hw, cy + hh]], dtype=np.float64)
            draw_calligraphic_polyline(canvas, pts, color_rgb, w * 1.20, style, kind="articulation")
    elif art == "marcato":
        hw, hh = sp * (0.44 if arch == "heavy_jazz_marker" else 0.36), sp * 0.55
        pts = np.array([[cx - hw, cy - s * hh * 0.5], [cx, cy + s * hh * 0.62], [cx + hw, cy - s * hh * 0.5]], dtype=np.float64)
        draw_calligraphic_polyline(canvas, pts, color_rgb, w * 1.25, style, kind="articulation")
    elif art == "fermata":
        hw, hh = sp * 0.96, sp * 0.80
        if arch == "tent_angular_modern":
            arch_pts = np.array([[cx - hw, cy], [cx, cy + s * hh * 1.18], [cx + hw, cy]], dtype=np.float64)
        elif arch == "flat_squared_fermata":
            arch_pts = np.array([
                [cx - hw, cy],
                [cx - hw * 0.82, cy + s * hh * 0.92],
                [cx + hw * 0.82, cy + s * hh * 0.92],
                [cx + hw, cy],
            ], dtype=np.float64)
        elif arch == "horseshoe_deep_fermata":
            arch_pts = _eval_cubic(
                np.array([cx - hw * 0.78, cy - s * hh * 0.20]),
                np.array([cx - hw * 1.15, cy + s * hh * 1.45]),
                np.array([cx + hw * 1.15, cy + s * hh * 1.45]),
                np.array([cx + hw * 0.78, cy - s * hh * 0.20]),
                18,
            )
        else:
            spread = 1.22 if arch == "romantic_expressive_trill" else 1.0
            arch_pts = _eval_cubic(
                np.array([cx - hw * spread, cy]),
                np.array([cx - hw * 0.65 * spread, cy + s * hh * 1.25]),
                np.array([cx + hw * 0.65 * spread, cy + s * hh * 1.25]),
                np.array([cx + hw * spread, cy]),
                16,
            )
        draw_calligraphic_polyline(canvas, arch_pts, color_rgb, w * 1.35, _st(style, taper_start=0.22, taper_end=0.22), kind="fermata")
        r = max(1.5, sp * 0.19)
        draw_filled_blob(canvas, cx, cy + s * hh * 0.26, r, r, 0.0, color_rgb, style)
    elif art in ("trill", "baroque_plus", "mordent", "inverted_mordent", "turn"):
        if art == "baroque_plus" or (art == "trill" and arch == "keil_baroque_wedge"):
            hl = sp * 0.44
            draw_calligraphic_polyline(canvas, np.array([[cx - hl, cy], [cx + hl, cy]], dtype=np.float64), color_rgb, w * 1.1, style, kind="articulation")
            draw_calligraphic_polyline(canvas, np.array([[cx, cy - hl], [cx, cy + hl]], dtype=np.float64), color_rgb, w * 1.1, style, kind="articulation")
        elif art in ("mordent", "inverted_mordent") or (art == "trill" and arch == "hurried_slash_flick"):
            xs = np.linspace(cx - sp * 0.65, cx + sp * 0.65, 7)
            ys = cy + np.array([0.0, -0.24, 0.20, -0.24, 0.20, -0.24, 0.0]) * sp
            draw_calligraphic_polyline(canvas, np.column_stack((xs, ys)), color_rgb, w * 1.05, style, kind="articulation")
            if art == "mordent":
                draw_calligraphic_polyline(canvas, np.array([[cx, cy - sp * 0.42], [cx, cy + sp * 0.42]], dtype=np.float64), color_rgb, w * 0.9, style, kind="articulation")
        elif art == "turn":
            pts = _xform(np.array([[-0.65, 0.18], [-0.30, -0.34], [0.0, 0.0], [0.30, 0.34], [0.65, -0.18]]), cx, cy, sp, rng, style)
            draw_calligraphic_polyline(canvas, _eval_catmull_rom(pts, 7), color_rgb, w * 1.05, style, kind="articulation")
        else:
            t_v = _xform(np.array([[-0.42, -0.58], [-0.46, 0.18], [-0.22, 0.24]]), cx, cy, sp, rng, style, extra_slant=12.0)
            t_h = _xform(np.array([[-0.64, -0.28], [-0.20, -0.32]]), cx, cy, sp, rng, style, extra_slant=12.0)
            r_p = _xform(np.array([[-0.06, 0.22], [0.02, -0.28], [0.28, -0.28], [0.36, -0.12]]), cx, cy, sp, rng, style, extra_slant=12.0)
            for p in (t_v, t_h, r_p):
                draw_calligraphic_polyline(canvas, p, color_rgb, w * 1.05, style, kind="articulation")
            if arch in ("romantic_expressive_trill", "heavy_jazz_marker"):
                wx = np.linspace(cx + sp * 0.48, cx + sp * 2.15, 11)
                wy = cy - sp * 0.05 + np.where(np.arange(11) % 2 == 0, -0.16, 0.16) * sp
                draw_calligraphic_polyline(canvas, np.column_stack((wx, wy)), color_rgb, w * 0.90, style, kind="articulation")


# ---------------------------------------------------------------------------
# 6. Augmentation Dots (6 Archetypes)
# ---------------------------------------------------------------------------

def draw_augmentation_dot_variant(
    canvas: StrokeCanvas, cx: float, cy: float, sp: float,
    color_rgb: RGBLike, style: dict, rng: np.random.Generator,
    archetype: Optional[str] = None,
) -> None:
    arch = archetype or resolve_expressive_archetype(style, "dot_archetype")
    cx = cx + float(rng.normal(0.0, 0.025 * sp))
    cy = cy + float(rng.normal(0.0, 0.025 * sp))
    if arch == "tiny_pinprick":
        r = max(0.9, sp * 0.085)
        draw_filled_blob(canvas, cx, cy, r, r, 0.0, color_rgb, style)
    elif arch == "heavy_ink_blot":
        r = max(1.9, sp * 0.24)
        draw_filled_blob(canvas, cx, cy, r * 1.15, r * 0.92, float(rng.uniform(-20, 20)), color_rgb, style)
    elif arch == "diagonal_pen_flick":
        d = sp * 0.18
        pts = np.array([[cx - d, cy + d * 0.85], [cx + d, cy - d * 0.85]], dtype=np.float64)
        draw_calligraphic_polyline(canvas, pts, color_rgb, max(1.1, sp * 0.14), style, kind="dot")
    elif arch == "horizontal_micro_dash":
        d = sp * 0.22
        draw_calligraphic_polyline(canvas, np.array([[cx - d, cy], [cx + d, cy]], dtype=np.float64), color_rgb, max(1.2, sp * 0.16), style, kind="dot")
    elif arch == "diamond_broad_nib":
        r = max(1.5, sp * 0.17)
        draw_filled_blob(canvas, cx, cy, r * 1.25, r * 0.65, -42.0, color_rgb, style)
    else:
        r = max(1.4, sp * 0.145)
        draw_filled_blob(canvas, cx, cy, r, r, 0.0, color_rgb, style)


# ---------------------------------------------------------------------------
# 7. Tuplet Brackets & Numerals (6 Archetypes)
# ---------------------------------------------------------------------------

def draw_tuplet_variant(
    canvas: StrokeCanvas, x0: float, y0: float, x1: float, y1: float,
    num: int, above: bool, sp: float, color_rgb: RGBLike,
    style: dict, rng: np.random.Generator, archetype: Optional[str] = None,
    normal_num: Optional[int] = None,
) -> None:
    arch = archetype or resolve_expressive_archetype(style, "tuplet_archetype")
    w = max(0.9, sp * 0.11)
    hdir = 1.0 if above else -1.0
    hlen = sp * 0.54
    xm, ym = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    label = f"{num}:{normal_num or (2 if num % 2 == 1 else 3)}" if arch == "ratio_colon_bracket" else str(num)
    gap = sp * (0.45 * len(label) + 0.35)

    if arch == "slur_arc_tuplet":
        draw_slur_variant(canvas, x0, y0, x1, y1, -1 if above else 1, sp, color_rgb, style, rng, is_tie=True, archetype="classic_tapered_arc")
        ym = ym - hdir * sp * 0.82
    elif arch == "continuous_over_bar":
        pts = np.array([[x0, y0 + hdir * hlen], [x0, y0], [x1, y1], [x1, y1 + hdir * hlen]], dtype=np.float64)
        draw_calligraphic_polyline(canvas, pts, color_rgb, w, style, kind="tuplet")
        ym = ym - hdir * sp * 0.72
    elif arch != "numeral_only_bare" and abs(x1 - x0) > gap * 2.1:
        tl = max(0.05, (xm - gap - x0) / max(1e-3, x1 - x0))
        tr = min(0.95, (xm + gap - x0) / max(1e-3, x1 - x0))
        yl_g, yr_g = y0 + tl * (y1 - y0), y0 + tr * (y1 - y0)
        hk_dx = sp * 0.28 if arch == "slanted_diagonal_hooks" else 0.0
        draw_calligraphic_polyline(canvas, np.array([[x0 + hk_dx, y0 + hdir * hlen], [x0, y0], [xm - gap, yl_g]], dtype=np.float64), color_rgb, w, style, kind="tuplet")
        draw_calligraphic_polyline(canvas, np.array([[xm + gap, yr_g], [x1, y1], [x1 - hk_dx, y1 + hdir * hlen]], dtype=np.float64), color_rgb, w, style, kind="tuplet")

    step = sp * 0.56
    sx0 = xm - 0.5 * step * (len(label) - 1)
    for i, ch in enumerate(label):
        lx = sx0 + i * step
        if ch == ":":
            draw_filled_blob(canvas, lx, ym - sp * 0.18, max(1.0, sp * 0.09), max(1.0, sp * 0.09), 0.0, color_rgb, style)
            draw_filled_blob(canvas, lx, ym + sp * 0.18, max(1.0, sp * 0.09), max(1.0, sp * 0.09), 0.0, color_rgb, style)
        elif ch.isdigit():
            _draw_ts_digit(canvas, ch, lx, ym, sp * 0.62, color_rgb, style, rng, 0.92, 0.92, 0.90, 8.0, "classic_engraved")


# ---------------------------------------------------------------------------
# 8. Grace Slashes, Arpeggio Squiggles, Courtesy Brackets, Braces, Voltas, Multi-Bar Rests
# ---------------------------------------------------------------------------

def draw_grace_slash_variant(
    canvas: StrokeCanvas, stem_x: float, sy_mid: float, eff_sp: float,
    color_rgb: RGBLike, style: dict, rng: np.random.Generator,
    archetype: Optional[str] = None,
) -> None:
    arch = archetype or resolve_expressive_archetype(style, "grace_slash_archetype")
    w = max(0.85, eff_sp * (0.20 if arch == "double_thick_jazz_cut" else 0.14))
    j = float(rng.normal(0.0, 0.04 * eff_sp))
    if arch == "steep_whip_slash":
        pts = _eval_quad(
            np.array([stem_x - eff_sp * 0.55, sy_mid + eff_sp * 0.95 + j]),
            np.array([stem_x + eff_sp * 0.10, sy_mid]),
            np.array([stem_x + eff_sp * 0.65, sy_mid - eff_sp * 0.85 - j]),
            10,
        )
    elif arch == "horizontal_crossbar":
        pts = np.array([[stem_x - eff_sp * 0.80, sy_mid + eff_sp * 0.12 + j], [stem_x + eff_sp * 0.85, sy_mid - eff_sp * 0.10 - j]], dtype=np.float64)
    elif arch == "curved_hook_cut":
        pts = _eval_quad(
            np.array([stem_x - eff_sp * 0.72, sy_mid + eff_sp * 0.58 + j]),
            np.array([stem_x - eff_sp * 0.08, sy_mid - eff_sp * 0.35]),
            np.array([stem_x + eff_sp * 0.90, sy_mid - eff_sp * 0.45 - j]),
            12,
        )
    else:
        pts = np.array([[stem_x - eff_sp * 0.75, sy_mid + eff_sp * 0.65 + j], [stem_x + eff_sp * 0.85, sy_mid - eff_sp * 0.55 - j]], dtype=np.float64)
    draw_calligraphic_polyline(canvas, pts, color_rgb, w, style, kind="stem")
    if arch == "double_thick_jazz_cut":
        draw_calligraphic_polyline(canvas, pts + [0.0, eff_sp * 0.12], color_rgb, w * 0.75, style, kind="stem")


def draw_arpeggio_variant(
    canvas: StrokeCanvas, x: float, y_top: float, y_bot: float, sp: float,
    color_rgb: RGBLike, style: dict, rng: np.random.Generator,
    archetype: Optional[str] = None,
) -> None:
    arch = archetype or resolve_expressive_archetype(style, "arpeggio_archetype")
    h = max(sp * 1.2, abs(y_bot - y_top))
    w = max(0.95, sp * 0.14)
    if arch == "bracket_non_arp_roll":
        pts = _eval_quad(np.array([x + sp * 0.22, y_top]), np.array([x - sp * 0.35, 0.5 * (y_top + y_bot)]), np.array([x + sp * 0.22, y_bot]), 16)
        draw_calligraphic_polyline(canvas, pts, color_rgb, w * 1.1, style, kind="articulation")
        return

    n_cycles = max(3, int(round(h / (sp * 0.65))))
    ys = np.linspace(y_top, y_bot, n_cycles * 4 + 1)
    amp = sp * 0.22 * float(rng.uniform(0.9, 1.12))
    if arch == "sharp_zigzag_sawtooth":
        xs = x + amp * np.where(np.arange(len(ys)) % 2 == 0, -1.0, 1.0)
    else:
        xs = x + amp * np.sin(np.linspace(0.0, n_cycles * 2.0 * math.pi, len(ys)))
    draw_calligraphic_polyline(canvas, np.column_stack((xs, ys)), color_rgb, w, style, kind="articulation")

    if arch == "up_arrow_arpeggio":
        aw = sp * 0.28
        draw_calligraphic_polyline(canvas, np.array([[x - aw, y_top + aw * 1.2], [x, y_top - aw * 0.2], [x + aw, y_top + aw * 1.2]], dtype=np.float64), color_rgb, w, style, kind="articulation")
    elif arch == "down_arrow_arpeggio":
        aw = sp * 0.28
        draw_calligraphic_polyline(canvas, np.array([[x - aw, y_bot - aw * 1.2], [x, y_bot + aw * 0.2], [x + aw, y_bot - aw * 1.2]], dtype=np.float64), color_rgb, w, style, kind="articulation")


def draw_courtesy_brackets_variant(
    canvas: StrokeCanvas, cx: float, cy: float, sp: float, is_double_flat: bool,
    color_rgb: RGBLike, style: dict, rng: np.random.Generator,
    archetype: Optional[str] = None,
) -> None:
    arch = archetype or resolve_expressive_archetype(style, "courtesy_archetype")
    px_off = sp * (0.88 if is_double_flat else 0.64) * (1.15 if arch == "wide_heavy_parens" else 1.0)
    ph = sp * (1.05 if arch == "tall_narrow_parens" else 0.56 if arch == "hurried_c_ticks" else 0.82)
    pw = sp * (0.16 if arch == "tall_narrow_parens" else 0.26)
    w = max(0.8, sp * float(style.get("acc_stroke_ratio", 0.145)) * (1.18 if arch == "wide_heavy_parens" else 0.78))

    for side in (-1.0, 1.0):
        bx = cx + side * px_off + float(rng.normal(0.0, 0.02 * sp))
        if arch == "square_editorial_brackets":
            pts = np.array([[bx - side * pw, cy - ph], [bx, cy - ph], [bx, cy + ph], [bx - side * pw, cy + ph]], dtype=np.float64)
        else:
            pts = _eval_quad(np.array([bx - side * pw, cy - ph]), np.array([bx + side * pw * 0.65, cy]), np.array([bx - side * pw, cy + ph]), 10)
        draw_calligraphic_polyline(canvas, pts, color_rgb, w, style, kind="accidental")


def draw_brace_variant(
    canvas: StrokeCanvas, x: float, y_top: float, y_bot: float, sp: float,
    color_rgb: RGBLike, style: dict, rng: np.random.Generator,
    archetype: Optional[str] = None,
) -> None:
    arch = archetype
    if arch is None:
        # Backward compatibility with legacy brace_kind == "bracket"
        legacy = str(style.get("brace_kind", "")).strip().lower()
        if legacy in ("bracket", "square_bracket_serif", "heavy_angled_c_bracket"):
            arch = "straight_square_bracket"
        elif legacy in ("hurried_single_swoop",):
            arch = "hurried_single_c_arc"
        elif legacy in ("baroque_double_quill_accolade",):
            arch = "ornate_double_swash"
        else:
            arch = resolve_expressive_archetype(style, "brace_archetype")

    h, ym = max(sp, abs(y_bot - y_top)), 0.5 * (y_top + y_bot)
    w_scale = float(np.clip(style.get("brace_width_scale", 1.0), 0.55, 1.90))
    w = max(1.1, sp * float(style.get("brace_stroke_ratio", 0.22))) * (0.75 + 0.25 * w_scale)

    if arch == "straight_square_bracket":
        bx, wing = x - sp * 0.25, sp * 0.82 * w_scale
        draw_calligraphic_polyline(canvas, np.array([[bx, y_top - 0.1 * sp], [bx, y_bot + 0.1 * sp]], dtype=np.float64), color_rgb, w * 1.4, style, kind="brace")
        draw_calligraphic_polyline(canvas, np.array([[bx, y_top - 0.1 * sp], [bx + wing, y_top - 0.42 * sp]], dtype=np.float64), color_rgb, w, style, kind="brace")
        draw_calligraphic_polyline(canvas, np.array([[bx, y_bot + 0.1 * sp], [bx + wing, y_bot + 0.42 * sp]], dtype=np.float64), color_rgb, w, style, kind="brace")
        return
    if arch == "hurried_single_c_arc":
        pts = _eval_quad(np.array([x, y_top]), np.array([x - sp * 1.15 * w_scale, ym]), np.array([x, y_bot]), 20)
        draw_calligraphic_polyline(canvas, pts, color_rgb, w, style, kind="brace")
        return
    if arch == "angular_chevron_brace":
        pts = np.array([[x, y_top], [x - sp * 1.05 * w_scale, ym], [x, y_bot]], dtype=np.float64)
        draw_calligraphic_polyline(canvas, pts, color_rgb, w * 1.15, style, kind="brace")
        return
    bow = sp * (1.25 if arch == "ornate_double_swash" else 0.98) * w_scale
    tip = np.array([x - bow * 1.15 + float(rng.normal(0.0, 0.05 * sp)), ym + float(rng.normal(0.0, 0.05 * sp))], dtype=np.float64)
    up = _eval_cubic(np.array([x, y_top]), np.array([x - bow, y_top + 0.14 * h]), np.array([x + bow * 0.2, ym - 0.16 * h]), tip, 18)
    lo = _eval_cubic(tip, np.array([x + bow * 0.2, ym + 0.16 * h]), np.array([x - bow, y_bot - 0.14 * h]), np.array([x, y_bot]), 18)
    draw_calligraphic_polyline(canvas, up, color_rgb, w * 1.15, style, kind="brace")
    draw_calligraphic_polyline(canvas, lo, color_rgb, w * 1.15, style, kind="brace")
    if arch == "ornate_double_swash":
        draw_calligraphic_polyline(canvas, up[4:-4] - [sp * 0.14, 0.0], color_rgb, w * 0.9, style, kind="brace")
        draw_calligraphic_polyline(canvas, lo[4:-4] - [sp * 0.14, 0.0], color_rgb, w * 0.9, style, kind="brace")


def draw_volta_variant(
    canvas: StrokeCanvas, x0: float, x1: float, y_top: float, label: str,
    draw_left_hook: bool, draw_right_hook: bool, sp: float,
    color_rgb: RGBLike, style: dict, rng: np.random.Generator,
    text_fn: Callable[..., Any], archetype: Optional[str] = None,
) -> None:
    arch = archetype or resolve_expressive_archetype(style, "volta_archetype")
    w = max(1.0, sp * (0.16 if arch == "angled_hook_jazz" else 0.12))
    hh = sp * 1.15
    y_r = y_top + (sp * float(rng.uniform(-0.22, 0.22)) if arch == "slanted_roof_hurried" else 0.0)
    draw_calligraphic_polyline(canvas, np.array([[x0, y_top], [x1, y_r]], dtype=np.float64), color_rgb, w, style, kind="volta")
    if arch == "double_roof_editorial":
        draw_calligraphic_polyline(canvas, np.array([[x0, y_top - sp * 0.14], [x1, y_r - sp * 0.14]], dtype=np.float64), color_rgb, w * 0.75, style, kind="volta")
    dx_hk = sp * 0.42 if arch == "angled_hook_jazz" else 0.0
    if draw_left_hook:
        draw_calligraphic_polyline(canvas, np.array([[x0, y_top], [x0 - dx_hk, y_top + hh]], dtype=np.float64), color_rgb, w, style, kind="volta")
        text_fn(label, x0 + sp * 0.45, y_top + sp * 0.85, sp * 0.82)
    if draw_right_hook or arch == "closed_box_both_hooks":
        draw_calligraphic_polyline(canvas, np.array([[x1, y_r], [x1 + dx_hk, y_r + hh]], dtype=np.float64), color_rgb, w, style, kind="volta")


def draw_multibar_rest_variant(
    canvas: StrokeCanvas, x0: float, x1: float, yc: float, count: int, sp: float,
    color_rgb: RGBLike, style: dict, rng: np.random.Generator,
    archetype: Optional[str] = None,
) -> None:
    arch = archetype or resolve_expressive_archetype(style, "multibar_archetype")
    pad = max(sp * 0.9, (x1 - x0) * 0.18)
    rx0, rx1 = x0 + pad, max(x0 + pad + sp * 1.5, x1 - pad)
    xm = 0.5 * (rx0 + rx1)
    sw = max(1.0, sp * 0.14)

    if arch == "thin_split_line_numeral":
        gap = sp * 0.75
        draw_calligraphic_polyline(canvas, np.array([[rx0, yc], [xm - gap, yc]], dtype=np.float64), color_rgb, sw * 1.2, style, kind="beam")
        draw_calligraphic_polyline(canvas, np.array([[xm + gap, yc], [rx1, yc]], dtype=np.float64), color_rgb, sw * 1.2, style, kind="beam")
        ym = yc
    else:
        if arch == "slanted_diagonal_bar":
            draw_calligraphic_polyline(canvas, np.array([[rx0, yc + sp * 0.55], [rx1, yc - sp * 0.55]], dtype=np.float64), color_rgb, max(2.2, sp * 0.52), style, kind="beam")
        elif arch == "church_ladder_bars":
            draw_calligraphic_polyline(canvas, np.array([[rx0, yc - sp * 0.45], [rx1, yc - sp * 0.45]], dtype=np.float64), color_rgb, max(2.0, sp * 0.36), style, kind="beam")
            draw_calligraphic_polyline(canvas, np.array([[rx0, yc + sp * 0.45], [rx1, yc + sp * 0.45]], dtype=np.float64), color_rgb, max(2.0, sp * 0.36), style, kind="beam")
        else:
            bw = max(2.4, sp * (0.72 if arch == "thick_marker_block" else 0.52))
            draw_calligraphic_polyline(canvas, np.array([[rx0, yc], [rx1, yc]], dtype=np.float64), color_rgb, bw, style, kind="beam")
        sh = sp * (1.25 if arch == "thick_marker_block" else 0.95)
        for sx in (rx0, rx1):
            draw_calligraphic_polyline(canvas, np.array([[sx, yc - sh], [sx, yc + sh]], dtype=np.float64), color_rgb, sw, style, kind="barline")
        ym = yc - sp * 2.85

    cs = str(count)
    step = sp * 0.72
    sx0 = xm - 0.5 * step * (len(cs) - 1)
    for i, ch in enumerate(cs):
        if ch.isdigit():
            _draw_ts_digit(canvas, ch, sx0 + i * step, ym, sp * 0.82, color_rgb, style, rng, 1.0, 1.0, 1.05, 4.0, "classic_engraved")
