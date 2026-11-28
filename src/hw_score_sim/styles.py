"""20-style handwritten music scribe profile registry.

Each style profile defines ink/nib physics, glyph quirks, spacing/layout
irregularities, and manuscript paper artifacts consumed by the stroke engine,
glyph variant generator, and page renderer.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

from .rest_variants import (
    ALL_NEW_PAUSE_ARCHETYPES,
    BAR_REST_ARCHETYPES,
    FLAGGED_REST_ARCHETYPES,
    NEW_QUARTER_REST_ARCHETYPES,
    draw_pause_variant,
)
from .structural_glyphs import (
    BARLINE_ARCHETYPES,
    CLEF_ARCHETYPES,
    KEYSIG_ARCHETYPES,
    resolve_structural_archetypes,
)

STYLE_PACKS_DIR: Path = Path(__file__).resolve().parent / "style_packs"
_REGISTRY_VERSION: int = 0


def _copy_style_dict(spec: dict[str, Any]) -> dict[str, Any]:
    """Fast shallow+nested copy for style dicts (avoids expensive copy.deepcopy)."""
    out = dict(spec)
    for k, v in out.items():
        if isinstance(v, dict):
            out[k] = dict(v)
        elif isinstance(v, list):
            out[k] = list(v)
        elif isinstance(v, set):
            out[k] = set(v)
    return out

REST_ARCHETYPES: tuple[str, ...] = (
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

NOTEHEAD_ARCHETYPES: tuple[str, ...] = (
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

PAPER_TYPES: tuple[str, ...] = (
    "aged_rag",
    "bright_white",
    "cream_manuscript",
    "spiral_notebook",
    "ozalid_blue",
    "soviet_green_tint",
)

STEM_ARCHETYPES: tuple[str, ...] = (
    "straight_classic",
    "hairline_needle",
    "heavy_thick",
    "tapered_quill",
    "bowed_right",
    "bowed_left",
    "s_curve_flex",
    "short_stubby",
    "tall_spindly",
    "steep_forward_slant",
    "back_slant_lefty",
    "detached_gap",
    "center_piercing",
    "overshoot_notehead",
    "tremor_shaky",
    "double_stroke_sketch",
    "flared_base",
    "whip_taper",
)

FLAG_ARCHETYPES: tuple[str, ...] = (
    "classic_droop",
    "baroque_curl",
    "sharp_angular_7",
    "hurried_slash",
    "tiny_flick",
    "long_dramatic_s",
    "closed_loop_flourish",
    "broad_nib_pennant",
    "detached_tick",
    "wide_horizontal_wing",
    "steep_vertical_drop",
    "zigzag_frantic",
    "hooked_j_tail",
    "calligraphic_ribbon",
    "stubby_wedge",
    "double_back_loop",
    "feathered_sweep",
    "flat_bar_flag",
    "reverse_kick",
    "minimal_dot_tail",
)

BEAM_ARCHETYPES: tuple[str, ...] = (
    "straight_classic",
    "thick_marker",
    "hairline_thin",
    "arched_convex",
    "sagged_concave",
    "tapered_wedge",
    "overhanging_ends",
    "short_clipped",
    "scribble_double_pass",
    "calligraphic_ribbon",
    "stepped_wobbly",
    "gapped_broken",
    "wide_spaced_bars",
    "tight_crammed_bars",
)

LEDGER_STYLES: tuple[str, ...] = (
    "crisp",
    "wide_sweep",
    "slanted",
    "short_tick",
    "wavy",
)

BEAM_SLOPE_STYLES: tuple[str, ...] = (
    "horizontal_flat",
    "engraver_gentle",
    "steep_contour",
    "exaggerated_swoop",
    "reverse_quirk",
    "concave_valley",
    "convex_arch",
    "stepped_secondary_tilt",
    "high_floating_beam",
    "tight_hug_beam",
)

BEAM_GROUPING_MODES: tuple[str, ...] = (
    "standard_beat",
    "all_flags_vocal",
    "mixed_split_flags",
    "long_phrase_beam",
    "pair_beams_2",
    "uneven_3_plus_1",
    "feathered_splay",
    "french_broken_secondary",
)

REQUIRED_STYLE_KEYS: tuple[str, ...] = (
    # Metadata
    "name",
    "description",
    "era",
    # Ink & nib
    "ink_rgb",
    "staff_ink_rgb",
    "nib_angle_deg",
    "nib_min_ratio",
    "stroke_scale",
    "wobble_px",
    "tremor_px",
    "ink_dropout",
    "grain_strength",
    "ink_var",
    "ink_pooling",
    # Glyph quirks
    "slant_deg",
    "notehead_archetype",
    "notehead_scale",
    "notehead_angle_deg",
    "hollow_open_prob",
    "rest_archetype_weights",
    "clef_ornate",
    "barline_overshoot_sp",
    # Spacing & layout quirks
    "spacing_jitter",
    "bar_width_var",
    "end_bar_cram",
    "staff_curve_px",
    "staff_slope_deg",
    "sp_jitter_frac",
    "blank_prerule_prob",
    # Paper & artifacts
    "paper_rgb",
    "paper_grain",
    "paper_mottle",
    "bleed_through",
    "eraser_smudge_prob",
    "margin_annotations_prob",
)


def _weights(
    lightning_z: float,
    backwards_7_hook: float,
    cursive_3: float,
    hurried_slash_zig: float,
    baroque_c_hook: float,
    calligraphic_s_curl: float,
    sharp_v_tick: float = 0.0,
    baroque_flag_loop: float = 0.0,
    modern_tall_z: float = 0.0,
    scribble_s: float = 0.0,
) -> dict[str, float]:
    raw = {
        "lightning_z": float(lightning_z),
        "backwards_7_hook": float(backwards_7_hook),
        "cursive_3": float(cursive_3),
        "hurried_slash_zig": float(hurried_slash_zig),
        "baroque_c_hook": float(baroque_c_hook),
        "calligraphic_s_curl": float(calligraphic_s_curl),
        "sharp_v_tick": float(sharp_v_tick),
        "baroque_flag_loop": float(baroque_flag_loop),
        "modern_tall_z": float(modern_tall_z),
        "scribble_s": float(scribble_s),
    }
    total = sum(raw.values())
    if total <= 0.0:
        raise ValueError("rest_archetype_weights must sum to a positive value")
    return {k: round(v / total, 6) for k, v in raw.items()}


SCRIBE_STYLES: dict[str, dict[str, Any]] = {
    "baroque_quill_neat": {
        "name": "baroque_quill_neat",
        "description": "Careful 18th-century Baroque fair copy written with a broad goose quill in dark iron-gall ink.",
        "era": "baroque",
        "ink_rgb": (0.19, 0.14, 0.10),
        "staff_ink_rgb": (0.28, 0.22, 0.17),
        "nib_angle_deg": 38.0,
        "nib_min_ratio": 0.18,
        "stroke_scale": 1.05,
        "wobble_px": 0.65,
        "tremor_px": 0.12,
        "ink_dropout": 0.06,
        "grain_strength": 0.10,
        "ink_var": 0.14,
        "ink_pooling": 0.24,
        "slant_deg": 6.5,
        "notehead_archetype": "teardrop",
        "notehead_scale": 1.02,
        "notehead_angle_deg": -32.0,
        "hollow_open_prob": 0.12,
        "rest_archetype_weights": _weights(0.02, 0.38, 0.08, 0.02, 0.18, 0.10, 0.0, 0.22, 0.0, 0.0),
        "ink_dip_period": 24,
        "rastrum_wobble": 1.2,
        "clef_ornate": 0.90,
        "barline_overshoot_sp": 0.28,
        "spacing_jitter": 0.11,
        "bar_width_var": 0.13,
        "end_bar_cram": 0.08,
        "staff_curve_px": 3.2,
        "staff_slope_deg": 0.45,
        "sp_jitter_frac": 0.025,
        "blank_prerule_prob": 0.15,
        "paper_rgb": (0.92, 0.88, 0.78),
        "paper_grain": 0.085,
        "paper_mottle": 0.14,
        "bleed_through": 0.15,
        "eraser_smudge_prob": 0.03,
        "margin_annotations_prob": 0.22,
    },
    "beethoven_hurried_autograph": {
        "name": "beethoven_hurried_autograph",
        "description": "Turbulent, heavily-inked autograph draft with aggressive slashes, cramped endings, and dense corrections.",
        "era": "classical_romantic",
        "ink_rgb": (0.14, 0.11, 0.09),
        "staff_ink_rgb": (0.30, 0.25, 0.20),
        "nib_angle_deg": 44.0,
        "nib_min_ratio": 0.22,
        "stroke_scale": 1.38,
        "wobble_px": 2.40,
        "tremor_px": 0.55,
        "ink_dropout": 0.15,
        "grain_strength": 0.18,
        "ink_var": 0.28,
        "ink_pooling": 0.42,
        "slant_deg": 13.5,
        "notehead_archetype": "slash",
        "notehead_scale": 1.12,
        "notehead_angle_deg": -40.0,
        "hollow_open_prob": 0.48,
        "rest_archetype_weights": _weights(0.14, 0.12, 0.08, 0.56, 0.06, 0.04),
        "stem_archetype": "overshoot_notehead",
        "clef_archetype": "beethoven_wild_slash",
        "beam_archetype": "scribble_double_pass",
        "ink_dip_period": 20,
        "clef_ornate": 0.35,
        "barline_overshoot_sp": 0.68,
        "spacing_jitter": 0.42,
        "bar_width_var": 0.33,
        "end_bar_cram": 0.38,
        "staff_curve_px": 11.5,
        "staff_slope_deg": 1.85,
        "sp_jitter_frac": 0.065,
        "blank_prerule_prob": 0.12,
        "paper_rgb": (0.89, 0.84, 0.74),
        "paper_grain": 0.11,
        "paper_mottle": 0.20,
        "bleed_through": 0.22,
        "eraser_smudge_prob": 0.18,
        "margin_annotations_prob": 0.72,
    },
    "mozart_flowing_cursive": {
        "name": "mozart_flowing_cursive",
        "description": "Fluid, graceful classical hand with strong forward cursive slant and light, confident quill motion.",
        "era": "classical",
        "ink_rgb": (0.21, 0.16, 0.12),
        "staff_ink_rgb": (0.32, 0.26, 0.21),
        "nib_angle_deg": 34.0,
        "nib_min_ratio": 0.24,
        "stroke_scale": 0.94,
        "wobble_px": 0.95,
        "tremor_px": 0.10,
        "ink_dropout": 0.07,
        "grain_strength": 0.09,
        "ink_var": 0.15,
        "ink_pooling": 0.18,
        "slant_deg": 17.5,
        "notehead_archetype": "teardrop",
        "notehead_scale": 0.96,
        "notehead_angle_deg": -28.0,
        "hollow_open_prob": 0.20,
        "rest_archetype_weights": _weights(0.03, 0.48, 0.12, 0.03, 0.06, 0.14, 0.0, 0.14, 0.0, 0.0),
        "stem_length_scale": 0.82,
        "stem_archetype": "tapered_quill",
        "ink_dip_period": 24,
        "clef_ornate": 0.72,
        "barline_overshoot_sp": 0.24,
        "spacing_jitter": 0.17,
        "bar_width_var": 0.16,
        "end_bar_cram": 0.14,
        "staff_curve_px": 4.8,
        "staff_slope_deg": 0.75,
        "sp_jitter_frac": 0.030,
        "blank_prerule_prob": 0.20,
        "paper_rgb": (0.93, 0.90, 0.82),
        "paper_grain": 0.075,
        "paper_mottle": 0.11,
        "bleed_through": 0.13,
        "eraser_smudge_prob": 0.04,
        "margin_annotations_prob": 0.28,
    },
    "schubert_dense_lieder": {
        "name": "schubert_dense_lieder",
        "description": "Compact early-Romantic song manuscript with tightly packed phrases and brisk steel-nib strokes.",
        "era": "early_romantic",
        "ink_rgb": (0.18, 0.15, 0.13),
        "staff_ink_rgb": (0.31, 0.27, 0.23),
        "nib_angle_deg": 40.0,
        "nib_min_ratio": 0.28,
        "stroke_scale": 0.90,
        "wobble_px": 1.15,
        "tremor_px": 0.18,
        "ink_dropout": 0.09,
        "grain_strength": 0.11,
        "ink_var": 0.18,
        "ink_pooling": 0.20,
        "slant_deg": 9.0,
        "notehead_archetype": "compact",
        "notehead_scale": 0.85,
        "notehead_angle_deg": -30.0,
        "hollow_open_prob": 0.25,
        "rest_archetype_weights": _weights(0.10, 0.34, 0.30, 0.12, 0.08, 0.06),
        "clef_ornate": 0.55,
        "barline_overshoot_sp": 0.32,
        "spacing_jitter": 0.28,
        "bar_width_var": 0.24,
        "end_bar_cram": 0.32,
        "staff_curve_px": 5.5,
        "staff_slope_deg": 0.90,
        "sp_jitter_frac": 0.040,
        "blank_prerule_prob": 0.10,
        "paper_rgb": (0.91, 0.87, 0.79),
        "paper_grain": 0.08,
        "paper_mottle": 0.15,
        "bleed_through": 0.16,
        "eraser_smudge_prob": 0.08,
        "margin_annotations_prob": 0.40,
    },
    "chopin_delicate_salon": {
        "name": "chopin_delicate_salon",
        "description": "Fine-nibbed Parisian salon manuscript with delicate hairlines, poised oval heads, and ornate clefs.",
        "era": "romantic",
        "ink_rgb": (0.20, 0.17, 0.16),
        "staff_ink_rgb": (0.36, 0.32, 0.29),
        "nib_angle_deg": 32.0,
        "nib_min_ratio": 0.30,
        "stroke_scale": 0.72,
        "wobble_px": 0.60,
        "tremor_px": 0.08,
        "ink_dropout": 0.08,
        "grain_strength": 0.07,
        "ink_var": 0.12,
        "ink_pooling": 0.10,
        "slant_deg": 11.5,
        "notehead_archetype": "oval",
        "notehead_scale": 0.90,
        "notehead_angle_deg": -24.0,
        "hollow_open_prob": 0.15,
        "rest_archetype_weights": _weights(0.05, 0.10, 0.25, 0.04, 0.08, 0.48),
        "stem_archetype": "tall_spindly",
        "stem_length_scale": 1.34,
        "beam_grouping_mode": "french_broken_secondary",
        "clef_ornate": 0.85,
        "barline_overshoot_sp": 0.16,
        "spacing_jitter": 0.15,
        "bar_width_var": 0.15,
        "end_bar_cram": 0.10,
        "staff_curve_px": 2.8,
        "staff_slope_deg": 0.40,
        "sp_jitter_frac": 0.022,
        "blank_prerule_prob": 0.22,
        "paper_rgb": (0.95, 0.93, 0.88),
        "paper_grain": 0.055,
        "paper_mottle": 0.07,
        "bleed_through": 0.08,
        "eraser_smudge_prob": 0.06,
        "margin_annotations_prob": 0.35,
    },
    "debussy_pencil_sketch": {
        "name": "debussy_pencil_sketch",
        "description": "Impressionist soft-graphite composer sketch with textured strokes, subtle erasures, and airy phrasing.",
        "era": "late_romantic_impressionist",
        "ink_rgb": (0.31, 0.32, 0.34),
        "staff_ink_rgb": (0.42, 0.41, 0.39),
        "nib_angle_deg": 28.0,
        "nib_min_ratio": 0.72,
        "stroke_scale": 0.98,
        "wobble_px": 1.35,
        "tremor_px": 0.22,
        "ink_dropout": 0.22,
        "grain_strength": 0.38,
        "ink_var": 0.26,
        "ink_pooling": 0.04,
        "slant_deg": 8.0,
        "notehead_archetype": "oval",
        "notehead_scale": 1.00,
        "notehead_angle_deg": -22.0,
        "hollow_open_prob": 0.35,
        "rest_archetype_weights": _weights(0.18, 0.14, 0.32, 0.16, 0.05, 0.15),
        "beam_grouping_mode": "french_broken_secondary",
        "clef_ornate": 0.48,
        "barline_overshoot_sp": 0.30,
        "spacing_jitter": 0.26,
        "bar_width_var": 0.25,
        "end_bar_cram": 0.16,
        "staff_curve_px": 4.2,
        "staff_slope_deg": 0.65,
        "sp_jitter_frac": 0.032,
        "blank_prerule_prob": 0.38,
        "paper_rgb": (0.94, 0.92, 0.87),
        "paper_grain": 0.11,
        "paper_mottle": 0.09,
        "bleed_through": 0.03,
        "eraser_smudge_prob": 0.50,
        "margin_annotations_prob": 0.48,
    },
    "bach_thick_iron_gall": {
        "name": "bach_thick_iron_gall",
        "description": "Heavy, oxidized iron-gall Baroque manuscript with broad calligraphic nib contrast and strong bleed-through.",
        "era": "baroque",
        "ink_rgb": (0.15, 0.10, 0.07),
        "staff_ink_rgb": (0.25, 0.19, 0.14),
        "nib_angle_deg": 45.0,
        "nib_min_ratio": 0.16,
        "stroke_scale": 1.52,
        "wobble_px": 1.05,
        "tremor_px": 0.20,
        "ink_dropout": 0.05,
        "grain_strength": 0.14,
        "ink_var": 0.22,
        "ink_pooling": 0.44,
        "slant_deg": 5.0,
        "notehead_archetype": "heavy_wedge",
        "notehead_scale": 1.15,
        "notehead_angle_deg": -36.0,
        "hollow_open_prob": 0.10,
        "rest_archetype_weights": _weights(0.02, 0.36, 0.05, 0.03, 0.18, 0.06, 0.01, 0.27, 0.01, 0.01),
        "ink_dip_period": 24,
        "rastrum_wobble": 2.1,
        "keysig_archetype": "baroque_vertical_flat",
        "clef_ornate": 0.88,
        "barline_overshoot_sp": 0.35,
        "spacing_jitter": 0.16,
        "bar_width_var": 0.18,
        "end_bar_cram": 0.20,
        "staff_curve_px": 5.2,
        "staff_slope_deg": 0.70,
        "sp_jitter_frac": 0.038,
        "blank_prerule_prob": 0.08,
        "paper_rgb": (0.87, 0.81, 0.69),
        "paper_grain": 0.12,
        "paper_mottle": 0.22,
        "bleed_through": 0.24,
        "eraser_smudge_prob": 0.02,
        "margin_annotations_prob": 0.30,
    },
    "romantic_copyist_oblong": {
        "name": "romantic_copyist_oblong",
        "description": "Disciplined 19th-century professional publisher's copyist hand on pre-ruled oblong manuscript paper.",
        "era": "romantic",
        "ink_rgb": (0.13, 0.12, 0.11),
        "staff_ink_rgb": (0.26, 0.24, 0.22),
        "nib_angle_deg": 36.0,
        "nib_min_ratio": 0.24,
        "stroke_scale": 1.00,
        "wobble_px": 0.42,
        "tremor_px": 0.05,
        "ink_dropout": 0.04,
        "grain_strength": 0.06,
        "ink_var": 0.09,
        "ink_pooling": 0.14,
        "slant_deg": 4.0,
        "notehead_archetype": "oval",
        "notehead_scale": 1.00,
        "notehead_angle_deg": -26.0,
        "hollow_open_prob": 0.06,
        "rest_archetype_weights": _weights(0.32, 0.14, 0.28, 0.04, 0.06, 0.16),
        "clef_ornate": 0.78,
        "barline_overshoot_sp": 0.10,
        "spacing_jitter": 0.09,
        "bar_width_var": 0.11,
        "end_bar_cram": 0.05,
        "staff_curve_px": 1.4,
        "staff_slope_deg": 0.25,
        "sp_jitter_frac": 0.015,
        "blank_prerule_prob": 0.42,
        "paper_rgb": (0.93, 0.90, 0.83),
        "paper_grain": 0.06,
        "paper_mottle": 0.08,
        "bleed_through": 0.09,
        "eraser_smudge_prob": 0.02,
        "margin_annotations_prob": 0.12,
    },
    "modern_jazz_lead_sheet": {
        "name": "modern_jazz_lead_sheet",
        "description": "Bold 20th-century Real-Book-inspired jazz chart written with a chisel felt-tip pen and angled slash heads.",
        "era": "modern",
        "ink_rgb": (0.07, 0.07, 0.08),
        "staff_ink_rgb": (0.22, 0.22, 0.24),
        "nib_angle_deg": 43.0,
        "nib_min_ratio": 0.42,
        "stroke_scale": 1.24,
        "wobble_px": 0.85,
        "tremor_px": 0.08,
        "ink_dropout": 0.05,
        "grain_strength": 0.08,
        "ink_var": 0.10,
        "ink_pooling": 0.16,
        "slant_deg": 7.0,
        "notehead_archetype": "slash",
        "notehead_scale": 1.08,
        "notehead_angle_deg": -38.0,
        "hollow_open_prob": 0.22,
        "rest_archetype_weights": _weights(0.45, 0.12, 0.08, 0.30, 0.02, 0.03),
        "beam_slope_style": "horizontal_flat",
        "beam_thickness_scale": 1.32,
        "barline_archetype": "heavy_thick_marker",
        "clef_ornate": 0.15,
        "barline_overshoot_sp": 0.22,
        "spacing_jitter": 0.20,
        "bar_width_var": 0.18,
        "end_bar_cram": 0.14,
        "staff_curve_px": 1.8,
        "staff_slope_deg": 0.30,
        "sp_jitter_frac": 0.018,
        "blank_prerule_prob": 0.36,
        "paper_rgb": (0.96, 0.95, 0.91),
        "paper_grain": 0.04,
        "paper_mottle": 0.03,
        "bleed_through": 0.04,
        "eraser_smudge_prob": 0.05,
        "margin_annotations_prob": 0.58,
    },
    "student_ballpoint_blue": {
        "name": "student_ballpoint_blue",
        "description": "Everyday conservatory homework written in blue ballpoint pen with uniform tip width and uneven measure planning.",
        "era": "contemporary",
        "ink_rgb": (0.10, 0.21, 0.56),
        "staff_ink_rgb": (0.34, 0.36, 0.40),
        "nib_angle_deg": 25.0,
        "nib_min_ratio": 0.92,
        "stroke_scale": 0.82,
        "wobble_px": 1.45,
        "tremor_px": 0.20,
        "ink_dropout": 0.14,
        "grain_strength": 0.12,
        "ink_var": 0.16,
        "ink_pooling": 0.12,
        "slant_deg": 5.5,
        "notehead_archetype": "compact",
        "notehead_scale": 0.92,
        "notehead_angle_deg": -18.0,
        "hollow_open_prob": 0.32,
        "rest_archetype_weights": _weights(0.36, 0.30, 0.18, 0.10, 0.03, 0.03),
        "clef_ornate": 0.22,
        "barline_overshoot_sp": 0.42,
        "spacing_jitter": 0.34,
        "bar_width_var": 0.28,
        "end_bar_cram": 0.30,
        "staff_curve_px": 3.5,
        "staff_slope_deg": 0.60,
        "sp_jitter_frac": 0.025,
        "blank_prerule_prob": 0.45,
        "paper_rgb": (0.96, 0.97, 0.98),
        "paper_type": "spiral_notebook",
        "paper_grain": 0.035,
        "paper_mottle": 0.02,
        "bleed_through": 0.06,
        "eraser_smudge_prob": 0.10,
        "margin_annotations_prob": 0.32,
    },
    "student_pencil_eraser": {
        "name": "student_pencil_eraser",
        "description": "Theory-class #2 graphite worksheet with heavy eraser smudges, tentative strokes, and crammed bar endings.",
        "era": "contemporary",
        "ink_rgb": (0.36, 0.36, 0.38),
        "staff_ink_rgb": (0.40, 0.40, 0.42),
        "nib_angle_deg": 30.0,
        "nib_min_ratio": 0.84,
        "stroke_scale": 0.92,
        "wobble_px": 1.65,
        "tremor_px": 0.26,
        "ink_dropout": 0.18,
        "grain_strength": 0.34,
        "ink_var": 0.24,
        "ink_pooling": 0.03,
        "slant_deg": 2.5,
        "notehead_archetype": "compact",
        "notehead_scale": 0.94,
        "notehead_angle_deg": -16.0,
        "hollow_open_prob": 0.44,
        "rest_archetype_weights": _weights(0.30, 0.28, 0.22, 0.14, 0.03, 0.03),
        "clef_ornate": 0.18,
        "barline_overshoot_sp": 0.46,
        "spacing_jitter": 0.36,
        "bar_width_var": 0.30,
        "end_bar_cram": 0.34,
        "staff_curve_px": 3.0,
        "staff_slope_deg": 0.55,
        "sp_jitter_frac": 0.028,
        "blank_prerule_prob": 0.50,
        "paper_rgb": (0.95, 0.94, 0.88),
        "paper_grain": 0.07,
        "paper_mottle": 0.05,
        "bleed_through": 0.02,
        "eraser_smudge_prob": 0.65,
        "margin_annotations_prob": 0.42,
    },
    "fountain_pen_calligraphy": {
        "name": "fountain_pen_calligraphy",
        "description": "High-contrast italic fountain pen in midnight blue-black ink with flourishes and crisp calligraphic curls.",
        "era": "modern_classic",
        "ink_rgb": (0.08, 0.11, 0.23),
        "staff_ink_rgb": (0.20, 0.22, 0.30),
        "nib_angle_deg": 42.0,
        "nib_min_ratio": 0.15,
        "stroke_scale": 1.12,
        "wobble_px": 0.48,
        "tremor_px": 0.06,
        "ink_dropout": 0.04,
        "grain_strength": 0.05,
        "ink_var": 0.11,
        "ink_pooling": 0.26,
        "slant_deg": 14.0,
        "notehead_archetype": "teardrop",
        "notehead_scale": 1.04,
        "notehead_angle_deg": -34.0,
        "hollow_open_prob": 0.08,
        "rest_archetype_weights": _weights(0.04, 0.06, 0.16, 0.02, 0.12, 0.60),
        "clef_ornate": 0.96,
        "barline_overshoot_sp": 0.18,
        "spacing_jitter": 0.10,
        "bar_width_var": 0.12,
        "end_bar_cram": 0.06,
        "staff_curve_px": 2.0,
        "staff_slope_deg": 0.30,
        "sp_jitter_frac": 0.018,
        "blank_prerule_prob": 0.25,
        "paper_rgb": (0.96, 0.94, 0.90),
        "paper_grain": 0.05,
        "paper_mottle": 0.04,
        "bleed_through": 0.07,
        "eraser_smudge_prob": 0.01,
        "margin_annotations_prob": 0.18,
    },
    "faded_sepia_archival": {
        "name": "faded_sepia_archival",
        "description": "Sun-faded 18th/19th-century archival folio with washed-out sepia ink, heavy foxing, and ghost bleed-through.",
        "era": "archival",
        "ink_rgb": (0.46, 0.35, 0.24),
        "staff_ink_rgb": (0.56, 0.46, 0.35),
        "nib_angle_deg": 37.0,
        "nib_min_ratio": 0.26,
        "stroke_scale": 0.96,
        "wobble_px": 1.20,
        "tremor_px": 0.24,
        "ink_dropout": 0.32,
        "grain_strength": 0.22,
        "ink_var": 0.32,
        "ink_pooling": 0.20,
        "slant_deg": 8.5,
        "notehead_archetype": "oval",
        "notehead_scale": 0.98,
        "notehead_angle_deg": -28.0,
        "hollow_open_prob": 0.22,
        "rest_archetype_weights": _weights(0.06, 0.24, 0.18, 0.06, 0.34, 0.12),
        "clef_ornate": 0.68,
        "barline_overshoot_sp": 0.30,
        "spacing_jitter": 0.22,
        "bar_width_var": 0.20,
        "end_bar_cram": 0.18,
        "staff_curve_px": 6.8,
        "staff_slope_deg": 1.05,
        "sp_jitter_frac": 0.045,
        "blank_prerule_prob": 0.18,
        "paper_rgb": (0.85, 0.78, 0.65),
        "paper_grain": 0.13,
        "paper_mottle": 0.24,
        "bleed_through": 0.23,
        "eraser_smudge_prob": 0.05,
        "margin_annotations_prob": 0.26,
    },
    "cramped_pocket_notebook": {
        "name": "cramped_pocket_notebook",
        "description": "Tightly squeezed field-notebook jottings in fine gel ink with severe end-of-bar cramming and small heads.",
        "era": "contemporary",
        "ink_rgb": (0.14, 0.15, 0.17),
        "staff_ink_rgb": (0.38, 0.38, 0.40),
        "nib_angle_deg": 22.0,
        "nib_min_ratio": 0.88,
        "stroke_scale": 0.74,
        "wobble_px": 1.30,
        "tremor_px": 0.16,
        "ink_dropout": 0.10,
        "grain_strength": 0.10,
        "ink_var": 0.14,
        "ink_pooling": 0.08,
        "slant_deg": 10.5,
        "notehead_archetype": "compact",
        "notehead_scale": 0.80,
        "notehead_angle_deg": -20.0,
        "hollow_open_prob": 0.36,
        "rest_archetype_weights": _weights(0.24, 0.26, 0.16, 0.28, 0.03, 0.03),
        "clef_ornate": 0.12,
        "barline_overshoot_sp": 0.50,
        "spacing_jitter": 0.38,
        "bar_width_var": 0.34,
        "end_bar_cram": 0.40,
        "staff_curve_px": 6.0,
        "staff_slope_deg": 1.20,
        "sp_jitter_frac": 0.048,
        "blank_prerule_prob": 0.55,
        "paper_rgb": (0.92, 0.89, 0.81),
        "paper_grain": 0.065,
        "paper_mottle": 0.08,
        "bleed_through": 0.11,
        "eraser_smudge_prob": 0.14,
        "margin_annotations_prob": 0.52,
    },
    "wide_conductor_score": {
        "name": "wide_conductor_score",
        "description": "Spacious orchestral conductor's score with large legible noteheads, wide bar spacing, and many blank staves.",
        "era": "modern",
        "ink_rgb": (0.09, 0.09, 0.10),
        "staff_ink_rgb": (0.25, 0.25, 0.26),
        "nib_angle_deg": 35.0,
        "nib_min_ratio": 0.36,
        "stroke_scale": 1.20,
        "wobble_px": 0.65,
        "tremor_px": 0.08,
        "ink_dropout": 0.04,
        "grain_strength": 0.07,
        "ink_var": 0.10,
        "ink_pooling": 0.16,
        "slant_deg": 3.0,
        "notehead_archetype": "oval",
        "notehead_scale": 1.22,
        "notehead_angle_deg": -25.0,
        "hollow_open_prob": 0.10,
        "rest_archetype_weights": _weights(0.42, 0.18, 0.22, 0.06, 0.04, 0.08),
        "clef_ornate": 0.52,
        "barline_overshoot_sp": 0.18,
        "spacing_jitter": 0.12,
        "bar_width_var": 0.12,
        "end_bar_cram": 0.04,
        "staff_curve_px": 2.2,
        "staff_slope_deg": 0.35,
        "sp_jitter_frac": 0.016,
        "blank_prerule_prob": 0.58,
        "paper_rgb": (0.94, 0.92, 0.86),
        "paper_grain": 0.055,
        "paper_mottle": 0.06,
        "bleed_through": 0.05,
        "eraser_smudge_prob": 0.08,
        "margin_annotations_prob": 0.56,
    },
    "left_handed_backslant": {
        "name": "left_handed_backslant",
        "description": "Distinctive left-handed scribe with pronounced backward slant, reversed nib angle, and dark teal-black ink.",
        "era": "contemporary",
        "ink_rgb": (0.10, 0.19, 0.22),
        "staff_ink_rgb": (0.30, 0.32, 0.33),
        "nib_angle_deg": -34.0,
        "nib_min_ratio": 0.76,
        "stroke_scale": 0.96,
        "wobble_px": 1.10,
        "tremor_px": 0.15,
        "ink_dropout": 0.09,
        "grain_strength": 0.11,
        "ink_var": 0.15,
        "ink_pooling": 0.14,
        "slant_deg": -13.5,
        "notehead_archetype": "hollow_diamond",
        "notehead_scale": 1.00,
        "notehead_angle_deg": -12.0,
        "hollow_open_prob": 0.26,
        "rest_archetype_weights": _weights(0.22, 0.30, 0.12, 0.10, 0.03, 0.03, 0.10, 0.02, 0.04, 0.04),
        "clef_ornate": 0.34,
        "barline_overshoot_sp": 0.28,
        "spacing_jitter": 0.24,
        "bar_width_var": 0.22,
        "end_bar_cram": 0.22,
        "staff_curve_px": 4.5,
        "staff_slope_deg": -0.95,
        "sp_jitter_frac": 0.032,
        "blank_prerule_prob": 0.26,
        "paper_rgb": (0.95, 0.96, 0.94),
        "paper_grain": 0.05,
        "paper_mottle": 0.04,
        "bleed_through": 0.06,
        "eraser_smudge_prob": 0.16,
        "margin_annotations_prob": 0.28,
    },
    "tremor_elderly_hand": {
        "name": "tremor_elderly_hand",
        "description": "Late-life composer manuscript marked by high-frequency hand tremor, wavering staff lines, and deliberate strokes.",
        "era": "late_romantic",
        "ink_rgb": (0.23, 0.19, 0.16),
        "staff_ink_rgb": (0.35, 0.31, 0.27),
        "nib_angle_deg": 33.0,
        "nib_min_ratio": 0.45,
        "stroke_scale": 1.04,
        "wobble_px": 2.35,
        "tremor_px": 2.60,
        "ink_dropout": 0.16,
        "grain_strength": 0.18,
        "ink_var": 0.24,
        "ink_pooling": 0.22,
        "slant_deg": 4.5,
        "notehead_archetype": "oval",
        "notehead_scale": 1.06,
        "notehead_angle_deg": -24.0,
        "hollow_open_prob": 0.38,
        "rest_archetype_weights": _weights(0.12, 0.30, 0.28, 0.08, 0.14, 0.08),
        "clef_ornate": 0.46,
        "barline_overshoot_sp": 0.52,
        "spacing_jitter": 0.31,
        "bar_width_var": 0.26,
        "end_bar_cram": 0.20,
        "staff_curve_px": 9.5,
        "staff_slope_deg": 1.40,
        "sp_jitter_frac": 0.075,
        "blank_prerule_prob": 0.16,
        "paper_rgb": (0.91, 0.87, 0.79),
        "paper_grain": 0.095,
        "paper_mottle": 0.16,
        "bleed_through": 0.14,
        "eraser_smudge_prob": 0.12,
        "margin_annotations_prob": 0.34,
    },
    "minimalist_quick_draft": {
        "name": "minimalist_quick_draft",
        "description": "Spare contemporary fineliner draft with bare-bones clefs, open slash noteheads, and rapid zigzag rests.",
        "era": "contemporary",
        "ink_rgb": (0.17, 0.17, 0.19),
        "staff_ink_rgb": (0.34, 0.34, 0.36),
        "nib_angle_deg": 20.0,
        "nib_min_ratio": 0.94,
        "stroke_scale": 0.76,
        "wobble_px": 0.90,
        "tremor_px": 0.09,
        "ink_dropout": 0.08,
        "grain_strength": 0.06,
        "ink_var": 0.10,
        "ink_pooling": 0.05,
        "slant_deg": 12.0,
        "notehead_archetype": "slash",
        "notehead_scale": 0.90,
        "notehead_angle_deg": -42.0,
        "hollow_open_prob": 0.56,
        "rest_archetype_weights": _weights(0.26, 0.10, 0.06, 0.54, 0.02, 0.02),
        "clef_ornate": 0.05,
        "barline_overshoot_sp": 0.36,
        "spacing_jitter": 0.30,
        "bar_width_var": 0.25,
        "end_bar_cram": 0.24,
        "staff_curve_px": 2.6,
        "staff_slope_deg": 0.45,
        "sp_jitter_frac": 0.022,
        "blank_prerule_prob": 0.48,
        "paper_rgb": (0.98, 0.98, 0.97),
        "paper_grain": 0.025,
        "paper_mottle": 0.02,
        "bleed_through": 0.03,
        "eraser_smudge_prob": 0.04,
        "margin_annotations_prob": 0.20,
    },
    "vintage_purple_hectograph": {
        "name": "vintage_purple_hectograph",
        "description": "Mid-century spirit-duplicator choral/band part in characteristic aniline violet ink on yellowed mimeo stock.",
        "era": "mid_century",
        "ink_rgb": (0.38, 0.17, 0.56),
        "staff_ink_rgb": (0.49, 0.31, 0.64),
        "nib_angle_deg": 30.0,
        "nib_min_ratio": 0.62,
        "stroke_scale": 1.06,
        "wobble_px": 1.00,
        "tremor_px": 0.14,
        "ink_dropout": 0.20,
        "grain_strength": 0.24,
        "ink_var": 0.26,
        "ink_pooling": 0.18,
        "slant_deg": 6.0,
        "notehead_archetype": "oval",
        "notehead_scale": 1.02,
        "notehead_angle_deg": -24.0,
        "hollow_open_prob": 0.18,
        "rest_archetype_weights": _weights(0.34, 0.16, 0.32, 0.08, 0.04, 0.06),
        "clef_ornate": 0.58,
        "barline_overshoot_sp": 0.20,
        "spacing_jitter": 0.18,
        "bar_width_var": 0.16,
        "end_bar_cram": 0.14,
        "staff_curve_px": 3.4,
        "staff_slope_deg": 0.50,
        "sp_jitter_frac": 0.026,
        "blank_prerule_prob": 0.24,
        "paper_rgb": (0.94, 0.91, 0.83),
        "paper_grain": 0.09,
        "paper_mottle": 0.13,
        "bleed_through": 0.12,
        "eraser_smudge_prob": 0.06,
        "margin_annotations_prob": 0.30,
    },
    "choral_director_red_pencil": {
        "name": "choral_director_red_pencil",
        "description": "Expressive red-pencil rehearsal realization over dark pre-ruled staves with bold slashes and margin cues.",
        "era": "contemporary",
        "ink_rgb": (0.72, 0.16, 0.14),
        "staff_ink_rgb": (0.28, 0.28, 0.30),
        "nib_angle_deg": 36.0,
        "nib_min_ratio": 0.68,
        "stroke_scale": 1.30,
        "wobble_px": 1.55,
        "tremor_px": 0.18,
        "ink_dropout": 0.16,
        "grain_strength": 0.42,
        "ink_var": 0.22,
        "ink_pooling": 0.06,
        "slant_deg": -4.0,
        "notehead_archetype": "slash",
        "notehead_scale": 1.16,
        "notehead_angle_deg": -32.0,
        "hollow_open_prob": 0.30,
        "rest_archetype_weights": _weights(0.40, 0.14, 0.10, 0.30, 0.03, 0.03),
        "clef_ornate": 0.26,
        "barline_overshoot_sp": 0.48,
        "spacing_jitter": 0.29,
        "bar_width_var": 0.22,
        "end_bar_cram": 0.26,
        "staff_curve_px": 4.0,
        "staff_slope_deg": -0.60,
        "sp_jitter_frac": 0.030,
        "blank_prerule_prob": 0.32,
        "paper_rgb": (0.95, 0.93, 0.88),
        "paper_grain": 0.075,
        "paper_mottle": 0.07,
        "bleed_through": 0.04,
        "eraser_smudge_prob": 0.36,
        "margin_annotations_prob": 0.75,
    },
}


# Snapshot of built-in styles for clean reloads.
_BUILTIN_STYLES: dict[str, dict[str, Any]] = copy.deepcopy(SCRIBE_STYLES)

# Numeric parameter bounds used for validation and safe random scribe sampling.
_NUMERIC_CLAMP_BOUNDS: dict[str, tuple[float, float]] = {
    "nib_angle_deg": (-65.0, 65.0),
    "nib_min_ratio": (0.15, 1.00),
    "stroke_scale": (0.45, 2.20),
    "wobble_px": (0.0, 4.0),
    "tremor_px": (0.0, 3.5),
    "ink_dropout": (0.0, 0.55),
    "grain_strength": (0.0, 0.75),
    "ink_var": (0.0, 0.55),
    "ink_pooling": (0.0, 0.75),
    "slant_deg": (-14.0, 18.0),
    "notehead_scale": (0.50, 1.45),
    "notehead_angle_deg": (-50.0, 10.0),
    "hollow_open_prob": (0.0, 1.0),
    "clef_ornate": (0.0, 1.0),
    "barline_overshoot_sp": (0.0, 1.0),
    "spacing_jitter": (0.08, 0.45),
    "bar_width_var": (0.10, 0.35),
    "end_bar_cram": (0.0, 0.70),
    "staff_curve_px": (0.0, 16.0),
    "staff_slope_deg": (-2.5, 2.5),
    "sp_jitter_frac": (0.0, 0.10),
    "blank_prerule_prob": (0.0, 0.60),
    "paper_grain": (0.0, 0.28),
    "paper_mottle": (0.0, 0.38),
    "bleed_through": (0.0, 0.25),
    "eraser_smudge_prob": (0.0, 1.0),
    "margin_annotations_prob": (0.0, 1.0),
    # Optional extended physics & paper parameters
    "ink_feather": (0.0, 1.0),
    "split_nib_prob": (0.0, 1.0),
    "rastrum_wobble": (0.0, 6.0),
    "stain_prob": (0.0, 1.0),
    "ink_dip_period": (0.0, 100.0),
    "notehead_aspect": (0.25, 2.20),
    "notehead_fill_density": (0.25, 1.50),
    "stem_length_scale": (0.65, 1.55),
    "beam_thickness_scale": (0.45, 2.10),
    "beam_split_to_flag_prob": (0.0, 1.0),
    # v0.3.0 Layer 1: Macro layout & staff ruling
    "first_system_indent_frac": (0.0, 0.28),
    "margin_left_frac": (0.025, 0.20),
    "margin_right_frac": (0.025, 0.22),
    "margin_top_frac": (0.03, 0.28),
    "margin_bottom_frac": (0.03, 0.22),
    "ragged_right_jitter_frac": (0.0, 0.22),
    "final_system_short_frac": (0.32, 1.0),
    "page_skew_drift_px": (-60.0, 60.0),
    "sp_scale_mult": (0.58, 1.55),
    "intra_gap_mult": (0.52, 1.85),
    "inter_system_gap_mult": (0.42, 2.20),
    "system_gap_jitter_frac": (0.0, 0.35),
    "sp_drift_per_system": (-0.08, 0.08),
    "staff_lr_fade": (-0.35, 0.55),
    "staff_end_stagger_sp": (0.0, 1.4),
    "staff_dry_skip_prob": (0.0, 0.65),
    "staff_line_width_ratio": (0.045, 0.165),
    "staff_mid_lift_prob": (0.0, 0.50),
    "brace_width_scale": (0.55, 1.90),
    # v0.3.0 Layer 3: Living manuscript overlays & wear
    "overlay_density": (0.0, 1.0),
    "scratch_out_prob": (0.0, 1.0),
    "chord_symbols_prob": (0.0, 1.0),
    "measure_numbers_prob": (0.0, 1.0),
    "ink_blot_prob": (0.0, 1.0),
    "palm_smear_prob": (0.0, 1.0),
    "quill_test_prob": (0.0, 1.0),
    "caret_callout_prob": (0.0, 1.0),
    "verso_bleed_strength": (0.0, 1.0),
    "dog_ear_prob": (0.0, 1.0),
    "fold_crease_prob": (0.0, 1.0),
    "foxing_cluster_prob": (0.0, 1.0),
    # v0.3.0 Layer 4: Spacing & micro-motor ductus
    "duration_spacing_gamma": (0.12, 1.35),
    "bar_end_panic_start": (0.45, 0.90),
    "bar_end_panic_strength": (0.0, 0.85),
    "cross_voice_x_slop_sp": (0.0, 0.42),
    "ligature_prob": (0.0, 0.85),
    "ligature_opacity": (0.12, 0.68),
    "duration_head_scale_slope": (-0.22, 0.25),
    "downbeat_weight_boost": (0.0, 0.42),
    "phrase_arc_mod": (0.0, 0.40),
    "stem_len_jitter_frac": (0.0, 0.28),
    "stem_center_mix_prob": (0.0, 0.45),
    "stem_wrong_side_prob": (0.0, 0.16),
    "beam_junction_overshoot_sp": (0.0, 0.42),
    "beam_junction_gap_prob": (0.0, 0.45),
    "beam_edge_overhang_sp": (-0.12, 0.55),
    "pitch_y_slop_sp": (0.0, 0.145),
    "chord_x_stagger_sp": (0.0, 0.28),
    "accidental_x_offset_sp": (0.92, 1.85),
    "accidental_x_jitter_sp": (0.0, 0.34),
    # v0.8.0 Chisel-nib physics, crisp iron-gall contrast & non-parallel rastrum
    "nib_anisotropy": (0.0, 1.0),
    "edge_crispness": (0.0, 1.0),
    "ink_mass": (0.0, 1.0),
    "rastrum_curve_var": (0.0, 0.35),
}

_OPTIONAL_NUMERIC_DEFAULTS: dict[str, float] = {
    "ink_feather": 0.0,
    "split_nib_prob": 0.0,
    "rastrum_wobble": 0.0,
    "stain_prob": 0.0,
    "ink_dip_period": 0.0,
    "notehead_aspect": 1.0,
    "notehead_fill_density": 1.0,
    "stem_length_scale": 1.0,
    "beam_thickness_scale": 1.0,
    "beam_split_to_flag_prob": 0.0,
    # v0.3.0 Layer 1 neutral defaults
    "first_system_indent_frac": 0.0,
    "margin_left_frac": 0.065,
    "margin_right_frac": 0.065,
    "margin_top_frac": 0.075,
    "margin_bottom_frac": 0.065,
    "ragged_right_jitter_frac": 0.0,
    "final_system_short_frac": 1.0,
    "page_skew_drift_px": 0.0,
    "sp_scale_mult": 1.0,
    "intra_gap_mult": 1.0,
    "inter_system_gap_mult": 1.0,
    "system_gap_jitter_frac": 0.0,
    "sp_drift_per_system": 0.0,
    "staff_lr_fade": 0.0,
    "staff_end_stagger_sp": 0.0,
    "staff_dry_skip_prob": 0.0,
    "staff_line_width_ratio": 0.085,
    "staff_mid_lift_prob": 0.0,
    "brace_width_scale": 1.0,
    # v0.3.0 Layer 3 neutral defaults
    "overlay_density": 0.0,
    "scratch_out_prob": 0.0,
    "chord_symbols_prob": 0.0,
    "measure_numbers_prob": 0.0,
    "ink_blot_prob": 0.0,
    "palm_smear_prob": 0.0,
    "quill_test_prob": 0.0,
    "caret_callout_prob": 0.0,
    "verso_bleed_strength": 0.0,
    "dog_ear_prob": 0.0,
    "fold_crease_prob": 0.0,
    "foxing_cluster_prob": 0.0,
    # v0.3.0 Layer 4 neutral defaults
    "duration_spacing_gamma": 1.0,
    "bar_end_panic_start": 0.68,
    "bar_end_panic_strength": 0.0,
    "cross_voice_x_slop_sp": 0.0,
    "ligature_prob": 0.0,
    "ligature_opacity": 0.34,
    "duration_head_scale_slope": 0.0,
    "downbeat_weight_boost": 0.0,
    "phrase_arc_mod": 0.0,
    "stem_len_jitter_frac": 0.03,
    "stem_center_mix_prob": 0.0,
    "stem_wrong_side_prob": 0.0,
    "beam_junction_overshoot_sp": 0.0,
    "beam_junction_gap_prob": 0.0,
    "beam_edge_overhang_sp": 0.0,
    "pitch_y_slop_sp": 0.0,
    "chord_x_stagger_sp": 0.0,
    "accidental_x_offset_sp": 1.25,
    "accidental_x_jitter_sp": 0.0,
    # v0.8.0 neutral defaults
    "nib_anisotropy": 0.0,
    "edge_crispness": 0.0,
    "ink_mass": 0.0,
    "rastrum_curve_var": 0.0,
}

LIGATURE_PATTERNS: tuple[str, ...] = (
    "notehead_to_notehead_swoop",
    "notehead_to_stem_base_hook",
    "flag_to_flag_return_loop",
    "chord_roll_drag",
)

SPACING_DUCTUS_PROFILES: dict[str, dict[str, Any]] = {
    "engraver_metronomic": {
        "duration_spacing_gamma": 0.62, "bar_end_panic_start": 0.85, "bar_end_panic_strength": 0.02,
        "cross_voice_x_slop_sp": 0.03, "ligature_prob": 0.0, "ligature_opacity": 0.22,
        "duration_head_scale_slope": 0.03, "downbeat_weight_boost": 0.04, "phrase_arc_mod": 0.03,
        "stem_len_jitter_frac": 0.03, "stem_center_mix_prob": 0.0, "stem_wrong_side_prob": 0.0,
        "beam_junction_overshoot_sp": 0.03, "beam_junction_gap_prob": 0.0, "beam_edge_overhang_sp": 0.02,
        "pitch_y_slop_sp": 0.020, "chord_x_stagger_sp": 0.02, "accidental_x_offset_sp": 1.28, "accidental_x_jitter_sp": 0.04,
    },
    "virtuoso_bunched_16ths": {
        "duration_spacing_gamma": 0.42, "bar_end_panic_start": 0.66, "bar_end_panic_strength": 0.28,
        "cross_voice_x_slop_sp": 0.18, "ligature_prob": 0.44, "ligature_opacity": 0.36,
        "duration_head_scale_slope": 0.18, "downbeat_weight_boost": 0.16, "phrase_arc_mod": 0.26,
        "stem_len_jitter_frac": 0.14, "stem_center_mix_prob": 0.10, "stem_wrong_side_prob": 0.0,
        "beam_junction_overshoot_sp": 0.24, "beam_junction_gap_prob": 0.14, "beam_edge_overhang_sp": 0.34,
        "pitch_y_slop_sp": 0.075, "chord_x_stagger_sp": 0.14, "accidental_x_offset_sp": 1.12, "accidental_x_jitter_sp": 0.18,
    },
    "panic_bar_end_cram": {
        "duration_spacing_gamma": 0.50, "bar_end_panic_start": 0.56, "bar_end_panic_strength": 0.72,
        "cross_voice_x_slop_sp": 0.26, "ligature_prob": 0.32, "ligature_opacity": 0.34,
        "duration_head_scale_slope": 0.14, "downbeat_weight_boost": 0.10, "phrase_arc_mod": 0.22,
        "stem_len_jitter_frac": 0.17, "stem_center_mix_prob": 0.14, "stem_wrong_side_prob": 0.02,
        "beam_junction_overshoot_sp": 0.30, "beam_junction_gap_prob": 0.22, "beam_edge_overhang_sp": 0.28,
        "pitch_y_slop_sp": 0.090, "chord_x_stagger_sp": 0.16, "accidental_x_offset_sp": 1.02, "accidental_x_jitter_sp": 0.22,
    },
    "novice_equal_spacing": {
        "duration_spacing_gamma": 0.18, "bar_end_panic_start": 0.60, "bar_end_panic_strength": 0.48,
        "cross_voice_x_slop_sp": 0.30, "ligature_prob": 0.08, "ligature_opacity": 0.28,
        "duration_head_scale_slope": -0.14, "downbeat_weight_boost": 0.05, "phrase_arc_mod": 0.08,
        "stem_len_jitter_frac": 0.22, "stem_center_mix_prob": 0.24, "stem_wrong_side_prob": 0.11,
        "beam_junction_overshoot_sp": 0.22, "beam_junction_gap_prob": 0.28, "beam_edge_overhang_sp": -0.06,
        "pitch_y_slop_sp": 0.110, "chord_x_stagger_sp": 0.20, "accidental_x_offset_sp": 1.52, "accidental_x_jitter_sp": 0.28,
    },
    "cursive_flying_ligatures": {
        "duration_spacing_gamma": 0.54, "bar_end_panic_start": 0.70, "bar_end_panic_strength": 0.22,
        "cross_voice_x_slop_sp": 0.16, "ligature_prob": 0.66, "ligature_opacity": 0.46,
        "duration_head_scale_slope": 0.11, "downbeat_weight_boost": 0.14, "phrase_arc_mod": 0.24,
        "stem_len_jitter_frac": 0.12, "stem_center_mix_prob": 0.08, "stem_wrong_side_prob": 0.0,
        "beam_junction_overshoot_sp": 0.20, "beam_junction_gap_prob": 0.08, "beam_edge_overhang_sp": 0.42,
        "pitch_y_slop_sp": 0.065, "chord_x_stagger_sp": 0.15, "accidental_x_offset_sp": 1.18, "accidental_x_jitter_sp": 0.14,
    },
    "heavy_downbeat_rubato": {
        "duration_spacing_gamma": 0.72, "bar_end_panic_start": 0.74, "bar_end_panic_strength": 0.18,
        "cross_voice_x_slop_sp": 0.14, "ligature_prob": 0.24, "ligature_opacity": 0.38,
        "duration_head_scale_slope": 0.15, "downbeat_weight_boost": 0.34, "phrase_arc_mod": 0.32,
        "stem_len_jitter_frac": 0.13, "stem_center_mix_prob": 0.06, "stem_wrong_side_prob": 0.0,
        "beam_junction_overshoot_sp": 0.26, "beam_junction_gap_prob": 0.06, "beam_edge_overhang_sp": 0.36,
        "pitch_y_slop_sp": 0.060, "chord_x_stagger_sp": 0.22, "accidental_x_offset_sp": 1.38, "accidental_x_jitter_sp": 0.16,
    },
    "tremulous_elder_hand": {
        "duration_spacing_gamma": 0.58, "bar_end_panic_start": 0.64, "bar_end_panic_strength": 0.35,
        "cross_voice_x_slop_sp": 0.28, "ligature_prob": 0.22, "ligature_opacity": 0.24,
        "duration_head_scale_slope": 0.06, "downbeat_weight_boost": 0.12, "phrase_arc_mod": 0.18,
        "stem_len_jitter_frac": 0.24, "stem_center_mix_prob": 0.20, "stem_wrong_side_prob": 0.04,
        "beam_junction_overshoot_sp": 0.28, "beam_junction_gap_prob": 0.36, "beam_edge_overhang_sp": 0.18,
        "pitch_y_slop_sp": 0.130, "chord_x_stagger_sp": 0.24, "accidental_x_offset_sp": 1.44, "accidental_x_jitter_sp": 0.26,
    },
}


def resolve_ductus_param(style: dict[str, Any], key: str, fallback: Any) -> Any:
    """Resolve a ductus parameter: explicit style key > named profile > fallback."""
    if key in style and style[key] is not None:
        return style[key]
    prof_name = style.get("spacing_ductus_profile")
    if isinstance(prof_name, str) and prof_name in SPACING_DUCTUS_PROFILES:
        prof = SPACING_DUCTUS_PROFILES[prof_name]
        if key in prof:
            return prof[key]
    return fallback

_INTERPOLATABLE_NUMERIC_KEYS: tuple[str, ...] = tuple(_NUMERIC_CLAMP_BOUNDS.keys())
_RGB_KEYS: tuple[str, ...] = ("ink_rgb", "staff_ink_rgb", "paper_rgb")


def _det_pick(domain: str, name: str, era: str, choices: tuple[str, ...]) -> str:
    """Deterministically pick an archetype from choices based on style name and era."""
    digest = hashlib.sha256(f"{domain}:{name}:{era}".encode("utf-8")).digest()
    idx = int.from_bytes(digest[:4], "big") % len(choices)
    return choices[idx]


def resolve_stem_archetype(style: dict[str, Any]) -> str:
    """Return explicit stem_archetype or deterministically derive one from name/era."""
    val = style.get("stem_archetype")
    if isinstance(val, str) and val in STEM_ARCHETYPES:
        return val
    name = str(style.get("name", ""))
    era = str(style.get("era", ""))
    low = name.lower()
    if "tremor" in low or "elderly" in low:
        return "tremor_shaky"
    if "left_hand" in low or "backslant" in low:
        return "back_slant_lefty"
    return _det_pick("stem", name, era, STEM_ARCHETYPES)


def resolve_flag_archetype(style: dict[str, Any]) -> str:
    """Return explicit flag_archetype or deterministically derive one from name/era."""
    val = style.get("flag_archetype")
    if isinstance(val, str) and val in FLAG_ARCHETYPES:
        return val
    name = str(style.get("name", ""))
    era = str(style.get("era", ""))
    return _det_pick("flag", name, era, FLAG_ARCHETYPES)


def resolve_beam_archetype(style: dict[str, Any]) -> str:
    """Return explicit beam_archetype or deterministically derive one from name/era."""
    val = style.get("beam_archetype")
    if isinstance(val, str) and val in BEAM_ARCHETYPES:
        return val
    name = str(style.get("name", ""))
    era = str(style.get("era", ""))
    return _det_pick("beam", name, era, BEAM_ARCHETYPES)


def resolve_ledger_style(style: dict[str, Any]) -> str:
    """Return explicit ledger_style or deterministically derive one from name/era."""
    val = style.get("ledger_style")
    if isinstance(val, str) and val in LEDGER_STYLES:
        return val
    name = str(style.get("name", ""))
    era = str(style.get("era", ""))
    return _det_pick("ledger", name, era, LEDGER_STYLES)


def resolve_beam_slope_style(style: dict[str, Any]) -> str:
    """Return explicit beam_slope_style or deterministically derive one from name/era."""
    val = style.get("beam_slope_style")
    if isinstance(val, str) and val in BEAM_SLOPE_STYLES:
        return val
    name = str(style.get("name", ""))
    era = str(style.get("era", ""))
    return _det_pick("beam_slope", name, era, BEAM_SLOPE_STYLES)


def resolve_beam_grouping_mode(style: dict[str, Any]) -> str:
    """Return explicit beam_grouping_mode or deterministically derive one from name/era."""
    val = style.get("beam_grouping_mode")
    if isinstance(val, str) and val in BEAM_GROUPING_MODES:
        return val
    name = str(style.get("name", ""))
    era = str(style.get("era", ""))
    return _det_pick("beam_grouping", name, era, BEAM_GROUPING_MODES)


def _normalize_rest_weights(raw_weights: dict[str, Any]) -> dict[str, float]:
    """Validate and normalize a rest_archetype_weights dict across all REST_ARCHETYPES."""
    if not isinstance(raw_weights, dict):
        raise ValueError("rest_archetype_weights must be a dict")
    unknown = set(raw_weights.keys()) - set(REST_ARCHETYPES)
    if unknown:
        raise ValueError(f"Unknown rest_archetype_weights keys: {sorted(unknown)}")
    merged: dict[str, float] = {}
    for key in REST_ARCHETYPES:
        val = float(raw_weights.get(key, 0.0))
        if val < 0.0:
            raise ValueError(f"rest_archetype_weights[{key!r}] must be >= 0, got {val}")
        merged[key] = val
    total = sum(merged.values())
    if total <= 0.0:
        raise ValueError("rest_archetype_weights must sum to a positive value")
    return {k: round(v / total, 6) for k, v in merged.items()}


def _merge_style_dicts(parent: dict[str, Any], child: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge child style overrides onto a copy of parent."""
    merged = _copy_style_dict(parent)
    for key, val in child.items():
        if key == "extends":
            continue
        if key == "rest_archetype_weights" and isinstance(val, dict) and isinstance(merged.get(key), dict):
            # Allow child to override a subset of rest weights or all of them.
            combined = dict(merged[key])
            combined.update(val)
            merged[key] = combined
        elif isinstance(val, dict):
            merged[key] = dict(val)
        elif isinstance(val, list):
            merged[key] = list(val)
        else:
            merged[key] = val
    return merged


def _validate_and_normalize_style(name: str, spec: dict[str, Any]) -> dict[str, Any]:
    """Validate a resolved style dict against REQUIRED_STYLE_KEYS and normalize types."""
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Style name must be a non-empty string")
    if not isinstance(spec, dict):
        raise ValueError(f"Style spec for {name!r} must be a dict")

    out = _copy_style_dict(spec)
    out.pop("extends", None)
    out["name"] = name

    missing = [k for k in REQUIRED_STYLE_KEYS if k not in out]
    if missing:
        raise ValueError(f"Style {name!r} is missing required keys: {missing}")

    # Normalize RGB channels to 3-tuples of floats in [0, 1].
    for rgb_key in _RGB_KEYS:
        rgb = out[rgb_key]
        if not isinstance(rgb, (list, tuple, np.ndarray)) or len(rgb) != 3:
            raise ValueError(f"Style {name!r}.{rgb_key} must be a 3-element RGB sequence")
        rgb_t = tuple(float(ch) for ch in rgb)
        for ch in rgb_t:
            if not (0.0 <= ch <= 1.0):
                raise ValueError(f"Style {name!r}.{rgb_key} channel {ch} out of [0, 1]")
        out[rgb_key] = rgb_t

    if out["notehead_archetype"] not in NOTEHEAD_ARCHETYPES:
        raise ValueError(
            f"Style {name!r} has invalid notehead_archetype {out['notehead_archetype']!r}; "
            f"expected one of {NOTEHEAD_ARCHETYPES}"
        )

    out["rest_archetype_weights"] = _normalize_rest_weights(out["rest_archetype_weights"])

    if "paper_type" in out and out["paper_type"] is not None:
        if out["paper_type"] not in PAPER_TYPES:
            raise ValueError(
                f"Style {name!r} has invalid paper_type {out['paper_type']!r}; "
                f"expected one of {PAPER_TYPES}"
            )

    for arch_key, valid_choices in (
        ("stem_archetype", STEM_ARCHETYPES),
        ("flag_archetype", FLAG_ARCHETYPES),
        ("beam_archetype", BEAM_ARCHETYPES),
        ("ledger_style", LEDGER_STYLES),
        ("beam_slope_style", BEAM_SLOPE_STYLES),
        ("beam_grouping_mode", BEAM_GROUPING_MODES),
        ("clef_archetype", CLEF_ARCHETYPES),
        ("keysig_archetype", KEYSIG_ARCHETYPES),
        ("barline_archetype", BARLINE_ARCHETYPES),
        ("pause_archetype", NEW_QUARTER_REST_ARCHETYPES),
        ("flagged_rest_archetype", FLAGGED_REST_ARCHETYPES),
        ("bar_rest_archetype", BAR_REST_ARCHETYPES),
    ):
        if arch_key in out and out[arch_key] is not None:
            if out[arch_key] not in valid_choices:
                raise ValueError(
                    f"Style {name!r} has invalid {arch_key} {out[arch_key]!r}; "
                    f"expected one of {valid_choices}"
                )

    if "tags" in out and out["tags"] is not None:
        if isinstance(out["tags"], str):
            out["tags"] = [out["tags"]]
        elif isinstance(out["tags"], (list, tuple, set)):
            out["tags"] = [str(t) for t in out["tags"]]
        else:
            raise ValueError(f"Style {name!r}.tags must be a list of strings")

    # Sanity-check numeric types and clamp to safe rendering bounds.
    for key in _INTERPOLATABLE_NUMERIC_KEYS:
        if key in out and out[key] is not None:
            out[key] = float(out[key])

    if out["stroke_scale"] <= 0.0:
        raise ValueError(f"Style {name!r}.stroke_scale must be > 0")
    if not (0.0 < out["nib_min_ratio"] <= 1.0):
        raise ValueError(f"Style {name!r}.nib_min_ratio must be in (0, 1]")
    if not (0.50 <= out["notehead_scale"] <= 1.45):
        raise ValueError(
            f"Style {name!r}.notehead_scale must be in [0.50, 1.45], got {out['notehead_scale']}"
        )
    if "notehead_aspect" in out and out["notehead_aspect"] is not None:
        if not (0.25 <= out["notehead_aspect"] <= 2.20):
            raise ValueError(
                f"Style {name!r}.notehead_aspect must be in [0.25, 2.20], got {out['notehead_aspect']}"
            )
    if "notehead_fill_density" in out and out["notehead_fill_density"] is not None:
        if not (0.25 <= out["notehead_fill_density"] <= 1.50):
            raise ValueError(
                f"Style {name!r}.notehead_fill_density must be in [0.25, 1.50], got {out['notehead_fill_density']}"
            )
    if "stem_length_scale" in out and out["stem_length_scale"] is not None:
        if not (0.65 <= out["stem_length_scale"] <= 1.55):
            raise ValueError(
                f"Style {name!r}.stem_length_scale must be in [0.65, 1.55], got {out['stem_length_scale']}"
            )
    if "beam_thickness_scale" in out and out["beam_thickness_scale"] is not None:
        if not (0.45 <= out["beam_thickness_scale"] <= 2.10):
            raise ValueError(
                f"Style {name!r}.beam_thickness_scale must be in [0.45, 2.10], got {out['beam_thickness_scale']}"
            )
    if "beam_split_to_flag_prob" in out and out["beam_split_to_flag_prob"] is not None:
        if not (0.0 <= out["beam_split_to_flag_prob"] <= 1.0):
            raise ValueError(
                f"Style {name!r}.beam_split_to_flag_prob must be in [0.0, 1.0], got {out['beam_split_to_flag_prob']}"
            )

    for key, (lo, hi) in _NUMERIC_CLAMP_BOUNDS.items():
        if key in out and out[key] is not None:
            v = out[key]
            out[key] = lo if v < lo else (hi if v > hi else v)

    return out


def validate_style(spec: dict[str, Any], name: Optional[str] = None) -> dict[str, Any]:
    """Validate and normalize a style dict against required keys and parameter bounds."""
    resolved_name = name if name is not None else str(spec.get("name", ""))
    return _validate_and_normalize_style(resolved_name, spec)


def register_style(
    name: str,
    spec: dict[str, Any],
    *,
    overwrite: bool = True,
) -> dict[str, Any]:
    """Validate, resolve optional ``extends`` inheritance, and register a scribe style."""
    global _REGISTRY_VERSION
    if not overwrite and name in SCRIBE_STYLES:
        raise ValueError(f"Style {name!r} is already registered")
    if not isinstance(spec, dict):
        raise ValueError(f"Style spec for {name!r} must be a dict")

    parent_name = spec.get("extends")
    if parent_name is not None:
        if parent_name == name:
            raise ValueError(f"Style {name!r} cannot extend itself")
        if parent_name not in SCRIBE_STYLES:
            raise ValueError(f"Style {name!r} extends unknown parent style {parent_name!r}")
        resolved = _merge_style_dicts(SCRIBE_STYLES[parent_name], spec)
    else:
        resolved = spec

    validated = _validate_and_normalize_style(name, resolved)
    SCRIBE_STYLES[name] = validated
    _REGISTRY_VERSION += 1
    return _copy_style_dict(validated)


def _extract_pack_entries(payload: Any, source_path: Path) -> dict[str, dict[str, Any]]:
    """Parse flexible JSON style-pack schemas into a {name: spec} mapping."""
    entries: dict[str, dict[str, Any]] = {}
    if isinstance(payload, list):
        for item in payload:
            if not isinstance(item, dict) or "name" not in item:
                raise ValueError(f"List entry in {source_path} must be a dict with a 'name' key")
            entries[str(item["name"])] = item
        return entries

    if not isinstance(payload, dict):
        raise ValueError(f"Style pack {source_path} must contain a JSON object or list")

    # Wrapper format: {"styles": {...}} or {"styles": [...]}
    if "styles" in payload and isinstance(payload["styles"], (dict, list)):
        return _extract_pack_entries(payload["styles"], source_path)

    # Single style dict format: {"name": "...", "extends": "...", ...}
    if "name" in payload and isinstance(payload["name"], str):
        entries[payload["name"]] = payload
        return entries

    # Mapping format: {"style_a": {...}, "style_b": {...}}
    for key, val in payload.items():
        if str(key).startswith("_") or str(key) in ("schema_version", "pack_name", "description"):
            continue
        if not isinstance(val, dict):
            raise ValueError(f"Entry {key!r} in {source_path} must be a dict")
        entries[str(key)] = val
    return entries


def reload_style_packs(
    packs_dir: Optional[Union[str, Path]] = None,
    *,
    reset_to_builtins: bool = False,
) -> list[str]:
    """Discover and load all ``*.json`` style packs from ``packs_dir`` (default: ``style_packs/``).

    Supports multi-level ``"extends"`` inheritance across built-ins and pack files,
    with circular-inheritance detection and full ``REQUIRED_STYLE_KEYS`` validation.
    Returns the list of style names loaded from disk.
    """
    global _REGISTRY_VERSION
    if reset_to_builtins:
        SCRIBE_STYLES.clear()
        for k, v in _BUILTIN_STYLES.items():
            SCRIBE_STYLES[k] = _copy_style_dict(v)
        _REGISTRY_VERSION += 1

    target_dir = Path(packs_dir) if packs_dir is not None else STYLE_PACKS_DIR
    if not target_dir.exists() or not target_dir.is_dir():
        return []

    pending: dict[str, dict[str, Any]] = {}
    for json_path in sorted(target_dir.glob("*.json")):
        with json_path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        entries = _extract_pack_entries(payload, json_path)
        for style_name, spec in entries.items():
            pending[style_name] = spec

    if not pending:
        return []

    # Resolve in dependency order so styles can extend other styles in the same or other packs.
    loaded: list[str] = []
    visiting: set[str] = set()

    def _resolve_and_register(style_name: str) -> None:
        if style_name not in pending:
            return
        if style_name in visiting:
            raise ValueError(f"Circular style inheritance detected involving {style_name!r}")
        visiting.add(style_name)
        spec = pending[style_name]
        parent_name = spec.get("extends")
        if parent_name is not None:
            if parent_name == style_name:
                raise ValueError(f"Style {style_name!r} cannot extend itself")
            if parent_name in pending:
                _resolve_and_register(parent_name)
            elif parent_name not in SCRIBE_STYLES:
                raise ValueError(
                    f"Style {style_name!r} extends unknown parent style {parent_name!r}"
                )
        register_style(style_name, spec, overwrite=True)
        pending.pop(style_name, None)
        visiting.discard(style_name)
        loaded.append(style_name)

    for style_name in list(pending.keys()):
        _resolve_and_register(style_name)

    return loaded


def list_styles(tag: Optional[str] = None) -> list[str]:
    """Return the ordered list of registered scribe style names, optionally filtered by ``tag``.

    When ``tag`` is provided, matches against explicit ``style["tags"]``, ``style["era"]``,
    ``style["paper_type"]``, or ``style["notehead_archetype"]`` (case-insensitive).
    """
    if tag is None:
        return list(SCRIBE_STYLES.keys())
    needle = str(tag).strip().lower()
    matched: list[str] = []
    for name, s in SCRIBE_STYLES.items():
        tags_lower = {str(t).strip().lower() for t in s.get("tags", ())}
        if (
            needle in tags_lower
            or str(s.get("era", "")).strip().lower() == needle
            or str(s.get("paper_type", "")).strip().lower() == needle
            or str(s.get("notehead_archetype", "")).strip().lower() == needle
        ):
            matched.append(name)
    return matched


def get_style(name: Union[str, dict[str, Any]]) -> dict[str, Any]:
    """Return an independent copy of a named scribe style or a provided style dict."""
    if isinstance(name, dict):
        return _copy_style_dict(name)
    if name not in SCRIBE_STYLES:
        valid = ", ".join(SCRIBE_STYLES.keys())
        raise KeyError(f"Unknown scribe style {name!r}. Valid styles: {valid}")
    return _copy_style_dict(SCRIBE_STYLES[name])


def interpolate_styles(
    name_a: Union[str, dict[str, Any]],
    name_b: Union[str, dict[str, Any]],
    alpha: float,
    rng: Optional[np.random.Generator] = None,
) -> dict[str, Any]:
    """Continuously interpolate between two scribe styles at blend factor ``alpha`` in ``[0, 1]``.

    ``alpha == 0.0`` yields style A; ``alpha == 1.0`` yields style B.
    Discrete attributes (``notehead_archetype``, ``era``, ``paper_type``) are chosen
    probabilistically using ``rng`` if provided, otherwise by thresholding ``alpha >= 0.5``.
    """
    t = float(alpha)
    if not (0.0 <= t <= 1.0):
        raise ValueError(f"alpha must be in [0.0, 1.0], got {alpha}")

    sa = get_style(name_a)
    sb = get_style(name_b)
    label_a = str(sa.get("name", "style_a"))
    label_b = str(sb.get("name", "style_b"))

    def _pick_b() -> bool:
        if rng is not None:
            return bool(rng.random() < t)
        return t >= 0.5

    out = _copy_style_dict(sa if t < 0.5 else sb)
    out["name"] = f"{label_a}__x{t:.2f}__{label_b}"
    out["description"] = f"Interpolated scribe ({1.0 - t:.2f} {label_a} + {t:.2f} {label_b})."
    out["era"] = sb["era"] if _pick_b() else sa["era"]
    out["notehead_archetype"] = sb["notehead_archetype"] if _pick_b() else sa["notehead_archetype"]

    # Optional paper_type
    pa = sa.get("paper_type")
    pb = sb.get("paper_type")
    if pa is not None or pb is not None:
        chosen_p = pb if _pick_b() else pa
        if chosen_p is not None:
            out["paper_type"] = chosen_p
        else:
            out.pop("paper_type", None)

    # Optional stem/flag/beam/ledger/beam-slope/beam-grouping discrete archetypes
    for disc_key, resolver in (
        ("stem_archetype", resolve_stem_archetype),
        ("flag_archetype", resolve_flag_archetype),
        ("beam_archetype", resolve_beam_archetype),
        ("ledger_style", resolve_ledger_style),
        ("beam_slope_style", resolve_beam_slope_style),
        ("beam_grouping_mode", resolve_beam_grouping_mode),
    ):
        if disc_key in sa or disc_key in sb:
            va = resolver(sa)
            vb = resolver(sb)
            out[disc_key] = vb if _pick_b() else va

    # Optional structural (clef/keysig/barline) discrete archetypes
    if any(k in sa or k in sb for k in ("clef_archetype", "keysig_archetype", "barline_archetype")):
        ca, ka, ba = resolve_structural_archetypes(sa)
        cb, kb, bb = resolve_structural_archetypes(sb)
        for disc_key, va, vb in (
            ("clef_archetype", ca, cb),
            ("keysig_archetype", ka, kb),
            ("barline_archetype", ba, bb),
        ):
            if disc_key in sa or disc_key in sb:
                out[disc_key] = vb if _pick_b() else va

    # Optional 30-pause discrete archetypes + v0.3.0 optional discrete keys
    for opt_disc_key in (
        "pause_archetype", "flagged_rest_archetype", "bar_rest_archetype",
        "page_layout_mode", "vertical_rhythm_mode", "staff_ruling_mode",
        "staff_line_weight_profile", "page_furniture_mode", "barline_span_mode",
        "left_barline_mode", "paper_border_mode", "paper_binding_mode",
        "paper_background_grid", "annotation_persona", "cancellation_style",
        "measure_number_style", "chord_symbol_style", "palm_smear_hand",
        "spacing_ductus_profile", "slur_archetype", "timesig_archetype",
        "dynamic_archetype", "hairpin_archetype", "articulation_archetype",
        "dot_archetype", "tuplet_archetype", "grace_slash_archetype",
        "arpeggio_archetype", "courtesy_archetype", "brace_archetype",
        "volta_archetype", "multibar_archetype",
    ):
        va = sa.get(opt_disc_key)
        vb = sb.get(opt_disc_key)
        if va is not None or vb is not None:
            chosen_v = vb if _pick_b() else va
            if chosen_v is not None:
                out[opt_disc_key] = chosen_v
            else:
                out.pop(opt_disc_key, None)

    # Interpolate RGB tuples
    for rgb_key in _RGB_KEYS:
        ra = np.asarray(sa[rgb_key], dtype=np.float64)
        rb = np.asarray(sb[rgb_key], dtype=np.float64)
        blended = np.clip((1.0 - t) * ra + t * rb, 0.0, 1.0)
        out[rgb_key] = tuple(round(float(c), 6) for c in blended)

    # Interpolate numeric parameters (required + optional)
    for key in _INTERPOLATABLE_NUMERIC_KEYS:
        if key in sa or key in sb:
            default_v = _OPTIONAL_NUMERIC_DEFAULTS.get(key, 0.0)
            va = float(sa.get(key, default_v))
            vb = float(sb.get(key, default_v))
            out[key] = (1.0 - t) * va + t * vb

    # Interpolate and renormalize rest_archetype_weights
    rwa = sa["rest_archetype_weights"]
    rwb = sb["rest_archetype_weights"]
    raw_rw = {
        k: (1.0 - t) * float(rwa.get(k, 0.0)) + t * float(rwb.get(k, 0.0))
        for k in REST_ARCHETYPES
    }
    out["rest_archetype_weights"] = _normalize_rest_weights(raw_rw)

    return out


def sample_random_scribe(
    rng: np.random.Generator,
    base_style: Optional[Union[str, dict[str, Any]]] = None,
    jitter: float = 0.20,
) -> dict[str, Any]:
    """Sample a continuous unseen handwritten scribe profile for high-variance training.

    If ``base_style`` is ``None``, randomly selects two registered styles and interpolates
    between them before applying controlled parameter jitter scaled by ``jitter``.
    """
    j = max(0.0, float(jitter))
    if base_style is None:
        names = list_styles()
        idx_a = int(rng.integers(0, len(names)))
        idx_b = int(rng.integers(0, len(names)))
        alpha = float(rng.uniform(0.0, 1.0))
        scribe = interpolate_styles(names[idx_a], names[idx_b], alpha, rng=rng)
    elif isinstance(base_style, str):
        scribe = get_style(base_style)
    else:
        scribe = _copy_style_dict(base_style)

    # Jitter RGB channels gently within [0, 1].
    if j > 0.0:
        for rgb_key in _RGB_KEYS:
            rgb = np.asarray(scribe[rgb_key], dtype=np.float64)
            scale = 0.12 if rgb_key == "paper_rgb" else 0.18
            delta = rng.normal(0.0, scale * j, size=3)
            # Keep paper reasonably bright unless explicitly dark.
            lo = 0.65 if rgb_key == "paper_rgb" else 0.02
            hi = 0.99 if rgb_key == "paper_rgb" else 0.85
            rgb_j = np.clip(rgb + delta, lo, hi)
            scribe[rgb_key] = tuple(round(float(c), 6) for c in rgb_j)

        # Jitter all required numeric keys within safe bounds.
        for key, (lo, hi) in _NUMERIC_CLAMP_BOUNDS.items():
            if key in _OPTIONAL_NUMERIC_DEFAULTS and key not in scribe:
                continue
            val = float(scribe[key])
            span = hi - lo
            # Mix multiplicative + additive jitter so zero-valued params can also explore.
            noise = float(rng.normal(0.0, j)) * (0.5 * abs(val) + 0.12 * span)
            scribe[key] = float(np.clip(val + noise, lo, hi))

        # Perturb rest archetype weights smoothly and renormalize.
        rw = scribe["rest_archetype_weights"]
        raw_rw: dict[str, float] = {}
        for k in REST_ARCHETYPES:
            w = float(rw.get(k, 0.0))
            w_new = max(0.0, w * float(np.exp(rng.normal(0.0, j * 1.4))) + max(0.0, float(rng.normal(0.0, 0.04 * j))))
            raw_rw[k] = w_new
        if sum(raw_rw.values()) <= 1e-9:
            raw_rw = {k: 1.0 for k in REST_ARCHETYPES}
        scribe["rest_archetype_weights"] = _normalize_rest_weights(raw_rw)

    # Populate extended stroke/paper physics parameters if not already set.
    if "ink_feather" not in scribe:
        scribe["ink_feather"] = float(np.clip(rng.uniform(0.0, 0.45) * (0.5 + j), 0.0, 1.0))
    if "split_nib_prob" not in scribe:
        scribe["split_nib_prob"] = float(np.clip(rng.uniform(0.0, 0.35) if rng.random() < 0.35 else 0.0, 0.0, 1.0))
    if "rastrum_wobble" not in scribe:
        scribe["rastrum_wobble"] = float(np.clip(rng.uniform(0.0, 2.5), 0.0, 6.0))
    if "stain_prob" not in scribe:
        scribe["stain_prob"] = float(np.clip(rng.uniform(0.0, 0.30) if rng.random() < 0.35 else 0.0, 0.0, 1.0))
    if base_style is None and j > 0.0:
        scribe["nib_angle_deg"] = float(rng.uniform(25.0, 55.0))
    if "nib_anisotropy" not in scribe:
        # Broad-nib / quill / fountain-pen scribes get authentic chisel-nib anisotropy.
        nmr = float(scribe.get("nib_min_ratio", 0.35))
        if nmr <= 0.65:
            scribe["nib_anisotropy"] = float(np.clip(rng.uniform(0.25, 0.85), 0.0, 1.0))
        else:
            scribe["nib_anisotropy"] = float(np.clip(rng.uniform(0.0, 0.20), 0.0, 1.0))
    if "edge_crispness" not in scribe:
        scribe["edge_crispness"] = float(np.clip(rng.uniform(0.20, 0.85), 0.0, 1.0))
    if "ink_mass" not in scribe:
        scribe["ink_mass"] = float(np.clip(rng.uniform(0.12, 0.65), 0.0, 1.0))
    if "rastrum_curve_var" not in scribe:
        scribe["rastrum_curve_var"] = float(np.clip(rng.uniform(0.0, 0.22), 0.0, 0.35))

    # Sample morphological notehead parameters & explore all 18 archetypes.
    # MUSCIMA++ biometrics: 76% core / 14% slim / 10% wide notehead_aspect.
    if base_style is None and j > 0.0 and rng.random() < 0.45:
        scribe["notehead_archetype"] = str(NOTEHEAD_ARCHETYPES[int(rng.integers(0, len(NOTEHEAD_ARCHETYPES)))])
    if "notehead_aspect" not in scribe:
        u = float(rng.random())
        if u < 0.14:
            aspect = float(rng.uniform(0.28, 0.65))
        elif u < 0.90:
            aspect = float(rng.uniform(0.72, 1.25))
        else:
            aspect = float(rng.uniform(1.30, 2.05))
        scribe["notehead_aspect"] = float(np.clip(aspect, 0.25, 2.20))
    if "notehead_fill_density" not in scribe:
        scribe["notehead_fill_density"] = float(np.clip(rng.uniform(0.30, 1.45), 0.25, 1.50))

    # MUSCIMA++ biometrics: 88% right-handed / 12% left-handed slant mixture for unconstrained scribes.
    if base_style is None and j > 0.0:
        if float(rng.random()) < 0.88:
            scribe["slant_deg"] = float(np.clip(rng.normal(6.5, 4.2), 0.0, 18.0))
        else:
            scribe["slant_deg"] = float(np.clip(rng.normal(-8.0, 3.2), -14.0, -1.0))

    # Sample morphological stem, flag, beam, and ledger archetypes across all 52 + 5 variants.
    if base_style is None or (j > 0.0 and rng.random() < 0.55):
        scribe["stem_archetype"] = str(STEM_ARCHETYPES[int(rng.integers(0, len(STEM_ARCHETYPES)))])
    elif "stem_archetype" not in scribe:
        scribe["stem_archetype"] = resolve_stem_archetype(scribe)

    if base_style is None or (j > 0.0 and rng.random() < 0.55):
        scribe["flag_archetype"] = str(FLAG_ARCHETYPES[int(rng.integers(0, len(FLAG_ARCHETYPES)))])
    elif "flag_archetype" not in scribe:
        scribe["flag_archetype"] = resolve_flag_archetype(scribe)

    if base_style is None or (j > 0.0 and rng.random() < 0.55):
        scribe["beam_archetype"] = str(BEAM_ARCHETYPES[int(rng.integers(0, len(BEAM_ARCHETYPES)))])
    elif "beam_archetype" not in scribe:
        scribe["beam_archetype"] = resolve_beam_archetype(scribe)

    if base_style is None or (j > 0.0 and rng.random() < 0.55):
        scribe["ledger_style"] = str(LEDGER_STYLES[int(rng.integers(0, len(LEDGER_STYLES)))])
    elif "ledger_style" not in scribe:
        scribe["ledger_style"] = resolve_ledger_style(scribe)

    # Sample structural clef, key-signature, and barline archetypes across all 54 variants.
    default_clef, default_keysig, default_barline = resolve_structural_archetypes(scribe)
    if base_style is None or (j > 0.0 and rng.random() < 0.55):
        scribe["clef_archetype"] = str(CLEF_ARCHETYPES[int(rng.integers(0, len(CLEF_ARCHETYPES)))])
    elif "clef_archetype" not in scribe:
        scribe["clef_archetype"] = default_clef

    if base_style is None or (j > 0.0 and rng.random() < 0.55):
        scribe["keysig_archetype"] = str(KEYSIG_ARCHETYPES[int(rng.integers(0, len(KEYSIG_ARCHETYPES)))])
    elif "keysig_archetype" not in scribe:
        scribe["keysig_archetype"] = default_keysig

    if base_style is None or (j > 0.0 and rng.random() < 0.55):
        scribe["barline_archetype"] = str(BARLINE_ARCHETYPES[int(rng.integers(0, len(BARLINE_ARCHETYPES)))])
    elif "barline_archetype" not in scribe:
        scribe["barline_archetype"] = default_barline

    if "stem_length_scale" not in scribe:
        # MUSCIMA++ empirical stem length distribution centered at 0.88.
        scribe["stem_length_scale"] = float(np.clip(rng.normal(0.88, 0.14), 0.65, 1.55))
    if "beam_thickness_scale" not in scribe:
        scribe["beam_thickness_scale"] = float(np.clip(rng.uniform(0.55, 1.95), 0.45, 2.10))

    # Sample the 30 pause/rest archetypes (18 quarter + 6 flagged + 6 bar/fermata).
    if base_style is None or (j > 0.0 and rng.random() < 0.55):
        scribe["pause_archetype"] = str(NEW_QUARTER_REST_ARCHETYPES[int(rng.integers(0, len(NEW_QUARTER_REST_ARCHETYPES)))])
    if base_style is None or (j > 0.0 and rng.random() < 0.55):
        scribe["flagged_rest_archetype"] = str(FLAGGED_REST_ARCHETYPES[int(rng.integers(0, len(FLAGGED_REST_ARCHETYPES)))])
    if base_style is None or (j > 0.0 and rng.random() < 0.55):
        scribe["bar_rest_archetype"] = str(BAR_REST_ARCHETYPES[int(rng.integers(0, len(BAR_REST_ARCHETYPES)))])

    # Sample beam leveling (slope/elevation) & beam-to-flag grouping across all 10x8 combinations.
    if base_style is None or (j > 0.0 and rng.random() < 0.65):
        scribe["beam_slope_style"] = str(BEAM_SLOPE_STYLES[int(rng.integers(0, len(BEAM_SLOPE_STYLES)))])
    elif "beam_slope_style" not in scribe:
        scribe["beam_slope_style"] = resolve_beam_slope_style(scribe)

    if base_style is None or (j > 0.0 and rng.random() < 0.65):
        scribe["beam_grouping_mode"] = str(BEAM_GROUPING_MODES[int(rng.integers(0, len(BEAM_GROUPING_MODES)))])
    elif "beam_grouping_mode" not in scribe:
        scribe["beam_grouping_mode"] = resolve_beam_grouping_mode(scribe)

    if "beam_split_to_flag_prob" not in scribe:
        if scribe["beam_grouping_mode"] == "mixed_split_flags":
            scribe["beam_split_to_flag_prob"] = float(np.clip(rng.uniform(0.25, 0.72), 0.0, 1.0))
        elif rng.random() < 0.28:
            scribe["beam_split_to_flag_prob"] = float(np.clip(rng.uniform(0.08, 0.35), 0.0, 1.0))
        else:
            scribe["beam_split_to_flag_prob"] = 0.0

    # Sample paper_type, expressive archetypes (86), ductus profile (7), and optional living-manuscript layers
    if base_style is None or (j > 0.0 and rng.random() < 0.40):
        scribe["paper_type"] = str(PAPER_TYPES[int(rng.integers(0, len(PAPER_TYPES)))])

    from .expressive_glyphs import EXPRESSIVE_REGISTRIES, resolve_expressive_archetypes
    resolved_expr = resolve_expressive_archetypes(scribe)
    for k, reg in EXPRESSIVE_REGISTRIES.items():
        if base_style is None or (j > 0.0 and rng.random() < 0.55):
            scribe[k] = str(reg[int(rng.integers(0, len(reg)))])
        elif k not in scribe:
            scribe[k] = resolved_expr[k]

    if base_style is None or (j > 0.0 and rng.random() < 0.55):
        ductus_names = tuple(SPACING_DUCTUS_PROFILES.keys())
        scribe["spacing_ductus_profile"] = str(ductus_names[int(rng.integers(0, len(ductus_names)))])

    token = int(rng.integers(0, 16**6))
    scribe["name"] = f"sampled_scribe_{token:06x}"
    return scribe


# Auto-discover any JSON style packs present in style_packs/ on module import.
reload_style_packs()
