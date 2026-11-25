"""Deterministic auto-diversification resolver across all 234 styles (Layers 1–4).

Given any registered or sampled style dict, ``resolve_page_and_expressive_profile(style)``
deterministically derives era-, medium-, and implement-coherent settings for:
  - Layer 1: Macro layout, staff ruling, page furniture, braces, paper borders/bindings
  - Layer 2: 13 expressive glyph archetypes (86 archetypes via ``expressive_glyphs``)
  - Layer 3: Living manuscript overlays, corrections, chords, wear (``manuscript_marks``)
  - Layer 4: Spacing habits, cursive ligatures & OMR-safe micro-motor ductus

Explicit keys already present in ``style`` ALWAYS win verbatim. Zero mutation of
``style`` or JSON packs.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, Sequence, Tuple

import numpy as np

from .catalog import (
    _blob,
    classify_era,
    classify_implement,
    classify_medium,
    compute_legibility_score,
    infer_paper_type,
    score_legibility,
)
from .expressive_glyphs import resolve_expressive_archetypes


# ---------------------------------------------------------------------------
# Curated Historical-Clean Whitelists (Quarantines Broken/Cartoonish Styles)
# ---------------------------------------------------------------------------

HISTORICAL_CLEAN_NOTEHEADS: Tuple[str, ...] = (
    "oval",
    "teardrop",
    "compact",
    "slash",
    "heavy_wedge",
    "broad_nib_lozenge",
)

HISTORICAL_CLEAN_STEMS: Tuple[str, ...] = (
    "straight_classic",
    "tapered_quill",
    "tall_spindly",
)

HISTORICAL_CLEAN_FLAGS: Tuple[str, ...] = (
    "classic_droop",
    "baroque_curl",
    "sharp_angular_7",
    "calligraphic_ribbon",
    "broad_nib_pennant",
    "steep_vertical_drop",
)

HISTORICAL_CLEAN_BEAMS: Tuple[str, ...] = (
    "straight_classic",
    "tapered_wedge",
    "calligraphic_ribbon",
    "tight_crammed_bars",
)

HISTORICAL_CLEAN_BEAM_SLOPES: Tuple[str, ...] = (
    "engraver_gentle",
    "horizontal_flat",
    "steep_contour",
)

HISTORICAL_CLEAN_BEAM_GROUPINGS: Tuple[str, ...] = (
    "standard_beat",
    "long_phrase_beam",
    "pair_beams_2",
    "french_broken_secondary",
)

HISTORICAL_CLEAN_CLEFS: Tuple[str, ...] = (
    "classic_engraved",
    "baroque_ornate_scroll",
    "chopin_delicate_spiral",
    "calligraphic_broad_nib",
    "hurried_single_loop",
    "figure_8_treble",
)

HISTORICAL_CLEAN_KEYSIGS: Tuple[str, ...] = (
    "classic_neat",
    "calligraphic_nib_acc",
    "baroque_vertical_flat",
    "pointed_teardrop_flat",
    "tall_narrow_sharp",
)

HISTORICAL_CLEAN_BARLINES: Tuple[str, ...] = (
    "crisp_vertical",
    "tapered_quill_bar",
    "tall_overshoot",
    "tight_double_gap",
)

HISTORICAL_CLEAN_PAPER_TYPES: Tuple[str, ...] = (
    "aged_rag",
    "cream_manuscript",
    "bright_white",
)

HISTORICAL_CLEAN_BINDINGS: Tuple[str, ...] = (
    "none",
    "book_gutter_shadow_left",
    "book_gutter_shadow_right",
)

HISTORICAL_CLEAN_GRIDS: Tuple[str, ...] = (
    "none",
    "heavy_laid_chainlines",
)

HISTORICAL_CLEAN_PERSONAS: Tuple[str, ...] = (
    "none",
    "archival_librarian",
    "composer_working_draft",
    "piano_conservatory",
    "choral_rehearsal",
    "string_section_pencil",
)

HISTORICAL_CLEAN_FURNITURE: Tuple[str, ...] = (
    "none",
    "baroque_archival_folio",
    "classical_fair_copy_header",
    "minimal_running_folio",
)

HISTORICAL_CLEAN_MEASURE_NUMBERS: Tuple[str, ...] = (
    "none",
    "plain",
    "italic_underlined",
)


def _deterministic_pick(domain: str, name: str, era: str, allowed: Sequence[str]) -> str:
    idx = int(_u01(domain, name, era) * len(allowed)) % len(allowed)
    return str(allowed[idx])


def sanitize_historical_clean_style(
    style: Dict[str, Any],
    *,
    clean_paper: bool = False,
) -> Dict[str, Any]:
    """Clamp & quarantine semantically broken or anachronistic parameters for historical archival realism.

    Returns a new style dict without mutating ``style``.
    """
    out = dict(style or {})
    name = str(out.get("name", "unnamed"))
    era = str(out.get("era", "classical"))

    # 1. Quarantine broken / cartoonish glyph archetypes
    def _enforce(key: str, allowed: Sequence[str], domain: str) -> None:
        val = out.get(key)
        if val not in allowed:
            out[key] = _deterministic_pick(domain, name, era, allowed)

    _enforce("notehead_archetype", HISTORICAL_CLEAN_NOTEHEADS, "hc_nh")
    _enforce("stem_archetype", HISTORICAL_CLEAN_STEMS, "hc_stem")
    _enforce("flag_archetype", HISTORICAL_CLEAN_FLAGS, "hc_flag")
    _enforce("beam_archetype", HISTORICAL_CLEAN_BEAMS, "hc_beam")
    _enforce("beam_slope_style", HISTORICAL_CLEAN_BEAM_SLOPES, "hc_bslope")
    _enforce("beam_grouping_mode", HISTORICAL_CLEAN_BEAM_GROUPINGS, "hc_bgrp")
    _enforce("clef_archetype", HISTORICAL_CLEAN_CLEFS, "hc_clef")
    _enforce("keysig_archetype", HISTORICAL_CLEAN_KEYSIGS, "hc_ksig")
    _enforce("barline_archetype", HISTORICAL_CLEAN_BARLINES, "hc_bar")

    # 2. Enforce solid filled noteheads & readable stroke/stem/beam physics
    # Keep notehead_fill_density >= 1.00 and tame phrase/ink variance so filled noteheads NEVER render hollow
    out["notehead_fill_density"] = float(np.clip(out.get("notehead_fill_density", 1.08), 1.00, 1.25))
    out["phrase_arc_mod"] = float(np.clip(out.get("phrase_arc_mod", 0.08), 0.0, 0.14))
    out["downbeat_weight_boost"] = float(np.clip(out.get("downbeat_weight_boost", 0.08), 0.0, 0.14))
    out["ink_var"] = float(np.clip(out.get("ink_var", 0.10), 0.04, 0.14))
    out["ink_dip_period"] = 0
    out["notehead_aspect"] = float(np.clip(out.get("notehead_aspect", 1.0), 0.80, 1.22))
    out["notehead_scale"] = float(np.clip(out.get("notehead_scale", 1.0), 0.90, 1.15))
    out["hollow_open_prob"] = float(np.clip(out.get("hollow_open_prob", 0.08), 0.0, 0.16))
    out["stroke_scale"] = float(np.clip(out.get("stroke_scale", 1.0), 0.82, 1.28))
    out["stem_length_scale"] = float(np.clip(out.get("stem_length_scale", 1.0), 0.85, 1.15))
    out["beam_thickness_scale"] = float(np.clip(out.get("beam_thickness_scale", 1.05), 0.85, 1.35))
    out["barline_overshoot_sp"] = float(np.clip(out.get("barline_overshoot_sp", 0.12), 0.0, 0.22))

    # Ban fake curly hooks on quarter-note stems & detached beams
    out["ligature_prob"] = 0.0
    out["stem_wrong_side_prob"] = 0.0
    out["stem_center_mix_prob"] = 0.0
    out["beam_junction_gap_prob"] = 0.0
    out["beam_junction_overshoot_sp"] = float(np.clip(out.get("beam_junction_overshoot_sp", 0.05), 0.0, 0.10))
    out["beam_edge_overhang_sp"] = float(np.clip(out.get("beam_edge_overhang_sp", 0.05), 0.0, 0.10))

    # 3. Clamp global italic shear <= 5 deg & tame wobble/tremor/skew
    out["slant_deg"] = float(np.clip(out.get("slant_deg", 2.0), -5.0, 5.0))
    out["wobble_px"] = float(np.clip(out.get("wobble_px", 0.65), 0.15, 1.05))
    out["tremor_px"] = float(np.clip(out.get("tremor_px", 0.10), 0.0, 0.22))
    out["staff_curve_px"] = float(np.clip(out.get("staff_curve_px", 2.0), 0.0, 3.5))
    out["staff_slope_deg"] = float(np.clip(out.get("staff_slope_deg", 0.30), -0.6, 0.6))
    out["sp_jitter_frac"] = float(np.clip(out.get("sp_jitter_frac", 0.02), 0.0, 0.03))
    out["page_skew_drift_px"] = 0.0
    out["ink_dropout"] = float(np.clip(out.get("ink_dropout", 0.06), 0.0, 0.14))
    out["split_nib_prob"] = float(np.clip(out.get("split_nib_prob", 0.04), 0.0, 0.12))

    # 4. Ban anachronistic modern clutter (timecodes, spiral rings, 3-hole punches, red/blue eyeglasses, jazz chords, border boxes)
    _enforce("annotation_persona", HISTORICAL_CLEAN_PERSONAS, "hc_pers")
    out["overlay_density"] = float(np.clip(out.get("overlay_density", 0.18), 0.0, 0.30))
    out["chord_symbols_prob"] = 0.0
    out["chord_symbol_style"] = "none"
    out["scratch_out_prob"] = 0.0
    out["caret_callout_prob"] = 0.0
    out["margin_annotations_prob"] = float(np.clip(out.get("margin_annotations_prob", 0.08), 0.0, 0.12))
    _enforce("page_furniture_mode", HISTORICAL_CLEAN_FURNITURE, "hc_furn")
    _enforce("measure_number_style", HISTORICAL_CLEAN_MEASURE_NUMBERS, "hc_mnum")
    if out["measure_number_style"] == "none":
        out["measure_numbers_prob"] = 0.0
    else:
        out["measure_numbers_prob"] = float(np.clip(out.get("measure_numbers_prob", 0.65), 0.0, 0.80))

    # 5. Historical paper stock, warm pre-ruled rastrum staves & high note-to-staff ink contrast
    _enforce("paper_type", HISTORICAL_CLEAN_PAPER_TYPES, "hc_paper")
    _enforce("paper_binding_mode", HISTORICAL_CLEAN_BINDINGS, "hc_bind")
    _enforce("paper_background_grid", HISTORICAL_CLEAN_GRIDS, "hc_grid")
    out["paper_border_mode"] = "none"
    out["paper_grain"] = float(np.clip(out.get("paper_grain", 0.065), 0.03, 0.10))
    out["paper_mottle"] = float(np.clip(out.get("paper_mottle", 0.07), 0.02, 0.12))
    out["bleed_through"] = float(np.clip(out.get("bleed_through", 0.08), 0.03, 0.14))
    out["verso_bleed_strength"] = 0.0  # handled cleanly by layout_quirks mirrored bleed-through
    out["stain_prob"] = float(np.clip(out.get("stain_prob", 0.02), 0.0, 0.05))
    out["ink_blot_prob"] = float(np.clip(out.get("ink_blot_prob", 0.03), 0.0, 0.08))
    out["palm_smear_prob"] = float(np.clip(out.get("palm_smear_prob", 0.05), 0.0, 0.12))
    out["dog_ear_prob"] = 0.0  # avoid painting corner fold over dark scanner-bed border

    # Darken/warm foreground ink if too pale/saturated so noteheads & beams pop sharply
    ink = np.asarray(out.get("ink_rgb", (0.14, 0.11, 0.09)), dtype=np.float64)
    lum = float(0.299 * ink[0] + 0.587 * ink[1] + 0.114 * ink[2])
    if lum > 0.22:
        ink = ink * (0.18 / max(lum, 1e-4))
    # Pull bright blue/red inks into archival iron-gall / sepia-charcoal range
    if float(np.max(ink) - np.min(ink)) > 0.14:
        ink = 0.35 * ink + 0.65 * np.array([0.14, 0.11, 0.09], dtype=np.float64)
    ink = np.clip(ink, 0.04, 0.24)
    out["ink_rgb"] = tuple(round(float(c), 4) for c in ink)

    # Lighter, warmer-brown pre-ruled rastrum staff lines for strong note-to-staff contrast
    rastrum_warm = np.array([0.58, 0.49, 0.39], dtype=np.float64)
    staff_ink = np.clip(0.25 * ink + 0.75 * rastrum_warm, 0.38, 0.66)
    out["staff_ink_rgb"] = tuple(round(float(c), 4) for c in staff_ink)
    out["staff_ruling_mode"] = "rastrum_coherent_quill"
    out["staff_line_width_ratio"] = 0.068
    out["rastrum_wobble"] = float(np.clip(out.get("rastrum_wobble", 0.85), 0.35, 1.35))

    # Tighter horizontal density (5-8 bars/system) & slightly compact sp + 1-3 unused rastrum staves
    out["sp_scale_mult"] = float(np.clip(out.get("sp_scale_mult", 0.90), 0.82, 0.96))
    out["inter_system_gap_mult"] = float(np.clip(out.get("inter_system_gap_mult", 0.88), 0.78, 0.98))
    out["intra_gap_mult"] = float(np.clip(out.get("intra_gap_mult", 0.90), 0.80, 1.00))
    out["blank_prerule_prob"] = max(float(out.get("blank_prerule_prob", 0.65)), 0.65)
    out["historical_clean"] = True

    if clean_paper:
        out["paper_binding_mode"] = "none"
        out["paper_background_grid"] = "none"
        out["paper_border_mode"] = "none"
        out["scanner_margin_frac"] = 0.0
        out["verso_bleed_strength"] = 0.0
        out["fold_crease_prob"] = 0.0
        out["foxing_cluster_prob"] = 0.0
        out["ink_blot_prob"] = 0.0
        out["palm_smear_prob"] = 0.0
        out["dog_ear_prob"] = 0.0
    else:
        if "scanner_margin_frac" not in out or out["scanner_margin_frac"] is None:
            out["scanner_margin_frac"] = round(0.015 + 0.018 * _u01("hc_scan", name, era), 4)

    return out


def _u01(domain: str, name: str, era: str) -> float:
    digest = hashlib.sha256(f"v03:{domain}:{name}:{era}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def _pick_weighted(domain: str, name: str, era: str, choices: Sequence[Tuple[str, float]]) -> str:
    total = sum(max(0.0, float(w)) for _, w in choices)
    if total <= 0.0:
        return choices[0][0]
    u = _u01(domain, name, era) * total
    acc = 0.0
    for val, w in choices:
        acc += max(0.0, float(w))
        if u <= acc:
            return val
    return choices[-1][0]


def _jitter(base: float, lo: float, hi: float, span: float, domain: str, name: str, era: str) -> float:
    return float(np.clip(base + (_u01(domain, name, era) - 0.5) * span, lo, hi))


def resolve_page_and_expressive_profile(
    style: Dict[str, Any],
    *,
    clean_paper: bool = False,
    historical_clean: bool = False,
) -> Dict[str, Any]:
    """Derive a complete, deterministic Layers 1–4 profile for any style dict.

    Returns a new dict containing all resolved v0.3.0 keys; explicit non-None keys
    present in ``style`` always take precedence (except paper/wear keys when
    ``clean_paper=True``).
    """
    st = style or {}
    name = str(st.get("name", "unnamed"))
    raw_era = str(st.get("era", ""))
    era = classify_era(st)
    medium = classify_medium(st)
    impl = classify_implement(st)
    paper = infer_paper_type(st)
    chaos = float(compute_legibility_score(st))
    tier = int(score_legibility(st))
    text = _blob(st)

    wobble = float(st.get("wobble_px", 0.7))
    stroke = float(st.get("stroke_scale", 1.0))
    slant = float(st.get("slant_deg", 6.0))
    cram = float(st.get("end_bar_cram", 0.15))
    mrg = float(st.get("margin_annotations_prob", 0.15))
    ers = float(st.get("eraser_smudge_prob", 0.05))
    bleed = float(st.get("bleed_through", 0.08))
    rastrum = float(st.get("rastrum_wobble", 0.0))
    bar_over = float(st.get("barline_overshoot_sp", 0.25))

    resolved: Dict[str, Any] = {}

    def _set(key: str, default_val: Any) -> Any:
        if key in st and st[key] is not None:
            resolved[key] = st[key]
        else:
            resolved[key] = default_val
        return resolved[key]

    # ------------------------------------------------------------------
    # Layer 1: Macro Layout, Staff Ruling, Furniture, Paper Borders/Bindings
    # ------------------------------------------------------------------
    if "choirbook" in text or (medium == "choral_liturgical" and era == "renaissance"):
        layout_mode_def = "engraved_first_indent"
    elif "pocket" in text or paper == "spiral_notebook":
        layout_mode_def = "cramped_edge_to_edge"
    elif medium == "autograph_sketch" and tier >= 4:
        layout_mode_def = "ragged_sketch_autograph"
    elif medium in ("fair_copy", "orchestral_conductor"):
        layout_mode_def = _pick_weighted("layout_mode", name, raw_era, [
            ("engraved_first_indent", 0.55), ("recto_verso_gutter", 0.30), ("standard_rect", 0.15),
        ])
    else:
        layout_mode_def = _pick_weighted("layout_mode", name, raw_era, [
            ("standard_rect", 0.40), ("engraved_first_indent", 0.30),
            ("recto_verso_gutter", 0.15), ("ragged_sketch_autograph", 0.15),
        ])
    layout_mode = _set("page_layout_mode", layout_mode_def)

    indent_def = 0.12 if layout_mode == "engraved_first_indent" else (0.06 if medium == "fair_copy" else 0.0)
    _set("first_system_indent_frac", _jitter(indent_def, 0.0, 0.22, 0.04, "indent", name, raw_era) if indent_def > 0 else 0.0)

    if layout_mode == "recto_verso_gutter":
        is_recto = _u01("recto", name, raw_era) < 0.5
        ml_def = 0.115 if is_recto else 0.052
        mr_def = 0.052 if is_recto else 0.115
    elif layout_mode == "cramped_edge_to_edge":
        ml_def = 0.095 if paper == "spiral_notebook" else 0.036
        mr_def = 0.032
    else:
        base_m = 0.088 if medium in ("fair_copy", "choral_liturgical") else 0.065
        ml_def = _jitter(base_m, 0.045, 0.125, 0.022, "ml", name, raw_era)
        mr_def = _jitter(base_m, 0.045, 0.125, 0.022, "mr", name, raw_era)
    _set("margin_left_frac", ml_def)
    _set("margin_right_frac", mr_def)

    rag_def = _jitter(0.11, 0.04, 0.18, 0.06, "rag", name, raw_era) if layout_mode == "ragged_sketch_autograph" else 0.0
    _set("ragged_right_jitter_frac", rag_def)
    _set("final_system_short_frac", 0.85 if layout_mode == "engraved_first_indent" else 1.0)
    _set("page_skew_drift_px", _jitter(0.0, -32.0, 32.0, 50.0, "skew", name, raw_era) if layout_mode == "ragged_sketch_autograph" else 0.0)

    # Vertical rhythm
    if medium == "jazz_lead_sheet":
        vr_def = "jumbo_beginner_primer"
    elif cram >= 0.34 or "pocket" in text:
        vr_def = "crushed_collision_sketch"
    elif medium in ("fair_copy", "orchestral_conductor") and tier <= 2:
        vr_def = "airy_conductor_justified"
    else:
        vr_def = _pick_weighted("vr", name, raw_era, [
            ("uniform", 0.45), ("grouped_choir_versus_piano", 0.25),
            ("progressive_squeeze_panic", 0.15), ("airy_conductor_justified", 0.15),
        ])
    vr = _set("vertical_rhythm_mode", vr_def)
    sp_mult_def = 1.12 if vr == "jumbo_beginner_primer" else (0.90 if vr == "crushed_collision_sketch" else _jitter(1.0, 0.86, 1.16, 0.14, "sp_m", name, raw_era))
    _set("sp_scale_mult", sp_mult_def)
    _set("intra_gap_mult", 1.0)
    _set("inter_system_gap_mult", 1.15 if vr == "airy_conductor_justified" else (0.85 if vr == "crushed_collision_sketch" else 1.0))

    # Staff ruling
    if paper == "spiral_notebook":
        sr_def = "crisp_printed_pre_ruled"
    elif rastrum >= 2.8 or "freehand" in text:
        sr_def = "freehand_rulerless_sketch"
    elif era in ("renaissance", "baroque", "classical") or rastrum >= 1.0:
        sr_def = "rastrum_coherent_quill"
    elif paper in ("ozalid_blue", "soviet_green_tint") or era == "impressionist":
        sr_def = "faint_ghost_staff"
    elif impl == "felt_marker":
        sr_def = "bold_marker_heavy_outer"
    else:
        sr_def = _pick_weighted("sr", name, raw_era, [
            ("crisp_printed_pre_ruled", 0.45), ("rastrum_coherent_quill", 0.35), ("independent_lines", 0.20),
        ])
    sr = _set("staff_ruling_mode", sr_def)
    lr_fade_def = _jitter(0.28, 0.12, 0.45, 0.16, "lrf", name, raw_era) if sr == "rastrum_coherent_quill" else (0.10 if sr == "freehand_rulerless_sketch" else 0.0)
    _set("staff_lr_fade", lr_fade_def)
    swp_def = (
        "alternating_worn_rastrum" if sr == "rastrum_coherent_quill"
        else "outer_heavy" if sr == "bold_marker_heavy_outer"
        else "uniform"
    )
    _set("staff_line_weight_profile", swp_def)
    _set("staff_end_stagger_sp", 0.35 if sr in ("rastrum_coherent_quill", "freehand_rulerless_sketch") else 0.0)
    _set("staff_dry_skip_prob", 0.28 if sr in ("faint_ghost_staff", "freehand_rulerless_sketch") else (0.12 if sr == "rastrum_coherent_quill" else 0.0))

    # Page furniture
    if medium == "jazz_lead_sheet":
        furn_def = "jazz_realbook_chart"
    elif medium == "choral_liturgical" or era in ("renaissance", "baroque"):
        furn_def = "baroque_archival_folio"
    elif medium in ("orchestral_conductor", "film_studio"):
        furn_def = "orchestral_conductor_part"
    elif medium == "student_everyday":
        furn_def = "student_homework_sheet"
    elif medium == "fair_copy":
        furn_def = "classical_fair_copy_header"
    else:
        furn_def = _pick_weighted("furn", name, raw_era, [
            ("classical_fair_copy_header", 0.35), ("minimal_running_folio", 0.35), ("none", 0.30),
        ])
    furn = _set("page_furniture_mode", furn_def)
    mt_def = 0.145 if furn != "none" else 0.075
    _set("margin_top_frac", _jitter(mt_def, 0.065, 0.175, 0.02, "mt", name, raw_era))
    _set("margin_bottom_frac", 0.065)

    # Barlines & Braces
    if era == "renaissance" and medium == "choral_liturgical":
        bspan_def = "mensurstrich_between"
    elif medium in ("orchestral_conductor", "film_studio"):
        bspan_def = "full_system_connected"
    else:
        bspan_def = "grand_staff_connected"
    _set("barline_span_mode", bspan_def)
    _set("left_barline_mode", "all_staves_including_solo" if medium in ("fair_copy", "orchestral_conductor") else "single_multi")

    # Paper borders, bindings, grids
    if paper == "spiral_notebook":
        bind_def, grid_def, border_def = "wire_spiral_left", "college_ruled", "red_margin_vertical"
    elif paper == "ozalid_blue":
        bind_def, grid_def, border_def = "none", "ozalid_cyan_streaks", "pre_ruled_header_box"
    elif paper == "aged_rag":
        bind_def = _pick_weighted("bind_rag", name, raw_era, [("book_gutter_shadow_left", 0.65), ("none", 0.35)])
        grid_def = "heavy_laid_chainlines"
        border_def = "double_engraved_frame" if era in ("renaissance", "baroque") else "none"
    elif paper == "soviet_green_tint":
        bind_def, grid_def, border_def = "book_gutter_shadow_left", "none", "none"
    elif paper == "bright_white":
        bind_def = _pick_weighted("bind_bw", name, raw_era, [("none", 0.65), ("three_hole_punch", 0.35)])
        grid_def, border_def = "none", ("publisher_footer_plate" if medium == "fair_copy" else "none")
    else:  # cream_manuscript
        bind_def = _pick_weighted("bind_cr", name, raw_era, [("none", 0.55), ("book_gutter_shadow_left", 0.25), ("three_hole_punch", 0.20)])
        grid_def, border_def = "none", ("pre_ruled_header_box" if medium == "jazz_lead_sheet" else "none")

    _set("paper_binding_mode", bind_def)
    _set("paper_background_grid", grid_def)
    _set("paper_border_mode", border_def)

    # ------------------------------------------------------------------
    # Layer 2: Expressive Glyph Archetypes (86 Archetypes)
    # ------------------------------------------------------------------
    expr = resolve_expressive_archetypes(st)
    for k, v in expr.items():
        _set(k, v)

    # ------------------------------------------------------------------
    # Layer 3: Living Manuscript Overlays, Corrections & Wear
    # ------------------------------------------------------------------
    if "mahler" in text or "bernstein" in text or "blue_crayon" in text or (medium == "orchestral_conductor" and mrg >= 0.35):
        pers_def = "orchestral_conductor"
    elif medium == "jazz_lead_sheet":
        pers_def = "jazz_realbook_chart"
    elif medium == "film_studio":
        pers_def = "film_studio_copyist"
    elif medium == "choral_liturgical" and era != "renaissance":
        pers_def = "choral_rehearsal"
    elif "violin" in text or "fiddle" in text or "paganini" in text or "abbey_road" in text:
        pers_def = "string_section_pencil"
    elif medium == "repro_archival" or (era in ("renaissance", "baroque") and "choirbook" in text):
        pers_def = "archival_librarian"
    elif medium == "autograph_sketch" and tier >= 4:
        pers_def = "composer_working_draft"
    elif medium == "fair_copy" and tier <= 2:
        pers_def = "none"
    else:
        pers_def = _pick_weighted("persona", name, raw_era, [
            ("none", 0.45), ("piano_conservatory", 0.20), ("composer_working_draft", 0.20), ("archival_librarian", 0.15),
        ])
    persona = _set("annotation_persona", pers_def)
    ov_dens_def = 0.0 if persona == "none" else float(np.clip(0.45 + 0.45 * mrg + 0.15 * chaos, 0.25, 0.95))
    _set("overlay_density", ov_dens_def)

    scratch_def = 0.0 if (medium == "fair_copy" and tier == 1) else float(np.clip(0.18 * chaos + 0.15 * ers + (0.12 if persona == "composer_working_draft" else 0.0), 0.0, 0.55))
    _set("scratch_out_prob", scratch_def)
    _set("cancellation_style", _pick_weighted("cancel", name, raw_era, [
        ("tight_scribble_hatch", 0.22), ("double_diagonal_slash", 0.18), ("heavy_x_cross", 0.16),
        ("wavy_loop_cancel", 0.14), ("vide_cut_bracket", 0.10), ("ink_blot_smear_out", 0.10), ("box_and_strike", 0.10),
    ]))

    is_chord_idiom = medium == "jazz_lead_sheet" or any(k in text for k in ("songwriter", "broadway", "tin_pan_alley", "guitar", "busker"))
    _set("chord_symbols_prob", 0.90 if is_chord_idiom else 0.0)
    chord_st_def = "jazz_realbook" if medium == "jazz_lead_sheet" else ("pop_slash" if is_chord_idiom else "none")
    _set("chord_symbol_style", chord_st_def)

    if medium in ("film_studio", "orchestral_conductor"):
        mn_st_def, mn_p_def = "boxed", 0.95
    elif medium == "jazz_lead_sheet":
        mn_st_def, mn_p_def = "circled", 0.90
    elif era == "renaissance":
        mn_st_def, mn_p_def = "none", 0.0
    else:
        mn_st_def = _pick_weighted("mnum", name, raw_era, [("plain", 0.40), ("italic_underlined", 0.30), ("none", 0.30)])
        mn_p_def = 0.0 if mn_st_def == "none" else 0.85
    _set("measure_number_style", mn_st_def)
    _set("measure_numbers_prob", mn_p_def)

    palm_def = float(np.clip(
        (0.32 if slant <= -3.0 else 0.06) + 0.20 * ers + (0.18 if impl in ("graphite_pencil", "gel_rollerball", "grease_pencil") else 0.0),
        0.0, 0.65,
    ))
    _set("palm_smear_prob", palm_def)
    _set("palm_smear_hand", "left" if slant <= -3.0 else "right")
    _set("ink_blot_prob", float(np.clip(0.35 * chaos if impl in ("quill", "fountain_pen") and tier >= 4 else 0.0, 0.0, 0.45)))
    _set("quill_test_prob", 0.35 if (impl == "quill" and medium in ("autograph_sketch", "choral_liturgical")) else 0.0)
    _set("caret_callout_prob", 0.40 if persona == "composer_working_draft" else 0.0)
    _set("verso_bleed_strength", float(np.clip(1.35 * bleed + (0.08 if paper == "aged_rag" else 0.0), 0.0, 0.45)))
    _set("fold_crease_prob", 0.55 if paper in ("soviet_green_tint", "aged_rag") else 0.0)
    _set("foxing_cluster_prob", 0.55 if paper == "aged_rag" else 0.0)
    _set("dog_ear_prob", 0.55 if paper in ("aged_rag", "soviet_green_tint", "spiral_notebook") else 0.10)

    # ------------------------------------------------------------------
    # Layer 4: Spacing Habits, Cursive Ligatures & OMR-Safe Ductus Physics
    # ------------------------------------------------------------------
    gamma_base = 0.66 if tier <= 2 else (0.55 if tier == 3 else 0.44)
    if era == "impressionist":
        gamma_base += 0.06
    _set("duration_spacing_gamma", _jitter(gamma_base, 0.25, 0.95, 0.14, "gamma", name, raw_era))
    _set("bar_end_panic_start", _jitter(0.68, 0.54, 0.84, 0.12, "pstart", name, raw_era))
    _set("bar_end_panic_strength", float(np.clip(cram * 1.15 + (0.14 if cram >= 0.28 else 0.0), 0.0, 0.80)))
    _set("cross_voice_x_slop_sp", float(np.clip(0.24 * chaos, 0.0, 0.34)))

    if impl in ("technical_pen", "mech_pencil") or era == "renaissance":
        lig_def = 0.0
    else:
        lig_def = float(np.clip(max(0.0, slant - 4.0) / 26.0 + (0.14 if impl in ("quill", "fountain_pen") else 0.0), 0.0, 0.60))
    _set("ligature_prob", lig_def)
    _set("ligature_opacity", 0.36)

    if slant >= 10.0 and chaos >= 0.35:
        dslope_def = 0.13
    elif era in ("renaissance", "baroque"):
        dslope_def = -0.05
    else:
        dslope_def = _jitter(0.02, -0.10, 0.14, 0.08, "dslope", name, raw_era)
    _set("duration_head_scale_slope", dslope_def)
    _set("downbeat_weight_boost", float(np.clip(0.06 + 0.18 * (1.0 if era == "romantic" else 0.4), 0.0, 0.35)))
    _set("phrase_arc_mod", float(np.clip(0.08 + 0.20 * chaos, 0.0, 0.35)))
    _set("stem_len_jitter_frac", float(np.clip(0.03 + 0.16 * chaos, 0.0, 0.24)))
    _set("stem_center_mix_prob", float(np.clip(0.18 * chaos if tier >= 4 else 0.0, 0.0, 0.30)))
    _set("stem_wrong_side_prob", 0.08 if medium == "student_everyday" and tier >= 4 else 0.0)
    _set("beam_junction_overshoot_sp", float(np.clip(0.28 * bar_over + 0.16 * chaos, 0.0, 0.38)))
    _set("beam_junction_gap_prob", float(np.clip(0.24 * chaos if tier >= 4 else 0.0, 0.0, 0.35)))
    _set("beam_edge_overhang_sp", float(np.clip(0.22 * chaos, 0.0, 0.42)))
    # Hard OMR-safe pitch-slop bound: <= 0.145 sp in renderer (+ 0.025 sp in draw_notehead <= 0.17 sp!)
    _set("pitch_y_slop_sp", float(np.clip(0.13 * chaos * chaos + 0.02 * (wobble / 3.0), 0.0, 0.145)))
    _set("chord_x_stagger_sp", float(np.clip(0.18 * chaos, 0.0, 0.24)))
    _set("accidental_x_offset_sp", _jitter(1.25, 1.02, 1.55, 0.24, "accoff", name, raw_era))
    _set("accidental_x_jitter_sp", float(np.clip(0.20 * chaos, 0.0, 0.28)))

    if clean_paper:
        resolved["paper_binding_mode"] = "none"
        resolved["paper_background_grid"] = "none"
        resolved["paper_border_mode"] = "none"
        resolved["scanner_margin_frac"] = 0.0
        resolved["verso_bleed_strength"] = 0.0
        resolved["fold_crease_prob"] = 0.0
        resolved["foxing_cluster_prob"] = 0.0
        resolved["ink_blot_prob"] = 0.0
        resolved["palm_smear_prob"] = 0.0
        resolved["dog_ear_prob"] = 0.0

    if historical_clean:
        resolved = sanitize_historical_clean_style(resolved, clean_paper=clean_paper)

    return resolved
