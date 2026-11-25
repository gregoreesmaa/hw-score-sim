"""Complete MuseScore (.mscx/.mscz) & MusicXML (.musicxml/.mxl) handwritten score renderer.

Integrates `musescore_compat.py`, `styles.py`, `layout_quirks.py`, `stroke_engine.py`,
and `glyph_variants.py` to render full real-world MuseScore and MusicXML notation
(polyphony, ties/slurs, tuplets, articulations, ornaments, grace notes, dynamics,
hairpins, tempo, lyrics, mid-measure clef/key/time changes, 8vb/8va clefs,
double/courtesy accidentals, multi-bar rests, and volta repeat brackets) into any
registered handwritten scribe style, with companion Bezier staff-curve GT JSON and
optional cropped per-system PNG + MusicXML training export.
"""

from __future__ import annotations

from functools import lru_cache
import json
import math
import os
import random
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image

from . import expressive_glyphs, glyph_variants, layout_quirks, manuscript_marks, profile_resolver, stroke_engine, styles
from .musescore_compat import (
    NoteEvent,
    ParsedScore,
    RestEvent,
    ScoreEvent,
    export_score_to_musicxml,
    parse_score,
)


# ---------------------------------------------------------------------------
# Page & System Layout
# ---------------------------------------------------------------------------


@dataclass
class StaffLayout:
    """Geometry for a single 5-line staff within a system."""

    part_index: int
    part_name: str
    brace_group: Optional[int]
    x_left: float
    x_right: float
    y_center: float
    sp: float
    ctrl_pts: List[List[float]] = field(default_factory=list)


@dataclass
class SystemLayout:
    """One system (horizontal row of 1..N staves) covering measures [m_start, m_end)."""

    system_index: int
    measure_start: int
    measure_end: int
    x_left: float
    x_right: float
    bar_x_bounds: List[Tuple[float, float]]  # (x0, x1) for each measure in system
    staves: List[StaffLayout] = field(default_factory=list)
    is_blank_prerule: bool = False


def layout_score_pages(
    score: ParsedScore,
    page_w: int = 1650,
    page_h: int = 2200,
    style: Optional[Dict[str, Any]] = None,
    rng: Optional[random.Random] = None,
    max_parts: int = 6,
) -> List[SystemLayout]:
    """Wrap score measures into systems that fit on page 1."""
    style = style or {}
    rng = rng or random.Random(42)

    parts = score.parts[:max_parts]
    n_staves = max(1, len(parts))
    total_measures = score.num_measures

    sp_base = float(style.get("sp", style.get("staff_space", 16.0)))
    if n_staves >= 5:
        sp = min(sp_base, 10.5)
    elif n_staves == 4:
        sp = min(sp_base, 12.0)
    elif n_staves == 3:
        sp = min(sp_base, 13.5)
    elif n_staves == 2:
        sp = min(sp_base, 15.0)
    else:
        sp = sp_base

    sp_scale_mult = float(np.clip(style.get("sp_scale_mult", 1.0), 0.58, 1.55))
    sp = float(np.clip(sp * sp_scale_mult, 8.0, 26.0))

    default_mx_frac = float(style.get("margin_x", page_w * 0.065)) / max(1.0, float(page_w))
    ml_frac = float(np.clip(style.get("margin_left_frac", default_mx_frac), 0.025, 0.20))
    mr_frac = float(np.clip(style.get("margin_right_frac", default_mx_frac), 0.025, 0.22))
    mt_frac = float(np.clip(style.get("margin_top_frac", float(style.get("margin_top", page_h * 0.075)) / max(1.0, float(page_h))), 0.03, 0.28))
    mb_frac = float(np.clip(style.get("margin_bottom_frac", float(style.get("margin_bottom", page_h * 0.065)) / max(1.0, float(page_h))), 0.03, 0.22))

    # Reserve left clearance for binding hardware (3-hole punch / spiral / gutter)
    bind_mode = str(style.get("paper_binding_mode") or "none").strip().lower()
    if bind_mode in ("three_hole_punch", "wire_spiral_left", "book_gutter_shadow_left"):
        ml_frac = max(ml_frac, 0.088)
    furn_mode = str(style.get("page_furniture_mode") or "none").strip().lower()
    if furn_mode != "none":
        mt_frac = max(mt_frac, 0.135)
    # Reserve clearance for optional dark scanner-bed margins
    scan_frac = float(np.clip(style.get("scanner_margin_frac", 0.0), 0.0, 0.05))
    if scan_frac >= 0.005:
        pad_s = scan_frac * 1.45
        ml_frac = max(ml_frac, pad_s + 0.040)
        mr_frac = max(mr_frac, pad_s + 0.038)
        mt_frac = max(mt_frac, pad_s + 0.045)
        mb_frac = max(mb_frac, pad_s + 0.040)

    x_left_base = page_w * ml_frac
    x_right_base = page_w * (1.0 - mr_frac)
    print_w = max(200.0, x_right_base - x_left_base)
    margin_top = page_h * mt_frac
    margin_bottom = page_h * mb_frac

    layout_mode = str(style.get("page_layout_mode") or "standard_rect").strip().lower()
    first_indent_frac = float(np.clip(style.get("first_system_indent_frac", 0.0), 0.0, 0.28))
    rag_jitter_frac = float(np.clip(style.get("ragged_right_jitter_frac", 0.0), 0.0, 0.22))
    final_short_frac = float(np.clip(style.get("final_system_short_frac", 1.0), 0.32, 1.0))
    skew_drift_px = float(np.clip(style.get("page_skew_drift_px", 0.0), -60.0, 60.0))

    staff_h = 4.0 * sp
    has_lyrics = any(
        isinstance(ev, NoteEvent) and bool(ev.lyrics)
        for p in parts
        for m in p.measures
        for ev in m.events
    )
    intra_gap_mult = float(np.clip(style.get("intra_gap_mult", 1.0), 0.52, 1.85))
    inter_gap_mult = float(np.clip(style.get("inter_system_gap_mult", 1.0), 0.42, 2.20))
    intra_gap = ((7.2 if has_lyrics else 6.5) * sp * intra_gap_mult) if n_staves > 1 else 0.0
    inter_system_gap = ((8.2 if has_lyrics else 7.5) * sp if n_staves == 1 else 6.2 * sp) * inter_gap_mult
    system_h = n_staves * staff_h + max(0, n_staves - 1) * intra_gap

    # Ensure 5-6 staff systems still fit cleanly inside available vertical page height (e.g. landscape).
    avail_h = max(120.0, float(page_h) - margin_top - margin_bottom)
    if n_staves > 1 and system_h > avail_h * 0.94:
        max_gap = max(3.6 * sp, (avail_h * 0.94 - n_staves * staff_h) / max(1, n_staves - 1))
        intra_gap = min(intra_gap, max_gap)
        system_h = n_staves * staff_h + (n_staves - 1) * intra_gap
        if system_h > avail_h * 0.94:
            shrink = (avail_h * 0.94) / max(1.0, system_h)
            sp = max(7.0, sp * shrink)
            staff_h = 4.0 * sp
            intra_gap = max(3.6 * sp, intra_gap * shrink)
            inter_system_gap = max(4.5 * sp, inter_system_gap * shrink)
            system_h = n_staves * staff_h + (n_staves - 1) * intra_gap

    is_hist_clean = bool(style.get("historical_clean", False))
    if is_hist_clean:
        bars_per_sys_default = 6 if n_staves >= 2 else 7
    else:
        bars_per_sys_default = 4 if n_staves >= 2 else 5
    bar_width_var = float(style.get("bar_width_var", 0.12))
    prerule_prob = float(style.get("blank_prerule_prob", 0.0))
    est_systems = max(2, int(math.ceil(max(1, total_measures) / float(bars_per_sys_default))))

    systems: List[SystemLayout] = []
    y_cursor = margin_top + staff_h * 0.5
    m_cursor = 0
    sys_idx = 0

    # In historical_clean mode, plan 1-3 unused pre-ruled rastrum staves (top / inter-system / bottom)
    hc_target_unused = rng.randint(1, 3) if is_hist_clean else 0
    hc_placed_unused = 0
    blank_step_h = staff_h + max(4.2 * sp, inter_system_gap * 0.72)
    # Reserve vertical space at bottom for at least 1 unused rastrum staff when page height allows
    hc_bottom_reserve = (inter_system_gap + staff_h) if (is_hist_clean and avail_h >= system_h + inter_system_gap + staff_h * 1.5) else 0.0
    hc_inter_after_sys = rng.randint(0, 1) if (is_hist_clean and hc_target_unused >= 2) else -1

    def _append_blank_staff() -> bool:
        nonlocal sys_idx, y_cursor, hc_placed_unused
        if y_cursor + staff_h * 0.5 > page_h - margin_bottom:
            return False
        staves_b = [
            StaffLayout(
                part_index=0,
                part_name="blank",
                brace_group=None,
                x_left=x_left_base,
                x_right=x_right_base,
                y_center=y_cursor,
                sp=sp,
            )
        ]
        systems.append(
            SystemLayout(
                system_index=sys_idx,
                measure_start=0,
                measure_end=0,
                x_left=x_left_base,
                x_right=x_right_base,
                bar_x_bounds=[],
                staves=staves_b,
                is_blank_prerule=True,
            )
        )
        sys_idx += 1
        hc_placed_unused += 1
        y_cursor += blank_step_h
        return True

    # Optional unused pre-ruled rastrum staff at top of page in historical_clean mode
    if (
        is_hist_clean
        and hc_target_unused >= 2
        and rng.random() < 0.45
        and avail_h >= system_h + blank_step_h * 2.2
    ):
        _append_blank_staff()

    active_sys_count = 0
    while y_cursor + system_h - staff_h * 0.5 <= page_h - margin_bottom - (hc_bottom_reserve if hc_placed_unused == 0 else 0.0):
        prog = sys_idx / float(max(1, est_systems - 1))
        drift = skew_drift_px * (prog - 0.5)
        indent = print_w * first_indent_frac if active_sys_count == 0 else 0.0
        left_jit = print_w * rng.uniform(-0.018, 0.018) if layout_mode == "ragged_sketch_autograph" else 0.0
        right_pull = print_w * rng.uniform(0.0, rag_jitter_frac) if rag_jitter_frac > 0.0 else 0.0
        x_left = max(16.0, x_left_base + indent + drift + left_jit)
        x_right = min(float(page_w - 16), x_right_base + drift - right_pull)
        if x_right <= x_left + 160.0:
            x_left, x_right = x_left_base, x_right_base

        if m_cursor < total_measures:
            delta_choices = [-1, 0, 0, 1] if (total_measures - m_cursor > bars_per_sys_default + 1) else [0]
            bars_this_sys = min(
                total_measures - m_cursor,
                bars_per_sys_default + rng.choice(delta_choices),
            )
            bars_this_sys = max(1, bars_this_sys)
            m_start = m_cursor
            m_end = m_cursor + bars_this_sys
            m_cursor = m_end

            if m_end >= total_measures and final_short_frac < 0.99 and bars_this_sys < bars_per_sys_default:
                x_right = x_left + max(0.32, final_short_frac) * (x_right - x_left)

            weights: List[float] = []
            for m_i in range(m_start, m_end):
                max_ev = 4
                for p in parts:
                    if m_i < len(p.measures):
                        max_ev = max(max_ev, len(p.measures[m_i].events))
                density_w = 0.65 + 0.12 * min(max_ev, 16)
                jitter = 1.0 + rng.uniform(-bar_width_var, bar_width_var)
                prefix_bonus = 0.35 if m_i == m_start else 0.0
                weights.append(max(0.4, (density_w + prefix_bonus) * jitter))

            w_sum = sum(weights)
            bar_bounds: List[Tuple[float, float]] = []
            bx = x_left
            avail_w = x_right - x_left
            for w in weights:
                bw = avail_w * (w / w_sum)
                bar_bounds.append((bx, bx + bw))
                bx += bw

            staves: List[StaffLayout] = []
            for s_i, p in enumerate(parts):
                yc = y_cursor + s_i * (staff_h + intra_gap)
                staves.append(
                    StaffLayout(
                        part_index=s_i,
                        part_name=p.name,
                        brace_group=p.brace_group,
                        x_left=x_left,
                        x_right=x_right,
                        y_center=yc,
                        sp=sp,
                    )
                )

            systems.append(
                SystemLayout(
                    system_index=sys_idx,
                    measure_start=m_start,
                    measure_end=m_end,
                    x_left=x_left,
                    x_right=x_right,
                    bar_x_bounds=bar_bounds,
                    staves=staves,
                    is_blank_prerule=False,
                )
            )
            sys_idx += 1
            active_sys_count += 1
            y_cursor += system_h + inter_system_gap

            # Optional inter-system unused pre-ruled staff in historical_clean mode
            if (
                is_hist_clean
                and hc_placed_unused < hc_target_unused - 1
                and active_sys_count - 1 == hc_inter_after_sys
                and m_cursor < total_measures
                and y_cursor + blank_step_h + system_h - staff_h * 0.5 <= page_h - margin_bottom
            ):
                _append_blank_staff()
        else:
            if is_hist_clean:
                if hc_placed_unused < hc_target_unused:
                    if not _append_blank_staff():
                        break
                else:
                    break
            elif prerule_prob > 0.0 and rng.random() < prerule_prob:
                staves = [
                    StaffLayout(
                        part_index=0,
                        part_name="blank",
                        brace_group=None,
                        x_left=x_left,
                        x_right=x_right,
                        y_center=y_cursor,
                        sp=sp,
                    )
                ]
                systems.append(
                    SystemLayout(
                        system_index=sys_idx,
                        measure_start=0,
                        measure_end=0,
                        x_left=x_left,
                        x_right=x_right,
                        bar_x_bounds=[],
                        staves=staves,
                        is_blank_prerule=True,
                    )
                )
                sys_idx += 1
                y_cursor += staff_h + inter_system_gap
            else:
                break

    # Ensure at least 1 (up to hc_target_unused) unused rastrum staff at bottom in historical_clean mode
    if is_hist_clean:
        while hc_placed_unused < hc_target_unused:
            if not _append_blank_staff():
                break

    return systems


# ---------------------------------------------------------------------------
# Adaptive Helpers for Engine Calls
# ---------------------------------------------------------------------------


def _eval_cubic_bezier_y(ctrl_pts: Sequence[Sequence[float]], x: float) -> float:
    if not ctrl_pts:
        return 0.0
    x0 = float(ctrl_pts[0][0])
    x3 = float(ctrl_pts[-1][0])
    if abs(x3 - x0) < 1e-6:
        return float(ctrl_pts[0][1])
    t = max(0.0, min(1.0, (x - x0) / (x3 - x0)))
    if len(ctrl_pts) == 4:
        mt = 1.0 - t
        return (
            (mt ** 3) * float(ctrl_pts[0][1])
            + 3.0 * (mt ** 2) * t * float(ctrl_pts[1][1])
            + 3.0 * mt * (t ** 2) * float(ctrl_pts[2][1])
            + (t ** 3) * float(ctrl_pts[3][1])
        )
    return (1.0 - t) * float(ctrl_pts[0][1]) + t * float(ctrl_pts[-1][1])


def _get_staff_curve(
    layout_mod: Any,
    x_left: float,
    x_right: float,
    y_center: float,
    sp: float,
    style: Dict[str, Any],
    np_rng: np.random.Generator,
) -> Tuple[Callable[[Any, int], Any], List[List[float]], float]:
    fn = getattr(layout_mod, "build_curved_staff_y", None)
    if fn is not None:
        res = fn(
            x0=x_left,
            x1=x_right,
            y_center=y_center,
            sp=sp,
            style=style,
            rng=np_rng,
        )
        if isinstance(res, tuple) and len(res) >= 3:
            y_fn, ctrl, sp_out = res[0], res[1], float(res[2])
            ctrl_list = [[float(pt[0]), float(pt[1])] for pt in ctrl]
            return y_fn, ctrl_list, sp_out
        if isinstance(res, tuple) and len(res) == 2:
            y_fn_1, ctrl = res[0], res[1]
            ctrl_list = [[float(pt[0]), float(pt[1])] for pt in ctrl]
            return (lambda x, li=0, f=y_fn_1, s=sp: f(x) + li * s), ctrl_list, sp

    sag = float(style.get("staff_curve_px", 4.0)) * float(np_rng.uniform(-1.0, 1.0))
    tilt = float(style.get("staff_slope_deg", 0.5)) * 4.0 * float(np_rng.uniform(-1.0, 1.0))
    w = x_right - x_left
    ctrl_list = [
        [x_left, y_center - tilt * 0.5],
        [x_left + w / 3.0, y_center + sag - tilt * 0.15],
        [x_left + 2.0 * w / 3.0, y_center + sag + tilt * 0.15],
        [x_right, y_center + tilt * 0.5],
    ]
    return (lambda x, li=0, c=ctrl_list, s=sp: _eval_cubic_bezier_y(c, float(x)) + li * s), ctrl_list, sp


def _compute_beat_x_map(
    layout_mod: Any,
    x_start: float,
    x_end: float,
    sp: float,
    events: List[ScoreEvent],
    extra_beats: Sequence[float],
    style: Dict[str, Any],
    np_rng: np.random.Generator,
) -> Tuple[List[float], Callable[[float], float]]:
    """Compute horizontal x positions aligned by beat_offset across all voices.

    Simultaneous voices at the same beat_offset share the same base x-position,
    while grace notes are tucked immediately left of their following main note.
    Returns (event_xs, beat_to_x_fn).
    """
    n = len(events)
    if x_end <= x_start + 4.0:
        x_end = x_start + 40.0

    # Collect unique non-grace beat positions (quantized to 1/96 beat)
    beat_keys: List[int] = sorted(
        {
            int(round(ev.beat_offset * 96.0))
            for ev in events
            if not (isinstance(ev, NoteEvent) and ev.is_grace)
        }
        | {int(round(b * 96.0)) for b in extra_beats}
    )
    if not beat_keys:
        beat_keys = [0]
    unique_beats = [k / 96.0 for k in beat_keys]

    fn = getattr(layout_mod, "compute_quirky_note_positions", None)
    beat_xs: Optional[List[float]] = None
    if fn is not None and x_end > x_start + 2.0:
        try:
            durs = [max(0.125, unique_beats[i + 1] - unique_beats[i]) for i in range(len(unique_beats) - 1)]
            durs.append(durs[-1] if durs else 1.0)
            xs = fn(
                beat_offsets=unique_beats,
                bar_x0=x_start,
                bar_x1=x_end,
                sp=sp,
                style=style,
                rng=np_rng,
                durations=durs,
            )
            if xs is not None and len(xs) == len(unique_beats):
                beat_xs = [float(x) for x in xs]
        except Exception:
            beat_xs = None

    if beat_xs is None:
        pad_l = (x_end - x_start) * 0.08
        pad_r = (x_end - x_start) * 0.08
        usable = max(8.0, (x_end - x_start) - pad_l - pad_r)
        b_min = unique_beats[0]
        b_span = max(unique_beats[-1] - b_min, 1e-3)
        beat_xs = [
            x_start + pad_l + ((b - b_min) / b_span if len(unique_beats) > 1 else 0.25) * usable
            for b in unique_beats
        ]

    key_to_x = {k: x for k, x in zip(beat_keys, beat_xs)}

    def beat_to_x(b: float) -> float:
        k = int(round(b * 96.0))
        if k in key_to_x:
            return key_to_x[k]
        if len(unique_beats) == 1:
            return beat_xs[0]
        if b <= unique_beats[0]:
            return beat_xs[0]
        if b >= unique_beats[-1]:
            return min(x_end - sp * 0.3, beat_xs[-1] + sp * 0.8)
        # Piecewise linear interpolation
        for i in range(len(unique_beats) - 1):
            if unique_beats[i] <= b <= unique_beats[i + 1]:
                t = (b - unique_beats[i]) / max(1e-6, unique_beats[i + 1] - unique_beats[i])
                return beat_xs[i] + t * (beat_xs[i + 1] - beat_xs[i])
        return beat_xs[-1]

    # Count consecutive grace notes preceding each beat so they space out cleanly
    grace_run_total: Dict[int, int] = {}
    grace_run_pos: Dict[int, int] = {}
    i = 0
    while i < n:
        ev = events[i]
        if isinstance(ev, NoteEvent) and ev.is_grace:
            j = i
            while j < n and isinstance(events[j], NoteEvent) and events[j].is_grace:
                j += 1
            count = j - i
            for k in range(i, j):
                grace_run_total[k] = count
                grace_run_pos[k] = k - i
            i = j
        else:
            i += 1

    ev_xs: List[float] = []
    voices_set = {getattr(ev, "voice", 1) for ev in events}
    cv_slop = float(np.clip(styles.resolve_ductus_param(style, "cross_voice_x_slop_sp", 0.0), 0.0, 0.42)) if len(voices_set) >= 2 else 0.0
    prev_x_by_voice: Dict[int, float] = {}
    for idx, ev in enumerate(events):
        base_x = beat_to_x(ev.beat_offset)
        if isinstance(ev, NoteEvent) and ev.is_grace:
            tot = grace_run_total.get(idx, 1)
            pos = grace_run_pos.get(idx, 0)
            gx = base_x - sp * (1.05 * (tot - pos))
            ev_xs.append(max(x_start + sp * 0.4, gx))
        else:
            nudge = getattr(ev, "x_nudge_sp", 0.0) * sp
            v = int(getattr(ev, "voice", 1))
            slop = float(np.clip(np_rng.normal(0.0, cv_slop * sp), -0.42 * sp, 0.42 * sp)) if (cv_slop > 0.0 and v > 1) else 0.0
            cand = min(x_end - sp * 0.3, max(x_start + sp * 0.4, base_x + nudge + slop))
            if v in prev_x_by_voice and cand < prev_x_by_voice[v] + 0.65 * sp:
                cand = min(x_end - sp * 0.3, prev_x_by_voice[v] + 0.65 * sp)
            prev_x_by_voice[v] = cand
            ev_xs.append(cand)

    return ev_xs, beat_to_x


# ---------------------------------------------------------------------------
# Local Calligraphic Primitives
# ---------------------------------------------------------------------------


@lru_cache(maxsize=64)
def _cubic_basis(n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    t = np.linspace(0.0, 1.0, n)[:, None]
    mt = 1.0 - t
    return mt ** 3, 3.0 * (mt ** 2) * t, 3.0 * mt * (t ** 2), t ** 3


@lru_cache(maxsize=64)
def _quad_basis(n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = np.linspace(0.0, 1.0, n)[:, None]
    mt = 1.0 - t
    return mt ** 2, 2.0 * mt * t, t ** 2


def _eval_cubic(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, n: int = 20) -> np.ndarray:
    b0, b1, b2, b3 = _cubic_basis(n)
    return b0 * p0 + b1 * p1 + b2 * p2 + b3 * p3


def _eval_quad(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, n: int = 14) -> np.ndarray:
    b0, b1, b2 = _quad_basis(n)
    return b0 * p0 + b1 * p1 + b2 * p2


def _draw_ledger_lines(
    canvas: Any,
    stroke_mod: Any,
    x: float,
    y_center: float,
    sp: float,
    steps: List[int],
    ink_rgb: Any,
    style: Dict[str, Any],
    rng: Any,
    glyph_mod: Any = None,
) -> None:
    if not steps:
        return
    if glyph_mod is not None and hasattr(glyph_mod, "draw_ledger_lines"):
        np_r = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(int(rng.randrange(0, 2**31)))
        glyph_mod.draw_ledger_lines(
            canvas=canvas,
            x=x,
            y_center=y_center,
            sp=sp,
            steps=steps,
            color_rgb=ink_rgb,
            style=style,
            rng=np_r,
        )
        return
    min_s = min(steps)
    max_s = max(steps)
    half_w = sp * 0.95
    draw_poly = getattr(stroke_mod, "draw_calligraphic_polyline", None)
    if draw_poly is None:
        return

    ledger_steps: List[int] = []
    if min_s <= -2:
        ledger_steps.extend(range(-2, min_s - 1, -2))
    if max_s >= 10:
        ledger_steps.extend(range(10, max_s + 1, 2))

    w_px = max(1.0, sp * 0.09 * float(style.get("stroke_scale", 1.0)))
    for ls in ledger_steps:
        ly = y_center - (ls - 4) * (sp * 0.5)
        pts = np.asarray(
            [
                [x - half_w, ly + rng.uniform(-0.5, 0.5)],
                [x, ly + rng.uniform(-0.4, 0.4)],
                [x + half_w, ly + rng.uniform(-0.5, 0.5)],
            ],
            dtype=np.float32,
        )
        draw_poly(
            canvas=canvas,
            pts=pts,
            color_rgb=ink_rgb,
            base_width=w_px,
            style=style,
            kind="ledger",
        )


def _draw_slur_or_tie(
    canvas: Any,
    stroke_mod: Any,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    arch_dir: int,  # -1 = arch upward (above), +1 = arch downward (below)
    sp: float,
    ink_rgb: Any,
    style: Dict[str, Any],
    np_rng: np.random.Generator,
    is_tie: bool = False,
) -> None:
    """Draw a tapered calligraphic hand-drawn slur or tie Bezier arc (10 archetypes)."""
    expressive_glyphs.draw_slur_variant(
        canvas, x0, y0, x1, y1, arch_dir, sp, ink_rgb, style, np_rng, is_tie=is_tie
    )


def _draw_double_or_courtesy_accidental(
    canvas: Any,
    stroke_mod: Any,
    glyph_mod: Any,
    acc: str,
    cx: float,
    cy: float,
    sp: float,
    ink_rgb: Any,
    style: Dict[str, Any],
    np_rng: np.random.Generator,
    courtesy: bool = False,
) -> None:
    """Draw sharp/flat/natural/double_sharp/double_flat with optional courtesy parentheses."""
    glyph_mod.draw_accidental(canvas, acc, cx, cy, sp, ink_rgb, style, np_rng)
    if courtesy:
        expressive_glyphs.draw_courtesy_brackets_variant(
            canvas, cx, cy, sp, acc == "double_flat", ink_rgb, style, np_rng
        )


def _draw_clef_with_octave(
    canvas: Any,
    glyph_mod: Any,
    clef_name: str,
    cx: float,
    cy: float,
    sp: float,
    ink_rgb: Any,
    style: Dict[str, Any],
    np_rng: np.random.Generator,
) -> None:
    """Draw clef plus small hand-drawn '8' below (8vb) or above (8va) when applicable."""
    glyph_mod.draw_clef(
        canvas=canvas,
        clef_name=clef_name,
        cx=cx,
        cy=cy,
        sp=sp,
        color_rgb=ink_rgb,
        style=style,
        rng=np_rng,
    )


def _draw_articulation(
    canvas: Any,
    stroke_mod: Any,
    art: str,
    cx: float,
    cy: float,
    above: bool,
    sp: float,
    ink_rgb: Any,
    style: Dict[str, Any],
    np_rng: np.random.Generator,
) -> None:
    """Draw staccato, accent, tenuto, fermata, trill, staccatissimo, or marcato (8 archetypes)."""
    expressive_glyphs.draw_articulation_variant(canvas, art, cx, cy, above, sp, ink_rgb, style, np_rng)


def _draw_tuplet_bracket(
    canvas: Any,
    stroke_mod: Any,
    glyph_mod: Any,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    num: int,
    above: bool,
    sp: float,
    ink_rgb: Any,
    style: Dict[str, Any],
    np_rng: np.random.Generator,
) -> None:
    """Draw a hand-drawn tuplet bracket with centered handwritten numeral (6 archetypes)."""
    expressive_glyphs.draw_tuplet_variant(canvas, x0, y0, x1, y1, num, above, sp, ink_rgb, style, np_rng)


def _draw_hairpin(
    canvas: Any,
    stroke_mod: Any,
    kind: str,  # 'crescendo' (<) or 'diminuendo' (>)
    x0: float,
    x1: float,
    yc: float,
    sp: float,
    ink_rgb: Any,
    style: Dict[str, Any],
    np_rng: Optional[np.random.Generator] = None,
) -> None:
    """Draw a hand-drawn crescendo (<) or diminuendo (>) hairpin wedge (8 archetypes)."""
    rng_eff = np_rng if np_rng is not None else getattr(canvas, "rng", np.random.default_rng(0))
    expressive_glyphs.draw_hairpin_variant(canvas, kind, x0, x1, yc, sp, ink_rgb, style, rng_eff)


def _draw_multi_bar_rest(
    canvas: Any,
    stroke_mod: Any,
    glyph_mod: Any,
    x0: float,
    x1: float,
    yc: float,
    count: int,
    sp: float,
    ink_rgb: Any,
    style: Dict[str, Any],
    np_rng: np.random.Generator,
) -> None:
    """Draw a multi-measure rest bar with end serifs and centered count numeral (5 archetypes)."""
    expressive_glyphs.draw_multibar_rest_variant(canvas, x0, x1, yc, count, sp, ink_rgb, style, np_rng)


def _draw_volta_bracket(
    canvas: Any,
    stroke_mod: Any,
    x0: float,
    x1: float,
    y_top: float,
    label: str,
    draw_left_hook: bool,
    draw_right_hook: bool,
    sp: float,
    ink_rgb: Any,
    style: Dict[str, Any],
    np_rng: np.random.Generator,
) -> None:
    """Draw a volta repeat bracket ('1.', '2.') above the top staff (5 archetypes)."""
    expressive_glyphs.draw_volta_variant(
        canvas, x0, x1, y_top, label, draw_left_hook, draw_right_hook, sp, ink_rgb, style, np_rng,
        text_fn=lambda s, tx, ty, sz: _draw_handwritten_text(canvas, stroke_mod, s, tx, ty, sz, ink_rgb, style, np_rng),
    )


# Compact vector stroke glyphs (normalized x in [0, 0.7], y in [-0.7, 0.4], baseline at 0)
_STROKE_FONT: Dict[str, List[List[Tuple[float, float]]]] = {
    "a": [[(0.50, -0.30), (0.20, -0.38), (0.05, -0.15), (0.20, 0.02), (0.50, -0.15)], [(0.52, -0.35), (0.50, 0.02)]],
    "b": [[(0.10, -0.78), (0.08, 0.02)], [(0.09, -0.28), (0.42, -0.36), (0.52, -0.14), (0.36, 0.02), (0.08, -0.02)]],
    "c": [[(0.50, -0.30), (0.22, -0.38), (0.06, -0.16), (0.22, 0.02), (0.52, -0.04)]],
    "d": [[(0.50, -0.78), (0.48, 0.02)], [(0.49, -0.30), (0.18, -0.36), (0.06, -0.14), (0.22, 0.02), (0.48, -0.04)]],
    "e": [[(0.08, -0.18), (0.50, -0.22), (0.42, -0.38), (0.14, -0.34), (0.08, -0.12), (0.26, 0.02), (0.52, -0.04)]],
    "f": [[(0.52, -0.74), (0.32, -0.82), (0.24, -0.58), (0.18, 0.24)], [(0.05, -0.34), (0.46, -0.36)]],
    "g": [[(0.50, -0.32), (0.20, -0.38), (0.06, -0.16), (0.22, 0.00), (0.48, -0.18)], [(0.50, -0.34), (0.46, 0.24), (0.24, 0.36), (0.06, 0.26)]],
    "h": [[(0.10, -0.78), (0.08, 0.02)], [(0.09, -0.22), (0.34, -0.38), (0.50, -0.24), (0.50, 0.02)]],
    "i": [[(0.24, -0.36), (0.22, 0.02)], [(0.25, -0.56), (0.27, -0.54)]],
    "j": [[(0.32, -0.36), (0.28, 0.24), (0.10, 0.34)], [(0.34, -0.56), (0.36, -0.54)]],
    "k": [[(0.10, -0.78), (0.08, 0.02)], [(0.48, -0.38), (0.10, -0.14), (0.50, 0.02)]],
    "l": [[(0.24, -0.78), (0.22, 0.00), (0.34, 0.02)]],
    "m": [[(0.04, -0.36), (0.04, 0.02)], [(0.04, -0.24), (0.22, -0.38), (0.34, -0.24), (0.34, 0.02)], [(0.34, -0.24), (0.52, -0.38), (0.64, -0.24), (0.64, 0.02)]],
    "n": [[(0.08, -0.36), (0.08, 0.02)], [(0.08, -0.24), (0.32, -0.38), (0.50, -0.24), (0.50, 0.02)]],
    "o": [[(0.28, -0.38), (0.06, -0.22), (0.12, 0.00), (0.36, 0.02), (0.52, -0.18), (0.28, -0.38)]],
    "p": [[(0.12, -0.36), (0.06, 0.36)], [(0.12, -0.30), (0.44, -0.38), (0.54, -0.16), (0.36, 0.02), (0.10, -0.02)]],
    "q": [[(0.48, -0.34), (0.44, 0.36)], [(0.48, -0.30), (0.18, -0.38), (0.06, -0.16), (0.22, 0.02), (0.46, -0.04)]],
    "r": [[(0.10, -0.36), (0.08, 0.02)], [(0.10, -0.22), (0.30, -0.38), (0.46, -0.32)]],
    "s": [[(0.48, -0.32), (0.22, -0.38), (0.10, -0.24), (0.44, -0.12), (0.42, 0.02), (0.10, 0.00)]],
    "t": [[(0.26, -0.68), (0.24, -0.04), (0.42, 0.02)], [(0.06, -0.38), (0.48, -0.40)]],
    "u": [[(0.08, -0.36), (0.08, -0.06), (0.26, 0.02), (0.48, -0.10)], [(0.50, -0.36), (0.48, 0.02)]],
    "v": [[(0.06, -0.36), (0.28, 0.02), (0.52, -0.36)]],
    "w": [[(0.04, -0.36), (0.18, 0.02), (0.34, -0.24), (0.50, 0.02), (0.64, -0.36)]],
    "x": [[(0.08, -0.36), (0.48, 0.02)], [(0.48, -0.36), (0.08, 0.02)]],
    "y": [[(0.08, -0.36), (0.28, -0.02)], [(0.50, -0.36), (0.26, 0.04), (0.10, 0.34)]],
    "z": [[(0.08, -0.36), (0.50, -0.36), (0.08, 0.02), (0.52, 0.02)]],
    "0": [[(0.28, -0.72), (0.08, -0.36), (0.28, 0.02), (0.48, -0.36), (0.28, -0.72)]],
    "1": [[(0.14, -0.54), (0.30, -0.72), (0.28, 0.02)]],
    "2": [[(0.10, -0.56), (0.30, -0.72), (0.48, -0.54), (0.08, 0.02), (0.52, 0.02)]],
    "3": [[(0.10, -0.64), (0.46, -0.68), (0.26, -0.36), (0.48, -0.14), (0.10, 0.00)]],
    "4": [[(0.42, -0.72), (0.08, -0.24), (0.54, -0.24)], [(0.42, -0.72), (0.40, 0.02)]],
    "5": [[(0.48, -0.70), (0.14, -0.70), (0.12, -0.38), (0.46, -0.32), (0.42, -0.04), (0.10, 0.00)]],
    "6": [[(0.44, -0.70), (0.14, -0.44), (0.10, -0.06), (0.36, 0.02), (0.48, -0.22), (0.14, -0.28)]],
    "7": [[(0.08, -0.70), (0.52, -0.70), (0.20, 0.02)]],
    "8": [[(0.28, -0.72), (0.10, -0.52), (0.46, -0.18), (0.28, 0.02), (0.10, -0.18), (0.46, -0.52), (0.28, -0.72)]],
    "9": [[(0.46, -0.48), (0.24, -0.72), (0.10, -0.48), (0.46, -0.42), (0.34, 0.02)]],
    ".": [[(0.18, -0.02), (0.22, 0.02)]],
    ",": [[(0.22, -0.02), (0.14, 0.14)]],
    "-": [[(0.08, -0.20), (0.44, -0.20)]],
    "=": [[(0.06, -0.32), (0.48, -0.32)], [(0.06, -0.14), (0.48, -0.14)]],
    "'": [[(0.22, -0.72), (0.18, -0.52)]],
    "!": [[(0.24, -0.72), (0.24, -0.18)], [(0.24, -0.02), (0.25, 0.02)]],
}


def _draw_handwritten_text(
    canvas: Any,
    stroke_mod: Any,
    text: str,
    x: float,
    y_baseline: float,
    size_px: float,
    ink_rgb: Any,
    style: Dict[str, Any],
    np_rng: np.random.Generator,
    italic: bool = False,
    bold: bool = False,
) -> float:
    """Render handwritten text in matching ink, slant, and calligraphic stroke physics."""
    if not text:
        return x
    slant_deg = float(style.get("slant_deg", 6.0)) + (12.0 if italic else 0.0)
    shear = math.tan(math.radians(float(np.clip(slant_deg, -24.0, 28.0))))
    w = max(0.85, size_px * (0.145 if bold else 0.105))
    cursor = x

    local_style = dict(style)
    local_style["wobble_px"] = min(float(style.get("wobble_px", 0.7)), size_px * 0.06)
    local_style["tremor_px"] = min(float(style.get("tremor_px", 0.2)), size_px * 0.03)

    for ch in text:
        if ch == " ":
            cursor += size_px * 0.42
            continue
        is_upper = ch.isupper()
        key = ch.lower()
        strokes = _STROKE_FONT.get(key)
        if strokes is None:
            cursor += size_px * 0.35
            continue
        scale_y = 1.32 if (is_upper and key.isalpha()) else 1.0
        scale_x = 1.15 if (is_upper and key.isalpha()) else 1.0
        adv = size_px * (0.72 if key in ("m", "w") else 0.58) * scale_x
        for st in strokes:
            pts_list: List[List[float]] = []
            for px, py in st:
                ly = py * size_px * scale_y
                lx = px * size_px * scale_x - ly * shear
                pts_list.append([cursor + lx, y_baseline + ly])
            pts = np.asarray(pts_list, dtype=np.float64)
            stroke_mod.draw_calligraphic_polyline(canvas, pts, ink_rgb, w, local_style, kind="text")
        cursor += adv + float(np_rng.normal(0.0, size_px * 0.02))
    return cursor


def _draw_dynamic_marking(
    canvas: Any,
    stroke_mod: Any,
    text: str,
    cx: float,
    cy: float,
    sp: float,
    ink_rgb: Any,
    style: Dict[str, Any],
    np_rng: np.random.Generator,
) -> None:
    """Draw calligraphic italic dynamics (8 archetypes via expressive_glyphs)."""
    expressive_glyphs.draw_dynamic_variant(canvas, text, cx, cy, sp, ink_rgb, style, np_rng)


def _modulate_event_style(
    base_style: Dict[str, Any],
    beat_offset: float,
    rel_m: int,
    n_meas_in_sys: int,
    u_in_bar: float,
    *,
    db: Optional[float] = None,
    pa: Optional[float] = None,
    base_stroke_scale: Optional[float] = None,
    base_fill_density: Optional[float] = None,
    base_opacity: Optional[float] = None,
    scratch: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Apply subtle metric downbeat weight boost & phrase-arc pressure modulation."""
    if db is None:
        raw_db = float(styles.resolve_ductus_param(base_style, "downbeat_weight_boost", 0.0))
        db = 0.0 if raw_db < 0.0 else (0.42 if raw_db > 0.42 else raw_db)
    if pa is None:
        raw_pa = float(styles.resolve_ductus_param(base_style, "phrase_arc_mod", 0.0))
        pa = 0.0 if raw_pa < 0.0 else (0.40 if raw_pa > 0.40 else raw_pa)
    if db <= 1e-4 and pa <= 1e-4:
        return base_style
    ev_style = scratch if scratch is not None else dict(base_style)
    is_downbeat = (beat_offset % 4.0) < 0.08
    sys_u = (rel_m + u_in_bar) / max(1.0, float(n_meas_in_sys))
    if sys_u < 0.0:
        sys_u = 0.0
    elif sys_u > 1.0:
        sys_u = 1.0
    arc = math.sin(math.pi * sys_u) - 0.45
    pressure = 1.0 + (db if is_downbeat else -0.15 * db * u_in_bar) + pa * arc
    bs = float(base_style.get("stroke_scale", 1.0)) if base_stroke_scale is None else base_stroke_scale
    bf = float(base_style.get("notehead_fill_density", 1.0)) if base_fill_density is None else base_fill_density
    bo = float(base_style.get("stroke_opacity", base_style.get("opacity", 0.92))) if base_opacity is None else base_opacity
    ss = bs * pressure
    ev_style["stroke_scale"] = 0.45 if ss < 0.45 else (2.20 if ss > 2.20 else ss)
    fd = bf * pressure
    ev_style["notehead_fill_density"] = 0.25 if fd < 0.25 else (1.50 if fd > 1.50 else fd)
    op = bo * (0.94 + 0.08 * pressure)
    ev_style["stroke_opacity"] = 0.25 if op < 0.25 else (1.0 if op > 1.0 else op)
    return ev_style


def _draw_page_furniture(
    canvas: Any,
    score: ParsedScore,
    systems: Sequence[SystemLayout],
    style: Dict[str, Any],
    np_rng: np.random.Generator,
) -> None:
    """Draw optional top-margin title header, composer/folio metadata, and part labels."""
    mode = str(style.get("page_furniture_mode") or "none").strip().lower()
    if mode == "none" or not systems:
        return
    ink_rgb = style.get("ink_rgb", (0.14, 0.12, 0.10))
    page_w, page_h = canvas.width, canvas.height
    title = (score.title or "Etude in D Major").strip()
    if len(title) > 36:
        title = title[:36].strip()
    top_y = max(48.0, min(float(systems[0].staves[0].y_center) - 5.6 * float(systems[0].staves[0].sp), page_h * 0.075))
    sp0 = float(systems[0].staves[0].sp)

    if mode == "jazz_realbook_chart":
        t_up = title.upper()
        sz = sp0 * 2.1
        est_w = len(t_up) * sz * 0.56
        tx = max(page_w * 0.08, 0.5 * (page_w - est_w))
        end_x = _draw_handwritten_text(canvas, stroke_engine, t_up, tx, top_y, sz, ink_rgb, style, np_rng, bold=True)
        stroke_engine.draw_calligraphic_polyline(
            canvas, np.array([[tx, top_y + sp0 * 0.45], [end_x, top_y + sp0 * 0.40]], dtype=np.float64),
            ink_rgb, max(1.8, sp0 * 0.22), style, kind="annotation",
        )
        _draw_handwritten_text(canvas, stroke_engine, "MEDIUM SWING", page_w * 0.74, top_y + sp0 * 1.4, sp0 * 0.95, ink_rgb, style, np_rng, bold=True)
    elif mode == "baroque_archival_folio":
        sz = sp0 * 1.65
        est_w = len(title) * sz * 0.52
        tx = max(page_w * 0.12, 0.5 * (page_w - est_w))
        _draw_handwritten_text(canvas, stroke_engine, title, tx, top_y, sz, ink_rgb, style, np_rng, italic=True)
        _draw_handwritten_text(canvas, stroke_engine, "fol. 4r", page_w * 0.86, top_y - sp0 * 0.2, sp0 * 0.95, ink_rgb, style, np_rng, italic=True)
    elif mode == "orchestral_conductor_part":
        _draw_handwritten_text(canvas, stroke_engine, "FULL SCORE", page_w * 0.08, top_y, sp0 * 1.4, ink_rgb, style, np_rng, bold=True)
        _draw_handwritten_text(canvas, stroke_engine, title, page_w * 0.36, top_y, sp0 * 1.5, ink_rgb, style, np_rng, bold=True)
        _draw_handwritten_text(canvas, stroke_engine, "1", page_w * 0.90, top_y, sp0 * 1.2, ink_rgb, style, np_rng, bold=True)
    elif mode == "student_homework_sheet":
        _draw_handwritten_text(canvas, stroke_engine, "Ex. 4", page_w * 0.11, top_y, sp0 * 1.35, ink_rgb, style, np_rng)
        _draw_handwritten_text(canvas, stroke_engine, "10/14", page_w * 0.84, top_y, sp0 * 1.15, ink_rgb, style, np_rng)
    elif mode == "classical_fair_copy_header":
        sz = sp0 * 1.75
        est_w = len(title) * sz * 0.52
        tx = max(page_w * 0.10, 0.5 * (page_w - est_w))
        end_x = _draw_handwritten_text(canvas, stroke_engine, title, tx, top_y, sz, ink_rgb, style, np_rng, italic=True)
        for dy in (0.35 * sp0, 0.58 * sp0):
            stroke_engine.draw_calligraphic_polyline(
                canvas, np.array([[tx, top_y + dy], [end_x, top_y + dy]], dtype=np.float64),
                ink_rgb, max(0.85, sp0 * 0.09), style, kind="annotation",
            )
        _draw_handwritten_text(canvas, stroke_engine, "Op. 27", page_w * 0.84, top_y + sp0 * 1.2, sp0 * 1.0, ink_rgb, style, np_rng, italic=True)
    elif mode == "minimal_running_folio":
        _draw_handwritten_text(canvas, stroke_engine, "- 3 -", page_w * 0.47, top_y, sp0 * 1.1, ink_rgb, style, np_rng)


# ---------------------------------------------------------------------------
# Per-System Training Export (`--export-train-dir`)
# ---------------------------------------------------------------------------


def _find_assembly_serialize_bin() -> Optional[Path]:
    """Locate an optional `assembly-serialize` binary via env var or PATH for bekern export."""
    env_bin = os.environ.get("ASSEMBLY_SERIALIZE_BIN")
    if env_bin:
        p = Path(env_bin)
        if p.exists():
            return p
    on_path = shutil.which("assembly-serialize")
    if on_path:
        return Path(on_path)
    return None


def _export_training_systems(
    score: ParsedScore,
    score_path: Path,
    final_img: Image.Image,
    systems: List[SystemLayout],
    export_dir: Path,
    stem: str,
    max_parts: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Export cropped per-system PNGs and per-system MusicXML targets (plus optional bekern)."""
    export_dir.mkdir(parents=True, exist_ok=True)
    bin_path = _find_assembly_serialize_bin()
    page_w, page_h = final_img.size

    exported: List[Dict[str, Any]] = []
    for sys_lay in systems:
        if sys_lay.is_blank_prerule or not sys_lay.staves:
            continue
        sp = max(st.sp for st in sys_lay.staves)
        all_cy = [pt[1] for st in sys_lay.staves for pt in (st.ctrl_pts or [[0.0, st.y_center]])]
        top_y = min(all_cy) - 5.4 * sp
        bot_y = max(all_cy) + 5.4 * sp
        left_x = max(0, int(math.floor(min(st.x_left for st in sys_lay.staves) - 3.0 * sp)))
        right_x = min(page_w, int(math.ceil(max(st.x_right for st in sys_lay.staves) + 1.8 * sp)))
        y0 = max(0, int(math.floor(top_y)))
        y1 = min(page_h, int(math.ceil(bot_y)))

        sys_stem = f"{stem}_sys{sys_lay.system_index:02d}"
        crop_png = export_dir / f"{sys_stem}.png"
        final_img.crop((left_x, y0, right_x, y1)).save(crop_png, compress_level=1)

        m1 = sys_lay.measure_start + 1
        m2 = sys_lay.measure_end
        sys_mxml = export_score_to_musicxml(
            score,
            export_dir / f"{sys_stem}.musicxml",
            measure_range=(sys_lay.measure_start, sys_lay.measure_end),
            max_parts=max_parts,
        )
        meas_arg = f"1..{max(1, m2 - m1 + 1)}"

        bekern_path = export_dir / f"{sys_stem}-bekern.txt"
        bekern_ok = False
        bekern_err: Optional[str] = None

        if bin_path is not None and sys_mxml.exists():
            cmd = [
                str(bin_path),
                "bekern",
                "--input",
                str(sys_mxml),
                "--measures",
                meas_arg,
                "--out",
                str(export_dir),
                "--stem",
                sys_stem,
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if proc.returncode == 0 and bekern_path.exists():
                bekern_ok = True
            else:
                bekern_err = (proc.stderr or proc.stdout or "").strip()[:300]

        exported.append(
            {
                "system_index": sys_lay.system_index,
                "measures_1based": [m1, m2],
                "crop_box": [left_x, y0, right_x, y1],
                "image": str(crop_png),
                "musicxml": str(sys_mxml),
                "bekern": str(bekern_path) if bekern_ok else None,
                "bekern_ok": bekern_ok,
                "bekern_error": bekern_err,
            }
        )

    manifest_path = export_dir / f"{stem}_train_manifest.json"
    manifest_path.write_text(
        json.dumps({"stem": stem, "source": str(score_path), "systems": exported}, indent=2),
        encoding="utf-8",
    )
    return exported


def plan_measure_beam_groups(
    events: Sequence[ScoreEvent],
    style: Dict[str, Any],
    rng: np.random.Generator,
) -> List[Optional[int]]:
    """Transform a measure's beam-group IDs according to ``beam_grouping_mode`` and ``beam_split_to_flag_prob``.

    Returns a list parallel to ``events`` where each entry is either an integer group ID
    (for beamed notes, guaranteed to have >= 2 notes) or ``None`` (for individual flagged
    or non-beamable notes). Does not mutate ``events``.
    """
    mode = styles.resolve_beam_grouping_mode(style)
    split_prob = float(np.clip(style.get("beam_split_to_flag_prob", 0.0), 0.0, 1.0))

    n = len(events)
    if mode == "all_flags_vocal":
        return [None] * n

    beamable_kinds = ("eighth", "16th", "32nd")
    eff: List[Optional[int]] = [
        ev.beam_group if (isinstance(ev, NoteEvent) and not ev.is_grace and ev.duration_kind in beamable_kinds) else None
        for ev in events
    ]

    next_gid = 1
    for g in eff:
        if g is not None and g >= next_gid:
            next_gid = g + 1

    def _collect_groups(arr: List[Optional[int]]) -> List[List[int]]:
        by_id: Dict[int, List[int]] = {}
        order: List[int] = []
        for idx, g in enumerate(arr):
            if g is not None:
                if g not in by_id:
                    by_id[g] = []
                    order.append(g)
                by_id[g].append(idx)
        return [by_id[g] for g in order]

    def _assign_chunks(out_arr: List[Optional[int]], chunks: List[List[int]]) -> None:
        nonlocal next_gid
        for ch in chunks:
            if len(ch) >= 2:
                gid = next_gid
                next_gid += 1
                for idx in ch:
                    out_arr[idx] = gid
            else:
                for idx in ch:
                    out_arr[idx] = None

    if mode == "long_phrase_beam":
        # Merge consecutive beamable notes per voice into sweeping 5-8 note phrase beams.
        new_eff: List[Optional[int]] = [None] * n
        voices = sorted({ev.voice for ev in events if isinstance(ev, (NoteEvent, RestEvent))})
        for v in voices:
            run: List[int] = []
            for idx, ev in enumerate(events):
                if getattr(ev, "voice", 1) != v:
                    continue
                if isinstance(ev, NoteEvent) and not ev.is_grace and ev.duration_kind in beamable_kinds:
                    run.append(idx)
                    if len(run) >= 8:
                        _assign_chunks(new_eff, [run])
                        run = []
                else:
                    if run:
                        _assign_chunks(new_eff, [run])
                        run = []
            if run:
                _assign_chunks(new_eff, [run])
        eff = new_eff

    elif mode == "pair_beams_2":
        # Split 4+ note groups into 2+2 pairs (trailing singleton becomes flagged).
        new_eff = [None] * n
        for grp in _collect_groups(eff):
            if len(grp) <= 2:
                _assign_chunks(new_eff, [grp])
            else:
                chunks = [grp[i : i + 2] for i in range(0, len(grp), 2)]
                _assign_chunks(new_eff, chunks)
        eff = new_eff

    elif mode == "uneven_3_plus_1":
        # Asymmetric 3+1 / 1+3 beamed + flagged splits.
        new_eff = [None] * n
        for g_ord, grp in enumerate(_collect_groups(eff)):
            L = len(grp)
            if L >= 4:
                chunks = []
                pos = 0
                flip = g_ord % 2
                while pos < L:
                    rem = L - pos
                    if rem >= 4:
                        if flip == 0:
                            chunks.append(grp[pos : pos + 3])
                            chunks.append(grp[pos + 3 : pos + 4])
                        else:
                            chunks.append(grp[pos : pos + 1])
                            chunks.append(grp[pos + 1 : pos + 4])
                        pos += 4
                        flip = 1 - flip
                    elif rem == 3:
                        chunks.append(grp[pos : pos + 2] if flip == 0 else grp[pos : pos + 1])
                        chunks.append(grp[pos + 2 : pos + 3] if flip == 0 else grp[pos + 1 : pos + 3])
                        pos += 3
                    else:
                        chunks.append(grp[pos:])
                        pos = L
                _assign_chunks(new_eff, chunks)
            elif L == 3:
                chunks = [grp[:2], grp[2:]] if (g_ord % 2 == 0) else [grp[:1], grp[1:]]
                _assign_chunks(new_eff, chunks)
            else:
                _assign_chunks(new_eff, [grp])
        eff = new_eff

    # Apply stochastic (seed-deterministic) beam-to-flag splitting for mixed_split_flags
    # or whenever explicit beam_split_to_flag_prob > 0.
    p_eff = (split_prob if split_prob > 0.0 else 0.45) if mode == "mixed_split_flags" else split_prob
    if p_eff > 0.0:
        new_eff = [None] * n
        for grp in _collect_groups(eff):
            if len(grp) < 2:
                continue
            chunks: List[List[int]] = []
            cur: List[int] = []
            for idx in grp:
                if float(rng.random()) < p_eff * 0.58:
                    # Detach this note as a standalone flagged note
                    if cur:
                        chunks.append(cur)
                        cur = []
                    chunks.append([idx])
                else:
                    if cur and float(rng.random()) < p_eff * 0.48:
                        chunks.append(cur)
                        cur = [idx]
                    else:
                        cur.append(idx)
            if cur:
                chunks.append(cur)
            _assign_chunks(new_eff, chunks)
        eff = new_eff

    return eff


# ---------------------------------------------------------------------------
# Main Handwritten Score Renderer
# ---------------------------------------------------------------------------


def render_score_handwritten(
    score_path: Union[str, Path],
    style_name: Union[str, Dict[str, Any]],
    out_png: Union[str, Path],
    seed: int = 42,
    page_w: int = 1650,
    page_h: int = 2200,
    export_train_dir: Optional[Union[str, Path]] = None,
    clean_paper: bool = False,
    max_parts: int = 6,
    historical_clean: bool = False,
    omit_cont_header_prob: Optional[float] = None,
) -> Dict[str, Any]:
    """Render a MusicXML / MuseScore file in the given handwritten style.

    Saves `out_png` and a companion Bezier staff-curve GT JSON at `Path(out_png).with_suffix('.json')`.
    If `export_train_dir` is provided, also exports cropped per-system PNGs and per-system
    MusicXML targets (plus optional bekern output when an external converter is available).
    When `clean_paper=True`, renders pure handwritten ink directly onto flat white `[1.0, 1.0, 1.0]`
    paper, skipping manuscript paper synthesis, pre-ink wear layers, and wet-ink/wear accidents.
    When `historical_clean=True`, activates the curated historical manuscript mode: quarantines
    semantically broken/cartoonish glyph archetypes and modern clutter, clamps italic shear <= 5 deg,
    enforces solid filled noteheads and high ink-to-staff contrast with lighter warm-brown rastrum
    staves, 1-3 unused pre-ruled staves, tighter 5-8 measures/system density, and archival paper.
    """
    score = parse_score(score_path)
    raw_style = styles.get_style(style_name) if hasattr(styles, "get_style") else styles.SCRIBE_STYLES[style_name]
    display_style_name = style_name if isinstance(style_name, str) else str(raw_style.get("name", "custom_style"))
    eff_hist_clean = bool(historical_clean or raw_style.get("historical_clean", False))
    # Merge deterministic v0.3.0 Layers 1–4 profile (explicit keys in raw_style always win)
    style = dict(
        profile_resolver.resolve_page_and_expressive_profile(
            raw_style, clean_paper=clean_paper, historical_clean=eff_hist_clean
        )
    )
    style.update(raw_style)
    if eff_hist_clean:
        style = profile_resolver.sanitize_historical_clean_style(style, clean_paper=clean_paper)
    elif clean_paper:
        style["paper_binding_mode"] = "none"
        style["paper_background_grid"] = "none"
        style["paper_border_mode"] = "none"
        style["scanner_margin_frac"] = 0.0
        style["verso_bleed_strength"] = 0.0
        style["fold_crease_prob"] = 0.0
        style["foxing_cluster_prob"] = 0.0
        style["ink_blot_prob"] = 0.0
        style["palm_smear_prob"] = 0.0
        style["dog_ear_prob"] = 0.0
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    ink_rgb = style.get("ink_rgb", (0.12, 0.10, 0.08))
    staff_ink_rgb = style.get("staff_ink_rgb", ink_rgb)
    paper_rgb = (1.0, 1.0, 1.0) if clean_paper else style.get("paper_rgb", (0.95, 0.92, 0.86))
    stroke_scale = float(style.get("stroke_scale", 1.0))

    # 1 & 2. Manuscript paper background + StrokeCanvas initialization
    if clean_paper:
        canvas = stroke_engine.StrokeCanvas(
            width=page_w,
            height=page_h,
            bg_rgb=(1.0, 1.0, 1.0),
            rng=np_rng,
        )
    else:
        paper_arr = layout_quirks.make_manuscript_paper(page_w, page_h, style, np_rng)
        if isinstance(paper_arr, Image.Image):
            paper_f32 = np.asarray(paper_arr.convert("RGB"), dtype=np.float32) / 255.0
        else:
            paper_f32 = np.asarray(paper_arr, dtype=np.float32)
            if float(np.max(paper_f32)) > 1.0 + 1e-3:
                paper_f32 = paper_f32 / 255.0
                np.clip(paper_f32, 0.0, 1.0, out=paper_f32)
        canvas = stroke_engine.StrokeCanvas(
            width=page_w,
            height=page_h,
            bg_rgb=paper_rgb,
            rng=np_rng,
            init_buf=paper_f32 if paper_f32.shape == (page_h, page_w, 3) else None,
        )
        if canvas.buf is not paper_f32 and hasattr(canvas, "buf") and canvas.buf.shape == paper_f32.shape:
            canvas.buf[:] = paper_f32

    # 2b. Pre-ink living-manuscript layers (mirrored verso bleed, fold crease, foxing clusters)
    manuscript_marks_log: List[Dict[str, Any]] = []
    if not clean_paper:
        manuscript_marks.apply_pre_ink_manuscript_layers(canvas, style, np_rng, manuscript_marks_log)

    # Hoist page-invariant ductus & beam parameters out of measure/note loops.
    page_beam_grouping_mode = styles.resolve_beam_grouping_mode(style)
    page_pitch_slop = float(np.clip(styles.resolve_ductus_param(style, "pitch_y_slop_sp", 0.0), 0.0, 0.145))
    page_chord_stag = float(np.clip(styles.resolve_ductus_param(style, "chord_x_stagger_sp", 0.0), 0.0, 0.28))
    page_acc_off_sp = float(np.clip(styles.resolve_ductus_param(style, "accidental_x_offset_sp", 1.25), 0.92, 1.85))
    page_acc_jit_sp = float(np.clip(styles.resolve_ductus_param(style, "accidental_x_jitter_sp", 0.0), 0.0, 0.34))
    page_lig_prob = float(np.clip(styles.resolve_ductus_param(style, "ligature_prob", 0.0), 0.0, 0.85))
    page_center_p = float(np.clip(styles.resolve_ductus_param(style, "stem_center_mix_prob", 0.0), 0.0, 0.45))
    page_wrong_p = float(np.clip(styles.resolve_ductus_param(style, "stem_wrong_side_prob", 0.0), 0.0, 0.16))
    page_db_boost = float(np.clip(styles.resolve_ductus_param(style, "downbeat_weight_boost", 0.0), 0.0, 0.42))
    page_phrase_arc = float(np.clip(styles.resolve_ductus_param(style, "phrase_arc_mod", 0.0), 0.0, 0.40))
    page_base_fill = float(style.get("notehead_fill_density", 1.0))
    page_base_op = float(style.get("stroke_opacity", style.get("opacity", 0.92)))
    ev_style_scratch = dict(style)

    # 3. Compute page systems & curved staff lines
    systems = layout_score_pages(score, page_w=page_w, page_h=page_h, style=style, rng=rng, max_parts=max_parts)
    _draw_page_furniture(canvas, score, systems, style, np_rng)
    gt_systems: List[Dict[str, Any]] = []

    # 18th/19th-century manuscript realism: optionally omit redundant opening clef & key signature
    # on Systems 2+ (active_sys_idx >= 1) when unchanged from System 0, keeping exact GT unchanged.
    if omit_cont_header_prob is not None:
        eff_omit_cont_prob = float(np.clip(omit_cont_header_prob, 0.0, 1.0))
    else:
        eff_omit_cont_prob = float(
            np.clip(style.get("omit_cont_header_prob", 0.35 if eff_hist_clean else 0.0), 0.0, 1.0)
        )
    omit_cont_header_active = bool(
        eff_omit_cont_prob > 0.0
        and float(np.random.default_rng(int(seed) ^ 0x4F6D6974).random()) < eff_omit_cont_prob
    )
    sys0_part_header: Dict[int, Tuple[str, int]] = {}
    active_sys_idx = 0

    for sys_lay in systems:
        sys_gt_staves: List[Dict[str, Any]] = []
        staff_curves: List[Tuple[Callable[[Any, int], Any], List[List[float]], float]] = []

        # Draw 5-line curved staves (rastrum coherent or independent lines)
        ruling_mode = str(style.get("staff_ruling_mode") or "independent_lines").strip().lower()
        use_rastrum = (ruling_mode != "independent_lines") or (float(style.get("rastrum_wobble", 0.0)) > 0.0)
        staff_lw_ratio = float(np.clip(style.get("staff_line_width_ratio", 0.085), 0.045, 0.165))
        for st_lay in sys_lay.staves:
            y_fn_2, ctrl_pts, sp_eff = _get_staff_curve(
                layout_quirks,
                st_lay.x_left,
                st_lay.x_right,
                st_lay.y_center,
                st_lay.sp,
                style,
                np_rng,
            )
            st_lay.sp = sp_eff
            st_lay.ctrl_pts = ctrl_pts
            staff_curves.append((y_fn_2, ctrl_pts, sp_eff))

            staff_lw = max(1.0, sp_eff * staff_lw_ratio * stroke_scale)
            if use_rastrum:
                stroke_engine.draw_rastrum_staff(
                    canvas=canvas,
                    x0=st_lay.x_left,
                    x1=st_lay.x_right,
                    y_fn=y_fn_2,
                    color_rgb=staff_ink_rgb,
                    base_width=staff_lw,
                    style=style,
                )
            else:
                for line_idx in range(-2, 3):
                    stroke_engine.draw_staff_line(
                        canvas=canvas,
                        x0=st_lay.x_left,
                        x1=st_lay.x_right,
                        y_fn=(lambda x, fn=y_fn_2, li=line_idx: fn(x, li)),
                        color_rgb=staff_ink_rgb,
                        base_width=staff_lw,
                        style=style,
                    )

            sys_gt_staves.append(
                {
                    "part_index": st_lay.part_index,
                    "part_name": st_lay.part_name,
                    "sp": round(float(sp_eff), 3),
                    "x_left": round(float(st_lay.x_left), 2),
                    "x_right": round(float(st_lay.x_right), 2),
                    "y_center": round(float(st_lay.y_center), 2),
                    "ctrl_pts": [[round(c[0], 2), round(c[1], 2)] for c in ctrl_pts],
                    "is_blank_prerule": sys_lay.is_blank_prerule,
                }
            )

        gt_systems.append(
            {
                "system_index": sys_lay.system_index,
                "measure_start": sys_lay.measure_start,
                "measure_end": sys_lay.measure_end,
                "is_blank_prerule": sys_lay.is_blank_prerule,
                "staves": sys_gt_staves,
            }
        )

        if sys_lay.is_blank_prerule:
            continue

        cur_active_sys_idx = active_sys_idx
        active_sys_idx += 1

        # Draw system-left connecting barline & grand-staff braces
        if len(sys_lay.staves) >= 2:
            top_st = sys_lay.staves[0]
            y_top = float(staff_curves[0][0](sys_lay.x_left, -2))
            y_bot = float(staff_curves[-1][0](sys_lay.x_left, 2))
            glyph_variants.draw_barline(
                canvas=canvas,
                x=sys_lay.x_left,
                y_top=y_top,
                y_bot=y_bot,
                sp=top_st.sp,
                kind="single",
                color_rgb=ink_rgb,
                style=style,
                rng=np_rng,
            )

            seen_groups: Dict[int, List[int]] = {}
            for idx, st_lay in enumerate(sys_lay.staves):
                if st_lay.brace_group is not None:
                    seen_groups.setdefault(st_lay.brace_group, []).append(idx)
            for idxs in seen_groups.values():
                if len(idxs) >= 2:
                    i0, i1 = idxs[0], idxs[-1]
                    by0 = float(staff_curves[i0][0](sys_lay.x_left, -2))
                    by1 = float(staff_curves[i1][0](sys_lay.x_left, 2))
                    glyph_variants.draw_brace(
                        canvas=canvas,
                        x=sys_lay.x_left - sys_lay.staves[i0].sp * 0.7,
                        y_top=by0,
                        y_bot=by1,
                        sp=sys_lay.staves[i0].sp,
                        color_rgb=ink_rgb,
                        style=style,
                        rng=np_rng,
                    )

        # Render each staff's measures in this system
        for s_idx, st_lay in enumerate(sys_lay.staves):
            part = score.parts[st_lay.part_index]
            y_fn_2, _, sp = staff_curves[s_idx]

            if cur_active_sys_idx == 0 and sys_lay.measure_start < len(part.measures):
                _m0 = part.measures[sys_lay.measure_start]
                sys0_part_header[st_lay.part_index] = (_m0.clef, int(_m0.key_fifths))

            def y_mid(x_val: float, _fn: Callable[[Any, int], Any] = y_fn_2) -> float:
                return float(_fn(x_val, 0))

            # Track rendered note coordinates in this system for ties/slurs/hairpins
            # (m_idx, ev_idx) -> (ex, {step: ny}, stem_dir, min_y, max_y)
            rendered_notes: Dict[Tuple[int, int], Tuple[float, Dict[int, float], int, float, float]] = {}
            measure_beat_to_x: Dict[int, Callable[[float], float]] = {}

            for rel_m, m_idx in enumerate(range(sys_lay.measure_start, sys_lay.measure_end)):
                if m_idx >= len(part.measures):
                    continue
                meas = part.measures[m_idx]
                bx0, bx1 = sys_lay.bar_x_bounds[rel_m]
                cursor_x = bx0 + sp * 0.7

                # Left repeat barline if present
                if meas.left_barline == "repeat_left":
                    glyph_variants.draw_barline(
                        canvas=canvas,
                        x=bx0 + sp * 0.15,
                        y_top=float(y_fn_2(bx0, -2)),
                        y_bot=float(y_fn_2(bx0, 2)),
                        sp=sp,
                        kind="repeat-left",
                        color_rgb=ink_rgb,
                        style=style,
                        rng=np_rng,
                    )
                    cursor_x += sp * 1.1

                sys0_hdr = sys0_part_header.get(st_lay.part_index)
                omit_cont_header_here = (
                    omit_cont_header_active
                    and cur_active_sys_idx >= 1
                    and rel_m == 0
                    and not meas.clef_change
                    and not meas.key_change
                    and sys0_hdr is not None
                    and (meas.clef, int(meas.key_fifths)) == sys0_hdr
                )

                # Clef at start of system or on clef change
                if (rel_m == 0 or meas.clef_change) and not omit_cont_header_here:
                    clef_x = cursor_x + sp * 0.9
                    _draw_clef_with_octave(
                        canvas=canvas,
                        glyph_mod=glyph_variants,
                        clef_name=meas.clef,
                        cx=clef_x,
                        cy=y_mid(clef_x),
                        sp=sp,
                        ink_rgb=ink_rgb,
                        style=style,
                        np_rng=np_rng,
                    )
                    cursor_x += sp * 2.5

                # Key signature at start of system or on key change
                if (rel_m == 0 or meas.key_change) and meas.key_fifths != 0 and not omit_cont_header_here:
                    end_x = glyph_variants.draw_key_sig(
                        canvas=canvas,
                        x_start=cursor_x,
                        y_lines=y_fn_2,
                        sp=sp,
                        fifths=meas.key_fifths,
                        clef_kind=meas.clef,
                        color_rgb=ink_rgb,
                        style=style,
                        rng=np_rng,
                    )
                    cursor_x = end_x + sp * 0.7

                # Time signature on measure 0 or on time change
                if m_idx == 0 or meas.time_change:
                    tx = cursor_x + sp * 0.8
                    glyph_variants.draw_time_sig(
                        canvas=canvas,
                        num=meas.time_sig[0],
                        den=meas.time_sig[1],
                        cx=tx,
                        cy=y_mid(tx),
                        sp=sp,
                        color_rgb=ink_rgb,
                        style=style,
                        rng=np_rng,
                    )
                    cursor_x += sp * 2.1

                # Volta bracket on top staff of system
                if s_idx == 0 and meas.volta is not None:
                    vb = meas.volta
                    vy = float(y_fn_2(0.5 * (bx0 + bx1), -2)) - sp * 3.1
                    _draw_volta_bracket(
                        canvas=canvas,
                        stroke_mod=stroke_engine,
                        x0=bx0 + sp * 0.15,
                        x1=bx1 - sp * 0.15,
                        y_top=vy,
                        label=vb.text,
                        draw_left_hook=(m_idx == vb.start_measure),
                        draw_right_hook=(m_idx == vb.end_measure and vb.closed_right),
                        sp=sp,
                        ink_rgb=ink_rgb,
                        style=style,
                        np_rng=np_rng,
                    )

                # Multi-bar rest shortcut
                if meas.multi_bar_rest > 1:
                    _draw_multi_bar_rest(
                        canvas=canvas,
                        stroke_mod=stroke_engine,
                        glyph_mod=glyph_variants,
                        x0=cursor_x,
                        x1=bx1 - sp * 0.4,
                        yc=y_mid(0.5 * (cursor_x + bx1)),
                        count=meas.multi_bar_rest,
                        sp=sp,
                        ink_rgb=ink_rgb,
                        style=style,
                        np_rng=np_rng,
                    )
                    measure_beat_to_x[m_idx] = (lambda _b, x=0.5 * (cursor_x + bx1): x)
                else:
                    # Compute horizontal positions aligned by beat across voices
                    extra_beats = (
                        [d.beat_offset for d in meas.dynamics]
                        + [t.beat_offset for t in meas.tempos]
                        + [mc.beat_offset for mc in meas.mid_changes]
                    )
                    ev_xs, beat_to_x = _compute_beat_x_map(
                        layout_quirks, cursor_x, bx1 - sp * 0.5, sp, meas.events, extra_beats, style, np_rng
                    )
                    measure_beat_to_x[m_idx] = beat_to_x

                    # Mid-measure clef / key / time changes
                    for mc in meas.mid_changes:
                        mx = max(cursor_x + sp * 0.4, beat_to_x(mc.beat_offset) - sp * 1.15)
                        my = y_mid(mx)
                        if mc.kind == "clef":
                            _draw_clef_with_octave(
                                canvas, glyph_variants, str(mc.value), mx, my, sp * 0.80, ink_rgb, style, np_rng
                            )
                        elif mc.kind == "key" and isinstance(mc.value, int) and mc.value != 0:
                            y_lines_mid = [float(y_fn_2(mx, li)) for li in range(-2, 3)]
                            glyph_variants.draw_key_sig(
                                canvas=canvas,
                                x_start=mx,
                                y_lines=y_lines_mid,
                                sp=sp * 0.88,
                                fifths=mc.value,
                                clef_kind=meas.clef,
                                color_rgb=ink_rgb,
                                style=style,
                                rng=np_rng,
                            )
                        elif mc.kind == "time" and isinstance(mc.value, tuple):
                            glyph_variants.draw_time_sig(
                                canvas, mc.value[0], mc.value[1], mx, my, sp * 0.90, ink_rgb, style, np_rng
                            )

                    # Pre-group beamed notes per effective beam_group ID (respecting beam_grouping_mode & split prob)
                    beam_grouping_mode = page_beam_grouping_mode
                    eff_beam_groups = plan_measure_beam_groups(meas.events, style, np_rng)
                    beam_group_dir: Dict[int, int] = {}
                    for ev_i, ev in enumerate(meas.events):
                        eff_bg = eff_beam_groups[ev_i]
                        if isinstance(ev, NoteEvent) and eff_bg is not None:
                            if eff_bg not in beam_group_dir:
                                if ev.stem_dir is not None:
                                    beam_group_dir[eff_bg] = ev.stem_dir
                                else:
                                    beam_group_dir[eff_bg] = 0
                    for bg_id, cur_dir in list(beam_group_dir.items()):
                        if cur_dir != 0:
                            continue
                        steps_in_bg = [
                            s
                            for ev_i, ev in enumerate(meas.events)
                            if isinstance(ev, NoteEvent) and eff_beam_groups[ev_i] == bg_id
                            for s in ev.staff_steps
                        ]
                        avg_s = sum(steps_in_bg) / max(1, len(steps_in_bg))
                        beam_group_dir[bg_id] = -1 if avg_s >= 4.0 else 1

                    beam_groups: Dict[int, List[Tuple[float, float, float, int, str]]] = {}
                    tuplet_groups: Dict[int, List[Tuple[float, float, float, int, int]]] = {}
                    prev_note_by_voice: Dict[int, Dict[str, Any]] = {}
                    n_meas_in_sys = max(1, sys_lay.measure_end - sys_lay.measure_start)
                    pitch_slop = page_pitch_slop
                    chord_stag = page_chord_stag
                    acc_off_sp = page_acc_off_sp
                    acc_jit_sp = page_acc_jit_sp
                    lig_prob = page_lig_prob
                    center_p = page_center_p
                    wrong_p = page_wrong_p

                    for ev_idx, (ev, ex) in enumerate(zip(meas.events, ev_xs)):
                        yc = y_mid(ex)

                        if isinstance(ev, RestEvent):
                            # Shift rests slightly up/down in polyphonic measures
                            ry = yc
                            if len(meas.voices_present) >= 2:
                                ry += -sp * 1.1 if (ev.voice % 2 == 1) else sp * 1.1
                            glyph_variants.draw_rest(
                                canvas=canvas,
                                rest_type=ev.duration_kind,
                                cx=ex,
                                cy=ry,
                                sp=sp,
                                color_rgb=ink_rgb,
                                style=style,
                                rng=np_rng,
                            )
                            if ev.dots > 0:
                                for d_i in range(ev.dots):
                                    expressive_glyphs.draw_augmentation_dot_variant(
                                        canvas,
                                        ex + sp * (0.85 + d_i * 0.42),
                                        ry - sp * 0.35,
                                        sp,
                                        ink_rgb,
                                        style,
                                        np_rng,
                                    )
                            for art in ev.articulations:
                                if art == "fermata":
                                    _draw_articulation(
                                        canvas, stroke_engine, "fermata", ex, yc - sp * 2.8, True, sp, ink_rgb, style, np_rng
                                    )
                            if ev.tuplet_id is not None and ev.tuplet_actual:
                                tuplet_groups.setdefault(ev.tuplet_id, []).append(
                                    (ex, yc - sp * 2.4, yc + sp * 2.4, ev.tuplet_actual, 1 if ev.voice % 2 == 1 else -1)
                                )
                            continue

                        # NoteEvent / Chord / Grace Note
                        eff_sp = sp * (0.64 if ev.is_grace else 1.0)
                        _draw_ledger_lines(
                            canvas, stroke_engine, ex, yc, sp, ev.staff_steps, ink_rgb, style, np_rng, glyph_mod=glyph_variants
                        )

                        eff_bg = eff_beam_groups[ev_idx]
                        if ev.is_grace:
                            stem_dir = 1
                        elif eff_bg is not None and eff_bg in beam_group_dir:
                            stem_dir = beam_group_dir[eff_bg]
                        elif ev.stem_dir is not None:
                            stem_dir = ev.stem_dir
                        else:
                            avg_step = sum(ev.staff_steps) / len(ev.staff_steps)
                            stem_dir = -1 if avg_step >= 4.0 else 1

                        u_in_bar = float(np.clip((ex - bx0) / max(1.0, bx1 - bx0), 0.0, 1.0))
                        ev_style = _modulate_event_style(
                            style,
                            ev.beat_offset,
                            rel_m,
                            n_meas_in_sys,
                            u_in_bar,
                            db=page_db_boost,
                            pa=page_phrase_arc,
                            base_stroke_scale=stroke_scale,
                            base_fill_density=page_base_fill,
                            base_opacity=page_base_op,
                            scratch=ev_style_scratch,
                        )

                        ys_note: List[float] = []
                        xs_note: List[float] = []
                        step_to_y: Dict[int, float] = {}
                        roll_lean = float(np_rng.normal(0.0, chord_stag * 0.45)) if (chord_stag > 0.0 and len(ev.staff_steps) >= 2) else 0.0
                        for h_i, (st_step, acc_kind) in enumerate(zip(ev.staff_steps, ev.accidentals)):
                            # Hard-clamp renderer pitch-slop to ±0.145*sp (+ ±0.025*sp in draw_notehead => <= ±0.17*sp total!)
                            y_slop = float(np.clip(np_rng.normal(0.0, pitch_slop * sp), -0.145 * sp, 0.145 * sp)) if pitch_slop > 0.0 else 0.0
                            ny = yc - (st_step - 4) * (sp * 0.5) + y_slop
                            ys_note.append(ny)
                            step_to_y[st_step] = ny
                            hx_off = (
                                ev.head_x_nudges_sp[h_i] * sp
                                if h_i < len(ev.head_x_nudges_sp)
                                else 0.0
                            )
                            c_off = 0.0
                            if chord_stag > 0.0 and len(ev.staff_steps) >= 2 and abs(hx_off) < 1e-3:
                                rank = (h_i - 0.5 * (len(ev.staff_steps) - 1))
                                c_off = float(np.clip(rank * roll_lean + np_rng.normal(0.0, chord_stag * 0.45), -0.32, 0.32)) * sp
                            hx = ex + hx_off + c_off
                            xs_note.append(hx)

                            if acc_kind:
                                is_court = (
                                    ev.courtesy_accidentals[h_i]
                                    if h_i < len(ev.courtesy_accidentals)
                                    else False
                                )
                                ax_j = float(np.clip(np_rng.normal(0.0, acc_jit_sp), -0.36, 0.36)) * eff_sp if acc_jit_sp > 0.0 else 0.0
                                ay_j = float(np.clip(np_rng.normal(0.0, acc_jit_sp * 0.42), -0.14, 0.14)) * eff_sp if acc_jit_sp > 0.0 else 0.0
                                _draw_double_or_courtesy_accidental(
                                    canvas=canvas,
                                    stroke_mod=stroke_engine,
                                    glyph_mod=glyph_variants,
                                    acc=acc_kind,
                                    cx=ex - eff_sp * ((acc_off_sp + 0.10) if is_court else acc_off_sp) + ax_j,
                                    cy=ny + ay_j,
                                    sp=eff_sp,
                                    ink_rgb=ink_rgb,
                                    style=ev_style,
                                    np_rng=np_rng,
                                    courtesy=is_court,
                                )

                            glyph_variants.draw_notehead(
                                canvas=canvas,
                                duration_kind=ev.duration_kind,
                                cx=hx,
                                cy=ny,
                                sp=eff_sp,
                                color_rgb=ink_rgb,
                                style=ev_style,
                                rng=np_rng,
                            )

                            if ev.dots > 0 and not ev.is_grace:
                                dot_y = ny - (sp * 0.28 if st_step % 2 == 0 else 0.0)
                                dot_base_x = max(ex, hx) + sp * 0.95
                                for d_i in range(ev.dots):
                                    dx = dot_base_x + d_i * sp * 0.45
                                    expressive_glyphs.draw_augmentation_dot_variant(
                                        canvas=canvas,
                                        cx=dx,
                                        cy=dot_y,
                                        sp=sp,
                                        color_rgb=ink_rgb,
                                        style=ev_style,
                                        rng=np_rng,
                                    )

                        # Chord roll drag ligature inside multi-note chords
                        if lig_prob > 0.0 and len(ys_note) >= 2 and float(np_rng.random()) < lig_prob * 0.55:
                            for ci in range(len(ys_note) - 1):
                                glyph_variants.draw_cursive_ligature(
                                    canvas, "chord_roll_drag",
                                    (xs_note[ci], ys_note[ci]), (xs_note[ci + 1], ys_note[ci + 1]),
                                    sp, ink_rgb, ev_style, np_rng, stem_dir=stem_dir,
                                )

                        # Stem & Flag
                        tip_y = min(ys_note) if stem_dir == 1 else max(ys_note)
                        stem_x = ex + (eff_sp * 0.52 if stem_dir == 1 else -eff_sp * 0.52)
                        far_y = max(ys_note) if stem_dir == 1 else min(ys_note)
                        n_flags_ev = 0
                        if ev.duration_kind != "whole":
                            anchor_y = min(ys_note) if stem_dir == 1 else max(ys_note)
                            far_y = max(ys_note) if stem_dir == 1 else min(ys_note)
                            stem_len_mult = glyph_variants.get_stem_length_multiplier(ev_style)
                            stem_len = max(3.3 * eff_sp * stem_len_mult, abs(far_y - anchor_y) + 2.8 * eff_sp * stem_len_mult)
                            edge_sign = 1.0 if stem_dir == 1 else -1.0
                            if eff_bg is None and wrong_p > 0.0 and float(np_rng.random()) < wrong_p:
                                edge_sign = -edge_sign
                                attach_frac = 0.48
                            elif center_p > 0.0 and float(np_rng.random()) < center_p:
                                attach_frac = float(np_rng.uniform(-0.10, 0.10))
                            else:
                                attach_frac = 0.52 + float(np_rng.normal(0.0, 0.03))
                            stem_x = ex + edge_sign * eff_sp * attach_frac
                            tip_y = anchor_y - stem_dir * stem_len

                            if ev.is_grace:
                                n_flags = max(1, {"eighth": 1, "16th": 2, "32nd": 3}.get(ev.duration_kind, 1))
                                n_flags_ev = n_flags
                                glyph_variants.draw_stem_and_flag(
                                    canvas=canvas,
                                    x=stem_x,
                                    y0=far_y,
                                    y1=tip_y,
                                    sp=eff_sp,
                                    flags=n_flags,
                                    stem_up=True,
                                    color_rgb=ink_rgb,
                                    style=ev_style,
                                    rng=np_rng,
                                    scale_length=False,
                                )
                                if ev.grace_slash:
                                    sy_mid = 0.45 * far_y + 0.55 * tip_y
                                    expressive_glyphs.draw_grace_slash_variant(
                                        canvas, stem_x, sy_mid, eff_sp, ink_rgb, ev_style, np_rng
                                    )
                            elif eff_bg is not None:
                                beam_groups.setdefault(eff_bg, []).append(
                                    (stem_x, far_y, tip_y, stem_dir, ev.duration_kind)
                                )
                            else:
                                n_flags = {"eighth": 1, "16th": 2, "32nd": 3}.get(ev.duration_kind, 0)
                                n_flags_ev = n_flags
                                glyph_variants.draw_stem_and_flag(
                                    canvas=canvas,
                                    x=stem_x,
                                    y0=far_y,
                                    y1=tip_y,
                                    sp=sp,
                                    flags=n_flags,
                                    stem_up=(stem_dir == 1),
                                    color_rgb=ink_rgb,
                                    style=ev_style,
                                    rng=np_rng,
                                    scale_length=False,
                                )

                        # Cursive pen-lift ligatures between consecutive same-voice notes
                        if not ev.is_grace:
                            v_id = int(getattr(ev, "voice", 1))
                            prev_n = prev_note_by_voice.get(v_id)
                            cur_y_head = ys_note[0]
                            if prev_n is not None and lig_prob > 0.0 and float(np_rng.random()) < lig_prob:
                                u_lig = float(np_rng.random())
                                if u_lig < 0.45:
                                    glyph_variants.draw_cursive_ligature(
                                        canvas, "notehead_to_notehead_swoop",
                                        (prev_n["x"] + 0.45 * sp, prev_n["y_head"]),
                                        (ex - 0.45 * sp, cur_y_head),
                                        sp, ink_rgb, ev_style, np_rng, stem_dir=stem_dir,
                                    )
                                elif u_lig < 0.78:
                                    glyph_variants.draw_cursive_ligature(
                                        canvas, "notehead_to_stem_base_hook",
                                        (prev_n["x"] + 0.40 * sp, prev_n["y_head"]),
                                        (stem_x, far_y),
                                        sp, ink_rgb, ev_style, np_rng, stem_dir=stem_dir,
                                    )
                                elif prev_n["n_flags"] > 0 or prev_n["beamed"]:
                                    glyph_variants.draw_cursive_ligature(
                                        canvas, "flag_to_flag_return_loop",
                                        (prev_n["stem_x"] + 0.50 * sp, prev_n["tip_y"]),
                                        (stem_x - 0.30 * sp, tip_y),
                                        sp, ink_rgb, ev_style, np_rng, stem_dir=stem_dir,
                                    )
                            prev_note_by_voice[v_id] = {
                                "x": ex, "y_head": cur_y_head, "stem_x": stem_x,
                                "tip_y": tip_y, "n_flags": n_flags_ev, "beamed": eff_bg is not None,
                            }

                        # Articulations & Ornaments
                        if ev.articulations:
                            # Place on notehead side (opposite stem), except fermata/trill always above
                            art_above = (stem_dir == -1)
                            cur_y_above = min(min(ys_note), yc - 2.0 * sp, tip_y if stem_dir == 1 else min(ys_note)) - sp * 0.95
                            cur_y_below = max(max(ys_note), yc + 2.0 * sp, tip_y if stem_dir == -1 else max(ys_note)) + sp * 0.95
                            for art in ev.articulations:
                                if art in ("fermata", "trill"):
                                    ay = cur_y_above - (sp * 0.35 if art == "fermata" else sp * 0.2)
                                    _draw_articulation(
                                        canvas, stroke_engine, art, ex, ay, True, sp, ink_rgb, style, np_rng
                                    )
                                    cur_y_above -= sp * 1.15
                                elif art_above:
                                    _draw_articulation(
                                        canvas, stroke_engine, art, ex, cur_y_above, True, sp, ink_rgb, style, np_rng
                                    )
                                    cur_y_above -= sp * 0.85
                                else:
                                    _draw_articulation(
                                        canvas, stroke_engine, art, ex, cur_y_below, False, sp, ink_rgb, style, np_rng
                                    )
                                    cur_y_below += sp * 0.85

                        # Under-staff Lyrics
                        if ev.lyrics:
                            for verse_num, syll_txt in ev.lyrics:
                                ly_y = yc + sp * (4.25 + (verse_num - 1) * 1.35)
                                _draw_handwritten_text(
                                    canvas,
                                    stroke_engine,
                                    syll_txt,
                                    ex - sp * 0.45,
                                    ly_y,
                                    sp * 0.92,
                                    ink_rgb,
                                    style,
                                    np_rng,
                                    italic=False,
                                )

                        # Track for tuplet brackets
                        if ev.tuplet_id is not None and ev.tuplet_actual and not ev.is_grace:
                            top_ref = min(min(ys_note), tip_y) - sp * 0.95
                            bot_ref = max(max(ys_note), tip_y) + sp * 0.95
                            tuplet_groups.setdefault(ev.tuplet_id, []).append(
                                (ex, top_ref, bot_ref, ev.tuplet_actual, stem_dir)
                            )

                        rendered_notes[(m_idx, ev_idx)] = (
                            ex,
                            step_to_y,
                            stem_dir,
                            min(min(ys_note), tip_y),
                            max(max(ys_note), tip_y),
                        )

                    # Draw beamed stems + beams
                    for grp_ord, grp_items in enumerate(beam_groups.values()):
                        if len(grp_items) == 1:
                            sx, fy, ty, sdir, dk = grp_items[0]
                            n_flags = {"eighth": 1, "16th": 2, "32nd": 3}.get(dk, 1)
                            glyph_variants.draw_stem_and_flag(
                                canvas=canvas,
                                x=sx,
                                y0=fy,
                                y1=ty,
                                sp=sp,
                                flags=n_flags,
                                stem_up=(sdir == 1),
                                color_rgb=ink_rgb,
                                style=style,
                                rng=np_rng,
                                scale_length=False,
                            )
                            continue

                        x0, _, _, sdir, _ = grp_items[0]
                        x1 = grp_items[-1][0]
                        dx_span = max(1e-3, x1 - x0)
                        geom = glyph_variants.compute_beam_geometry(grp_items, sp, style, rng=np_rng)
                        y0 = geom["y0"]
                        y1 = geom["y1"]
                        curve_sag_px = geom["curve_sag_px"]
                        secondary_tilt_px = geom["secondary_tilt_px"]

                        def _beam_y(t_norm: float) -> float:
                            return y0 + t_norm * (y1 - y0) + 4.0 * t_norm * (1.0 - t_norm) * curve_sag_px

                        for sx, fy, _, _, _ in grp_items:
                            t = (sx - x0) / dx_span
                            by = _beam_y(t)
                            glyph_variants.draw_stem_and_flag(
                                canvas=canvas,
                                x=sx,
                                y0=fy,
                                y1=by,
                                sp=sp,
                                flags=0,
                                stem_up=(sdir == 1),
                                color_rgb=ink_rgb,
                                style=style,
                                rng=np_rng,
                                beamed=True,
                                scale_length=False,
                            )

                        beam_levels = [
                            {"eighth": 1, "16th": 2, "32nd": 3}.get(it[4], 1) for it in grp_items
                        ]
                        n_beams = max(beam_levels)
                        beam_step = glyph_variants.get_beam_spacing_sp(style) * sp
                        splay_out = (grp_ord % 2 == 0)
                        for b_i in range(n_beams):
                            b_off = b_i * beam_step * sdir
                            if b_i == 0:
                                glyph_variants.draw_beam(
                                    canvas=canvas,
                                    x0=x0,
                                    y0=y0 + b_off,
                                    x1=x1,
                                    y1=y1 + b_off,
                                    sp=sp,
                                    color_rgb=ink_rgb,
                                    style=style,
                                    rng=np_rng,
                                    extra_sag_px=curve_sag_px,
                                )
                            else:
                                req = b_i + 1
                                if beam_grouping_mode == "french_broken_secondary":
                                    # Break 16th/32nd sub-beams into alternating short left/right hooks
                                    # under the continuous primary beam.
                                    for idx_g, lv in enumerate(beam_levels):
                                        if lv < req:
                                            continue
                                        sx_i = grp_items[idx_g][0]
                                        if idx_g == 0:
                                            point_right = True
                                        elif idx_g == len(grp_items) - 1:
                                            point_right = False
                                        else:
                                            point_right = (idx_g % 2 == 0)
                                        if point_right and idx_g + 1 < len(grp_items):
                                            local_dx = max(sp * 0.6, grp_items[idx_g + 1][0] - sx_i)
                                            stub = min(sp * 0.92, local_dx * 0.45)
                                            bx_a, bx_b = sx_i, sx_i + stub
                                        else:
                                            local_dx = max(sp * 0.6, sx_i - grp_items[max(0, idx_g - 1)][0])
                                            stub = min(sp * 0.92, local_dx * 0.45)
                                            bx_a, bx_b = sx_i - stub, sx_i
                                        ta = (bx_a - x0) / dx_span
                                        tb = (bx_b - x0) / dx_span
                                        tilt_a = (ta - 0.5) * secondary_tilt_px * b_i
                                        tilt_b = (tb - 0.5) * secondary_tilt_px * b_i
                                        glyph_variants.draw_beam(
                                            canvas=canvas,
                                            x0=bx_a,
                                            y0=_beam_y(ta) + b_off + tilt_a,
                                            x1=bx_b,
                                            y1=_beam_y(tb) + b_off + tilt_b,
                                            sp=sp,
                                            color_rgb=ink_rgb,
                                            style=style,
                                            rng=np_rng,
                                        )
                                    continue

                                # Draw contiguous secondary/partial beams for notes needing level >= b_i + 1
                                run_start: Optional[int] = None
                                for idx_g, lv in enumerate(beam_levels + [0]):
                                    if lv >= req and run_start is None:
                                        run_start = idx_g
                                    elif lv < req and run_start is not None:
                                        run_end = idx_g - 1
                                        if run_start == run_end:
                                            # Partial stub beam pointing inward toward group center
                                            sx_i = grp_items[run_start][0]
                                            stub_len = min(sp * 0.95, dx_span * 0.35)
                                            if run_start == len(grp_items) - 1:
                                                bx_a, bx_b = sx_i - stub_len, sx_i
                                            else:
                                                bx_a, bx_b = sx_i, sx_i + stub_len
                                        else:
                                            bx_a = grp_items[run_start][0]
                                            bx_b = grp_items[run_end][0]
                                        ta = (bx_a - x0) / dx_span
                                        tb = (bx_b - x0) / dx_span
                                        if beam_grouping_mode == "feathered_splay":
                                            # Fan secondary/tertiary beams outward (accelerando) or inward (ritardando).
                                            f_a = (0.24 + 1.52 * ta) if splay_out else (1.76 - 1.52 * ta)
                                            f_b = (0.24 + 1.52 * tb) if splay_out else (1.76 - 1.52 * tb)
                                            off_a = b_i * beam_step * f_a * sdir
                                            off_b = b_i * beam_step * f_b * sdir
                                        else:
                                            off_a = b_off + (ta - 0.5) * secondary_tilt_px * b_i
                                            off_b = b_off + (tb - 0.5) * secondary_tilt_px * b_i
                                        sub_sag = curve_sag_px * ((tb - ta) ** 2)
                                        glyph_variants.draw_beam(
                                            canvas=canvas,
                                            x0=bx_a,
                                            y0=_beam_y(ta) + off_a,
                                            x1=bx_b,
                                            y1=_beam_y(tb) + off_b,
                                            sp=sp,
                                            color_rgb=ink_rgb,
                                            style=style,
                                            rng=np_rng,
                                            extra_sag_px=sub_sag,
                                        )
                                        run_start = None

                    # Draw Tuplet Brackets
                    for t_items in tuplet_groups.values():
                        if len(t_items) < 2:
                            continue
                        t_items.sort(key=lambda it: it[0])
                        tx0 = t_items[0][0] - sp * 0.35
                        tx1 = t_items[-1][0] + sp * 0.35
                        t_num = t_items[0][3]
                        sdir_avg = sum(it[4] for it in t_items)
                        above = sdir_avg >= 0
                        if above:
                            ty = min(it[1] for it in t_items) - sp * 0.35
                        else:
                            ty = max(it[2] for it in t_items) + sp * 0.35
                        _draw_tuplet_bracket(
                            canvas, stroke_engine, glyph_variants, tx0, ty, tx1, ty, t_num, above, sp, ink_rgb, style, np_rng
                        )

                    # Dynamics
                    for dyn in meas.dynamics:
                        dx = beat_to_x(dyn.beat_offset)
                        dy = y_mid(dx) + (sp * 3.25 if dyn.placement != "above" else -sp * 3.15)
                        _draw_dynamic_marking(canvas, stroke_engine, dyn.text, dx, dy, sp, ink_rgb, style, np_rng)

                    # Tempo / Staff Text (above staff)
                    for tmp in meas.tempos:
                        tx = beat_to_x(tmp.beat_offset)
                        ty = float(y_fn_2(tx, -2)) - sp * 2.15
                        _draw_handwritten_text(
                            canvas, stroke_engine, tmp.text, tx - sp * 0.3, ty, sp * 1.05, ink_rgb, style, np_rng, bold=True
                        )

                # Measure right barline
                bar_y0 = float(y_fn_2(bx1, -2))
                bar_y1 = float(y_fn_2(bx1, 2))
                glyph_variants.draw_barline(
                    canvas=canvas,
                    x=bx1,
                    y_top=bar_y0,
                    y_bot=bar_y1,
                    sp=sp,
                    kind=meas.barline,
                    color_rgb=ink_rgb,
                    style=style,
                    rng=np_rng,
                )

            # Draw Ties & Slurs for this staff within this system
            m_lo, m_hi = sys_lay.measure_start, sys_lay.measure_end
            for span in part.ties + part.slurs:
                is_tie = span.kind == "tie"
                s_in = m_lo <= span.start_measure < m_hi
                e_in = m_lo <= span.end_measure < m_hi
                if not s_in and not e_in:
                    continue

                p_start = rendered_notes.get((span.start_measure, span.start_event_idx)) if s_in else None
                p_end = rendered_notes.get((span.end_measure, span.end_event_idx)) if e_in else None
                if p_start is None and p_end is None:
                    continue

                if p_start is not None and p_end is not None:
                    sx, smap, sdir, smin_y, smax_y = p_start
                    ex, emap, edir, emin_y, emax_y = p_end
                    arch_dir = 1 if sdir == 1 else -1
                    if is_tie:
                        sy = smap.get(span.start_step, y_mid(sx)) + arch_dir * sp * 0.42
                        ey = emap.get(span.end_step, y_mid(ex)) + arch_dir * sp * 0.42
                        sx += sp * 0.42
                        ex -= sp * 0.42
                    else:
                        sy = (smax_y + sp * 0.55) if arch_dir == 1 else (smin_y - sp * 0.55)
                        ey = (emax_y + sp * 0.55) if arch_dir == 1 else (emin_y - sp * 0.55)
                    _draw_slur_or_tie(
                        canvas, stroke_engine, sx, sy, ex, ey, arch_dir, sp, ink_rgb, style, np_rng, is_tie=is_tie
                    )
                elif p_start is not None:
                    # Crosses right system edge: draw outgoing half-arc
                    sx, smap, sdir, smin_y, smax_y = p_start
                    arch_dir = 1 if sdir == 1 else -1
                    sy = smap.get(span.start_step, y_mid(sx)) + arch_dir * sp * 0.45
                    _draw_slur_or_tie(
                        canvas, stroke_engine, sx + sp * 0.35, sy, sys_lay.x_right, sy, arch_dir, sp, ink_rgb, style, np_rng, is_tie=is_tie
                    )
                elif p_end is not None:
                    # Crosses left system edge: draw incoming half-arc
                    ex, emap, edir, emin_y, emax_y = p_end
                    arch_dir = 1 if edir == 1 else -1
                    ey = emap.get(span.end_step, y_mid(ex)) + arch_dir * sp * 0.45
                    _draw_slur_or_tie(
                        canvas, stroke_engine, sys_lay.x_left + sp * 2.6, ey, ex - sp * 0.35, ey, arch_dir, sp, ink_rgb, style, np_rng, is_tie=is_tie
                    )

            # Draw Hairpins (< / >) for this staff within this system
            for hp in part.hairpins:
                if hp.end_measure < m_lo or hp.start_measure >= m_hi:
                    continue
                if hp.start_measure in measure_beat_to_x:
                    hx0 = measure_beat_to_x[hp.start_measure](hp.start_beat)
                else:
                    hx0 = sys_lay.x_left + sp * 3.0
                if hp.end_measure in measure_beat_to_x:
                    hx1 = measure_beat_to_x[hp.end_measure](hp.end_beat)
                else:
                    hx1 = sys_lay.x_right - sp * 0.5
                if hx1 <= hx0 + sp * 0.5:
                    hx1 = min(sys_lay.x_right - sp * 0.3, hx0 + sp * 3.5)
                hy = y_mid(0.5 * (hx0 + hx1)) + (sp * 3.35 if hp.placement != "above" else -sp * 3.15)
                _draw_hairpin(canvas, stroke_engine, hp.kind, hx0, hx1, hy, sp, ink_rgb, style, np_rng=np_rng)

        # Cross-staff barlines (grand_staff_connected / full_system_connected / mensurstrich_between)
        bar_span_mode = str(style.get("barline_span_mode") or "per_staff").strip().lower()
        if len(sys_lay.staves) >= 2 and bar_span_mode in ("grand_staff_connected", "full_system_connected", "mensurstrich_between"):
            for _rel_m, (_bx0, bx1) in enumerate(sys_lay.bar_x_bounds):
                for s_a in range(len(sys_lay.staves) - 1):
                    st_a, st_b = sys_lay.staves[s_a], sys_lay.staves[s_a + 1]
                    if bar_span_mode == "grand_staff_connected":
                        if st_a.brace_group is None or st_a.brace_group != st_b.brace_group:
                            continue
                    y_gap0 = float(staff_curves[s_a][0](bx1, 2))
                    y_gap1 = float(staff_curves[s_a + 1][0](bx1, -2))
                    glyph_variants.draw_barline(
                        canvas=canvas, x=bx1, y_top=y_gap0, y_bot=y_gap1,
                        sp=st_a.sp, kind="single", color_rgb=ink_rgb, style=style, rng=np_rng,
                    )

    # Post-notation living-manuscript overlays, corrections, chords, measure numbers, smears & wear
    manuscript_summary = manuscript_marks.apply_post_notation_manuscript_marks(
        canvas=canvas,
        systems=systems,
        style=style,
        rng=np_rng,
        marks_out=manuscript_marks_log,
        clean_paper=clean_paper,
    )

    # Optional legacy margin annotations (boxed rehearsal marks & conductor cues)
    margin_prob = float(np.clip(style.get("margin_annotations_prob", 0.0), 0.0, 1.0))
    if margin_prob > 0.0 and not manuscript_marks_log and float(np_rng.random()) < margin_prob:
        staff_ys = [st.y_center for s in systems if not s.is_blank_prerule for st in s.staves]
        sps = [st.sp for s in systems if not s.is_blank_prerule for st in s.staves]
        avg_sp = float(sum(sps) / len(sps)) if sps else 12.0
        layout_quirks.draw_margin_annotations(canvas, style, np_rng, staff_ys=staff_ys, sp=avg_sp)

    # 4. Composite & save output PNG + companion Bezier GT JSON
    out_path = Path(out_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    final_img = canvas.to_pil()
    final_img.save(out_path, compress_level=1)

    gt_payload: Dict[str, Any] = {
        "score_title": score.title,
        "source_path": str(score_path),
        "style": display_style_name,
        "seed": seed,
        "page_w": page_w,
        "page_h": page_h,
        "clean_paper": bool(clean_paper),
        "historical_clean": bool(eff_hist_clean),
        "omit_cont_header_prob": float(eff_omit_cont_prob),
        "omit_cont_header": bool(omit_cont_header_active),
        "systems": gt_systems,
    }
    if manuscript_marks_log:
        gt_payload["manuscript_marks"] = manuscript_marks_log
        gt_payload["manuscript_summary"] = manuscript_summary

    if export_train_dir is not None:
        train_exports = _export_training_systems(
            score=score,
            score_path=Path(score_path),
            final_img=final_img,
            systems=systems,
            export_dir=Path(export_train_dir),
            stem=out_path.stem,
            max_parts=max_parts,
        )
        gt_payload["train_exports"] = train_exports

    gt_path = out_path.with_suffix(".json")
    gt_path.write_text(json.dumps(gt_payload, indent=2), encoding="utf-8")
    return gt_payload


# Crisp public alias
render_score = render_score_handwritten

