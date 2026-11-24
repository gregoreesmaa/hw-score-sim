"""Unit tests for the 52 stem/flag/beam archetypes and 5 ledger line styles."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from hw_score_sim.glyph_variants import (
    BEAM_ARCHETYPES,
    BEAM_GROUPING_MODES,
    BEAM_SLOPE_STYLES,
    FLAG_ARCHETYPES,
    LEDGER_STYLES,
    STEM_ARCHETYPES,
    compute_beam_geometry,
    draw_beam,
    draw_ledger_line,
    draw_ledger_lines,
    draw_notehead,
    draw_stem_and_flag,
    get_beam_spacing_sp,
    get_stem_length_multiplier,
)
from hw_score_sim.musescore_compat import NoteEvent
from hw_score_sim.renderer import plan_measure_beam_groups, render_score_handwritten
from hw_score_sim.stroke_engine import StrokeCanvas
from hw_score_sim.styles import (
    SCRIBE_STYLES,
    get_style,
    interpolate_styles,
    list_styles,
    register_style,
    reload_style_packs,
    resolve_beam_archetype,
    resolve_beam_grouping_mode,
    resolve_beam_slope_style,
    resolve_flag_archetype,
    resolve_ledger_style,
    resolve_stem_archetype,
    sample_random_scribe,
    validate_style,
)

BG_WHITE = np.array([0.98, 0.98, 0.97], dtype=np.float32)
INK_BLACK = np.array([0.08, 0.08, 0.08], dtype=np.float32)


def _base_clean_style(**overrides) -> dict:
    s = get_style("romantic_copyist_oblong")
    s["wobble_px"] = 0.25
    s["tremor_px"] = 0.0
    s["ink_var"] = 0.0
    s["ink_dropout"] = 0.0
    s["slant_deg"] = 0.0
    s.update(overrides)
    return s


def _ink_mask(buf: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(buf - BG_WHITE[None, None, :]), axis=2) > 0.05


def _mean_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))))


class TestStemsFlagsBeamsRegistryAndValidation(unittest.TestCase):
    def test_archetype_counts_total_52_plus_5_ledgers(self) -> None:
        self.assertEqual(len(STEM_ARCHETYPES), 18)
        self.assertEqual(len(FLAG_ARCHETYPES), 20)
        self.assertEqual(len(BEAM_ARCHETYPES), 14)
        self.assertEqual(len(STEM_ARCHETYPES) + len(FLAG_ARCHETYPES) + len(BEAM_ARCHETYPES), 52)
        self.assertEqual(len(LEDGER_STYLES), 5)
        self.assertEqual(len(set(STEM_ARCHETYPES)), 18)
        self.assertEqual(len(set(FLAG_ARCHETYPES)), 20)
        self.assertEqual(len(set(BEAM_ARCHETYPES)), 14)
        self.assertEqual(len(set(LEDGER_STYLES)), 5)

    def test_all_existing_styles_validate_and_gain_diverse_defaults(self) -> None:
        names = list_styles()
        self.assertGreaterEqual(len(names), 234)
        seen_stems = set()
        seen_flags = set()
        seen_beams = set()
        seen_ledgers = set()
        for name in names:
            s = get_style(name)
            validated = validate_style(s, name=name)
            self.assertEqual(validated["name"], name)
            sa = resolve_stem_archetype(validated)
            fa = resolve_flag_archetype(validated)
            ba = resolve_beam_archetype(validated)
            la = resolve_ledger_style(validated)
            self.assertIn(sa, STEM_ARCHETYPES)
            self.assertIn(fa, FLAG_ARCHETYPES)
            self.assertIn(ba, BEAM_ARCHETYPES)
            self.assertIn(la, LEDGER_STYLES)
            # Deterministic across repeated calls
            self.assertEqual(sa, resolve_stem_archetype(validated))
            self.assertEqual(fa, resolve_flag_archetype(validated))
            self.assertEqual(ba, resolve_beam_archetype(validated))
            self.assertEqual(la, resolve_ledger_style(validated))
            seen_stems.add(sa)
            seen_flags.add(fa)
            seen_beams.add(ba)
            seen_ledgers.add(la)

        # All 234 existing styles should immediately span a wide diversity of archetypes
        self.assertGreaterEqual(len(seen_stems), 15)
        self.assertGreaterEqual(len(seen_flags), 16)
        self.assertGreaterEqual(len(seen_beams), 12)
        self.assertEqual(len(seen_ledgers), 5)

    def test_validate_style_accepts_and_rejects_optional_keys(self) -> None:
        base = get_style("baroque_quill_neat")
        valid = dict(base)
        valid.update(
            stem_archetype="s_curve_flex",
            flag_archetype="baroque_curl",
            beam_archetype="calligraphic_ribbon",
            ledger_style="wavy",
            stem_length_scale=1.35,
            beam_thickness_scale=1.65,
        )
        out = validate_style(valid)
        self.assertEqual(out["stem_archetype"], "s_curve_flex")
        self.assertEqual(out["flag_archetype"], "baroque_curl")
        self.assertEqual(out["beam_archetype"], "calligraphic_ribbon")
        self.assertEqual(out["ledger_style"], "wavy")
        self.assertAlmostEqual(out["stem_length_scale"], 1.35, places=6)
        self.assertAlmostEqual(out["beam_thickness_scale"], 1.65, places=6)

        for key, bad_val in (
            ("stem_archetype", "not_a_stem"),
            ("flag_archetype", "not_a_flag"),
            ("beam_archetype", "not_a_beam"),
            ("ledger_style", "not_a_ledger"),
            ("stem_length_scale", 0.20),
            ("stem_length_scale", 2.50),
            ("beam_thickness_scale", 0.10),
            ("beam_thickness_scale", 3.50),
        ):
            bad = dict(base)
            bad[key] = bad_val
            with self.assertRaises(ValueError, msg=f"Expected ValueError for {key}={bad_val!r}"):
                validate_style(bad)

    def test_interpolate_styles_handles_new_keys(self) -> None:
        sa = _base_clean_style(
            stem_archetype="hairline_needle",
            flag_archetype="tiny_flick",
            beam_archetype="hairline_thin",
            ledger_style="short_tick",
            stem_length_scale=0.70,
            beam_thickness_scale=0.60,
        )
        sb = _base_clean_style(
            stem_archetype="heavy_thick",
            flag_archetype="baroque_curl",
            beam_archetype="thick_marker",
            ledger_style="wide_sweep",
            stem_length_scale=1.50,
            beam_thickness_scale=2.00,
        )
        mid = interpolate_styles(sa, sb, 0.5)
        self.assertAlmostEqual(mid["stem_length_scale"], 1.10, places=5)
        self.assertAlmostEqual(mid["beam_thickness_scale"], 1.30, places=5)
        self.assertIn(mid["stem_archetype"], ("hairline_needle", "heavy_thick"))
        self.assertIn(mid["flag_archetype"], ("tiny_flick", "baroque_curl"))
        self.assertIn(mid["beam_archetype"], ("hairline_thin", "thick_marker"))
        self.assertIn(mid["ledger_style"], ("short_tick", "wide_sweep"))

    def test_sample_random_scribe_explores_all_52_archetypes(self) -> None:
        rng = np.random.default_rng(20261124)
        seen_s, seen_f, seen_b, seen_l = set(), set(), set(), set()
        for _ in range(300):
            sc = sample_random_scribe(rng)
            validated = validate_style(sc)
            seen_s.add(validated["stem_archetype"])
            seen_f.add(validated["flag_archetype"])
            seen_b.add(validated["beam_archetype"])
            seen_l.add(validated["ledger_style"])
            self.assertTrue(0.65 <= validated["stem_length_scale"] <= 1.55)
            self.assertTrue(0.45 <= validated["beam_thickness_scale"] <= 2.10)
        self.assertEqual(seen_s, set(STEM_ARCHETYPES))
        self.assertEqual(seen_f, set(FLAG_ARCHETYPES))
        self.assertEqual(seen_b, set(BEAM_ARCHETYPES))
        self.assertEqual(seen_l, set(LEDGER_STYLES))


class TestStemArchetypesRendering(unittest.TestCase):
    def test_all_18_stems_render_up_and_down_and_are_pairwise_distinct(self) -> None:
        sp = 20.0
        for stem_up in (True, False):
            renders = {}
            for arch in STEM_ARCHETYPES:
                rng = np.random.default_rng(101)
                canvas = StrokeCanvas(140, 180, BG_WHITE, rng=rng)
                style = _base_clean_style(stem_archetype=arch, stem_length_scale=1.0)
                y0 = 130.0 if stem_up else 50.0
                y1 = 60.0 if stem_up else 120.0
                draw_stem_and_flag(canvas, 70.0, y0, y1, sp, 0, stem_up, INK_BLACK, style, rng)
                ink_px = int(np.sum(_ink_mask(canvas.buf)))
                self.assertGreater(ink_px, 20, f"Stem {arch!r} (stem_up={stem_up}) drew too little ink")
                renders[arch] = canvas.buf.copy()

            names = list(STEM_ARCHETYPES)
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    diff = _mean_abs_diff(renders[names[i]], renders[names[j]])
                    self.assertGreater(
                        diff,
                        0.0012,
                        f"Stems {names[i]!r} and {names[j]!r} (stem_up={stem_up}) too similar: diff={diff:.6f}",
                    )

    def test_stem_length_scale_changes_ink_extent(self) -> None:
        sp = 20.0
        extents = {}
        for scale in (0.65, 1.55):
            rng = np.random.default_rng(7)
            canvas = StrokeCanvas(120, 220, BG_WHITE, rng=rng)
            style = _base_clean_style(stem_archetype="straight_classic", stem_length_scale=scale)
            draw_stem_and_flag(canvas, 60.0, 170.0, 100.0, sp, 0, True, INK_BLACK, style, rng)
            ys = np.where(_ink_mask(canvas.buf))[0]
            extents[scale] = int(ys.max() - ys.min())
        self.assertGreater(extents[1.55], extents[0.65] * 1.6)
        self.assertLess(get_stem_length_multiplier({"stem_archetype": "short_stubby"}), 0.8)
        self.assertGreater(get_stem_length_multiplier({"stem_archetype": "tall_spindly"}), 1.3)


class TestFlagArchetypesRendering(unittest.TestCase):
    def test_all_20_flags_render_up_and_down_with_1_2_3_flags_and_are_distinct(self) -> None:
        sp = 20.0
        for stem_up in (True, False):
            renders_1 = {}
            for arch in FLAG_ARCHETYPES:
                for n_flags in (1, 2, 3):
                    rng = np.random.default_rng(202)
                    canvas = StrokeCanvas(160, 180, BG_WHITE, rng=rng)
                    style = _base_clean_style(stem_archetype="straight_classic", flag_archetype=arch)
                    y0 = 135.0 if stem_up else 45.0
                    y1 = 55.0 if stem_up else 125.0
                    draw_stem_and_flag(canvas, 55.0, y0, y1, sp, n_flags, stem_up, INK_BLACK, style, rng)
                    ink_px = int(np.sum(_ink_mask(canvas.buf)))
                    self.assertGreater(
                        ink_px, 40, f"Flag {arch!r} (stem_up={stem_up}, n={n_flags}) drew too little ink"
                    )
                    if n_flags == 1:
                        renders_1[arch] = canvas.buf.copy()

            names = list(FLAG_ARCHETYPES)
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    diff = _mean_abs_diff(renders_1[names[i]], renders_1[names[j]])
                    self.assertGreater(
                        diff,
                        0.0012,
                        f"Flags {names[i]!r} and {names[j]!r} (stem_up={stem_up}) too similar: diff={diff:.6f}",
                    )


class TestBeamAndLedgerArchetypesRendering(unittest.TestCase):
    def test_all_14_beams_render_multi_beam_and_are_pairwise_distinct(self) -> None:
        sp = 20.0
        renders = {}
        for arch in BEAM_ARCHETYPES:
            rng = np.random.default_rng(303)
            canvas = StrokeCanvas(220, 140, BG_WHITE, rng=rng)
            style = _base_clean_style(beam_archetype=arch, beam_thickness_scale=1.0)
            spacing = get_beam_spacing_sp(style) * sp
            # Primary beam + secondary beam + partial tertiary beam
            draw_beam(canvas, 35.0, 45.0, 185.0, 58.0, sp, INK_BLACK, style, rng)
            draw_beam(canvas, 35.0, 45.0 + spacing, 185.0, 58.0 + spacing, sp, INK_BLACK, style, rng)
            draw_beam(canvas, 110.0, 51.5 + 2.0 * spacing, 185.0, 58.0 + 2.0 * spacing, sp, INK_BLACK, style, rng)
            ink_px = int(np.sum(_ink_mask(canvas.buf)))
            self.assertGreater(ink_px, 120, f"Beam {arch!r} drew too little ink")
            renders[arch] = canvas.buf.copy()

        names = list(BEAM_ARCHETYPES)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                diff = _mean_abs_diff(renders[names[i]], renders[names[j]])
                self.assertGreater(
                    diff,
                    0.0015,
                    f"Beams {names[i]!r} and {names[j]!r} too similar: diff={diff:.6f}",
                )

    def test_beam_thickness_scale_changes_ink_mass(self) -> None:
        sp = 20.0
        masses = {}
        for sc in (0.45, 2.10):
            rng = np.random.default_rng(19)
            canvas = StrokeCanvas(200, 100, BG_WHITE, rng=rng)
            style = _base_clean_style(beam_archetype="straight_classic", beam_thickness_scale=sc)
            draw_beam(canvas, 30.0, 50.0, 170.0, 50.0, sp, INK_BLACK, style, rng)
            masses[sc] = int(np.sum(_ink_mask(canvas.buf)))
        self.assertGreater(masses[2.10], masses[0.45] * 2.2)

    def test_all_5_ledger_styles_render_and_are_pairwise_distinct(self) -> None:
        sp = 20.0
        renders = {}
        for lstyle in LEDGER_STYLES:
            rng = np.random.default_rng(404)
            canvas = StrokeCanvas(140, 100, BG_WHITE, rng=rng)
            style = _base_clean_style(ledger_style=lstyle)
            draw_ledger_line(canvas, 70.0, 50.0, sp, INK_BLACK, style, rng)
            draw_ledger_lines(canvas, 70.0, 50.0, sp, [-4, 12], INK_BLACK, style, rng)
            ink_px = int(np.sum(_ink_mask(canvas.buf)))
            self.assertGreater(ink_px, 25, f"Ledger style {lstyle!r} drew too little ink")
            renders[lstyle] = canvas.buf.copy()

        names = list(LEDGER_STYLES)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                diff = _mean_abs_diff(renders[names[i]], renders[names[j]])
                self.assertGreater(
                    diff,
                    0.0006,
                    f"Ledger styles {names[i]!r} and {names[j]!r} too similar: diff={diff:.6f}",
                )


_BEAM_RICH_MUSICXML = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 3.1 Partwise//EN" "http://www.musicxml.org/dtds/partwise.dtd">
<score-partwise version="3.1">
  <work><work-title>Beam Leveling and Grouping Test</work-title></work>
  <part-list><score-part id="P1"><part-name>Piano</part-name></score-part></part-list>
  <part id="P1">
    <measure number="1">
      <attributes>
        <divisions>8</divisions>
        <key><fifths>0</fifths></key>
        <time><beats>4</beats><beat-type>4</beat-type></time>
        <clef><sign>G</sign><line>2</line></clef>
      </attributes>
      <note><pitch><step>C</step><octave>4</octave></pitch><duration>4</duration><type>eighth</type><stem>up</stem><beam number="1">begin</beam></note>
      <note><pitch><step>E</step><octave>4</octave></pitch><duration>4</duration><type>eighth</type><stem>up</stem><beam number="1">continue</beam></note>
      <note><pitch><step>G</step><octave>4</octave></pitch><duration>4</duration><type>eighth</type><stem>up</stem><beam number="1">continue</beam></note>
      <note><pitch><step>B</step><octave>4</octave></pitch><duration>4</duration><type>eighth</type><stem>up</stem><beam number="1">end</beam></note>
      <note><pitch><step>C</step><octave>5</octave></pitch><duration>2</duration><type>16th</type><stem>down</stem><beam number="1">begin</beam><beam number="2">begin</beam></note>
      <note><pitch><step>E</step><octave>5</octave></pitch><duration>2</duration><type>16th</type><stem>down</stem><beam number="1">continue</beam><beam number="2">continue</beam></note>
      <note><pitch><step>G</step><octave>5</octave></pitch><duration>2</duration><type>16th</type><stem>down</stem><beam number="1">continue</beam><beam number="2">continue</beam></note>
      <note><pitch><step>B</step><octave>5</octave></pitch><duration>2</duration><type>16th</type><stem>down</stem><beam number="1">end</beam><beam number="2">end</beam></note>
      <note><pitch><step>A</step><octave>5</octave></pitch><duration>2</duration><type>16th</type><stem>down</stem><beam number="1">begin</beam><beam number="2">begin</beam></note>
      <note><pitch><step>F</step><octave>5</octave></pitch><duration>2</duration><type>16th</type><stem>down</stem><beam number="1">continue</beam><beam number="2">continue</beam></note>
      <note><pitch><step>D</step><octave>5</octave></pitch><duration>2</duration><type>16th</type><stem>down</stem><beam number="1">continue</beam><beam number="2">continue</beam></note>
      <note><pitch><step>B</step><octave>4</octave></pitch><duration>2</duration><type>16th</type><stem>down</stem><beam number="1">end</beam><beam number="2">end</beam></note>
    </measure>
  </part>
</score-partwise>
"""


class TestBeamLevelingAndGrouping(unittest.TestCase):
    def tearDown(self) -> None:
        reload_style_packs(reset_to_builtins=True)

    def test_registries_and_234_styles_diversity(self) -> None:
        self.assertEqual(len(BEAM_SLOPE_STYLES), 10)
        self.assertEqual(len(set(BEAM_SLOPE_STYLES)), 10)
        self.assertEqual(len(BEAM_GROUPING_MODES), 8)
        self.assertEqual(len(set(BEAM_GROUPING_MODES)), 8)

        seen_slopes = set()
        seen_modes = set()
        for name in list_styles():
            s = validate_style(get_style(name), name=name)
            sl = resolve_beam_slope_style(s)
            gm = resolve_beam_grouping_mode(s)
            self.assertIn(sl, BEAM_SLOPE_STYLES)
            self.assertIn(gm, BEAM_GROUPING_MODES)
            # Deterministic across repeated calls
            self.assertEqual(sl, resolve_beam_slope_style(s))
            self.assertEqual(gm, resolve_beam_grouping_mode(s))
            seen_slopes.add(sl)
            seen_modes.add(gm)
        self.assertEqual(seen_slopes, set(BEAM_SLOPE_STYLES))
        self.assertEqual(seen_modes, set(BEAM_GROUPING_MODES))

    def test_validation_interpolation_and_sampling_10x8(self) -> None:
        base = get_style("baroque_quill_neat")
        valid = dict(base)
        valid.update(
            beam_slope_style="exaggerated_swoop",
            beam_grouping_mode="french_broken_secondary",
            beam_split_to_flag_prob=0.35,
        )
        out = validate_style(valid)
        self.assertEqual(out["beam_slope_style"], "exaggerated_swoop")
        self.assertEqual(out["beam_grouping_mode"], "french_broken_secondary")
        self.assertAlmostEqual(out["beam_split_to_flag_prob"], 0.35, places=6)

        for key, bad_val in (
            ("beam_slope_style", "not_a_slope"),
            ("beam_grouping_mode", "not_a_mode"),
            ("beam_split_to_flag_prob", -0.1),
            ("beam_split_to_flag_prob", 1.25),
        ):
            bad = dict(base)
            bad[key] = bad_val
            with self.assertRaises(ValueError):
                validate_style(bad)

        sa = _base_clean_style(
            beam_slope_style="horizontal_flat",
            beam_grouping_mode="all_flags_vocal",
            beam_split_to_flag_prob=0.10,
        )
        sb = _base_clean_style(
            beam_slope_style="steep_contour",
            beam_grouping_mode="long_phrase_beam",
            beam_split_to_flag_prob=0.90,
        )
        mid = interpolate_styles(sa, sb, 0.5)
        self.assertIn(mid["beam_slope_style"], ("horizontal_flat", "steep_contour"))
        self.assertIn(mid["beam_grouping_mode"], ("all_flags_vocal", "long_phrase_beam"))
        self.assertAlmostEqual(mid["beam_split_to_flag_prob"], 0.50, places=5)

        rng = np.random.default_rng(20261124)
        seen_slopes, seen_modes, seen_pairs = set(), set(), set()
        for _ in range(500):
            sc = validate_style(sample_random_scribe(rng))
            seen_slopes.add(sc["beam_slope_style"])
            seen_modes.add(sc["beam_grouping_mode"])
            seen_pairs.add((sc["beam_slope_style"], sc["beam_grouping_mode"]))
            self.assertTrue(0.0 <= sc["beam_split_to_flag_prob"] <= 1.0)
        self.assertEqual(seen_slopes, set(BEAM_SLOPE_STYLES))
        self.assertEqual(seen_modes, set(BEAM_GROUPING_MODES))
        self.assertGreaterEqual(len(seen_pairs), 70)

    def test_compute_beam_geometry_distinct_across_10_slope_styles(self) -> None:
        sp = 20.0
        # Ascending 4-note stem-up group (tip_y decreases left-to-right: 110 -> 50)
        grp = [
            (40.0, 180.0, 110.0, 1, "16th"),
            (85.0, 160.0, 90.0, 1, "16th"),
            (130.0, 140.0, 70.0, 1, "16th"),
            (175.0, 120.0, 50.0, 1, "16th"),
        ]
        geoms = {
            st: compute_beam_geometry(grp, sp, _base_clean_style(beam_slope_style=st))
            for st in BEAM_SLOPE_STYLES
        }
        # horizontal_flat has zero slope
        self.assertAlmostEqual(geoms["horizontal_flat"]["y0"], geoms["horizontal_flat"]["y1"], places=5)
        # steep_contour steeper than engraver_gentle, exaggerated_swoop steeper still
        dy_gentle = abs(geoms["engraver_gentle"]["y1"] - geoms["engraver_gentle"]["y0"])
        dy_steep = abs(geoms["steep_contour"]["y1"] - geoms["steep_contour"]["y0"])
        dy_swoop = abs(geoms["exaggerated_swoop"]["y1"] - geoms["exaggerated_swoop"]["y0"])
        self.assertGreater(dy_steep, dy_gentle * 1.4)
        self.assertGreater(dy_swoop, dy_steep * 1.1)
        # reverse_quirk reverses slope direction relative to engraver_gentle
        self.assertLess(
            (geoms["reverse_quirk"]["y1"] - geoms["reverse_quirk"]["y0"])
            * (geoms["engraver_gentle"]["y1"] - geoms["engraver_gentle"]["y0"]),
            0.0,
        )
        # concave_valley vs convex_arch have opposite parabolic curve sag
        self.assertGreater(geoms["concave_valley"]["curve_sag_px"], 5.0)
        self.assertLess(geoms["convex_arch"]["curve_sag_px"], -5.0)
        # stepped_secondary_tilt produces non-zero secondary tilt
        self.assertGreater(abs(geoms["stepped_secondary_tilt"]["secondary_tilt_px"]), 4.0)
        # high_floating_beam sits higher (smaller y for stem-up) than engraver_gentle, tight_hug_beam sits lower (larger y)
        mid_high = 0.5 * (geoms["high_floating_beam"]["y0"] + geoms["high_floating_beam"]["y1"])
        mid_gentle = 0.5 * (geoms["engraver_gentle"]["y0"] + geoms["engraver_gentle"]["y1"])
        mid_tight = 0.5 * (geoms["tight_hug_beam"]["y0"] + geoms["tight_hug_beam"]["y1"])
        self.assertLess(mid_high, mid_gentle - 15.0)
        self.assertGreater(mid_tight, mid_gentle + 10.0)

    def test_plan_measure_beam_groups_all_8_modes_and_split_prob(self) -> None:
        # 8 beamable 16th notes split into two 4-note beat groups
        events = [
            NoteEvent(
                staff_steps=[i % 6],
                accidentals=[None],
                duration_kind="16th",
                beat_offset=i * 0.25,
                beam_group=(1 if i < 4 else 2),
                voice=1,
            )
            for i in range(8)
        ]

        g_std = plan_measure_beam_groups(events, _base_clean_style(beam_grouping_mode="standard_beat"), np.random.default_rng(1))
        self.assertEqual(len({g for g in g_std if g is not None}), 2)
        self.assertTrue(all(g is not None for g in g_std))

        g_vocal = plan_measure_beam_groups(events, _base_clean_style(beam_grouping_mode="all_flags_vocal"), np.random.default_rng(1))
        self.assertTrue(all(g is None for g in g_vocal))

        g_long = plan_measure_beam_groups(events, _base_clean_style(beam_grouping_mode="long_phrase_beam"), np.random.default_rng(1))
        self.assertEqual(len({g for g in g_long if g is not None}), 1)
        self.assertTrue(all(g is not None for g in g_long))

        g_pair = plan_measure_beam_groups(events, _base_clean_style(beam_grouping_mode="pair_beams_2"), np.random.default_rng(1))
        self.assertEqual(len({g for g in g_pair if g is not None}), 4)
        self.assertTrue(all(g is not None for g in g_pair))

        g_31 = plan_measure_beam_groups(events, _base_clean_style(beam_grouping_mode="uneven_3_plus_1"), np.random.default_rng(1))
        # First 4 -> 3 beamed + 1 flagged; second 4 -> 1 flagged + 3 beamed
        self.assertEqual(sum(1 for g in g_31 if g is None), 2)
        self.assertEqual(len({g for g in g_31 if g is not None}), 2)

        g_mix_a = plan_measure_beam_groups(
            events,
            _base_clean_style(beam_grouping_mode="mixed_split_flags", beam_split_to_flag_prob=0.55),
            np.random.default_rng(123),
        )
        g_mix_b = plan_measure_beam_groups(
            events,
            _base_clean_style(beam_grouping_mode="mixed_split_flags", beam_split_to_flag_prob=0.55),
            np.random.default_rng(123),
        )
        self.assertEqual(g_mix_a, g_mix_b)  # deterministic by seed
        self.assertGreater(sum(1 for g in g_mix_a if g is None), 0)

    def test_end_to_end_render_all_10_slopes_and_8_grouping_modes(self) -> None:
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            xml_path = tmp / "beam_rich.musicxml"
            xml_path.write_text(_BEAM_RICH_MUSICXML, encoding="utf-8")

            slope_imgs = {}
            for sl in BEAM_SLOPE_STYLES:
                st = _base_clean_style(
                    beam_slope_style=sl,
                    beam_grouping_mode="standard_beat",
                    beam_archetype="straight_classic",
                    stem_archetype="straight_classic",
                    flag_archetype="classic_droop",
                    staff_curve_px=0.0,
                    staff_slope_deg=0.0,
                    sp=18.0,
                    margin_x=36.0,
                    margin_top=70.0,
                    margin_bottom=40.0,
                )
                reg = f"_test_beam_slope_{sl}"
                register_style(reg, st, overwrite=True)
                out_png = tmp / f"slope_{sl}.png"
                render_score_handwritten(xml_path, reg, out_png, seed=77, page_w=960, page_h=240)
                arr = np.asarray(Image.open(out_png).convert("RGB"), dtype=np.float32) / 255.0
                slope_imgs[sl] = arr

            # Every pair of the 10 slope styles should produce visibly distinct pixels
            sl_list = list(BEAM_SLOPE_STYLES)
            for i in range(len(sl_list)):
                for j in range(i + 1, len(sl_list)):
                    diff = _mean_abs_diff(slope_imgs[sl_list[i]], slope_imgs[sl_list[j]])
                    self.assertGreater(
                        diff,
                        0.0004,
                        f"Beam slope styles {sl_list[i]!r} and {sl_list[j]!r} too similar: diff={diff:.6f}",
                    )

            mode_imgs = {}
            for gm in BEAM_GROUPING_MODES:
                st = _base_clean_style(
                    beam_slope_style="engraver_gentle",
                    beam_grouping_mode=gm,
                    beam_split_to_flag_prob=(0.50 if gm == "mixed_split_flags" else 0.0),
                    beam_archetype="straight_classic",
                    stem_archetype="straight_classic",
                    flag_archetype="classic_droop",
                    staff_curve_px=0.0,
                    staff_slope_deg=0.0,
                    sp=18.0,
                    margin_x=36.0,
                    margin_top=70.0,
                    margin_bottom=40.0,
                )
                reg = f"_test_beam_mode_{gm}"
                register_style(reg, st, overwrite=True)
                out_png = tmp / f"mode_{gm}.png"
                render_score_handwritten(xml_path, reg, out_png, seed=88, page_w=960, page_h=240)
                arr = np.asarray(Image.open(out_png).convert("RGB"), dtype=np.float32) / 255.0
                mode_imgs[gm] = arr

            # Every pair of the 8 grouping modes should produce visibly distinct pixels
            gm_list = list(BEAM_GROUPING_MODES)
            for i in range(len(gm_list)):
                for j in range(i + 1, len(gm_list)):
                    diff = _mean_abs_diff(mode_imgs[gm_list[i]], mode_imgs[gm_list[j]])
                    self.assertGreater(
                        diff,
                        0.0004,
                        f"Beam grouping modes {gm_list[i]!r} and {gm_list[j]!r} too similar: diff={diff:.6f}",
                    )


if __name__ == "__main__":
    unittest.main()
