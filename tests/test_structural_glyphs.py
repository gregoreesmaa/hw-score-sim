"""Unit tests for the 54 structural handwritten glyph archetypes in structural_glyphs.py."""

from __future__ import annotations

import unittest

import numpy as np

from hw_score_sim.stroke_engine import StrokeCanvas
from hw_score_sim import structural_glyphs as sg


BASE_STYLE = {
    "name": "test_scribe",
    "era": "romantic",
    "ink_rgb": (0.10, 0.09, 0.08),
    "nib_angle_deg": 32.0,
    "nib_min_ratio": 0.35,
    "stroke_scale": 1.0,
    "wobble_px": 0.45,
    "tremor_px": 0.18,
    "ink_dropout": 0.02,
    "grain_strength": 0.04,
    "ink_var": 0.02,
    "ink_pooling": 0.18,
    "slant_deg": 0.0,
}


def _ink_mask_sum(canvas: StrokeCanvas, bg: tuple[float, float, float] = (0.98, 0.96, 0.92)) -> float:
    bg_arr = np.asarray(bg, dtype=np.float32).reshape(1, 1, 3)
    diff = np.abs(canvas.buf - bg_arr).mean(axis=2)
    return float(np.sum(diff > 0.04))


def _l1_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))))


class TestStructuralGlyphs(unittest.TestCase):
    def test_registry_counts_and_uniqueness(self) -> None:
        self.assertEqual(len(sg.CLEF_ARCHETYPES), 18)
        self.assertEqual(len(sg.KEYSIG_ARCHETYPES), 18)
        self.assertEqual(len(sg.BARLINE_ARCHETYPES), 18)
        self.assertEqual(len(set(sg.CLEF_ARCHETYPES)), 18)
        self.assertEqual(len(set(sg.KEYSIG_ARCHETYPES)), 18)
        self.assertEqual(len(set(sg.BARLINE_ARCHETYPES)), 18)

    def test_resolve_structural_archetypes_explicit_and_deterministic(self) -> None:
        explicit = dict(
            BASE_STYLE,
            clef_archetype="figure_8_treble",
            keysig_archetype="jagged_zigzag_acc",
            barline_archetype="slashed_repeat_ticks",
        )
        self.assertEqual(
            sg.resolve_structural_archetypes(explicit),
            ("figure_8_treble", "jagged_zigzag_acc", "slashed_repeat_ticks"),
        )

        # Deterministic across calls for named styles
        for sname, era in [
            ("beethoven_sketch", "romantic"),
            ("chopin_salon", "romantic"),
            ("jazz_realbook", "modern"),
            ("bach_baroque", "baroque"),
            ("unknown_scribe_42", "contemporary"),
        ]:
            st = dict(BASE_STYLE, name=sname, era=era)
            r1 = sg.resolve_structural_archetypes(st)
            r2 = sg.resolve_structural_archetypes(st)
            self.assertEqual(r1, r2)
            self.assertIn(r1[0], sg.CLEF_ARCHETYPES)
            self.assertIn(r1[1], sg.KEYSIG_ARCHETYPES)
            self.assertIn(r1[2], sg.BARLINE_ARCHETYPES)

    def test_all_18_clefs_render_all_kinds_and_are_pairwise_distinct(self) -> None:
        bg = (0.98, 0.96, 0.92)
        sp = 16.0
        renders: dict[str, np.ndarray] = {}

        for arch in sg.CLEF_ARCHETYPES:
            for kind in ("G", "F", "C", "G8vb", "G8va"):
                rng = np.random.default_rng(101)
                c = StrokeCanvas(120, 160, bg, rng=rng)
                sg.draw_clef_variant(c, 60.0, 80.0, sp, kind, arch, BASE_STYLE, rng)
                ink = _ink_mask_sum(c, bg)
                self.assertGreater(ink, 45.0, f"Clef {arch}/{kind} produced too little ink ({ink})")

            # Composite strip (G, F, C) for pairwise distinctness
            rng = np.random.default_rng(2026)
            strip = StrokeCanvas(300, 150, bg, rng=rng)
            sg.draw_clef_variant(strip, 55.0, 75.0, sp, "G", arch, BASE_STYLE, rng)
            sg.draw_clef_variant(strip, 150.0, 75.0, sp, "F", arch, BASE_STYLE, rng)
            sg.draw_clef_variant(strip, 245.0, 75.0, sp, "C", arch, BASE_STYLE, rng)
            renders[arch] = strip.buf.copy()

        arches = list(sg.CLEF_ARCHETYPES)
        for i in range(len(arches)):
            for j in range(i + 1, len(arches)):
                d = _l1_diff(renders[arches[i]], renders[arches[j]])
                self.assertGreater(
                    d, 0.0025, f"Clef archetypes {arches[i]} and {arches[j]} are too similar (L1={d:.6f})"
                )

    def test_all_18_keysigs_and_accidentals_render_and_are_pairwise_distinct(self) -> None:
        bg = (0.98, 0.96, 0.92)
        sp = 16.0
        y_lines = [40.0 + k * sp for k in range(5)]  # top-to-bottom
        renders: dict[str, np.ndarray] = {}

        # Zero fifths should return x_start and draw nothing
        c0 = StrokeCanvas(120, 120, bg, rng=np.random.default_rng(7))
        end_x = sg.draw_keysig_variant(c0, 20.0, y_lines, sp, 0, "G", "classic_neat", BASE_STYLE, np.random.default_rng(7))
        self.assertEqual(end_x, 20.0)
        self.assertEqual(_ink_mask_sum(c0, bg), 0.0)

        for arch in sg.KEYSIG_ARCHETYPES:
            for acc in ("sharp", "flat", "natural", "double_sharp", "double_flat"):
                rng = np.random.default_rng(303)
                c = StrokeCanvas(90, 110, bg, rng=rng)
                sg.draw_accidental_variant(c, 45.0, 55.0, sp, acc, arch, BASE_STYLE, rng)
                ink = _ink_mask_sum(c, bg)
                self.assertGreater(ink, 18.0, f"Accidental {arch}/{acc} produced too little ink ({ink})")

            # Composite strip: 4 sharps + 4 flats + natural
            rng = np.random.default_rng(404)
            strip = StrokeCanvas(380, 145, bg, rng=rng)
            x1 = sg.draw_keysig_variant(strip, 30.0, y_lines, sp, 4, "G", arch, BASE_STYLE, rng)
            self.assertGreater(x1, 30.0)
            x2 = sg.draw_keysig_variant(strip, 195.0, y_lines, sp, -4, "F", arch, BASE_STYLE, rng)
            self.assertGreater(x2, 195.0)
            sg.draw_accidental_variant(strip, 345.0, y_lines[2], sp, "natural", arch, BASE_STYLE, rng)
            renders[arch] = strip.buf.copy()

        arches = list(sg.KEYSIG_ARCHETYPES)
        for i in range(len(arches)):
            for j in range(i + 1, len(arches)):
                d = _l1_diff(renders[arches[i]], renders[arches[j]])
                self.assertGreater(
                    d, 0.0018, f"Keysig archetypes {arches[i]} and {arches[j]} are too similar (L1={d:.6f})"
                )

    def test_all_18_barlines_render_all_kinds_and_are_pairwise_distinct(self) -> None:
        bg = (0.98, 0.96, 0.92)
        sp = 16.0
        y_top, y_bot = 36.0, 36.0 + 4.0 * sp
        renders: dict[str, np.ndarray] = {}

        for arch in sg.BARLINE_ARCHETYPES:
            for bkind in ("single", "double", "final", "repeat_start", "repeat_end", "repeat_both"):
                rng = np.random.default_rng(505)
                c = StrokeCanvas(110, 140, bg, rng=rng)
                sg.draw_barline_variant(c, 55.0, y_top, y_bot, sp, bkind, arch, BASE_STYLE, rng)
                ink = _ink_mask_sum(c, bg)
                self.assertGreater(ink, 22.0, f"Barline {arch}/{bkind} produced too little ink ({ink})")

            # Composite strip: single, double, final, repeat_both
            rng = np.random.default_rng(606)
            strip = StrokeCanvas(340, 140, bg, rng=rng)
            sg.draw_barline_variant(strip, 45.0, y_top, y_bot, sp, "single", arch, BASE_STYLE, rng)
            sg.draw_barline_variant(strip, 120.0, y_top, y_bot, sp, "double", arch, BASE_STYLE, rng)
            sg.draw_barline_variant(strip, 200.0, y_top, y_bot, sp, "final", arch, BASE_STYLE, rng)
            sg.draw_barline_variant(strip, 285.0, y_top, y_bot, sp, "repeat_both", arch, BASE_STYLE, rng)
            renders[arch] = strip.buf.copy()

        arches = list(sg.BARLINE_ARCHETYPES)
        for i in range(len(arches)):
            for j in range(i + 1, len(arches)):
                d = _l1_diff(renders[arches[i]], renders[arches[j]])
                self.assertGreater(
                    d, 0.0012, f"Barline archetypes {arches[i]} and {arches[j]} are too similar (L1={d:.6f})"
                )

    def test_engine_wrapper_and_y_lines_formats(self) -> None:
        class _EngineWrapper:
            def __init__(self, canvas: StrokeCanvas) -> None:
                self.canvas = canvas

        bg = (0.98, 0.96, 0.92)
        sp = 15.0
        for y_spec in (
            [30.0, 45.0, 60.0, 75.0, 90.0],  # top-to-bottom
            [90.0, 75.0, 60.0, 45.0, 30.0],  # bottom-to-top
            60.0,  # scalar y_center
            lambda x, li: 60.0 + li * sp,  # callable y_fn
        ):
            rng = np.random.default_rng(808)
            c = StrokeCanvas(220, 120, bg, rng=rng)
            eng = _EngineWrapper(c)
            end_x = sg.draw_keysig_variant(eng, 30.0, y_spec, sp, 3, "G", "classic_neat", BASE_STYLE, rng)
            self.assertGreater(end_x, 50.0)
            self.assertGreater(_ink_mask_sum(c, bg), 60.0)

    def test_styles_wiring_validation_interpolation_and_random_sampling(self) -> None:
        from hw_score_sim import styles as styles_mod

        # Re-exports match structural_glyphs
        self.assertEqual(styles_mod.CLEF_ARCHETYPES, sg.CLEF_ARCHETYPES)
        self.assertEqual(styles_mod.KEYSIG_ARCHETYPES, sg.KEYSIG_ARCHETYPES)
        self.assertEqual(styles_mod.BARLINE_ARCHETYPES, sg.BARLINE_ARCHETYPES)

        # Validation accepts valid optional keys and rejects invalid ones
        base = styles_mod.get_style("baroque_quill_neat")
        valid = styles_mod.validate_style(
            dict(
                base,
                clef_archetype="figure_8_treble",
                keysig_archetype="wide_spread_keysig",
                barline_archetype="slashed_repeat_ticks",
            ),
            name="test_struct_valid",
        )
        self.assertEqual(valid["clef_archetype"], "figure_8_treble")
        self.assertEqual(valid["keysig_archetype"], "wide_spread_keysig")
        self.assertEqual(valid["barline_archetype"], "slashed_repeat_ticks")

        for bad_kw in (
            {"clef_archetype": "not_a_clef"},
            {"keysig_archetype": "not_a_keysig"},
            {"barline_archetype": "not_a_barline"},
        ):
            with self.assertRaises(ValueError):
                styles_mod.validate_style(dict(base, **bad_kw), name="test_struct_bad")

        # interpolate_styles preserves explicit structural archetypes at endpoints
        sa = dict(
            base,
            clef_archetype="tall_spindly_g",
            keysig_archetype="crammed_tight_cluster",
            barline_archetype="tall_overshoot",
        )
        sb = dict(
            styles_mod.get_style("modern_jazz_lead_sheet"),
            clef_archetype="jazz_realbook_bold",
            keysig_archetype="jazz_heavy_marker_acc",
            barline_archetype="heavy_thick_marker",
        )
        m0 = styles_mod.interpolate_styles(sa, sb, 0.0)
        self.assertEqual(m0["clef_archetype"], "tall_spindly_g")
        self.assertEqual(m0["keysig_archetype"], "crammed_tight_cluster")
        self.assertEqual(m0["barline_archetype"], "tall_overshoot")

        m1 = styles_mod.interpolate_styles(sa, sb, 1.0)
        self.assertEqual(m1["clef_archetype"], "jazz_realbook_bold")
        self.assertEqual(m1["keysig_archetype"], "jazz_heavy_marker_acc")
        self.assertEqual(m1["barline_archetype"], "heavy_thick_marker")

        # sample_random_scribe covers all 54 archetypes over enough draws
        rng = np.random.default_rng(202611)
        seen_c: set[str] = set()
        seen_k: set[str] = set()
        seen_b: set[str] = set()
        for _ in range(240):
            sc = styles_mod.sample_random_scribe(rng, jitter=0.25)
            self.assertIn(sc["clef_archetype"], sg.CLEF_ARCHETYPES)
            self.assertIn(sc["keysig_archetype"], sg.KEYSIG_ARCHETYPES)
            self.assertIn(sc["barline_archetype"], sg.BARLINE_ARCHETYPES)
            seen_c.add(sc["clef_archetype"])
            seen_k.add(sc["keysig_archetype"])
            seen_b.add(sc["barline_archetype"])
        self.assertEqual(seen_c, set(sg.CLEF_ARCHETYPES))
        self.assertEqual(seen_k, set(sg.KEYSIG_ARCHETYPES))
        self.assertEqual(seen_b, set(sg.BARLINE_ARCHETYPES))

        # All registered styles resolve deterministically to valid archetypes
        all_names = styles_mod.list_styles()
        self.assertGreaterEqual(len(all_names), 20)
        for sname in all_names:
            c, k, b = sg.resolve_structural_archetypes(styles_mod.get_style(sname))
            self.assertIn(c, sg.CLEF_ARCHETYPES)
            self.assertIn(k, sg.KEYSIG_ARCHETYPES)
            self.assertIn(b, sg.BARLINE_ARCHETYPES)

    def test_glyph_variants_delegation_and_legacy_compat(self) -> None:
        from hw_score_sim import glyph_variants as gv

        bg = (0.98, 0.96, 0.92)
        sp = 16.0
        y_lines = [40.0 + k * sp for k in range(5)]

        # Legacy clef_archetype_weights still works
        for legacy in ("ornate", "simplified", "hurried"):
            rng = np.random.default_rng(11)
            c = StrokeCanvas(120, 140, bg, rng=rng)
            gv.draw_clef(c, "G-2", 60.0, 70.0, sp, (0.1, 0.1, 0.1), dict(BASE_STYLE, clef_archetype_weights={legacy: 1.0}), rng)
            self.assertGreater(_ink_mask_sum(c, bg), 40.0)

        # Explicit clef/keysig/barline archetypes alter glyph_variants outputs
        c_a = StrokeCanvas(260, 140, bg, rng=np.random.default_rng(77))
        c_b = StrokeCanvas(260, 140, bg, rng=np.random.default_rng(77))
        st_a = dict(BASE_STYLE, clef_archetype="tall_spindly_g", keysig_archetype="crammed_tight_cluster", barline_archetype="hairline_faint")
        st_b = dict(BASE_STYLE, clef_archetype="jazz_realbook_bold", keysig_archetype="wide_spread_keysig", barline_archetype="heavy_thick_marker")
        for canvas, st in ((c_a, st_a), (c_b, st_b)):
            r = np.random.default_rng(77)
            gv.draw_clef(canvas, "G8vb", 40.0, 72.0, sp, (0.1, 0.1, 0.1), st, r)
            end_x = gv.draw_key_sig(canvas, 80.0, y_lines, sp, 3, "G-2", (0.1, 0.1, 0.1), st, r)
            self.assertGreater(end_x, 100.0)
            gv.draw_accidental(canvas, "double_sharp", 185.0, 72.0, sp, (0.1, 0.1, 0.1), st, r)
            gv.draw_barline(canvas, 225.0, 40.0, 104.0, sp, "repeat-right", (0.1, 0.1, 0.1), st, r)
        self.assertGreater(_l1_diff(c_a.buf, c_b.buf), 0.005)

    def test_precedence_baroque_tremor_jazz_over_wild_and_heavy_stroke(self) -> None:
        # Wild Baroque style must get baroque_ornate_scroll, not beethoven_wild_slash
        wild_baroque = dict(BASE_STYLE, name="wild_hist_vivaldi_baroque", era="baroque", clef_ornate=0.90)
        c, k, b = sg.resolve_structural_archetypes(wild_baroque)
        self.assertEqual(c, "baroque_ornate_scroll")
        self.assertEqual(k, "baroque_vertical_flat")
        self.assertEqual(b, "tapered_quill_bar")

        # Heavy Bach iron-gall (stroke_scale >= 1.45) must get baroque_vertical_flat, not jazz_heavy_marker_acc
        heavy_bach = dict(BASE_STYLE, name="bach_thick_iron_gall", era="baroque", stroke_scale=1.52)
        c2, k2, b2 = sg.resolve_structural_archetypes(heavy_bach)
        self.assertEqual(c2, "baroque_ornate_scroll")
        self.assertEqual(k2, "baroque_vertical_flat")
        self.assertEqual(b2, "tapered_quill_bar")

        # Wild arthritis tremor must get tremor_shaky_clef, not beethoven_wild_slash
        wild_tremor = dict(BASE_STYLE, name="wild_ext_arthritis_tremor", era="contemporary", tremor_px=2.1)
        c3, _, b3 = sg.resolve_structural_archetypes(wild_tremor)
        self.assertEqual(c3, "tremor_shaky_clef")
        self.assertEqual(b3, "s_wobble_freehand")

        # Wild Real Book must get jazz_realbook_bold, not beethoven_wild_slash
        wild_jazz = dict(BASE_STYLE, name="wild_vern_realbook_70s_felt_tip", era="1970s_jazz")
        c4, k4, b4 = sg.resolve_structural_archetypes(wild_jazz)
        self.assertEqual(c4, "jazz_realbook_bold")
        self.assertEqual(k4, "jazz_heavy_marker_acc")
        self.assertEqual(b4, "heavy_thick_marker")


if __name__ == "__main__":
    unittest.main()
