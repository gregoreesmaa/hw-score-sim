"""Unit tests for the 30 brand-new handwritten pause/rest variations in rest_variants.py."""

from __future__ import annotations

import unittest

import numpy as np

from hw_score_sim.glyph_variants import EIGHTH_REST_ARCHETYPES, QUARTER_REST_ARCHETYPES
from hw_score_sim.rest_variants import (
    ALL_NEW_PAUSE_ARCHETYPES,
    BAR_REST_ARCHETYPES,
    FLAGGED_REST_ARCHETYPES,
    NEW_QUARTER_REST_ARCHETYPES,
    PAUSE_ARCHETYPE_DESCRIPTIONS,
    draw_pause_variant,
)
from hw_score_sim.stroke_engine import StrokeCanvas
from hw_score_sim.styles import REST_ARCHETYPES


PAPER_BG = (0.98, 0.96, 0.92)
INK_RGB = (0.10, 0.09, 0.08)

BASE_STYLE = {
    "name": "test_pause_scribe",
    "era": "romantic",
    "ink_rgb": INK_RGB,
    "nib_angle_deg": 32.0,
    "nib_min_ratio": 0.34,
    "stroke_scale": 1.0,
    "wobble_px": 0.42,
    "tremor_px": 0.18,
    "ink_dropout": 0.02,
    "grain_strength": 0.04,
    "ink_var": 0.02,
    "ink_pooling": 0.22,
    "slant_deg": 0.0,
}


def _ink_pixels(canvas: StrokeCanvas, bg: tuple[float, float, float] = PAPER_BG) -> float:
    bg_arr = np.asarray(bg, dtype=np.float32).reshape(1, 1, 3)
    diff = np.abs(canvas.buf - bg_arr).mean(axis=2)
    return float(np.sum(diff > 0.04))


def _l1_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))))


class TestRestVariantsRegistry(unittest.TestCase):
    def test_exact_counts_and_disjoint_registries(self) -> None:
        self.assertEqual(len(NEW_QUARTER_REST_ARCHETYPES), 18)
        self.assertEqual(len(FLAGGED_REST_ARCHETYPES), 6)
        self.assertEqual(len(BAR_REST_ARCHETYPES), 6)
        self.assertEqual(len(ALL_NEW_PAUSE_ARCHETYPES), 30)
        self.assertEqual(len(set(ALL_NEW_PAUSE_ARCHETYPES)), 30)
        self.assertEqual(
            set(ALL_NEW_PAUSE_ARCHETYPES),
            set(PAUSE_ARCHETYPE_DESCRIPTIONS.keys()),
        )

    def test_zero_collisions_with_existing_rest_archetypes(self) -> None:
        existing = set(REST_ARCHETYPES) | set(QUARTER_REST_ARCHETYPES) | set(EIGHTH_REST_ARCHETYPES)
        overlap = existing.intersection(set(ALL_NEW_PAUSE_ARCHETYPES))
        self.assertEqual(overlap, set(), f"Found unexpected archetype collisions: {overlap}")


class TestRestVariantsRendering(unittest.TestCase):
    def test_all_18_quarter_rests_render_and_are_pairwise_distinct(self) -> None:
        sp = 18.0
        renders: dict[str, np.ndarray] = {}
        for arch in NEW_QUARTER_REST_ARCHETYPES:
            rng = np.random.default_rng(1001)
            c = StrokeCanvas(110, 130, PAPER_BG, rng=rng)
            used = draw_pause_variant(c, "quarter", 55.0, 65.0, sp, INK_RGB, BASE_STYLE, rng, archetype=arch)
            self.assertEqual(used, arch)
            ink = _ink_pixels(c)
            self.assertGreater(ink, 20.0, f"Quarter rest {arch!r} produced too little ink ({ink})")
            renders[arch] = c.buf.copy()

        arches = list(NEW_QUARTER_REST_ARCHETYPES)
        for i in range(len(arches)):
            for j in range(i + 1, len(arches)):
                d = _l1_diff(renders[arches[i]], renders[arches[j]])
                self.assertGreater(
                    d,
                    0.0020,
                    f"Quarter rest archetypes {arches[i]!r} and {arches[j]!r} too similar (L1={d:.6f})",
                )

    def test_all_6_flagged_rests_render_all_durations_and_are_pairwise_distinct(self) -> None:
        sp = 18.0
        renders: dict[str, np.ndarray] = {}
        for arch in FLAGGED_REST_ARCHETYPES:
            for kind in ("eighth", "16th", "32nd"):
                rng = np.random.default_rng(2002)
                c = StrokeCanvas(110, 140, PAPER_BG, rng=rng)
                used = draw_pause_variant(c, kind, 55.0, 70.0, sp, INK_RGB, BASE_STYLE, rng, archetype=arch)
                self.assertEqual(used, arch)
                ink = _ink_pixels(c)
                self.assertGreater(ink, 25.0, f"Flagged rest {arch!r}/{kind} produced too little ink ({ink})")

            # Composite strip (eighth + 16th + 32nd) for pairwise distinctness
            rng = np.random.default_rng(2026)
            strip = StrokeCanvas(260, 140, PAPER_BG, rng=rng)
            draw_pause_variant(strip, "eighth", 48.0, 70.0, sp, INK_RGB, BASE_STYLE, rng, archetype=arch)
            draw_pause_variant(strip, "16th", 130.0, 70.0, sp, INK_RGB, BASE_STYLE, rng, archetype=arch)
            draw_pause_variant(strip, "32nd", 212.0, 70.0, sp, INK_RGB, BASE_STYLE, rng, archetype=arch)
            renders[arch] = strip.buf.copy()

        arches = list(FLAGGED_REST_ARCHETYPES)
        for i in range(len(arches)):
            for j in range(i + 1, len(arches)):
                d = _l1_diff(renders[arches[i]], renders[arches[j]])
                self.assertGreater(
                    d,
                    0.0030,
                    f"Flagged rest archetypes {arches[i]!r} and {arches[j]!r} too similar (L1={d:.6f})",
                )

    def test_all_6_bar_rests_and_fermatas_render_and_are_pairwise_distinct(self) -> None:
        sp = 18.0
        renders: dict[str, np.ndarray] = {}
        for arch in BAR_REST_ARCHETYPES:
            for kind in ("whole", "half", "fermata"):
                rng = np.random.default_rng(3003)
                c = StrokeCanvas(110, 110, PAPER_BG, rng=rng)
                used = draw_pause_variant(c, kind, 55.0, 55.0, sp, INK_RGB, BASE_STYLE, rng, archetype=arch)
                self.assertEqual(used, arch)
                ink = _ink_pixels(c)
                self.assertGreater(ink, 18.0, f"Bar/fermata {arch!r}/{kind} produced too little ink ({ink})")

            # Composite strip (whole + half + fermata) for pairwise distinctness
            rng = np.random.default_rng(3026)
            strip = StrokeCanvas(260, 110, PAPER_BG, rng=rng)
            draw_pause_variant(strip, "whole", 48.0, 55.0, sp, INK_RGB, BASE_STYLE, rng, archetype=arch)
            draw_pause_variant(strip, "half", 130.0, 55.0, sp, INK_RGB, BASE_STYLE, rng, archetype=arch)
            draw_pause_variant(strip, "fermata", 212.0, 62.0, sp, INK_RGB, BASE_STYLE, rng, archetype=arch)
            renders[arch] = strip.buf.copy()

        arches = list(BAR_REST_ARCHETYPES)
        for i in range(len(arches)):
            for j in range(i + 1, len(arches)):
                d = _l1_diff(renders[arches[i]], renders[arches[j]])
                self.assertGreater(
                    d,
                    0.0025,
                    f"Bar/fermata archetypes {arches[i]!r} and {arches[j]!r} too similar (L1={d:.6f})",
                )

    def test_zero_glyph_reuse_consecutive_instances_unique_across_all_30(self) -> None:
        sp = 18.0
        cases = (
            [("quarter", a) for a in NEW_QUARTER_REST_ARCHETYPES]
            + [("eighth", a) for a in FLAGGED_REST_ARCHETYPES]
            + [("whole", a) for a in BAR_REST_ARCHETYPES]
        )
        self.assertEqual(len(cases), 30)

        for idx, (kind, arch) in enumerate(cases):
            rng = np.random.default_rng(4000 + idx)
            c1 = StrokeCanvas(110, 130, PAPER_BG, rng=np.random.default_rng(9000 + idx * 2))
            draw_pause_variant(c1, kind, 55.0, 65.0, sp, INK_RGB, BASE_STYLE, rng, archetype=arch)
            c2 = StrokeCanvas(110, 130, PAPER_BG, rng=np.random.default_rng(9000 + idx * 2 + 1))
            draw_pause_variant(c2, kind, 55.0, 65.0, sp, INK_RGB, BASE_STYLE, rng, archetype=arch)
            a, b = c1.to_uint8(), c2.to_uint8()
            diff_sum = int(np.abs(a.astype(np.int16) - b.astype(np.int16)).sum())
            self.assertGreater(
                diff_sum,
                0,
                f"Pause archetype {arch!r} ({kind}) produced identical consecutive stamps!",
            )

    def test_default_sampling_and_error_handling(self) -> None:
        rng = np.random.default_rng(555)
        c = StrokeCanvas(120, 120, PAPER_BG, rng=rng)
        for kind, expected_pool in (
            ("quarter", NEW_QUARTER_REST_ARCHETYPES),
            ("eighth", FLAGGED_REST_ARCHETYPES),
            ("16th", FLAGGED_REST_ARCHETYPES),
            ("32nd", FLAGGED_REST_ARCHETYPES),
            ("half", BAR_REST_ARCHETYPES),
            ("whole", BAR_REST_ARCHETYPES),
            ("fermata", BAR_REST_ARCHETYPES),
        ):
            chosen = draw_pause_variant(c, kind, 60.0, 60.0, 16.0, INK_RGB, BASE_STYLE, rng)
            self.assertIn(chosen, expected_pool)

        with self.assertRaises(ValueError):
            draw_pause_variant(c, "quarter", 60.0, 60.0, 16.0, INK_RGB, BASE_STYLE, rng, archetype="nonexistent")
        with self.assertRaises(ValueError):
            draw_pause_variant(c, "unknown_pause", 60.0, 60.0, 16.0, INK_RGB, BASE_STYLE, rng)


if __name__ == "__main__":
    unittest.main()
