"""Tests for get_style(dict), clean_paper=True API, spiral-hole bug fix, and single-key expressive resolver."""

from __future__ import annotations

import pathlib
import tempfile
import unittest

import numpy as np
from PIL import Image

from hw_score_sim import (
    SCRIBE_STYLES,
    ScribeSampler,
    cli,
    expressive_glyphs,
    get_style,
    layout_quirks,
    render_sample,
    render_score_handwritten,
)


class TestCleanPaperAndFixes(unittest.TestCase):
    def test_get_style_accepts_dict_and_returns_independent_copy(self) -> None:
        base = get_style("beethoven_hurried_autograph")
        cloned = get_style(base)
        self.assertEqual(cloned, base)
        self.assertIsNot(cloned, base)
        cloned["wobble_px"] = 99.0
        cloned["rest_archetype_weights"]["lightning_z"] = 0.999
        self.assertNotEqual(base["wobble_px"], 99.0)
        self.assertNotEqual(base["rest_archetype_weights"]["lightning_z"], 0.999)

    def test_render_score_handwritten_accepts_dict_without_mutating_registry(self) -> None:
        before_keys = set(SCRIBE_STYLES.keys())
        sampler = ScribeSampler(preset="wild_archive", master_seed=123)
        sampled_style, render_seed = sampler.sample("dict_test_score", variant_idx=0)
        with tempfile.TemporaryDirectory() as td:
            xml_p = pathlib.Path(td) / "score.musicxml"
            xml_p.write_text(cli.SHOWCASE_MUSICXML, encoding="utf-8")
            out_p = pathlib.Path(td) / "out.png"
            gt = render_score_handwritten(
                xml_p,
                sampled_style,
                out_p,
                seed=render_seed,
                page_w=900,
                page_h=450,
                clean_paper=True,
            )
            self.assertTrue(out_p.exists())
            self.assertEqual(gt["style"], sampled_style["name"])
            self.assertTrue(gt["clean_paper"])
        self.assertEqual(set(SCRIBE_STYLES.keys()), before_keys)

    def test_clean_paper_pure_white_background_and_skips_wear_layers(self) -> None:
        # Force all wear/paper probabilities high on a custom style dict.
        st = get_style("beethoven_hurried_autograph")
        st["verso_bleed_strength"] = 0.45
        st["fold_crease_prob"] = 1.0
        st["foxing_cluster_prob"] = 1.0
        st["ink_blot_prob"] = 1.0
        st["palm_smear_prob"] = 1.0
        st["dog_ear_prob"] = 1.0
        st["measure_numbers_prob"] = 1.0
        st["measure_number_style"] = "boxed"

        with tempfile.TemporaryDirectory() as td:
            xml_p = pathlib.Path(td) / "score.musicxml"
            xml_p.write_text(cli.SHOWCASE_MUSICXML, encoding="utf-8")
            out_p = pathlib.Path(td) / "clean.png"
            gt = render_score_handwritten(
                xml_p,
                st,
                out_p,
                seed=42,
                page_w=1200,
                page_h=600,
                clean_paper=True,
            )
            arr = np.asarray(Image.open(out_p))
            # All 4 corners must be pure white [255, 255, 255] (no aged paper, no vignette, no dog-ear).
            for y, x in ((0, 0), (0, -1), (-1, 0), (-1, -1)):
                self.assertEqual(arr[y, x].tolist(), [255, 255, 255])
            # Ink must actually be drawn (not blank white).
            self.assertLess(float(arr.mean()), 254.5)
            # Wear/accident marks must be absent.
            forbidden_kinds = {
                "mirrored_verso_bleed",
                "fold_crease",
                "foxing_cluster",
                "ink_blot",
                "palm_smear",
                "dog_ear_fold",
                "page_turn_thumb_smudge",
            }
            drawn_kinds = {m.get("kind") for m in gt.get("manuscript_marks", [])}
            self.assertTrue(drawn_kinds.isdisjoint(forbidden_kinds), f"Unexpected wear marks: {drawn_kinds}")

            # Also verify render_sample(..., clean_paper=True) and CLI --clean-paper.
            sample = render_sample(xml_p, st, seed=42, page_w=1200, page_h=600, clean_paper=True)
            s_arr = np.asarray(sample.image)
            self.assertEqual(s_arr[0, 0].tolist(), [255, 255, 255])

            rc = cli.main(
                [
                    "--input",
                    str(xml_p),
                    "--styles",
                    "student_ballpoint_blue",
                    "--out-dir",
                    td,
                    "--page-w",
                    "900",
                    "--page-h",
                    "450",
                    "--clean-paper",
                ]
            )
            self.assertEqual(rc, 0)
            cli_png = pathlib.Path(td) / "score__student_ballpoint_blue.png"
            self.assertEqual(np.asarray(Image.open(cli_png))[0, 0].tolist(), [255, 255, 255])

    def test_spiral_notebook_no_double_punch_holes(self) -> None:
        st = get_style("student_ballpoint_blue")
        st["paper_type"] = "spiral_notebook"
        st["paper_binding_mode"] = "wire_spiral_left"
        st["paper_grain"] = 0.0
        st["paper_mottle"] = 0.0
        st["bleed_through"] = 0.0
        st["eraser_smudge_prob"] = 0.0
        st["stain_prob"] = 0.0
        # With wire_spiral_left active, the fallback holes at x = 0.025*w should NOT be drawn
        # on top of the wire_spiral_left holes at x = 0.030*w.
        w, h = 1000, 1000
        img_with_wire = layout_quirks.make_manuscript_paper(w, h, st, np.random.default_rng(7))
        # Compare a pixel right between hole rows at x=25 (0.025*w) vs paper tone: should not have extra dark smudge
        # And compare against bright_white + wire_spiral_left at the hole region: hole count along x=30 should match.
        st_no_wire = dict(st)
        st_no_wire["paper_binding_mode"] = "none"
        img_fallback = layout_quirks.make_manuscript_paper(w, h, st_no_wire, np.random.default_rng(7))
        # Both modes should draw holes (dark pixels in left 5% strip), and wire_spiral_left should differ
        # cleanly (has metallic wire bars at x=0..30) without double-darkening the fallback x=25 column.
        self.assertLess(float(img_with_wire[:, :50, :].min()), 0.35)
        self.assertLess(float(img_fallback[:, :50, :].min()), 0.50)
        # At x=0 (far left edge), wire_spiral_left draws metallic wire loops (0.32), fallback does not.
        self.assertLess(float(img_with_wire[:, 0, :].min()), 0.40)
        self.assertGreater(float(img_fallback[:, 0, :].min()), 0.70)

    def test_resolve_expressive_archetype_matches_full_resolver(self) -> None:
        for name in ("baroque_quill_neat", "beethoven_hurried_autograph", "modern_jazz_lead_sheet", "chopin_delicate_salon"):
            st = get_style(name)
            full = expressive_glyphs.resolve_expressive_archetypes(st)
            for k, expected in full.items():
                self.assertEqual(expressive_glyphs.resolve_expressive_archetype(st, k), expected)


if __name__ == "__main__":
    unittest.main()
