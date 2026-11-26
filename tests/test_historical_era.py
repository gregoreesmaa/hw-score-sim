"""Tests for Iter 13 historical era preset (`--era historical` / `historical_era=True`)."""

from __future__ import annotations

import json
import pathlib
import tempfile
import unittest

import numpy as np

from hw_score_sim import (
    HISTORICAL_ERAS,
    HISTORICAL_FORBIDDEN_MEDIA,
    HISTORICAL_IMPLEMENTS,
    ScribeSampler,
    apply_historical_era_profile,
    catalog,
    cli,
    get_style,
    list_historical_styles,
    profile_resolver,
    render_sample,
    render_score_handwritten,
)


def _make_multi_measure_xml(n_measures: int = 20) -> str:
    steps = ["C", "D", "E", "F", "G", "A", "B"]
    out = []
    for m in range(1, n_measures + 1):
        attrs = ""
        if m == 1:
            attrs = (
                "<attributes><divisions>4</divisions><key><fifths>1</fifths></key>"
                "<time><beats>4</beats><beat-type>4</beat-type></time>"
                "<clef><sign>G</sign><line>2</line></clef></attributes>"
            )
        s0, s1, s2, s3 = steps[(m * 2) % 7], steps[(m * 2 + 1) % 7], steps[(m * 2 + 2) % 7], steps[(m * 2 + 3) % 7]
        notes = (
            f"<note><pitch><step>{s0}</step><octave>4</octave></pitch><duration>4</duration><type>quarter</type></note>"
            f"<note><pitch><step>{s1}</step><octave>4</octave></pitch><duration>4</duration><type>quarter</type></note>"
            f"<note><pitch><step>{s2}</step><octave>4</octave></pitch><duration>4</duration><type>quarter</type></note>"
            f"<note><pitch><step>{s3}</step><octave>4</octave></pitch><duration>4</duration><type>quarter</type></note>"
        )
        out.append(f'    <measure number="{m}">{attrs}{notes}</measure>')
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<score-partwise version="3.1">\n'
        '  <work><work-title>Historical Era Study</work-title></work>\n'
        '  <part-list><score-part id="P1"><part-name>Violin</part-name></score-part></part-list>\n'
        '  <part id="P1">\n' + "\n".join(out) + "\n  </part>\n</score-partwise>\n"
    )


class TestHistoricalEraPreset(unittest.TestCase):
    def test_list_historical_styles_excludes_modern_jazz_film(self) -> None:
        hist = list_historical_styles()
        self.assertGreater(len(hist), 40)
        for name in hist:
            desc = catalog.describe_style(name)
            self.assertIn(desc["era"], HISTORICAL_ERAS, msg=name)
            self.assertIn(desc["implement"], HISTORICAL_IMPLEMENTS, msg=name)
            self.assertNotIn(desc["medium"], HISTORICAL_FORBIDDEN_MEDIA, msg=name)

        # filter_styles(era="historical") alias works
        via_filter = catalog.filter_styles(era="historical")
        self.assertGreaterEqual(len(via_filter), len(hist))
        for name in via_filter:
            self.assertIn(catalog.describe_style(name)["era"], HISTORICAL_ERAS, msg=name)

    def test_apply_historical_era_profile_physics(self) -> None:
        modern = get_style("modern_jazz_lead_sheet")
        hist_st = apply_historical_era_profile(modern)

        self.assertTrue(hist_st["historical_era"])
        self.assertEqual(hist_st["chord_symbols_prob"], 0.0)
        self.assertEqual(hist_st["chord_symbol_style"], "none")
        self.assertIn(hist_st["annotation_persona"], profile_resolver.HISTORICAL_CLEAN_PERSONAS)
        self.assertIn(hist_st["page_furniture_mode"], profile_resolver.HISTORICAL_CLEAN_FURNITURE)
        self.assertIn(hist_st["paper_type"], profile_resolver.HISTORICAL_CLEAN_PAPER_TYPES)
        self.assertGreaterEqual(hist_st["bleed_through"], 0.28)
        self.assertGreaterEqual(hist_st["verso_bleed_strength"], 0.28)
        self.assertGreaterEqual(hist_st["scratch_out_prob"], 0.25)
        self.assertGreaterEqual(hist_st["in_bar_crossout_prob"], 0.25)
        self.assertGreaterEqual(hist_st["ink_var"], 0.18)
        # Aged iron-gall / sepia ink is warm (R >= B) and not pure neon blue
        r, g, b = hist_st["ink_rgb"]
        self.assertGreaterEqual(r, b)
        self.assertLessEqual(r, 0.36)

    def test_sampler_era_historical_filters_pool_and_applies_profile(self) -> None:
        sampler = ScribeSampler(preset="wild_archive", master_seed=123, era="historical")
        pool = sampler._resolve_pool((1, 2, 3, 4, 5))
        self.assertTrue(pool)
        allowed = set(list_historical_styles())
        self.assertTrue(set(pool).issubset(allowed))

        st, seed = sampler.sample("quartet_001", variant_idx=0)
        self.assertIsInstance(seed, int)
        self.assertTrue(st["historical_era"])
        self.assertGreaterEqual(st["bleed_through"], 0.28)
        self.assertGreaterEqual(st["verso_bleed_strength"], 0.28)
        self.assertGreaterEqual(st["in_bar_crossout_prob"], 0.25)
        self.assertEqual(st["chord_symbols_prob"], 0.0)

    def test_render_and_cli_historical_era_with_in_bar_crossout(self) -> None:
        xml = _make_multi_measure_xml(n_measures=20)
        with tempfile.TemporaryDirectory() as td:
            xml_p = pathlib.Path(td) / "hist_study.musicxml"
            xml_p.write_text(xml, encoding="utf-8")

            # Force in-bar cross-out probability to 1.0 to verify OMR-safe rendering & mark logging
            custom = get_style("baroque_quill_neat")
            custom["in_bar_crossout_prob"] = 1.0
            out_p = pathlib.Path(td) / "hist_study.png"
            gt = render_score_handwritten(
                xml_p,
                custom,
                out_p,
                seed=77,
                page_w=1400,
                page_h=1800,
                historical_era=True,
            )
            self.assertTrue(out_p.exists())
            self.assertTrue(gt["historical_era"])
            marks = gt.get("manuscript_marks", [])
            kinds = {m.get("kind") for m in marks}
            self.assertIn("mirrored_verso_bleed", kinds)
            self.assertIn("in_bar_crossout", kinds)
            for m in marks:
                self.assertTrue(m.get("gt_safe", False))
            forbidden = {"studio_timecode", "eyeglasses_cue", "chord_symbol", "section_header"}
            self.assertTrue(kinds.isdisjoint(forbidden), f"Forbidden modern marks: {kinds}")

            # Verify render_sample(..., historical_era=True)
            sample = render_sample(
                xml_p,
                "baroque_quill_neat",
                seed=77,
                page_w=1000,
                page_h=1300,
                historical_era=True,
            )
            self.assertTrue(sample.meta["historical_era"])
            self.assertEqual(sample.image.size, (1000, 1300))

            # Verify CLI --era historical
            rc = cli.main(
                [
                    "--input",
                    str(xml_p),
                    "--styles",
                    "beethoven_hurried_autograph",
                    "--out-dir",
                    td,
                    "--page-w",
                    "1000",
                    "--page-h",
                    "1300",
                    "--era",
                    "historical",
                ]
            )
            self.assertEqual(rc, 0)
            cli_json = pathlib.Path(td) / "hist_study__beethoven_hurried_autograph.json"
            self.assertTrue(cli_json.exists())
            payload = json.loads(cli_json.read_text(encoding="utf-8"))
            self.assertTrue(payload["historical_era"])


if __name__ == "__main__":
    unittest.main()
