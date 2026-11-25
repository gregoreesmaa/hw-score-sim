"""Tests for curated historical_clean mode, archival paper realism, and toxic-style quarantine."""

from __future__ import annotations

import pathlib
import tempfile
import unittest

import numpy as np
from PIL import Image

from hw_score_sim import (
    ScribeSampler,
    cli,
    get_style,
    layout_quirks,
    list_styles,
    profile_resolver,
    render_sample,
    render_score_handwritten,
    sanitize_historical_clean_style,
)


def _make_multi_measure_xml(n_measures: int = 24, two_staves: bool = False) -> str:
    """Build a valid multi-measure MusicXML score for density & unused-staff tests."""
    part_list = '<score-part id="P1"><part-name>Violin</part-name></score-part>'
    if two_staves:
        part_list += '\n    <score-part id="P2"><part-name>Cello</part-name></score-part>'

    def _measures_xml(clef_sign: str, clef_line: int, octave: int) -> str:
        out = []
        steps = ["C", "D", "E", "F", "G", "A", "B"]
        for m in range(1, n_measures + 1):
            attrs = ""
            if m == 1:
                attrs = (
                    f"<attributes><divisions>4</divisions><key><fifths>1</fifths></key>"
                    f"<time><beats>4</beats><beat-type>4</beat-type></time>"
                    f"<clef><sign>{clef_sign}</sign><line>{clef_line}</line></clef></attributes>"
                )
            s0 = steps[(m * 2) % 7]
            s1 = steps[(m * 2 + 1) % 7]
            s2 = steps[(m * 2 + 2) % 7]
            s3 = steps[(m * 2 + 3) % 7]
            notes = (
                f"<note><pitch><step>{s0}</step><octave>{octave}</octave></pitch><duration>4</duration><type>quarter</type></note>"
                f"<note><pitch><step>{s1}</step><octave>{octave}</octave></pitch><duration>2</duration><type>eighth</type><beam number=\"1\">begin</beam></note>"
                f"<note><pitch><step>{s2}</step><octave>{octave}</octave></pitch><duration>2</duration><type>eighth</type><beam number=\"1\">end</beam></note>"
                f"<note><pitch><step>{s3}</step><octave>{octave}</octave></pitch><duration>4</duration><type>quarter</type></note>"
                f"<note><pitch><step>{s1}</step><octave>{octave}</octave></pitch><duration>4</duration><type>quarter</type></note>"
            )
            out.append(f'    <measure number="{m}">{attrs}{notes}</measure>')
        return "\n".join(out)

    parts_xml = f'  <part id="P1">\n{_measures_xml("G", 2, 4)}\n  </part>'
    if two_staves:
        parts_xml += f'\n  <part id="P2">\n{_measures_xml("F", 4, 3)}\n  </part>'

    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<score-partwise version="3.1">\n'
        '  <work><work-title>Archival Quartet Study</work-title></work>\n'
        f"  <part-list>\n    {part_list}\n  </part-list>\n"
        f"{parts_xml}\n"
        "</score-partwise>\n"
    )


class TestHistoricalCleanMode(unittest.TestCase):
    def test_sanitize_quarantines_all_toxic_settings(self) -> None:
        toxic = get_style("beethoven_hurried_autograph")
        toxic.update(
            {
                "notehead_archetype": "open_loop",
                "notehead_fill_density": 0.30,
                "notehead_aspect": 0.28,
                "stem_archetype": "whip_taper",
                "flag_archetype": "zigzag_frantic",
                "beam_archetype": "stepped_wobbly",
                "beam_slope_style": "high_floating_beam",
                "barline_archetype": "bowed_curved_bar",
                "slant_deg": 17.5,
                "ligature_prob": 0.80,
                "annotation_persona": "film_studio_copyist",
                "chord_symbols_prob": 0.95,
                "chord_symbol_style": "jazz_realbook",
                "paper_type": "spiral_notebook",
                "paper_binding_mode": "wire_spiral_left",
                "paper_background_grid": "yellow_legal_pad",
                "paper_border_mode": "double_engraved_frame",
                "page_furniture_mode": "jazz_realbook_chart",
                "measure_number_style": "pit_every_bar",
                "ink_rgb": (0.10, 0.21, 0.72),
                "staff_ink_rgb": (0.12, 0.12, 0.12),
            }
        )

        clean = sanitize_historical_clean_style(toxic)

        self.assertIn(clean["notehead_archetype"], profile_resolver.HISTORICAL_CLEAN_NOTEHEADS)
        self.assertGreaterEqual(clean["notehead_fill_density"], 0.90)
        self.assertGreaterEqual(clean["notehead_aspect"], 0.80)
        self.assertIn(clean["stem_archetype"], profile_resolver.HISTORICAL_CLEAN_STEMS)
        self.assertIn(clean["flag_archetype"], profile_resolver.HISTORICAL_CLEAN_FLAGS)
        self.assertIn(clean["beam_archetype"], profile_resolver.HISTORICAL_CLEAN_BEAMS)
        self.assertIn(clean["beam_slope_style"], profile_resolver.HISTORICAL_CLEAN_BEAM_SLOPES)
        self.assertIn(clean["barline_archetype"], profile_resolver.HISTORICAL_CLEAN_BARLINES)
        self.assertLessEqual(abs(clean["slant_deg"]), 5.0)
        self.assertEqual(clean["ligature_prob"], 0.0)
        self.assertIn(clean["annotation_persona"], profile_resolver.HISTORICAL_CLEAN_PERSONAS)
        self.assertEqual(clean["chord_symbols_prob"], 0.0)
        self.assertEqual(clean["chord_symbol_style"], "none")
        self.assertIn(clean["paper_type"], profile_resolver.HISTORICAL_CLEAN_PAPER_TYPES)
        self.assertIn(clean["paper_binding_mode"], profile_resolver.HISTORICAL_CLEAN_BINDINGS)
        self.assertIn(clean["paper_background_grid"], profile_resolver.HISTORICAL_CLEAN_GRIDS)
        self.assertEqual(clean["paper_border_mode"], "none")
        self.assertIn(clean["page_furniture_mode"], profile_resolver.HISTORICAL_CLEAN_FURNITURE)
        self.assertIn(clean["measure_number_style"], profile_resolver.HISTORICAL_CLEAN_MEASURE_NUMBERS)
        self.assertGreaterEqual(clean["scanner_margin_frac"], 0.01)
        self.assertLessEqual(clean["scanner_margin_frac"], 0.04)

        # Ink-to-staff contrast: foreground ink dark, staff ink lighter warm brown
        ink_lum = 0.299 * clean["ink_rgb"][0] + 0.587 * clean["ink_rgb"][1] + 0.114 * clean["ink_rgb"][2]
        staff_lum = (
            0.299 * clean["staff_ink_rgb"][0]
            + 0.587 * clean["staff_ink_rgb"][1]
            + 0.114 * clean["staff_ink_rgb"][2]
        )
        self.assertLessEqual(ink_lum, 0.24)
        self.assertGreater(staff_lum - ink_lum, 0.20)
        self.assertGreater(clean["staff_ink_rgb"][0], clean["staff_ink_rgb"][2])

    def test_sanitize_all_registered_styles(self) -> None:
        for name in list_styles():
            st = profile_resolver.resolve_page_and_expressive_profile(get_style(name), historical_clean=True)
            self.assertLessEqual(abs(st["slant_deg"]), 5.0, msg=name)
            self.assertGreaterEqual(st["notehead_fill_density"], 0.90, msg=name)
            self.assertIn(st["notehead_archetype"], profile_resolver.HISTORICAL_CLEAN_NOTEHEADS, msg=name)
            self.assertIn(st["beam_archetype"], profile_resolver.HISTORICAL_CLEAN_BEAMS, msg=name)
            self.assertIn(st["barline_archetype"], profile_resolver.HISTORICAL_CLEAN_BARLINES, msg=name)
            self.assertEqual(st["paper_border_mode"], "none", msg=name)
            self.assertEqual(st["chord_symbols_prob"], 0.0, msg=name)
            self.assertNotIn(
                st["annotation_persona"],
                ("film_studio_copyist", "orchestral_conductor", "jazz_realbook_chart"),
                msg=name,
            )

    def test_scanner_bed_margins_and_subtle_paper_texture(self) -> None:
        st = sanitize_historical_clean_style(get_style("baroque_quill_neat"))
        st["scanner_margin_frac"] = 0.025
        paper = layout_quirks.make_manuscript_paper(800, 1000, st, np.random.default_rng(7))
        # Outer border pixels must be dark scanner bed (< 0.22)
        self.assertLess(float(paper[2, 2].mean()), 0.22)
        self.assertLess(float(paper[-3, -3].mean()), 0.22)
        # Interior paper must be warm bright archival stock (> 0.72) with low per-channel spatial std
        interior = paper[80:-80, 80:-80]
        self.assertGreater(float(interior.mean()), 0.72)
        self.assertLess(float(interior.std(axis=(0, 1)).max()), 0.045)

    def test_historical_clean_density_and_unused_staves_and_cli(self) -> None:
        xml = _make_multi_measure_xml(n_measures=28, two_staves=False)
        with tempfile.TemporaryDirectory() as td:
            xml_p = pathlib.Path(td) / "quartet.musicxml"
            xml_p.write_text(xml, encoding="utf-8")
            out_p = pathlib.Path(td) / "quartet_hc.png"
            gt = render_score_handwritten(
                xml_p,
                "beethoven_hurried_autograph",
                out_p,
                seed=101,
                page_w=1650,
                page_h=2200,
                historical_clean=True,
            )
            self.assertTrue(out_p.exists())
            self.assertTrue(gt["historical_clean"])

            active = [s for s in gt["systems"] if not s["is_blank_prerule"]]
            blank = [s for s in gt["systems"] if s["is_blank_prerule"]]
            self.assertGreaterEqual(len(active), 3)
            self.assertGreaterEqual(len(blank), 1)
            self.assertLessEqual(len(blank), 3)
            # First active systems should pack 5-8 measures/system
            for s in active[:-1]:
                n_bars = s["measure_end"] - s["measure_start"]
                self.assertGreaterEqual(n_bars, 5)
                self.assertLessEqual(n_bars, 8)

            # Verify no forbidden modern marks (timecodes, eyeglasses, jazz chords)
            forbidden_kinds = {"studio_timecode", "eyeglasses_cue", "chord_symbol", "section_header"}
            drawn_kinds = {m.get("kind") for m in gt.get("manuscript_marks", [])}
            self.assertTrue(drawn_kinds.isdisjoint(forbidden_kinds), f"Forbidden marks: {drawn_kinds}")

            # Verify render_sample + ScribeSampler + CLI --historical-clean
            sampler = ScribeSampler(preset="wild_archive", master_seed=77, historical_clean=True)
            sampled_st, s_seed = sampler.sample("quartet", variant_idx=0)
            self.assertTrue(sampled_st.get("historical_clean"))
            self.assertLessEqual(abs(sampled_st["slant_deg"]), 5.0)
            sample = render_sample(xml_p, sampled_st, seed=s_seed, page_w=1000, page_h=1300, historical_clean=True)
            self.assertEqual(sample.image.size, (1000, 1300))

            rc = cli.main(
                [
                    "--input",
                    str(xml_p),
                    "--styles",
                    "modern_jazz_lead_sheet",
                    "--out-dir",
                    td,
                    "--page-w",
                    "1000",
                    "--page-h",
                    "1300",
                    "--historical-clean",
                ]
            )
            self.assertEqual(rc, 0)
            cli_png = pathlib.Path(td) / "quartet__modern_jazz_lead_sheet.png"
            self.assertTrue(cli_png.exists())
            arr = np.asarray(Image.open(cli_png))
            # Dark scanner bed border present on historical-clean output
            self.assertLess(float(arr[2, 2].mean()), 60.0)

    def test_omit_cont_header_prob_suppresses_sys2_clef_key_preserving_gt(self) -> None:
        # Default prob: 0.0 when historical_clean=False, 0.35 when historical_clean=True
        base_st = get_style("baroque_quill_neat")
        res_off = profile_resolver.resolve_page_and_expressive_profile(base_st, historical_clean=False)
        res_hc = profile_resolver.resolve_page_and_expressive_profile(base_st, historical_clean=True)
        self.assertEqual(res_off["omit_cont_header_prob"], 0.0)
        self.assertAlmostEqual(res_hc["omit_cont_header_prob"], 0.35, places=6)

        xml = _make_multi_measure_xml(n_measures=24, two_staves=False)
        with tempfile.TemporaryDirectory() as td:
            xml_p = pathlib.Path(td) / "cont_hdr.musicxml"
            xml_p.write_text(xml, encoding="utf-8")

            s_drawn = render_sample(
                xml_p,
                "baroque_quill_neat",
                seed=314,
                page_w=1400,
                page_h=1800,
                clean_paper=True,
                historical_clean=True,
                omit_cont_header_prob=0.0,
            )
            s_omit = render_sample(
                xml_p,
                "baroque_quill_neat",
                seed=314,
                page_w=1400,
                page_h=1800,
                clean_paper=True,
                historical_clean=True,
                omit_cont_header_prob=1.0,
            )

            self.assertFalse(s_drawn.meta["omit_cont_header"])
            self.assertTrue(s_omit.meta["omit_cont_header"])

            # Exact GT MusicXML slices (and therefore exact bekern GT) must be identical
            self.assertEqual(s_drawn.musicxml_systems, s_omit.musicxml_systems)
            self.assertGreaterEqual(len(s_drawn.musicxml_systems), 2)

            active_drawn = [s for s in s_drawn.meta["systems"] if not s["is_blank_prerule"]]
            active_omit = [s for s in s_omit.meta["systems"] if not s["is_blank_prerule"]]
            self.assertGreaterEqual(len(active_drawn), 2)
            self.assertEqual(len(active_drawn), len(active_omit))

            arr_drawn = np.asarray(s_drawn.image, dtype=np.float32)
            arr_omit = np.asarray(s_omit.image, dtype=np.float32)

            def _header_ink_count(arr: np.ndarray, staff_dict: dict) -> int:
                sp = float(staff_dict["sp"])
                x0 = int(max(0, staff_dict["x_left"] + sp * 0.4))
                x1 = int(min(arr.shape[1], staff_dict["x_left"] + sp * 3.3))
                y0 = int(max(0, staff_dict["y_center"] - sp * 3.5))
                y1 = int(min(arr.shape[0], staff_dict["y_center"] + sp * 3.5))
                patch = arr[y0:y1, x0:x1]
                # Dark foreground ink (note/clef/key ink < 100 vs staff lines ~140 and white paper 255)
                return int((patch.mean(axis=2) < 95.0).sum())

            # System 1 (active index 0): clef & key signature MUST be preserved
            s0_drawn_ink = _header_ink_count(arr_drawn, active_drawn[0]["staves"][0])
            s0_omit_ink = _header_ink_count(arr_omit, active_omit[0]["staves"][0])
            self.assertGreater(s0_drawn_ink, 80)
            self.assertGreater(s0_omit_ink, 80)

            # System 2 (active index 1): visual clef & key signature suppressed on left margin
            st1_d = active_drawn[1]["staves"][0]
            st1_o = active_omit[1]["staves"][0]
            sp1 = float(st1_d["sp"])
            x0 = int(max(0, st1_d["x_left"] + sp1 * 0.4))
            x1 = int(min(arr_drawn.shape[1], st1_d["x_left"] + sp1 * 3.0))
            y0 = int(max(0, st1_d["y_center"] - sp1 * 3.5))
            y1 = int(min(arr_drawn.shape[0], st1_d["y_center"] + sp1 * 3.5))
            patch_d = arr_drawn[y0:y1, x0:x1]
            patch_o = arr_omit[y0:y1, x0:x1]
            # The two renders must differ in System 2's opening header region and have fewer dark ink pixels when omitted
            self.assertGreater(float(np.abs(patch_d - patch_o).mean()), 2.0)
            self.assertLess(
                int((patch_o.mean(axis=2) < 95.0).sum()),
                int((patch_d.mean(axis=2) < 95.0).sum()),
            )

            # Verify CLI --omit-cont-header-prob flag
            rc = cli.main(
                [
                    "--input",
                    str(xml_p),
                    "--styles",
                    "baroque_quill_neat",
                    "--out-dir",
                    td,
                    "--page-w",
                    "1000",
                    "--page-h",
                    "1300",
                    "--historical-clean",
                    "--omit-cont-header-prob",
                    "1.0",
                ]
            )
            self.assertEqual(rc, 0)
            cli_json = pathlib.Path(td) / "cont_hdr__baroque_quill_neat.json"
            self.assertTrue(cli_json.exists())
            import json as _json
            payload = _json.loads(cli_json.read_text(encoding="utf-8"))
            self.assertTrue(payload["omit_cont_header"])
            self.assertAlmostEqual(payload["omit_cont_header_prob"], 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
