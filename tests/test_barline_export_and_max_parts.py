"""Regression tests for MusicXML barline/measure export integrity and 5-6 staff layout."""

from __future__ import annotations

import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from hw_score_sim.musescore_compat import (
    MeasureData,
    NoteEvent,
    ParsedScore,
    PartData,
    RestEvent,
    export_score_to_musicxml,
)
from hw_score_sim.renderer import layout_score_pages, render_score_handwritten


def _make_multi_part_score(n_parts: int = 3, n_measures: int = 8) -> ParsedScore:
    parts = []
    for p_idx in range(n_parts):
        measures = []
        for m_idx in range(n_measures):
            # Mix pitched notes, chords, rests, and an empty measure to stress all branches.
            if m_idx == 2 and p_idx == 1:
                events = []
            elif m_idx == 3 and p_idx == 0:
                events = [RestEvent(duration_kind="whole", is_measure_rest=True, voice=1)]
            else:
                events = [
                    NoteEvent(
                        staff_steps=[2 + (m_idx % 4)],
                        accidentals=[None],
                        duration_kind="half",
                        beat_offset=0.0,
                        voice=1,
                        pitches_info=[("G", 4 if p_idx == 0 else 3, 0)],
                    ),
                    NoteEvent(
                        staff_steps=[4, 6],
                        accidentals=[None, "sharp"],
                        duration_kind="half",
                        beat_offset=2.0,
                        voice=1,
                        pitches_info=[("B", 4 if p_idx == 0 else 3, 0), ("D", 5 if p_idx == 0 else 4, 1)],
                    ),
                ]
            measures.append(
                MeasureData(
                    number=m_idx + 1,
                    clef="F-4" if p_idx == n_parts - 1 and n_parts >= 2 else "G-2",
                    key_fifths=1,
                    time_sig=(4, 4),
                    barline="final" if m_idx == n_measures - 1 else "single",
                    events=events,
                )
            )
        parts.append(PartData(name=f"Part {p_idx + 1}", staff_index=p_idx, measures=measures))
    return ParsedScore(title="Barline Export Test", source_path="synthetic", parts=parts)


class TestBarlineExportAndMaxParts(unittest.TestCase):
    def test_export_preserves_every_measure_as_direct_child_of_part(self) -> None:
        """Regression test for p_el variable shadowing that nested <measure> inside <pitch>."""
        score = _make_multi_part_score(n_parts=3, n_measures=8)
        with tempfile.TemporaryDirectory() as td:
            out_xml = export_score_to_musicxml(score, Path(td) / "slice.musicxml", measure_range=(2, 6))
            root = ET.fromstring(out_xml.read_text(encoding="utf-8"))
            parts = root.findall("part")
            self.assertEqual(len(parts), 3)
            for p_el in parts:
                direct_measures = p_el.findall("measure")
                self.assertEqual(
                    len(direct_measures),
                    4,
                    f"part {p_el.attrib.get('id')} must have exactly 4 direct <measure> children",
                )
                self.assertEqual([m.attrib["number"] for m in direct_measures], ["1", "2", "3", "4"])
                # Ensure no <measure> is ever nested inside <pitch> or <note>
                for m_el in direct_measures:
                    self.assertEqual(m_el.findall(".//measure"), [])
                    self.assertGreater(len(m_el.findall("note")), 0)

    def test_export_pads_short_part_and_empty_measure_with_measure_rest(self) -> None:
        score = _make_multi_part_score(n_parts=2, n_measures=5)
        # Truncate part 2 to 3 measures
        score.parts[1].measures = score.parts[1].measures[:3]
        with tempfile.TemporaryDirectory() as td:
            out_xml = export_score_to_musicxml(score, Path(td) / "padded.musicxml", measure_range=(1, 5))
            root = ET.fromstring(out_xml.read_text(encoding="utf-8"))
            p2 = root.findall("part")[1]
            p2_measures = p2.findall("measure")
            self.assertEqual(len(p2_measures), 4)
            # Last 2 measures of P2 were padded with measure rests
            for m_el in p2_measures[2:]:
                rests = m_el.findall("note/rest[@measure='yes']")
                self.assertEqual(len(rests), 1)

    def test_export_respects_max_parts(self) -> None:
        score = _make_multi_part_score(n_parts=7, n_measures=4)
        with tempfile.TemporaryDirectory() as td:
            out_xml = export_score_to_musicxml(score, Path(td) / "cap6.musicxml", max_parts=6)
            root = ET.fromstring(out_xml.read_text(encoding="utf-8"))
            self.assertEqual(len(root.findall("part")), 6)
            self.assertEqual(len(root.findall("part-list/score-part")), 6)

    def test_layout_and_render_six_parts_portrait_and_landscape(self) -> None:
        score = _make_multi_part_score(n_parts=6, n_measures=8)
        # Portrait
        sys_p = layout_score_pages(score, page_w=1650, page_h=2200, max_parts=6)
        self.assertGreaterEqual(len(sys_p), 1)
        self.assertEqual(len(sys_p[0].staves), 6)
        self.assertLessEqual(sys_p[0].staves[0].sp, 10.5 + 1e-6)
        # Landscape
        sys_l = layout_score_pages(score, page_w=2200, page_h=1550, max_parts=6)
        self.assertGreaterEqual(len(sys_l), 1)
        self.assertEqual(len(sys_l[0].staves), 6)
        self.assertLessEqual(sys_l[0].staves[-1].y_center + 2.0 * sys_l[0].staves[-1].sp, 1550.0)

        # Full render round-trip with 6 parts + export_train_dir
        with tempfile.TemporaryDirectory() as td:
            xml_in = export_score_to_musicxml(score, Path(td) / "six.musicxml")
            out_png = Path(td) / "six.png"
            gt = render_score_handwritten(
                score_path=xml_in,
                style_name="beethoven_hurried_autograph",
                out_png=out_png,
                seed=7,
                page_w=2200,
                page_h=1550,
                export_train_dir=Path(td) / "exp",
                clean_paper=True,
                max_parts=6,
            )
            self.assertTrue(out_png.exists())
            self.assertGreaterEqual(len(gt["systems"]), 1)
            self.assertEqual(len(gt["systems"][0]["staves"]), 6)


if __name__ == "__main__":
    unittest.main()
