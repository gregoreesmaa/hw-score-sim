"""Tests for hw-score-sim v0.3.0 Hyper-Diversity Upgrade (Layers 1–4)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

import hw_score_sim as hss
from hw_score_sim import expressive_glyphs, glyph_variants, layout_quirks, manuscript_marks, profile_resolver, stroke_engine, styles


_SAMPLE_XML = """<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="3.1">
  <work><work-title>v0.3 Diversity Test</work-title></work>
  <part-list><score-part id="P1"><part-name>Violin</part-name></score-part></part-list>
  <part id="P1">
    <measure number="1">
      <attributes>
        <divisions>4</divisions>
        <key><fifths>2</fifths></key>
        <time><beats>4</beats><beat-type>4</beat-type></time>
        <clef><sign>G</sign><line>2</line></clef>
      </attributes>
      <direction placement="below"><direction-type><dynamics><mf/></dynamics></direction-type></direction>
      <note><pitch><step>D</step><octave>4</octave></pitch><duration>4</duration><type>quarter</type><notations><slur type="start" number="1"/><articulations><staccato/></articulations></notations></note>
      <note><pitch><step>F</step><alter>1</alter><octave>4</octave></pitch><duration>2</duration><type>eighth</type><beam number="1">begin</beam></note>
      <note><pitch><step>A</step><octave>4</octave></pitch><duration>2</duration><type>eighth</type><beam number="1">end</beam></note>
      <note><pitch><step>D</step><octave>5</octave></pitch><duration>8</duration><type>half</type><notations><slur type="stop" number="1"/><fermata/></notations></note>
    </measure>
    <measure number="2">
      <direction placement="below"><direction-type><wedge type="crescendo"/></direction-type></direction>
      <note><pitch><step>E</step><octave>5</octave></pitch><duration>4</duration><type>quarter</type><dot/></note>
      <note><pitch><step>D</step><octave>5</octave></pitch><duration>2</duration><type>eighth</type></note>
      <note><pitch><step>C</step><alter>1</alter><octave>5</octave></pitch><duration>2</duration><type>eighth</type></note>
      <note><pitch><step>B</step><octave>4</octave></pitch><duration>4</duration><type>quarter</type></note>
      <note><rest/><duration>4</duration><type>quarter</type></note>
      <direction placement="below"><direction-type><wedge type="stop"/></direction-type></direction>
    </measure>
  </part>
</score-partwise>
"""


class TestExpressiveGlyphs(unittest.TestCase):
    def test_86_archetypes_registered(self) -> None:
        total = sum(len(v) for v in expressive_glyphs.EXPRESSIVE_REGISTRIES.values())
        self.assertEqual(len(expressive_glyphs.EXPRESSIVE_REGISTRIES), 13)
        self.assertEqual(total, 86)

    def test_resolve_expressive_explicit_override_and_determinism(self) -> None:
        st = styles.get_style("hero_vivaldi_turin_concerto_rush")
        r1 = expressive_glyphs.resolve_expressive_archetypes(st)
        r2 = expressive_glyphs.resolve_expressive_archetypes(st)
        self.assertEqual(r1, r2)
        st_ov = dict(st, slur_archetype="dashed_editorial_slur", timesig_archetype="common_cut_c_symbol")
        r3 = expressive_glyphs.resolve_expressive_archetypes(st_ov)
        self.assertEqual(r3["slur_archetype"], "dashed_editorial_slur")
        self.assertEqual(r3["timesig_archetype"], "common_cut_c_symbol")

    def test_smoke_all_expressive_drawers(self) -> None:
        rng = np.random.default_rng(7)
        canvas = stroke_engine.StrokeCanvas(320, 220, (0.96, 0.94, 0.90), rng=rng)
        st = styles.get_style("bach_leipzig_late_quill")
        ink = st["ink_rgb"]
        for arch in expressive_glyphs.SLUR_ARCHETYPES:
            expressive_glyphs.draw_slur_variant(canvas, 20, 60, 140, 60, -1, 14.0, ink, st, rng, archetype=arch)
        for arch in expressive_glyphs.TIMESIG_ARCHETYPES:
            expressive_glyphs.draw_time_sig_variant(canvas, 4, 4, 60, 110, 14.0, ink, st, rng, archetype=arch)
        for arch in expressive_glyphs.DYNAMIC_ARCHETYPES:
            expressive_glyphs.draw_dynamic_variant(canvas, "mf", 120, 160, 14.0, ink, st, rng, archetype=arch)
        for arch in expressive_glyphs.HAIRPIN_ARCHETYPES:
            expressive_glyphs.draw_hairpin_variant(canvas, "crescendo", 180, 280, 160, 14.0, ink, st, rng, archetype=arch)
        for arch in expressive_glyphs.ARTICULATION_ARCHETYPES:
            for art in ("staccato", "accent", "fermata", "trill"):
                expressive_glyphs.draw_articulation_variant(canvas, art, 220, 80, True, 14.0, ink, st, rng, archetype=arch)
        for arch in expressive_glyphs.DOT_ARCHETYPES:
            expressive_glyphs.draw_augmentation_dot_variant(canvas, 260, 80, 14.0, ink, st, rng, archetype=arch)
        for arch in expressive_glyphs.TUPLET_ARCHETYPES:
            expressive_glyphs.draw_tuplet_variant(canvas, 30, 40, 120, 40, 3, True, 14.0, ink, st, rng, archetype=arch)
        for arch in expressive_glyphs.GRACE_SLASH_ARCHETYPES:
            expressive_glyphs.draw_grace_slash_variant(canvas, 180, 120, 10.0, ink, st, rng, archetype=arch)
        for arch in expressive_glyphs.ARPEGGIO_ARCHETYPES:
            expressive_glyphs.draw_arpeggio_variant(canvas, 290, 50, 130, 14.0, ink, st, rng, archetype=arch)
        for arch in expressive_glyphs.COURTESY_ARCHETYPES:
            expressive_glyphs.draw_courtesy_brackets_variant(canvas, 240, 130, 14.0, False, ink, st, rng, archetype=arch)
        for arch in expressive_glyphs.BRACE_ARCHETYPES:
            expressive_glyphs.draw_brace_variant(canvas, 30, 40, 180, 14.0, ink, st, rng, archetype=arch)
        for arch in expressive_glyphs.MULTIBAR_ARCHETYPES:
            expressive_glyphs.draw_multibar_rest_variant(canvas, 80, 220, 110, 8, 14.0, ink, st, rng, archetype=arch)
        self.assertLess(float(np.min(canvas.buf)), 0.6)


class TestManuscriptMarksAndSafety(unittest.TestCase):
    def test_all_personas_and_cancellations_gt_safe_and_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            xml_path = Path(td) / "test.musicxml"
            xml_path.write_text(_SAMPLE_XML, encoding="utf-8")
            base = styles.get_style("hero_mahler_vienna_blue_crayon")
            for persona in manuscript_marks.ANNOTATION_PERSONAS[1:]:
                st = dict(
                    base,
                    annotation_persona=persona,
                    overlay_density=0.95,
                    scratch_out_prob=1.0,
                    chord_symbols_prob=0.9,
                    chord_symbol_style="jazz_realbook",
                    measure_numbers_prob=1.0,
                    measure_number_style="boxed",
                    palm_smear_prob=1.0,
                    ink_blot_prob=1.0,
                    quill_test_prob=1.0,
                    caret_callout_prob=1.0,
                    verso_bleed_strength=0.4,
                    fold_crease_prob=1.0,
                    foxing_cluster_prob=1.0,
                    dog_ear_prob=1.0,
                )
                s1 = hss.render_sample(xml_path, st, seed=99, page_w=900, page_h=1100)
                s2 = hss.render_sample(xml_path, st, seed=99, page_w=900, page_h=1100)
                self.assertEqual(s1.image.tobytes(), s2.image.tobytes())
                marks = s1.meta.get("manuscript_marks", [])
                summary = s1.meta.get("manuscript_summary", {})
                self.assertGreater(len(marks), 0)
                self.assertFalse(summary.get("omr_gt_corrupted", True))
                # Verify OMR GT safety: none of the localized marks overlap active staff core boxes
                active_boxes = []
                for sys_gt in s1.meta["systems"]:
                    if sys_gt.get("is_blank_prerule"):
                        continue
                    for st_gt in sys_gt["staves"]:
                        sp = float(st_gt["sp"])
                        yc = float(st_gt["y_center"])
                        active_boxes.append((st_gt["x_left"], yc - 2.0 * sp, st_gt["x_right"], yc + 2.0 * sp))
                for m in marks:
                    self.assertTrue(m["gt_safe"])
                    if m["layer"] in ("scratch_out", "chord_symbols", "measure_numbers", "caret_callout"):
                        mx0, my0, mx1, my1 = m["bbox_xyxy"]
                        for px0, py0, px1, py1 in active_boxes:
                            overlaps = not (mx1 < px0 or mx0 > px1 or my1 < py0 or my0 > py1)
                            self.assertFalse(overlaps, f"Mark {m} overlaps active staff {px0, py0, px1, py1}")


class TestProfileResolverAndDuctus(unittest.TestCase):
    def test_profile_resolver_all_234_styles(self) -> None:
        names = styles.list_styles()
        self.assertGreaterEqual(len(names), 234)
        seen_signatures = set()
        for name in names:
            st = styles.get_style(name)
            p1 = profile_resolver.resolve_page_and_expressive_profile(st)
            p2 = profile_resolver.resolve_page_and_expressive_profile(st)
            self.assertEqual(p1, p2)
            # Hard OMR-safe pitch-slop bound: renderer part <= 0.145 sp (+ <=0.025 sp in draw_notehead <= 0.17 sp)
            self.assertLessEqual(p1["pitch_y_slop_sp"], 0.145)
            self.assertGreaterEqual(p1["pitch_y_slop_sp"], 0.0)
            sig = (
                p1["page_layout_mode"], p1["staff_ruling_mode"], p1["page_furniture_mode"],
                p1["slur_archetype"], p1["timesig_archetype"], p1["annotation_persona"],
                p1["paper_binding_mode"],
            )
            seen_signatures.add(sig)
        # Ensure high diversity across the 234 styles
        self.assertGreater(len(seen_signatures), 60)

        # Explicit override precedence
        base = styles.get_style("hero_ravel_watchmaker_micro_nib")
        ov = dict(base, annotation_persona="orchestral_conductor", pitch_y_slop_sp=0.01)
        rp = profile_resolver.resolve_page_and_expressive_profile(ov)
        self.assertEqual(rp["annotation_persona"], "orchestral_conductor")
        self.assertEqual(rp["pitch_y_slop_sp"], 0.01)

    def test_ductus_pitch_slop_and_ligatures(self) -> None:
        for prof_name, prof in styles.SPACING_DUCTUS_PROFILES.items():
            self.assertLessEqual(prof["pitch_y_slop_sp"], 0.145, f"Profile {prof_name} exceeds OMR pitch-slop bound")
        rng = np.random.default_rng(123)
        canvas = stroke_engine.StrokeCanvas(240, 120, (0.96, 0.94, 0.90), rng=rng)
        st = dict(styles.get_style("hero_vivaldi_turin_concerto_rush"), spacing_ductus_profile="cursive_flying_ligatures")
        for pat in styles.LIGATURE_PATTERNS:
            glyph_variants.draw_cursive_ligature(canvas, pat, (40.0, 60.0), (110.0, 56.0), 14.0, st["ink_rgb"], st, rng)
        self.assertLess(float(np.min(canvas.buf)), 0.95)


if __name__ == "__main__":
    unittest.main()
