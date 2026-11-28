"""Unit tests for hw-score-sim v0.8 5-axis realism upgrades:
1. Chisel-nib anisotropy, crisp iron-gall edge contrast, ink_mass pooling & quill-dab noteheads
2. Dense archival page layouts (oblong & portrait, 8-14 staves/page, non-parallel rastrum curvature)
3. Broad scribe/paper diversity (v08 style pack + ScribeSampler archival_diversity mode)
4. Full determinism, valid output ranges, and backward compatibility
"""

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np

import hw_score_sim as hss
from hw_score_sim import (
    ARCHIVAL_LAYOUT_PROFILES,
    ARCHIVAL_V08_PRESETS,
    ScribeSampler,
    apply_dense_archival_profile,
    get_archival_v08_preset,
    get_style,
    layout_quirks,
    layout_score_pages,
    list_archival_v08_presets,
    list_styles,
    parse_score,
    render_score_handwritten,
)
from hw_score_sim.glyph_variants import draw_notehead
from hw_score_sim.stroke_engine import StrokeCanvas, draw_calligraphic_polyline, draw_filled_blob
from hw_score_sim.styles import validate_style


def _make_quartet_xml(n_measures: int = 24) -> str:
    parts_meta = [
        ("P1", "Violin I", "G", 2, 5),
        ("P2", "Violin II", "G", 2, 4),
        ("P3", "Viola", "C", 3, 4),
        ("P4", "Cello", "F", 4, 3),
    ]
    part_list = "\n    ".join(
        f'<score-part id="{pid}"><part-name>{pname}</part-name></score-part>'
        for pid, pname, _, _, _ in parts_meta
    )
    steps = ["C", "D", "E", "F", "G", "A", "B"]
    parts_xml = []
    for p_idx, (pid, _, clef_s, clef_l, octv) in enumerate(parts_meta):
        measures = []
        for m in range(1, n_measures + 1):
            attrs = ""
            if m == 1:
                attrs = (
                    f"<attributes><divisions>4</divisions><key><fifths>1</fifths></key>"
                    f"<time><beats>4</beats><beat-type>4</beat-type></time>"
                    f"<clef><sign>{clef_s}</sign><line>{clef_l}</line></clef></attributes>"
                )
            notes = "".join(
                f"<note><pitch><step>{steps[(m + p_idx + k) % 7]}</step><octave>{octv}</octave></pitch>"
                f"<duration>4</duration><type>quarter</type></note>"
                for k in range(4)
            )
            measures.append(f'    <measure number="{m}">{attrs}{notes}</measure>')
        parts_xml.append(f'  <part id="{pid}">\n' + "\n".join(measures) + "\n  </part>")
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<score-partwise version="3.1">\n'
        '  <work><work-title>v08 Archival Test</work-title></work>\n'
        f"  <part-list>\n    {part_list}\n  </part-list>\n"
        + "\n".join(parts_xml)
        + "\n</score-partwise>\n"
    )


def _edge_grad_q95(buf: np.ndarray) -> float:
    gray = 0.299 * buf[:, :, 0] + 0.587 * buf[:, :, 1] + 0.114 * buf[:, :, 2]
    gx = np.diff(gray, axis=1)
    gy = np.diff(gray, axis=0)
    mag = np.hypot(gx[:-1, :], gy[:, :-1])
    nonzero = mag[mag > 1e-4]
    return float(np.quantile(nonzero, 0.95)) if nonzero.size > 0 else 0.0


class TestV08ChiselNibAndIronGall(unittest.TestCase):
    def test_chisel_nib_anisotropy_and_edge_crispness_sharpen_gradients(self) -> None:
        pts = np.array([[20.0, 60.0], [100.0, 60.0], [140.0, 25.0], [180.0, 95.0]], dtype=np.float64)
        base_style = get_style("baroque_quill_neat")
        soft_style = dict(base_style, edge_crispness=0.0, nib_anisotropy=0.0, ink_mass=0.0, ink_feather=0.45)
        crisp_style = dict(base_style, nib_angle_deg=42.0, edge_crispness=0.88, nib_anisotropy=0.82, ink_mass=0.65, ink_feather=0.45)

        c_soft = StrokeCanvas(220, 130, (0.95, 0.92, 0.85), rng=np.random.default_rng(101))
        draw_calligraphic_polyline(c_soft, pts, (0.12, 0.09, 0.07), 5.0, soft_style)

        c_crisp = StrokeCanvas(220, 130, (0.95, 0.92, 0.85), rng=np.random.default_rng(101))
        draw_calligraphic_polyline(c_crisp, pts, (0.12, 0.09, 0.07), 5.0, crisp_style)

        # Deterministic check
        c_crisp_2 = StrokeCanvas(220, 130, (0.95, 0.92, 0.85), rng=np.random.default_rng(101))
        draw_calligraphic_polyline(c_crisp_2, pts, (0.12, 0.09, 0.07), 5.0, crisp_style)
        np.testing.assert_array_equal(c_crisp.buf, c_crisp_2.buf)

        # Valid pixel bounds & higher edge gradient contrast on crisp iron-gall chisel stroke
        self.assertGreaterEqual(float(c_crisp.buf.min()), 0.0)
        self.assertLessEqual(float(c_crisp.buf.max()), 1.0)
        self.assertGreater(_edge_grad_q95(c_crisp.buf), _edge_grad_q95(c_soft.buf))

    def test_quill_dab_oval_notehead_and_ink_mass_rim(self) -> None:
        st_plain = dict(get_style("baroque_quill_neat"), notehead_archetype="oval", nib_anisotropy=0.0, edge_crispness=0.0, ink_mass=0.0)
        st_quill = dict(get_style("baroque_quill_neat"), notehead_archetype="oval", nib_angle_deg=40.0, nib_anisotropy=0.75, edge_crispness=0.80, ink_mass=0.60)

        c1 = StrokeCanvas(100, 100, (0.95, 0.93, 0.88), rng=np.random.default_rng(7))
        draw_notehead(c1, "quarter", 50.0, 50.0, 12.0, (0.12, 0.09, 0.07), st_plain, np.random.default_rng(7))

        c2 = StrokeCanvas(100, 100, (0.95, 0.93, 0.88), rng=np.random.default_rng(7))
        draw_notehead(c2, "quarter", 50.0, 50.0, 12.0, (0.12, 0.09, 0.07), st_quill, np.random.default_rng(7))

        self.assertGreater(float(np.mean(np.abs(c1.buf - c2.buf))), 1e-4)
        self.assertGreaterEqual(float(c2.buf.min()), 0.0)
        self.assertLessEqual(float(c2.buf.max()), 1.0)


class TestV08DenseArchivalLayouts(unittest.TestCase):
    def test_apply_dense_archival_profile_all_profiles_non_mutating(self) -> None:
        base = get_style("baroque_quill_neat")
        snap = copy.deepcopy(base)
        for prof in ARCHIVAL_LAYOUT_PROFILES:
            out = apply_dense_archival_profile(base, layout_profile=prof)
            self.assertEqual(base, snap)
            self.assertEqual(out, apply_dense_archival_profile(base, layout_profile=prof))
            self.assertTrue(out["dense_archival"])
            self.assertEqual(out["archival_layout_profile"], prof)
            self.assertGreaterEqual(out["target_staves_per_page"], 8)
            self.assertLessEqual(out["target_staves_per_page"], 14)
            self.assertGreaterEqual(out["nib_angle_deg"], 25.0)
            self.assertLessEqual(out["nib_angle_deg"], 55.0)
            self.assertEqual(out["barline_span_mode"], "full_system_connected")
            self.assertGreater(out["rastrum_curve_var"], 0.0)
        with self.assertRaises(ValueError):
            apply_dense_archival_profile(base, layout_profile="nonexistent_layout")

    def test_dense_archival_oblong_and_portrait_staff_density_and_scale(self) -> None:
        xml = _make_quartet_xml(n_measures=24)
        with tempfile.TemporaryDirectory() as td:
            xml_path = Path(td) / "quartet.musicxml"
            xml_path.write_text(xml, encoding="utf-8")
            score = parse_score(xml_path)

            # Oblong 12-staff landscape
            st_ob = apply_dense_archival_profile(get_archival_v08_preset("v08_dark_irongall_fair_oblong"), layout_profile="oblong_12")
            sys_ob = layout_score_pages(score, page_w=2200, page_h=1600, style=st_ob)
            total_staves_ob = sum(len(s.staves) for s in sys_ob)
            self.assertGreaterEqual(total_staves_ob, 8)
            self.assertLessEqual(total_staves_ob, 14)
            sp_ob_1024 = sys_ob[0].staves[0].sp * (1024.0 / 2200.0)
            self.assertGreaterEqual(sp_ob_1024, 4.5)
            self.assertLessEqual(sp_ob_1024, 10.5)

            # Tall portrait 14-staff
            st_pt = apply_dense_archival_profile(get_archival_v08_preset("v08_dark_irongall_portrait_14"), layout_profile="portrait_14")
            sys_pt = layout_score_pages(score, page_w=1600, page_h=2200, style=st_pt)
            total_staves_pt = sum(len(s.staves) for s in sys_pt)
            self.assertGreaterEqual(total_staves_pt, 10)
            self.assertLessEqual(total_staves_pt, 14)

            # Full end-to-end render determinism
            out_a = Path(td) / "a.png"
            out_b = Path(td) / "b.png"
            gt_a = render_score_handwritten(xml_path, st_ob, out_a, seed=19, page_w=1400, page_h=1050, dense_archival=True, archival_layout="oblong_12")
            gt_b = render_score_handwritten(xml_path, st_ob, out_b, seed=19, page_w=1400, page_h=1050, dense_archival=True, archival_layout="oblong_12")
            self.assertTrue(gt_a["dense_archival"])
            self.assertEqual(gt_a["archival_layout_profile"], "oblong_12")
            self.assertEqual(out_a.read_bytes(), out_b.read_bytes())

    def test_non_parallel_rastrum_curvature_preserves_exact_centerline(self) -> None:
        st = dict(get_style("baroque_quill_neat"), rastrum_curve_var=0.25)
        y_fn, ctrl, sp_out = layout_quirks.build_curved_staff_y(100.0, 900.0, 400.0, 12.0, st, np.random.default_rng(33))
        # Center line (line_idx == 0) must still match cubic Bezier ctrl_pts at endpoints & midpoint
        self.assertAlmostEqual(float(y_fn(100.0, 0)), float(ctrl[0, 1]), places=6)
        self.assertAlmostEqual(float(y_fn(900.0, 0)), float(ctrl[3, 1]), places=6)
        # Outer lines must deviate non-linearly (non-parallel bow) between midpoint and linear average of endpoints
        mid_diff = abs((float(y_fn(500.0, -2)) - float(y_fn(500.0, 0))) - 0.5 * ((float(y_fn(100.0, -2)) - float(y_fn(100.0, 0))) + (float(y_fn(900.0, -2)) - float(y_fn(900.0, 0)))))
        self.assertGreater(mid_diff, 1e-4)
        self.assertEqual(sp_out, 12.0)


class TestV08StylePackAndSamplerDiversity(unittest.TestCase):
    def test_v08_diversity_style_presets_loaded_and_valid(self) -> None:
        v08_styles = list_archival_v08_presets()
        self.assertGreaterEqual(len(v08_styles), 10)
        self.assertEqual(len(list_styles()), 234)
        for name in v08_styles:
            s = get_archival_v08_preset(name)
            validate_style(s)
            self.assertGreaterEqual(s["nib_angle_deg"], 25.0)
            self.assertLessEqual(s["nib_angle_deg"], 55.0)
        with self.assertRaises(KeyError):
            get_archival_v08_preset("nonexistent_v08_preset")

    def test_sampler_archival_diversity_deterministic_and_broad(self) -> None:
        sampler = ScribeSampler(preset="wild_archive", master_seed=77, archival_diversity=True)
        s0, seed0 = sampler.sample("piece_0", variant_idx=0)
        s0_rep, seed0_rep = sampler.sample("piece_0", variant_idx=0)
        self.assertEqual(s0, s0_rep)
        self.assertEqual(seed0, seed0_rep)

        inks_r = []
        nib_angles = []
        layouts = set()
        has_verso = 0
        for i in range(40):
            st, _ = sampler.sample(f"piece_{i}", variant_idx=0)
            inks_r.append(st["ink_rgb"][0])
            nib_angles.append(st["nib_angle_deg"])
            layouts.add(st.get("archival_layout_profile"))
            if st.get("verso_bleed_alpha", 0.0) > 0.05:
                has_verso += 1

        # Broad ink luminance spread (dark iron-gall to light sepia/graphite)
        self.assertLess(min(inks_r), 0.14)
        self.assertGreater(max(inks_r), 0.32)
        # Broad nib angle spread inside [25, 55]
        self.assertGreaterEqual(min(nib_angles), 25.0)
        self.assertLessEqual(max(nib_angles), 55.0)
        self.assertGreater(max(nib_angles) - min(nib_angles), 20.0)
        # Multiple dense archival layouts & verso bleed variants sampled
        self.assertGreaterEqual(len(layouts), 4)
        self.assertGreater(has_verso, 5)


if __name__ == "__main__":
    unittest.main()
