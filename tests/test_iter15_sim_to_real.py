"""Unit tests for Iter 15 sim-to-real upgrades:
1. Aged sepia paper tint (chroma 0.08-0.22) + warm brown iron-gall ink + faint thin-stroke sepia mode
2. Subtle horizontally-mirrored inverted verso bleed-through
3. Dense 12-staff landscape archival preset (zero extra inter-system gap, forced left-margin braces, optional dark scanner margin)
"""

from __future__ import annotations

import copy
import json
import pathlib
import tempfile
import unittest

import numpy as np
from PIL import Image

from hw_score_sim import (
    SEPIA_MODES,
    ScribeSampler,
    apply_dense_landscape_profile,
    apply_sepia_archival_profile,
    cli,
    get_style,
    layout_quirks,
    render_sample,
    render_score_handwritten,
)


def _make_quartet_xml(n_measures: int = 24) -> str:
    """Build a 4-part string quartet MusicXML (no brace_group, 4 independent score-parts)."""
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
                    f"<attributes><divisions>4</divisions><key><fifths>-1</fifths></key>"
                    f"<time><beats>4</beats><beat-type>4</beat-type></time>"
                    f"<clef><sign>{clef_s}</sign><line>{clef_l}</line></clef></attributes>"
                )
            s0 = steps[(m + p_idx) % 7]
            s1 = steps[(m + p_idx + 2) % 7]
            s2 = steps[(m + p_idx + 4) % 7]
            s3 = steps[(m + p_idx + 1) % 7]
            notes = (
                f"<note><pitch><step>{s0}</step><octave>{octv}</octave></pitch><duration>4</duration><type>quarter</type></note>"
                f"<note><pitch><step>{s1}</step><octave>{octv}</octave></pitch><duration>4</duration><type>quarter</type></note>"
                f"<note><pitch><step>{s2}</step><octave>{octv}</octave></pitch><duration>4</duration><type>quarter</type></note>"
                f"<note><pitch><step>{s3}</step><octave>{octv}</octave></pitch><duration>4</duration><type>quarter</type></note>"
            )
            measures.append(f'    <measure number="{m}">{attrs}{notes}</measure>')
        parts_xml.append(f'  <part id="{pid}">\n' + "\n".join(measures) + "\n  </part>")

    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<score-partwise version="3.1">\n'
        '  <work><work-title>String Quartet Archival Study</work-title></work>\n'
        f"  <part-list>\n    {part_list}\n  </part-list>\n"
        + "\n".join(parts_xml)
        + "\n</score-partwise>\n"
    )


class TestIter15SepiaAndIronGall(unittest.TestCase):
    def test_sepia_modes_and_profile_chroma_bounds(self) -> None:
        self.assertEqual(SEPIA_MODES, ("none", "aged", "faint_thin"))
        with self.assertRaises(ValueError):
            apply_sepia_archival_profile(get_style("baroque_quill_neat"), mode="neon")

        for sname in ("modern_jazz_lead_sheet", "student_ballpoint_blue", "baroque_quill_neat", "chopin_delicate_salon"):
            base = get_style(sname)
            snap = copy.deepcopy(base)

            aged = apply_sepia_archival_profile(base, mode="aged")
            faint = apply_sepia_archival_profile(base, mode="faint_thin")
            # Non-mutating & deterministic
            self.assertEqual(base, snap)
            self.assertEqual(aged, apply_sepia_archival_profile(base, mode="aged"))
            self.assertEqual(faint, apply_sepia_archival_profile(base, mode="faint_thin"))

            for prof in (aged, faint):
                # Warm sepia paper with R > G > B and chroma in [0.08, 0.22]
                pr, pg, pb = prof["paper_rgb"]
                self.assertGreater(pr, pg)
                self.assertGreater(pg, pb)
                chroma = pr - pb
                self.assertGreaterEqual(chroma, 0.08 - 1e-4, msg=sname)
                self.assertLessEqual(chroma, 0.22 + 1e-4, msg=sname)
                # Warm brown iron-gall ink (R > G > B)
                ir, ig, ib = prof["ink_rgb"]
                self.assertGreater(ir, ig)
                self.assertGreater(ig, ib)

            # faint_thin has lighter ink and thinner strokes than aged
            self.assertGreater(faint["ink_rgb"][0], aged["ink_rgb"][0])
            self.assertGreaterEqual(faint["stroke_scale"], 0.58)
            self.assertLessEqual(faint["stroke_scale"], 0.78)
            self.assertGreaterEqual(faint["stroke_opacity"], 0.66)
            self.assertLessEqual(faint["stroke_opacity"], 0.82)

    def test_rendered_sepia_page_measured_median_chroma(self) -> None:
        xml = _make_quartet_xml(n_measures=8)
        for mode in ("aged", "faint_thin"):
            s1 = render_sample(
                xml,
                "modern_jazz_lead_sheet",
                seed=2026,
                page_w=900,
                page_h=1100,
                include_musicxml=False,
                sepia_mode=mode,
            )
            s2 = render_sample(
                xml,
                "modern_jazz_lead_sheet",
                seed=2026,
                page_w=900,
                page_h=1100,
                include_musicxml=False,
                sepia_mode=mode,
            )
            self.assertEqual(s1.meta["sepia_mode"], mode)
            arr1 = np.asarray(s1.image, dtype=np.float32) / 255.0
            arr2 = np.asarray(s2.image, dtype=np.float32) / 255.0
            # Exact pixel determinism
            np.testing.assert_array_equal(arr1, arr2)
            # Measured median chroma across interior pixels in [0.08, 0.22]
            interior = arr1[40:-40, 40:-40]
            chroma_map = interior.max(axis=2) - interior.min(axis=2)
            med_chroma = float(np.median(chroma_map))
            self.assertGreaterEqual(med_chroma, 0.08, msg=f"mode={mode} med_chroma={med_chroma:.4f}")
            self.assertLessEqual(med_chroma, 0.22, msg=f"mode={mode} med_chroma={med_chroma:.4f}")
            # Warm sepia hue: median R > median G > median B
            self.assertGreater(float(np.median(interior[:, :, 0])), float(np.median(interior[:, :, 1])))
            self.assertGreater(float(np.median(interior[:, :, 1])), float(np.median(interior[:, :, 2])))


class TestIter15SubtleVersoBleed(unittest.TestCase):
    def test_subtle_inverted_verso_bleed_in_paper_synthesis(self) -> None:
        base = get_style("student_pencil_eraser")
        base["bleed_through"] = 0.0  # force zero legacy bleed

        no_bleed = layout_quirks.make_manuscript_paper(640, 800, base, np.random.default_rng(19))

        st_verso = dict(base)
        st_verso["verso_bleed_alpha"] = 0.12
        st_verso["verso_bleed_invert"] = True
        with_bleed_a = layout_quirks.make_manuscript_paper(640, 800, st_verso, np.random.default_rng(19))
        with_bleed_b = layout_quirks.make_manuscript_paper(640, 800, st_verso, np.random.default_rng(19))

        # Determinism
        np.testing.assert_array_equal(with_bleed_a, with_bleed_b)

        # Subtle visible difference (not zero, not overpowering)
        diff = np.abs(with_bleed_a - no_bleed)
        self.assertGreater(float(diff.max()), 0.025)
        self.assertLess(float(diff.mean()), 0.03)

        # Vertically inverted differs from non-inverted
        st_noninv = dict(st_verso)
        st_noninv["verso_bleed_invert"] = False
        noninv = layout_quirks.make_manuscript_paper(640, 800, st_noninv, np.random.default_rng(19))
        self.assertGreater(float(np.abs(with_bleed_a - noninv).max()), 0.01)


class TestIter15DenseLandscapeArchival(unittest.TestCase):
    def test_dense_12_staff_landscape_zero_inter_gap_and_system_braces(self) -> None:
        xml = _make_quartet_xml(n_measures=24)
        sample = render_sample(
            xml,
            "baroque_quill_neat",
            seed=77,
            page_w=2200,
            page_h=1650,
            include_musicxml=False,
            sepia_mode="aged",
            verso_bleed=0.10,
            dense_landscape=True,
            scanner_margin_frac=0.025,
        )
        self.assertTrue(sample.meta["dense_landscape"])
        self.assertEqual(sample.meta["sepia_mode"], "aged")
        self.assertAlmostEqual(sample.meta["verso_bleed_alpha"], 0.10, places=4)

        systems = sample.meta["systems"]
        active = [s for s in systems if not s["is_blank_prerule"]]
        total_staves = sum(len(s["staves"]) for s in systems)
        # 4-part quartet x 3 systems = 12 staves total on the landscape sheet
        self.assertGreaterEqual(len(active), 2)
        self.assertEqual(total_staves, 12)

        # Check zero extra inter-system gap: gap between last staff of sys 0 and first staff of sys 1
        # matches intra-system staff spacing within [0.85, 1.25]
        s0_staves = active[0]["staves"]
        s1_staves = active[1]["staves"]
        intra_step = s0_staves[1]["y_center"] - s0_staves[0]["y_center"]
        inter_step = s1_staves[0]["y_center"] - s0_staves[-1]["y_center"]
        ratio = inter_step / max(intra_step, 1e-6)
        self.assertGreaterEqual(ratio, 0.85)
        self.assertLessEqual(ratio, 1.25)

        arr = np.asarray(sample.image, dtype=np.float32)
        # Dark scanner-bed outer border present (< 60/255)
        self.assertLess(float(arr[3, 3].mean()), 60.0)
        self.assertLess(float(arr[-4, -4].mean()), 60.0)

        # Forced left-margin system brace drawn left of x_left for quartet systems
        # even though parts had no brace_group!
        st_top = s0_staves[0]
        st_bot = s0_staves[-1]
        sp = float(st_top["sp"])
        bx0 = max(0, int(st_top["x_left"] - sp * 1.8))
        bx1 = max(bx0 + 2, int(st_top["x_left"] - sp * 0.2))
        by0 = max(0, int(st_top["y_center"] - 2.0 * sp))
        by1 = min(arr.shape[0], int(st_bot["y_center"] + 2.0 * sp))
        brace_patch = arr[by0:by1, bx0:bx1]
        dark_brace_px = int((brace_patch.mean(axis=2) < 130.0).sum())
        self.assertGreater(dark_brace_px, 60, msg="Expected left-margin system brace ink pixels")

    def test_sampler_and_cli_integration(self) -> None:
        sampler = ScribeSampler(
            preset="wild_archive",
            master_seed=99,
            era="historical",
            sepia_mode="faint_thin",
            verso_bleed=0.11,
            dense_landscape=True,
            scanner_margin_frac=0.02,
        )
        st, seed = sampler.sample("mozart_k387", variant_idx=0)
        self.assertEqual(st["sepia_mode"], "faint_thin")
        self.assertTrue(st["dense_landscape"])
        self.assertAlmostEqual(st["verso_bleed_alpha"], 0.11, places=4)
        self.assertTrue(st["verso_bleed_invert"])
        self.assertAlmostEqual(st["scanner_margin_frac"], 0.02, places=4)

        # CLI --dense-landscape defaults to 2200x1650 when --page-w/--page-h omitted
        xml = _make_quartet_xml(n_measures=16)
        with tempfile.TemporaryDirectory() as td:
            xml_p = pathlib.Path(td) / "k387.musicxml"
            xml_p.write_text(xml, encoding="utf-8")
            rc = cli.main(
                [
                    "--input",
                    str(xml_p),
                    "--styles",
                    "baroque_quill_neat",
                    "--out-dir",
                    td,
                    "--sepia-mode",
                    "aged",
                    "--verso-bleed",
                    "0.10",
                    "--dense-landscape",
                    "--scanner-margin",
                    "0.02",
                ]
            )
            self.assertEqual(rc, 0)
            out_png = pathlib.Path(td) / "k387__baroque_quill_neat.png"
            out_json = pathlib.Path(td) / "k387__baroque_quill_neat.json"
            self.assertTrue(out_png.exists())
            self.assertTrue(out_json.exists())
            with Image.open(out_png) as im:
                self.assertEqual(im.size, (2200, 1650))
            payload = json.loads(out_json.read_text(encoding="utf-8"))
            self.assertTrue(payload["dense_landscape"])
            self.assertEqual(payload["sepia_mode"], "aged")
            self.assertAlmostEqual(payload["verso_bleed_alpha"], 0.10, places=4)


if __name__ == "__main__":
    unittest.main()
