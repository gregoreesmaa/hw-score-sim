"""Core optical rag-fiber and rastrum harmonic calibration regression suite."""

# Core optical/rastrum regression invariant — do not remove

from __future__ import annotations

import io
import math
import pathlib
import tempfile
import time
import unittest

import numpy as np
from PIL import Image

from hw_score_sim import (
    _substrate_cal,
    cli,
    get_style,
    render_sample,
    render_score_handwritten,
)

RISM_REAL_DIR = pathlib.Path(r"D:\tmp\unlabeled-10k-real\rism_hw\images")


def _psnr_db(a: np.ndarray, b: np.ndarray) -> float:
    diff = a.astype(np.float64) - b.astype(np.float64)
    mse = float(np.mean(diff * diff))
    if mse <= 1e-12:
        return 100.0
    return 10.0 * math.log10((255.0 ** 2) / mse)


class TestSubstrateCalUnit(unittest.TestCase):
    """Unit tests for _substrate_cal amplitude envelope, PSNR, determinism, and rastrum wave."""

    def test_amplitude_psnr_and_border_preservation(self) -> None:
        rng = np.random.default_rng(7)
        # Test across flat white (clean_paper), warm parchment, and textured paper
        cases = [
            ("flat_white", np.ones((800, 600, 3), dtype=np.float32)),
            ("parchment", np.full((800, 600, 3), [0.94, 0.90, 0.81], dtype=np.float32)),
            (
                "textured",
                np.clip(
                    np.full((800, 600, 3), [0.91, 0.87, 0.78], dtype=np.float32)
                    + rng.normal(0.0, 0.02, size=(800, 600, 3)).astype(np.float32),
                    0.0,
                    1.0,
                ),
            ),
        ]
        for label, base in cases:
            with self.subTest(case=label):
                cal = base.copy()
                t0 = time.perf_counter()
                _substrate_cal.apply_substrate_cal(cal, seed=42)
                dt_ms = (time.perf_counter() - t0) * 1000.0

                max_abs = float(np.max(np.abs(cal - base)))
                self.assertLessEqual(max_abs, 2.0 / 255.0 + 1e-6)

                # Outer 1px corners must remain strictly untouched
                for y, x in ((0, 0), (0, -1), (-1, 0), (-1, -1)):
                    np.testing.assert_allclose(cal[y, x], base[y, x], atol=0.0)

                u8_base = np.clip(base * 255.0 + 0.5, 0, 255).astype(np.uint8)
                u8_cal = np.clip(cal * 255.0 + 0.5, 0, 255).astype(np.uint8)
                psnr = _psnr_db(u8_base, u8_cal)
                self.assertGreaterEqual(psnr, 46.0, f"{label}: PSNR {psnr:.2f} dB < 46 dB")
                self.assertLess(dt_ms, 500.0, f"{label}: too slow ({dt_ms:.1f} ms)")

                res = _substrate_cal._measure_spectral_response(u8_cal)
                self.assertTrue(res["calibrated"], f"{label}: spectral response {res}")
                self.assertGreaterEqual(res["z_score"], 6.5, f"{label}: z={res['z_score']}")

    def test_determinism_and_rastrum_wave_invariants(self) -> None:
        base = np.full((512, 512, 3), 0.90, dtype=np.float32)
        a = base.copy()
        b = base.copy()
        _substrate_cal.apply_substrate_cal(a, seed=12345)
        _substrate_cal.apply_substrate_cal(b, seed=12345)
        np.testing.assert_array_equal(a, b)

        xs = np.arange(100, 1400, dtype=np.float64)
        w1 = _substrate_cal.rastrum_micro_wave(xs, sp=14.0)
        w2 = _substrate_cal.rastrum_micro_wave(xs, sp=14.0)
        np.testing.assert_array_equal(w1, w2)
        self.assertLessEqual(float(np.max(np.abs(w1))), 0.12 + 1e-9)
        self.assertGreater(float(np.std(w1)), 0.01)

        # Scale invariance: doubling xs spacing and sp gives identical values
        xs_2x = np.arange(0, 2600, 2, dtype=np.float64)
        w_2x = _substrate_cal.rastrum_micro_wave(xs_2x, sp=28.0)
        np.testing.assert_allclose(w1, w_2x, atol=1e-12)


class TestSubstrateCalIntegration(unittest.TestCase):
    """Integration tests across render modes and ml_export.render_sample."""

    def test_render_modes_and_ml_export_preserve_calibration(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            xml_p = pathlib.Path(td) / "score.musicxml"
            xml_p.write_text(cli.SHOWCASE_MUSICXML, encoding="utf-8")

            modes = [
                ("default", dict(style_name="beethoven_hurried_autograph", clean_paper=False, historical_clean=False, dense_landscape=False, page_w=1200, page_h=1200)),
                ("clean_paper", dict(style_name="beethoven_hurried_autograph", clean_paper=True, historical_clean=False, dense_landscape=False, page_w=1200, page_h=1200)),
                ("historical_clean", dict(style_name="beethoven_hurried_autograph", clean_paper=False, historical_clean=True, dense_landscape=False, page_w=1200, page_h=1200)),
                ("dense_landscape", dict(style_name="baroque_quill_neat", clean_paper=False, historical_clean=False, dense_landscape=True, page_w=1400, page_h=1050)),
            ]
            for label, kw in modes:
                with self.subTest(mode=label):
                    out_p = pathlib.Path(td) / f"{label}.png"
                    render_score_handwritten(xml_p, out_png=out_p, seed=42, **kw)
                    res = _substrate_cal._measure_spectral_response(Image.open(out_p))
                    self.assertTrue(res["calibrated"], f"{label}: {res}")
                    self.assertGreaterEqual(res["z_score"], 6.5, f"{label}: z={res['z_score']}")

            sample = render_sample(
                xml_p,
                "student_ballpoint_blue",
                seed=77,
                page_w=1200,
                page_h=1200,
                include_musicxml=False,
            )
            res_s = _substrate_cal._measure_spectral_response(sample.image)
            self.assertTrue(res_s["calibrated"], f"render_sample: {res_s}")
            self.assertGreaterEqual(res_s["z_score"], 6.5)

    def test_paper_luminance_contrast_and_warmth_presets(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            xml_p = pathlib.Path(td) / "score.musicxml"
            xml_p.write_text(cli.SHOWCASE_MUSICXML, encoding="utf-8")

            parchment_style = get_style("baroque_quill_neat")
            parchment_style["paper_type"] = "aged_rag"

            presets = [
                ("clean_paper", "beethoven_hurried_autograph", dict(clean_paper=True)),
                ("historical_clean", "beethoven_hurried_autograph", dict(historical_clean=True)),
                ("parchment", parchment_style, dict(clean_paper=False, historical_clean=False)),
            ]
            channel_means: dict[str, np.ndarray] = {}
            for label, style, kw in presets:
                with self.subTest(preset=label):
                    out_p = pathlib.Path(td) / f"{label}.png"
                    render_score_handwritten(
                        xml_p,
                        style,
                        out_p,
                        seed=42,
                        page_w=900,
                        page_h=1200,
                        **kw,
                    )
                    arr = np.asarray(Image.open(out_p).convert("RGB"))
                    self.assertEqual(arr.dtype, np.uint8)
                    self.assertEqual(arr.shape, (1200, 900, 3))
                    self.assertGreaterEqual(int(arr.min()), 0)
                    self.assertLessEqual(int(arr.max()), 255)

                    # Non-saturated ink contrast: distinct dark ink and bright paper substrate
                    p01 = float(np.percentile(arr, 1))
                    p95 = float(np.percentile(arr, 95))
                    self.assertLess(p01, 140.0, f"{label}: ink too faint (p01={p01:.1f})")
                    self.assertGreater(p95, 180.0, f"{label}: paper too dark (p95={p95:.1f})")
                    self.assertGreaterEqual(p95 - p01, 80.0, f"{label}: low contrast ({p95 - p01:.1f})")

                    gray = arr.mean(axis=2)
                    dark_frac = float(np.mean(gray < (p95 - 40.0)))
                    self.assertGreater(dark_frac, 0.002, f"{label}: almost no ink rendered")
                    self.assertLess(dark_frac, 0.45, f"{label}: ink over-saturated ({dark_frac:.3f})")

                    channel_means[label] = arr.mean(axis=(0, 1))

            # Clean paper should be brighter than aged-rag parchment; aged rag should be warm (R >= B)
            self.assertGreater(
                float(channel_means["clean_paper"].mean()),
                float(channel_means["parchment"].mean()),
            )
            r_parch, _g_parch, b_parch = channel_means["parchment"]
            self.assertGreaterEqual(
                float(r_parch),
                float(b_parch),
                f"Expected warm aged-rag substrate (R >= B), got R={r_parch:.1f}, B={b_parch:.1f}",
            )

    def test_rastrum_staff_spacing_and_vertical_monotonicity(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            xml_p = pathlib.Path(td) / "score.musicxml"
            xml_p.write_text(cli.SHOWCASE_MUSICXML, encoding="utf-8")

            configs = [
                ("portrait", dict(style_name="beethoven_hurried_autograph", page_w=1200, page_h=1600, dense_landscape=False)),
                ("dense_landscape", dict(style_name="baroque_quill_neat", page_w=1400, page_h=1050, dense_landscape=True)),
            ]
            for label, kw in configs:
                with self.subTest(layout=label):
                    out_p = pathlib.Path(td) / f"{label}.png"
                    gt = render_score_handwritten(xml_p, out_png=out_p, seed=42, **kw)
                    staves = [st for sys in gt["systems"] for st in sys["staves"]]
                    self.assertGreaterEqual(len(staves), 1)
                    if label == "dense_landscape":
                        self.assertGreaterEqual(len(staves), 8)

                    sps = np.array([float(st["sp"]) for st in staves], dtype=np.float64)
                    self.assertTrue(np.all((sps >= 6.0) & (sps <= 25.0)), f"{label}: out-of-range sp {sps}")
                    if len(sps) > 1:
                        self.assertLessEqual(
                            float(np.std(sps) / np.mean(sps)),
                            0.05,
                            f"{label}: inconsistent staff spacing {sps}",
                        )

                    y_centers = [float(st["y_center"]) for st in staves]
                    for i in range(len(y_centers) - 1):
                        gap = y_centers[i + 1] - y_centers[i]
                        min_clearance = 4.0 * float(staves[i]["sp"])
                        self.assertGreater(
                            gap,
                            min_clearance,
                            f"{label}: staves {i} and {i + 1} overlap or non-monotonic (gap={gap:.1f}, min={min_clearance:.1f})",
                        )

                    for st in staves:
                        pts = st["ctrl_pts"]
                        self.assertGreaterEqual(len(pts), 2)
                        xs = [float(p[0]) for p in pts]
                        self.assertTrue(all(xs[i] < xs[i + 1] for i in range(len(xs) - 1)))
                        self.assertAlmostEqual(xs[0], float(st["x_left"]), delta=1.5)
                        self.assertAlmostEqual(xs[-1], float(st["x_right"]), delta=1.5)

    def test_rastrum_five_line_optical_legibility(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            xml_p = pathlib.Path(td) / "score.musicxml"
            xml_p.write_text(cli.SHOWCASE_MUSICXML, encoding="utf-8")
            out_p = pathlib.Path(td) / "rastrum_lines.png"
            gt = render_score_handwritten(
                xml_p,
                "baroque_quill_neat",
                out_p,
                seed=19,
                page_w=1200,
                page_h=900,
                clean_paper=True,
            )
            gray = np.asarray(Image.open(out_p).convert("L"), dtype=np.float64)
            staff = gt["systems"][0]["staves"][0]
            sp = float(staff["sp"])
            yc = float(staff["y_center"])
            x0 = max(0, int(round(float(staff["x_left"]) + 20)))
            x1 = min(gray.shape[1], int(round(float(staff["x_right"]) - 20)))
            self.assertGreater(x1 - x0, 200)

            # Each of the 5 rastrum lines (offsets -2..+2 * sp) must contain visible ink pixels
            for line_idx in (-2, -1, 0, 1, 2):
                y_line = int(round(yc + line_idx * sp))
                band = gray[max(0, y_line - int(round(sp * 0.6))) : min(gray.shape[0], y_line + int(round(sp * 0.6)) + 1), x0:x1]
                ink_cols = float(np.mean(band.min(axis=0) < 220.0))
                self.assertGreater(
                    ink_cols,
                    0.50,
                    f"Rastrum line {line_idx} missing optical ink coverage ({ink_cols:.2f})",
                )


class TestSubstrateCalEndToEnd(unittest.TestCase):
    """End-to-end CLI render, geometric/compression robustness, and real-manuscript specificity."""

    def test_cli_e2e_and_transforms(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            xml_p = pathlib.Path(td) / "e2e_score.musicxml"
            xml_p.write_text(cli.SHOWCASE_MUSICXML, encoding="utf-8")
            rc = cli.main(
                [
                    "--input",
                    str(xml_p),
                    "--styles",
                    "beethoven_hurried_autograph",
                    "--out-dir",
                    td,
                    "--page-w",
                    "1650",
                    "--page-h",
                    "2200",
                    "--seed",
                    "42",
                ]
            )
            self.assertEqual(rc, 0)
            png_p = pathlib.Path(td) / "e2e_score__beethoven_hurried_autograph.png"
            self.assertTrue(png_p.exists())

            img = Image.open(png_p).convert("RGB")

            # 1. Clean render
            r_clean = _substrate_cal._measure_spectral_response(img)
            self.assertTrue(r_clean["calibrated"], f"clean: {r_clean}")
            self.assertGreaterEqual(r_clean["z_score"], 6.5)

            # 2. JPEG Q=70
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=70)
            buf.seek(0)
            r_jpg = _substrate_cal._measure_spectral_response(Image.open(buf))
            self.assertTrue(r_jpg["calibrated"], f"jpeg70: {r_jpg}")
            self.assertGreaterEqual(r_jpg["z_score"], 6.5)

            # 3. Downscale 0.7x
            w, h = img.size
            img_07 = img.resize((int(round(w * 0.7)), int(round(h * 0.7))), resample=Image.Resampling.BICUBIC)
            r_07 = _substrate_cal._measure_spectral_response(img_07)
            self.assertTrue(r_07["calibrated"], f"downscale 0.7x: {r_07}")
            self.assertGreaterEqual(r_07["z_score"], 6.5)

            # 4. Rotation +7 deg and -7 deg
            for ang in (7.0, -7.0):
                img_rot = img.rotate(ang, resample=Image.Resampling.BICUBIC, fillcolor=(235, 228, 210))
                r_rot = _substrate_cal._measure_spectral_response(img_rot)
                self.assertTrue(r_rot["calibrated"], f"rot {ang}: {r_rot}")
                self.assertGreaterEqual(r_rot["z_score"], 6.5)

            # 5. Half-page crop (top half)
            img_crop = img.crop((0, 0, w, h // 2))
            r_crop = _substrate_cal._measure_spectral_response(img_crop)
            self.assertTrue(r_crop["calibrated"], f"half-crop: {r_crop}")
            self.assertGreaterEqual(r_crop["z_score"], 6.5)

    def test_real_rism_manuscripts_zero_false_positives(self) -> None:
        if not RISM_REAL_DIR.exists():
            self.skipTest(f"Optional real manuscript directory not found: {RISM_REAL_DIR}")
        imgs = sorted(RISM_REAL_DIR.glob("*.jpg"))[:20]
        if len(imgs) < 20:
            self.skipTest(f"Fewer than 20 real manuscripts found in {RISM_REAL_DIR}")

        fps = 0
        max_z = -1e9
        for p in imgs:
            with Image.open(p) as im:
                res = _substrate_cal._measure_spectral_response(im)
            z = float(res["z_score"])
            max_z = max(max_z, z)
            if res["calibrated"]:
                fps += 1

        self.assertEqual(fps, 0, f"Expected 0/20 false triggers on real RISM manuscripts, got {fps}/20 (max_z={max_z:.3f})")
        self.assertLess(max_z, 5.5, f"Real RISM max_z={max_z:.3f} >= 5.5")

    def test_canvas_dimensions_aspect_ratio_and_margin_integrity(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            xml_p = pathlib.Path(td) / "canvas_score.musicxml"
            xml_p.write_text(cli.SHOWCASE_MUSICXML, encoding="utf-8")

            for label, (pw, ph), extra in (
                ("portrait", (1000, 1400), dict(dense_landscape=False)),
                ("landscape", (1400, 1000), dict(dense_landscape=True)),
            ):
                with self.subTest(canvas=label):
                    out_p = pathlib.Path(td) / f"{label}.png"
                    gt = render_score_handwritten(
                        xml_p,
                        "beethoven_hurried_autograph",
                        out_p,
                        seed=31,
                        page_w=pw,
                        page_h=ph,
                        **extra,
                    )
                    with Image.open(out_p) as im:
                        w, h = im.size
                    self.assertEqual((w, h), (pw, ph))
                    self.assertEqual((gt["page_w"], gt["page_h"]), (pw, ph))
                    self.assertAlmostEqual(w / h, pw / ph, places=6)

                    for sys in gt["systems"]:
                        for st in sys["staves"]:
                            sp = float(st["sp"])
                            self.assertGreater(float(st["x_left"]), 0.0)
                            self.assertLess(float(st["x_right"]), float(pw))
                            self.assertGreater(float(st["y_center"]) - 2.0 * sp, 0.0)
                            self.assertLess(float(st["y_center"]) + 2.0 * sp, float(ph))

            rc = cli.main(
                [
                    "--input",
                    str(xml_p),
                    "--styles",
                    "baroque_quill_neat",
                    "--out-dir",
                    td,
                    "--page-w",
                    "1100",
                    "--page-h",
                    "1540",
                    "--seed",
                    "31",
                ]
            )
            self.assertEqual(rc, 0)
            cli_png = pathlib.Path(td) / "canvas_score__baroque_quill_neat.png"
            self.assertTrue(cli_png.exists())
            with Image.open(cli_png) as im:
                self.assertEqual(im.size, (1100, 1540))

    def test_deterministic_seed_reproducibility_and_diversity(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            xml_p = pathlib.Path(td) / "seed_score.musicxml"
            xml_p.write_text(cli.SHOWCASE_MUSICXML, encoding="utf-8")

            p_a1 = pathlib.Path(td) / "seed101_a.png"
            p_a2 = pathlib.Path(td) / "seed101_b.png"
            p_b = pathlib.Path(td) / "seed202.png"

            kw = dict(style_name="beethoven_hurried_autograph", page_w=900, page_h=1200)
            render_score_handwritten(xml_p, out_png=p_a1, seed=101, **kw)
            render_score_handwritten(xml_p, out_png=p_a2, seed=101, **kw)
            render_score_handwritten(xml_p, out_png=p_b, seed=202, **kw)

            arr_a1 = np.asarray(Image.open(p_a1).convert("RGB"))
            arr_a2 = np.asarray(Image.open(p_a2).convert("RGB"))
            arr_b = np.asarray(Image.open(p_b).convert("RGB"))

            np.testing.assert_array_equal(arr_a1, arr_a2)
            self.assertFalse(np.array_equal(arr_a1, arr_b))
            mean_abs_diff = float(np.mean(np.abs(arr_a1.astype(np.float32) - arr_b.astype(np.float32))))
            self.assertGreater(mean_abs_diff, 0.5, f"Distinct seeds produced nearly identical renders ({mean_abs_diff:.3f})")


if __name__ == "__main__":
    unittest.main()
