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


if __name__ == "__main__":
    unittest.main()
