"""Fast smoke tests verifying all runnable examples import and execute cleanly."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from hw_score_sim import parse_score, reload_style_packs

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = REPO_ROOT / "examples"


def _load_example_module(filename: str):
    """Load an example script from `examples/` by filename via importlib."""
    path = EXAMPLES_DIR / filename
    if str(EXAMPLES_DIR) not in sys.path:
        sys.path.insert(0, str(EXAMPLES_DIR))
    mod_name = f"hw_score_sim_example_{path.stem}"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load example module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


class TestExamplesSmoke(unittest.TestCase):
    """Verify bundled sample score and all four runnable examples."""

    def tearDown(self) -> None:
        reload_style_packs(reset_to_builtins=True)

    def test_sample_score_parses_cleanly(self) -> None:
        mod = _load_example_module("_sample_score.py")
        with tempfile.TemporaryDirectory() as tmp:
            xml_path = mod.write_sample_score(Path(tmp) / "sample.musicxml")
            score = parse_score(xml_path)
            self.assertEqual(score.num_measures, 8)
            self.assertEqual(len(score.parts), 1)
            part = score.parts[0]
            self.assertGreaterEqual(len(part.ties), 2)
            self.assertGreaterEqual(len(part.slurs), 2)
            self.assertGreaterEqual(len(part.hairpins), 2)

    def test_01_time_machine_strip(self) -> None:
        mod = _load_example_module("01_time_machine_strip.py")
        with tempfile.TemporaryDirectory() as tmp:
            strip_path = mod.run(tmp, seed=11, page_w=540, page_h=680, panel_w=200)
            self.assertTrue(strip_path.exists())
            with Image.open(strip_path) as im:
                self.assertGreater(im.width, 900)
                self.assertGreater(im.height, 200)

    def test_02_scribe_morph(self) -> None:
        mod = _load_example_module("02_scribe_morph.py")
        with tempfile.TemporaryDirectory() as tmp:
            strip_path = mod.run(tmp, seed=22, page_w=540, page_h=680, panel_w=200)
            self.assertTrue(strip_path.exists())
            with Image.open(strip_path) as im:
                self.assertGreater(im.width, 900)
                self.assertGreater(im.height, 200)

    def test_03_infinite_omr_dataset(self) -> None:
        mod = _load_example_module("03_infinite_omr_dataset.py")
        with tempfile.TemporaryDirectory() as tmp:
            index_path = mod.run(tmp, count=2, seed=33, page_w=540, page_h=680)
            self.assertTrue(index_path.exists())
            html_text = index_path.read_text(encoding="utf-8")
            self.assertIn("2 Unseen Procedural Scribes", html_text)

            manifest = json.loads((Path(tmp) / "dataset_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(len(manifest["samples"]), 2)
            for item in manifest["samples"]:
                png_path = Path(tmp) / item["png"]
                gt_path = Path(tmp) / item["gt_json"]
                self.assertTrue(png_path.exists())
                self.assertTrue(gt_path.exists())
                with Image.open(png_path) as im:
                    self.assertEqual(im.size, (540, 680))
                gt = json.loads(gt_path.read_text(encoding="utf-8"))
                self.assertGreaterEqual(len(gt["systems"]), 1)

    def test_04_custom_scribe_from_json(self) -> None:
        mod = _load_example_module("04_custom_scribe_from_json.py")
        with tempfile.TemporaryDirectory() as tmp:
            ab_path = mod.run(tmp, seed=44, page_w=540, page_h=680, panel_w=240)
            self.assertTrue(ab_path.exists())
            self.assertTrue((Path(tmp) / "custom_scribe.json").exists())
            with Image.open(ab_path) as im:
                self.assertGreater(im.width, 480)
                self.assertGreater(im.height, 200)


if __name__ == "__main__":
    unittest.main()
