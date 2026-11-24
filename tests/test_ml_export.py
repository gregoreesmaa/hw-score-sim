"""Unit tests for `hw_score_sim.ml_export` (in-memory rendering, COCO/YOLO, atomic export, datasets)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

import hw_score_sim
from hw_score_sim import (
    HandwrittenScoreDataset,
    RenderSample,
    export_dataset_sample,
    render_sample,
    sample_random_scribe,
    to_coco_annotations,
    to_hf_generator,
    to_yolo_labels,
)
from hw_score_sim.styles import SCRIBE_STYLES

MINIMAL_MSCX = """<?xml version="1.0" encoding="UTF-8"?>
<museScore version="3.01">
  <Score>
    <metaTag name="workTitle">ML Export Synthetic Test</metaTag>
    <Part>
      <Staff id="1"><defaultClef>G</defaultClef></Staff>
      <Staff id="2"><defaultClef>F</defaultClef></Staff>
      <trackName>Piano</trackName>
    </Part>
    <Staff id="1">
      <Measure>
        <voice>
          <KeySig><accidental>1</accidental></KeySig>
          <TimeSig><sigN>4</sigN><sigD>4</sigD></TimeSig>
          <Chord>
            <durationType>quarter</durationType>
            <Note><pitch>60</pitch><tpc>14</tpc></Note>
            <Note><pitch>64</pitch><tpc>18</tpc></Note>
          </Chord>
          <Chord>
            <durationType>eighth</durationType>
            <Note><pitch>66</pitch><tpc>20</tpc></Note>
          </Chord>
          <Chord>
            <durationType>eighth</durationType>
            <Note><pitch>67</pitch><tpc>15</tpc></Note>
          </Chord>
          <Rest><durationType>half</durationType></Rest>
        </voice>
      </Measure>
      <Measure>
        <voice>
          <Chord>
            <durationType>half</durationType>
            <dots>1</dots>
            <Note><pitch>72</pitch><tpc>14</tpc></Note>
          </Chord>
          <Rest><durationType>quarter</durationType></Rest>
          <BarLine><subtype>end</subtype></BarLine>
        </voice>
      </Measure>
    </Staff>
    <Staff id="2">
      <Measure>
        <voice>
          <KeySig><accidental>1</accidental></KeySig>
          <TimeSig><sigN>4</sigN><sigD>4</sigD></TimeSig>
          <Chord>
            <durationType>whole</durationType>
            <Note><pitch>48</pitch><tpc>14</tpc></Note>
          </Chord>
        </voice>
      </Measure>
      <Measure>
        <voice>
          <Rest><durationType>measure</durationType></Rest>
          <BarLine><subtype>end</subtype></BarLine>
        </voice>
      </Measure>
    </Staff>
  </Score>
</museScore>
"""


class TestMLExport(unittest.TestCase):
    def test_render_sample_named_style_and_schema(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            mscx_path = Path(td) / "mini.mscx"
            mscx_path.write_text(MINIMAL_MSCX, encoding="utf-8")

            sample = render_sample(
                mscx_path,
                "beethoven_hurried_autograph",
                seed=123,
                page_w=900,
                page_h=1200,
                include_musicxml=True,
            )

            self.assertIsInstance(sample, RenderSample)
            self.assertIsInstance(sample.image, Image.Image)
            self.assertEqual(sample.image.size, (900, 1200))
            self.assertEqual(sample.image.mode, "RGB")

            meta = sample.meta
            self.assertEqual(meta["schema_version"], "1.0")
            self.assertEqual(
                meta["coord_system"],
                {
                    "origin": "top_left",
                    "units": "px",
                    "bbox_format": "xyxy",
                    "bezier_eval": "uniform_x_cubic",
                },
            )
            self.assertEqual(meta["reproducibility"]["seed"], 123)
            self.assertEqual(meta["reproducibility"]["style_name"], "beethoven_hurried_autograph")
            self.assertEqual(len(meta["reproducibility"]["style_hash"]), 16)
            self.assertEqual(meta["style_metadata"]["name"], "beethoven_hurried_autograph")
            self.assertFalse(meta["truncated"])
            self.assertGreater(len(meta["systems"]), 0)

            sys0 = meta["systems"][0]
            self.assertEqual(len(sys0["bbox_xyxy"]), 4)
            x0, y0, x1, y1 = sys0["bbox_xyxy"]
            self.assertLess(x0, x1)
            self.assertLess(y0, y1)
            self.assertGreaterEqual(x0, 0.0)
            self.assertLessEqual(x1, 900.0)
            self.assertGreaterEqual(y0, 0.0)
            self.assertLessEqual(y1, 1200.0)

            self.assertEqual(len(sys0["staves"]), 2)
            for st in sys0["staves"]:
                self.assertEqual(len(st["ctrl_pts"]), 4)
                self.assertEqual(len(st["bbox_xyxy"]), 4)
                self.assertLess(st["bbox_xyxy"][0], st["bbox_xyxy"][2])
                self.assertLess(st["bbox_xyxy"][1], st["bbox_xyxy"][3])

            # Pure-Python MusicXML slices
            self.assertGreaterEqual(len(sample.musicxml_systems), 1)
            self.assertIn("<score-partwise", sample.musicxml_systems[0])

    def test_render_sample_custom_dict_does_not_mutate_global_styles(self) -> None:
        before_keys = set(SCRIBE_STYLES.keys())
        rng = np.random.default_rng(999)
        custom_scribe = sample_random_scribe(rng, base_style="mozart_flowing_cursive", jitter=0.25)
        custom_scribe["name"] = "ephemeral_custom_test_scribe"

        sample = render_sample(
            MINIMAL_MSCX,
            custom_scribe,
            seed=77,
            page_w=800,
            page_h=1050,
            include_musicxml=False,
        )

        after_keys = set(SCRIBE_STYLES.keys())
        self.assertEqual(before_keys, after_keys)
        self.assertNotIn("ephemeral_custom_test_scribe", SCRIBE_STYLES)
        self.assertEqual(sample.meta["schema_version"], "1.0")
        self.assertEqual(sample.meta["style"], "ephemeral_custom_test_scribe")
        self.assertEqual(sample.meta["style_metadata"]["name"], "ephemeral_custom_test_scribe")
        self.assertEqual(sample.musicxml_systems, [])

    def test_to_coco_annotations_and_to_yolo_labels(self) -> None:
        sample = render_sample(
            MINIMAL_MSCX,
            "baroque_quill_neat",
            seed=42,
            page_w=850,
            page_h=1100,
        )
        coco = to_coco_annotations(sample.meta, image_id=7)
        self.assertEqual(len(coco["images"]), 1)
        self.assertEqual(coco["images"][0]["id"], 7)
        self.assertEqual(coco["images"][0]["width"], 850)
        self.assertEqual(coco["images"][0]["height"], 1100)
        self.assertEqual([c["name"] for c in coco["categories"]], ["system", "staff"])

        sys_anns = [a for a in coco["annotations"] if a["category_id"] == 1]
        staff_anns = [a for a in coco["annotations"] if a["category_id"] == 2]
        self.assertGreaterEqual(len(sys_anns), 1)
        self.assertGreaterEqual(len(staff_anns), 2)
        for ann in coco["annotations"]:
            self.assertEqual(ann["image_id"], 7)
            bx, by, bw, bh = ann["bbox"]
            self.assertGreater(bw, 0.0)
            self.assertGreater(bh, 0.0)
            self.assertGreaterEqual(bx, 0.0)
            self.assertGreaterEqual(by, 0.0)
            self.assertAlmostEqual(ann["area"], round(bw * bh, 2), places=2)

        yolo_lines = to_yolo_labels(sample.meta)
        self.assertEqual(len(yolo_lines), len(coco["annotations"]))
        for line in yolo_lines:
            parts = line.split()
            self.assertEqual(len(parts), 5)
            cls_id = int(parts[0])
            self.assertIn(cls_id, (0, 1))
            cx, cy, w, h = [float(x) for x in parts[1:]]
            for val in (cx, cy, w, h):
                self.assertGreaterEqual(val, 0.0)
                self.assertLessEqual(val, 1.0)
            self.assertGreater(w, 0.0)
            self.assertGreater(h, 0.0)

    def test_export_dataset_sample_atomic_writes(self) -> None:
        sample = render_sample(
            MINIMAL_MSCX,
            "student_ballpoint_blue",
            seed=55,
            page_w=800,
            page_h=1000,
            include_musicxml=True,
        )
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td) / "export_out"
            written = export_dataset_sample(
                sample,
                out_dir,
                "sample_0001",
                formats=("json", "coco", "yolo", "musicxml"),
            )

            self.assertTrue(Path(written["image"]).exists())
            self.assertTrue(Path(written["json"]).exists())
            self.assertTrue(Path(written["coco"]).exists())
            self.assertTrue(Path(written["yolo"]).exists())
            self.assertGreaterEqual(len(written["musicxml"]), 1)
            for mp in written["musicxml"]:
                self.assertTrue(Path(mp).exists())
                self.assertIn("<score-partwise", Path(mp).read_text(encoding="utf-8"))

            # Verify no leftover .tmp files remain after atomic rename
            leftover_tmps = list(out_dir.glob("*.tmp"))
            self.assertEqual(leftover_tmps, [])

            # Verify JSON & COCO parse cleanly
            meta_loaded = json.loads(Path(written["json"]).read_text(encoding="utf-8"))
            self.assertEqual(meta_loaded["schema_version"], "1.0")
            coco_loaded = json.loads(Path(written["coco"]).read_text(encoding="utf-8"))
            self.assertEqual(coco_loaded["images"][0]["file_name"], "sample_0001.png")

            # Verify saved PNG matches in-memory image size
            with Image.open(written["image"]) as saved_img:
                self.assertEqual(saved_img.size, (800, 1000))

    def test_dataset_and_hf_generator(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p1 = Path(td) / "s1.mscx"
            p2 = Path(td) / "s2.mscx"
            p1.write_text(MINIMAL_MSCX, encoding="utf-8")
            p2.write_text(MINIMAL_MSCX, encoding="utf-8")

            ds = HandwrittenScoreDataset(
                [p1, p2],
                scribes=["baroque_quill_neat", "debussy_pencil_sketch"],
                seed=101,
                page_w=750,
                page_h=950,
            )
            self.assertEqual(len(ds), 2)
            s0 = ds[0]
            self.assertIsInstance(s0, RenderSample)
            self.assertEqual(s0.meta["style"], "baroque_quill_neat")
            self.assertEqual(s0.image.size, (750, 950))

            rows = list(
                to_hf_generator(
                    [p1, p2],
                    scribes=["baroque_quill_neat", "debussy_pencil_sketch"],
                    seed=101,
                    page_w=750,
                    page_h=950,
                    include_coco=True,
                    include_yolo=True,
                )
            )
            self.assertEqual(len(rows), 2)
            self.assertIsInstance(rows[0]["image"], Image.Image)
            self.assertEqual(rows[0]["style"], "baroque_quill_neat")
            self.assertEqual(rows[1]["style"], "debussy_pencil_sketch")
            self.assertIn("coco_json", rows[0])
            self.assertIn("yolo_labels", rows[0])
            self.assertEqual(json.loads(rows[0]["meta_json"])["schema_version"], "1.0")
            self.assertGreaterEqual(len(rows[0]["musicxml_systems"]), 1)

    def test_top_level_exports(self) -> None:
        for sym in (
            "RenderSample",
            "render_sample",
            "to_coco_annotations",
            "to_yolo_labels",
            "export_dataset_sample",
            "HandwrittenScoreDataset",
            "to_hf_generator",
        ):
            self.assertTrue(hasattr(hw_score_sim, sym), f"Missing top-level export: {sym}")


if __name__ == "__main__":
    unittest.main()
