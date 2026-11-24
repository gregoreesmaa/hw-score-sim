"""Example 04 — "Design Your Own Scribe in 20 Lines of JSON".

Demonstrates how to define a brand-new handwritten scribe in ~20 lines of JSON
using `"extends"` inheritance, validate and register it with `hw_score_sim`, and
render a side-by-side Before / After A/B comparison PNG against its parent style.

Usage:
    python examples/04_custom_scribe_from_json.py --out-dir out/04_custom_scribe
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageDraw, ImageFont

from hw_score_sim import register_style, render_score
from hw_score_sim.styles import validate_style

try:
    from _sample_score import write_sample_score
except ImportError:
    from examples._sample_score import write_sample_score

CUSTOM_SCRIBE_JSON = """{
  "name": "midnight_indigo_architect",
  "extends": "romantic_copyist_oblong",
  "description": "Crisp midnight-indigo architect pen with dramatic italic slant and sapphire-tinted rag paper.",
  "era": "modern_classic",
  "ink_rgb": [0.06, 0.13, 0.34],
  "staff_ink_rgb": [0.24, 0.30, 0.44],
  "paper_rgb": [0.93, 0.95, 0.98],
  "nib_angle_deg": 42.0,
  "nib_min_ratio": 0.20,
  "stroke_scale": 1.18,
  "slant_deg": 14.5,
  "wobble_px": 0.55,
  "notehead_archetype": "teardrop",
  "stem_archetype": "tapered_quill",
  "flag_archetype": "calligraphic_ribbon",
  "beam_archetype": "arched_convex",
  "clef_archetype": "calligraphic_broad_nib",
  "tags": ["custom", "indigo", "calligraphic"]
}
"""


def _stitch_ab_comparison(
    before_label: str,
    before_png: Path,
    after_label: str,
    after_png: Path,
    out_png: Path,
    panel_w: int = 560,
) -> Path:
    """Create a side-by-side Before / After A/B comparison PNG."""
    pad = 18
    title_h = 46
    label_h = 34
    imgs: list[tuple[str, Image.Image]] = []

    for label, path in ((before_label, before_png), (after_label, after_png)):
        with Image.open(path) as im:
            rgb = im.convert("RGB")
            scale = panel_w / float(rgb.width)
            target_h = max(1, int(round(rgb.height * scale)))
            imgs.append((label, rgb.resize((panel_w, target_h), Image.Resampling.LANCZOS)))

    max_h = max(im.height for _, im in imgs)
    total_w = pad + len(imgs) * (panel_w + pad)
    total_h = title_h + label_h + max_h + pad

    canvas = Image.new("RGB", (total_w, total_h), (24, 24, 28))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((pad, 16), "Custom Scribe from 20 Lines of JSON — Before / After A/B", fill=(244, 244, 248), font=font)

    for idx, (label, im) in enumerate(imgs):
        x0 = pad + idx * (panel_w + pad)
        y_img = title_h + label_h
        draw.rectangle([x0, title_h, x0 + panel_w, y_img - 4], fill=(38, 38, 44))
        draw.text((x0 + 8, title_h + 10), label, fill=(244, 244, 248), font=font)
        canvas.paste(im, (x0, y_img))

    out_png.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_png)
    return out_png


def run(
    out_dir: str | Path = "out/04_custom_scribe",
    *,
    seed: int = 42,
    page_w: int = 1200,
    page_h: int = 1400,
    panel_w: int = 560,
) -> Path:
    """Register the custom JSON scribe, validate it, and render an A/B comparison PNG."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    score_path = write_sample_score(out_path / "sample_score.musicxml")

    json_path = out_path / "custom_scribe.json"
    json_path.write_text(CUSTOM_SCRIBE_JSON, encoding="utf-8")

    spec = json.loads(json_path.read_text(encoding="utf-8"))
    parent_name = str(spec["extends"])
    custom_name = str(spec["name"])

    # Resolve inheritance ("extends") + validate all required keys & bounds.
    registered = register_style(custom_name, spec, overwrite=True)
    validate_style(registered, name=custom_name)

    before_png = out_path / f"before_{parent_name}.png"
    after_png = out_path / f"after_{custom_name}.png"

    render_score(score_path, parent_name, before_png, seed=seed, page_w=page_w, page_h=page_h)
    render_score(score_path, custom_name, after_png, seed=seed, page_w=page_w, page_h=page_h)

    ab_path = _stitch_ab_comparison(
        before_label=f"BEFORE (parent): {parent_name}",
        before_png=before_png,
        after_label=f"AFTER (custom JSON): {custom_name}",
        after_png=after_png,
        out_png=out_path / "custom_scribe_ab.png",
        panel_w=panel_w,
    )
    print(f"[ok] Validated '{custom_name}' and wrote A/B comparison -> {ab_path.resolve()}")
    return ab_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Design Your Own Scribe in 20 Lines of JSON.")
    parser.add_argument("--out-dir", default="out/04_custom_scribe", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic RNG seed")
    parser.add_argument("--page-w", type=int, default=1200, help="Rendered page width in pixels")
    parser.add_argument("--page-h", type=int, default=1400, help="Rendered page height in pixels")
    parser.add_argument("--panel-w", type=int, default=560, help="Width of each A/B panel")
    args = parser.parse_args(argv)
    run(args.out_dir, seed=args.seed, page_w=args.page_w, page_h=args.page_h, panel_w=args.panel_w)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
