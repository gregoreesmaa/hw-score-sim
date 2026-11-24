"""Example 01 — "One Score, Five Centuries" (Time Machine Strip).

Renders the bundled 8-bar sample score across five iconic historical eras and
writing implements, then stitches a crisp side-by-side comparison PNG:
  1. 1600s · Frescobaldi lozenge choirbook (`wild_hist_frescobaldi_lozenge_choirbook`)
  2. 1730s · Baroque goose quill fair copy (`baroque_quill_neat`)
  3. 1826  · Beethoven Op. 131 blotted autograph (`wild_rom_beethoven_op131_blots`)
  4. 1970s · Real Book felt-tip lead sheet (`real_book_1970s_felt_tip`)
  5. 2020s · 0.25mm mechanical pencil (`wild_ext_025mm_mech_pencil`)

Usage:
    python examples/01_time_machine_strip.py --out-dir out/01_time_machine
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageDraw, ImageFont

from hw_score_sim import render_score

try:
    from _sample_score import write_sample_score
except ImportError:
    from examples._sample_score import write_sample_score

FIVE_CENTURIES: tuple[tuple[str, str], ...] = (
    ("wild_hist_frescobaldi_lozenge_choirbook", "1600s · Frescobaldi Choirbook"),
    ("baroque_quill_neat", "1730s · Baroque Goose Quill"),
    ("wild_rom_beethoven_op131_blots", "1826 · Beethoven Op. 131"),
    ("real_book_1970s_felt_tip", "1970s · Real Book Felt-Tip"),
    ("wild_ext_025mm_mech_pencil", "2020s · 0.25mm Mech Pencil"),
)


def _stitch_labeled_strip(
    panels: list[tuple[str, str, Path]],
    out_png: Path,
    title: str,
    panel_w: int = 440,
) -> Path:
    """Resize panel PNGs to a uniform width and stitch them side-by-side with labels."""
    pad = 16
    title_h = 46
    label_h = 42
    resized: list[tuple[str, str, Image.Image]] = []

    for style_name, caption, img_path in panels:
        with Image.open(img_path) as im:
            rgb = im.convert("RGB")
            scale = panel_w / float(rgb.width)
            target_h = max(1, int(round(rgb.height * scale)))
            resized.append((style_name, caption, rgb.resize((panel_w, target_h), Image.Resampling.LANCZOS)))

    max_h = max(im.height for _, _, im in resized)
    total_w = pad + len(resized) * (panel_w + pad)
    total_h = title_h + label_h + max_h + pad

    canvas = Image.new("RGB", (total_w, total_h), (24, 24, 28))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    draw.text((pad, 16), title, fill=(244, 244, 248), font=font)

    for idx, (style_name, caption, im) in enumerate(resized):
        x0 = pad + idx * (panel_w + pad)
        y_img = title_h + label_h
        draw.rectangle([x0, title_h, x0 + panel_w, y_img - 4], fill=(38, 38, 44))
        draw.text((x0 + 8, title_h + 6), caption, fill=(244, 244, 248), font=font)
        draw.text((x0 + 8, title_h + 22), style_name, fill=(161, 161, 174), font=font)
        canvas.paste(im, (x0, y_img))

    out_png.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_png)
    return out_png


def run(
    out_dir: str | Path = "out/01_time_machine",
    *,
    seed: int = 101,
    page_w: int = 1200,
    page_h: int = 1400,
    panel_w: int = 440,
) -> Path:
    """Render all five eras and return the path to the stitched comparison PNG."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    score_path = write_sample_score(out_path / "sample_score.musicxml")

    panels: list[tuple[str, str, Path]] = []
    for idx, (style_name, caption) in enumerate(FIVE_CENTURIES):
        png_path = out_path / f"{idx + 1:02d}_{style_name}.png"
        render_score(
            score_path=score_path,
            style_name=style_name,
            out_png=png_path,
            seed=seed + idx,
            page_w=page_w,
            page_h=page_h,
        )
        panels.append((style_name, caption, png_path))

    strip_path = _stitch_labeled_strip(
        panels,
        out_path / "time_machine_strip.png",
        title="One Score, Five Centuries — hw-score-sim Time Machine",
        panel_w=panel_w,
    )
    print(f"[ok] Wrote 5-century comparison strip -> {strip_path.resolve()}")
    return strip_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="One Score, Five Centuries comparison strip.")
    parser.add_argument("--out-dir", default="out/01_time_machine", help="Output directory")
    parser.add_argument("--seed", type=int, default=101, help="Base deterministic RNG seed")
    parser.add_argument("--page-w", type=int, default=1200, help="Rendered page width in pixels")
    parser.add_argument("--page-h", type=int, default=1400, help="Rendered page height in pixels")
    parser.add_argument("--panel-w", type=int, default=440, help="Width of each panel in the strip")
    args = parser.parse_args(argv)
    run(args.out_dir, seed=args.seed, page_w=args.page_w, page_h=args.page_h, panel_w=args.panel_w)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
