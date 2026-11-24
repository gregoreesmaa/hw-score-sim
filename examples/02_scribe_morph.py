"""Example 02 — "Continuous Scribe Interpolation: Chopin -> Beethoven".

Uses `interpolate_styles` across `alpha in [0.0, 0.25, 0.5, 0.75, 1.0]` with a
fixed deterministic seed to render a smooth 5-panel morph strip from delicate
Parisian salon handwriting (`chopin_delicate_salon`) to turbulent, heavily-inked
Beethoven autograph drafts (`beethoven_hurried_autograph`).

Usage:
    python examples/02_scribe_morph.py --out-dir out/02_scribe_morph
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageDraw, ImageFont

from hw_score_sim import interpolate_styles, register_style, render_score

try:
    from _sample_score import write_sample_score
except ImportError:
    from examples._sample_score import write_sample_score

MORPH_ALPHAS: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)


def _stitch_morph_strip(
    panels: list[tuple[float, str, Path]],
    out_png: Path,
    style_a: str,
    style_b: str,
    panel_w: int = 440,
) -> Path:
    """Stitch interpolated panels into a horizontal 5-panel morph strip."""
    pad = 16
    title_h = 46
    label_h = 42
    resized: list[tuple[float, str, Image.Image]] = []

    for alpha, notehead, img_path in panels:
        with Image.open(img_path) as im:
            rgb = im.convert("RGB")
            scale = panel_w / float(rgb.width)
            target_h = max(1, int(round(rgb.height * scale)))
            resized.append((alpha, notehead, rgb.resize((panel_w, target_h), Image.Resampling.LANCZOS)))

    max_h = max(im.height for _, _, im in resized)
    total_w = pad + len(resized) * (panel_w + pad)
    total_h = title_h + label_h + max_h + pad

    canvas = Image.new("RGB", (total_w, total_h), (24, 24, 28))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    draw.text(
        (pad, 16),
        f"Continuous Scribe Morph: {style_a}  ->  {style_b}",
        fill=(244, 244, 248),
        font=font,
    )

    for idx, (alpha, notehead, im) in enumerate(resized):
        x0 = pad + idx * (panel_w + pad)
        y_img = title_h + label_h
        draw.rectangle([x0, title_h, x0 + panel_w, y_img - 4], fill=(38, 38, 44))
        draw.text((x0 + 8, title_h + 6), f"alpha = {alpha:.2f}", fill=(244, 244, 248), font=font)
        draw.text((x0 + 8, title_h + 22), f"notehead: {notehead}", fill=(161, 161, 174), font=font)
        canvas.paste(im, (x0, y_img))

    out_png.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_png)
    return out_png


def run(
    out_dir: str | Path = "out/02_scribe_morph",
    *,
    style_a: str = "chopin_delicate_salon",
    style_b: str = "beethoven_hurried_autograph",
    seed: int = 2026,
    page_w: int = 1200,
    page_h: int = 1400,
    panel_w: int = 440,
) -> Path:
    """Render the 5-step continuous morph and return the stitched strip path."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    score_path = write_sample_score(out_path / "sample_score.musicxml")

    panels: list[tuple[float, str, Path]] = []
    for idx, alpha in enumerate(MORPH_ALPHAS):
        blended = interpolate_styles(style_a, style_b, alpha)
        reg_name = f"morph_{idx:02d}_a{int(round(alpha * 100)):03d}"
        register_style(reg_name, blended, overwrite=True)

        png_path = out_path / f"{idx + 1:02d}_alpha_{alpha:.2f}.png"
        render_score(
            score_path=score_path,
            style_name=reg_name,
            out_png=png_path,
            seed=seed,  # Fixed seed keeps rhythm/layout aligned across morph steps
            page_w=page_w,
            page_h=page_h,
        )
        panels.append((alpha, str(blended.get("notehead_archetype", "oval")), png_path))

    strip_path = _stitch_morph_strip(
        panels,
        out_path / "scribe_morph_strip.png",
        style_a=style_a,
        style_b=style_b,
        panel_w=panel_w,
    )
    print(f"[ok] Wrote 5-panel scribe morph strip -> {strip_path.resolve()}")
    return strip_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Continuous Scribe Interpolation: Chopin -> Beethoven.")
    parser.add_argument("--out-dir", default="out/02_scribe_morph", help="Output directory")
    parser.add_argument("--style-a", default="chopin_delicate_salon", help="Start scribe style (alpha=0)")
    parser.add_argument("--style-b", default="beethoven_hurried_autograph", help="End scribe style (alpha=1)")
    parser.add_argument("--seed", type=int, default=2026, help="Fixed RNG seed across all morph frames")
    parser.add_argument("--page-w", type=int, default=1200, help="Rendered page width in pixels")
    parser.add_argument("--page-h", type=int, default=1400, help="Rendered page height in pixels")
    parser.add_argument("--panel-w", type=int, default=440, help="Width of each panel in the strip")
    args = parser.parse_args(argv)
    run(
        args.out_dir,
        style_a=args.style_a,
        style_b=args.style_b,
        seed=args.seed,
        page_w=args.page_w,
        page_h=args.page_h,
        panel_w=args.panel_w,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
