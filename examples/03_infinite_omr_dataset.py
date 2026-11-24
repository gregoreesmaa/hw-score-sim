"""Example 03 — "Generate N Unseen Scribes + ML Ground Truth".

Uses `sample_random_scribe` to procedurally synthesize `--count` (default 6)
brand-new unseen handwriting scribes with paired rendered PNGs, Bezier
staff-curve ground-truth JSONs, a dataset manifest, and an interactive HTML
contact sheet (`index.html`).

Usage:
    python examples/03_infinite_omr_dataset.py --count 6 --out-dir out/03_infinite_omr
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from hw_score_sim import register_style, render_score, sample_random_scribe

try:
    from _sample_score import write_sample_score
except ImportError:
    from examples._sample_score import write_sample_score


def _build_html_index(records: list[dict[str, Any]], out_dir: Path) -> Path:
    """Write a self-contained dark-mode HTML contact sheet for the generated dataset."""
    cards: list[str] = []
    for rec in records:
        png_name = html.escape(rec["png"])
        gt_name = html.escape(rec["gt_json"])
        scribe_name = html.escape(rec["scribe_name"])
        traits = (
            f"era={html.escape(rec['era'])} | "
            f"head={html.escape(rec['notehead_archetype'])} | "
            f"stem={html.escape(rec['stem_archetype'])} | "
            f"beam={html.escape(rec['beam_archetype'])} | "
            f"slant={rec['slant_deg']:.1f} deg | "
            f"systems={rec['num_systems']}"
        )
        cards.append(
            f'<div class="card">'
            f'<a href="{png_name}" target="_blank"><img src="{png_name}" loading="lazy"/></a>'
            f'<div class="cap">'
            f'<b>#{rec["index"]:02d}</b> <code>{scribe_name}</code> '
            f'&middot; <a href="{gt_name}" target="_blank">Bezier GT JSON</a>'
            f'<div class="meta">{traits}</div>'
            f'</div></div>'
        )

    page_html = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>Infinite OMR Dataset — hw-score-sim</title>"
        "<style>"
        "body{font-family:system-ui,sans-serif;background:#141418;color:#f4f4f6;margin:24px}"
        "a{color:#7dd3fc;text-decoration:none}"
        ".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:16px}"
        ".card{background:#222229;border-radius:8px;overflow:hidden;border:1px solid #383844}"
        ".card img{width:100%;display:block;background:#fff}"
        ".cap{padding:10px 12px;font-size:13px}"
        ".meta{color:#a1a1aa;font-size:11px;margin-top:4px}"
        "</style></head><body>"
        f"<h1>Infinite OMR Dataset ({len(records)} Unseen Procedural Scribes)</h1>"
        "<p>Generated with <code>sample_random_scribe()</code> + Bezier GT JSON.</p>"
        f"<div class='grid'>{''.join(cards)}</div>"
        "</body></html>"
    )
    index_path = out_dir / "index.html"
    index_path.write_text(page_html, encoding="utf-8")
    return index_path


def run(
    out_dir: str | Path = "out/03_infinite_omr",
    *,
    count: int = 6,
    seed: int = 777,
    jitter: float = 0.25,
    page_w: int = 1200,
    page_h: int = 1400,
) -> Path:
    """Generate `count` unseen scribes + Bezier GT JSON + HTML index."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    score_path = write_sample_score(out_path / "sample_score.musicxml")

    rng = np.random.default_rng(seed)
    records: list[dict[str, Any]] = []

    for idx in range(count):
        scribe = sample_random_scribe(rng, base_style=None, jitter=jitter)
        scribe_name = f"unseen_{idx + 1:02d}_{scribe['name']}"
        register_style(scribe_name, scribe, overwrite=True)

        png_path = out_path / f"{idx + 1:02d}_{scribe_name}.png"
        gt = render_score(
            score_path=score_path,
            style_name=scribe_name,
            out_png=png_path,
            seed=seed + idx,
            page_w=page_w,
            page_h=page_h,
        )
        gt_path = png_path.with_suffix(".json")
        records.append(
            {
                "index": idx + 1,
                "scribe_name": scribe_name,
                "png": png_path.name,
                "gt_json": gt_path.name,
                "era": str(scribe.get("era", "unknown")),
                "notehead_archetype": str(scribe.get("notehead_archetype", "oval")),
                "stem_archetype": str(scribe.get("stem_archetype", "straight_classic")),
                "flag_archetype": str(scribe.get("flag_archetype", "classic_droop")),
                "beam_archetype": str(scribe.get("beam_archetype", "straight_classic")),
                "slant_deg": float(scribe.get("slant_deg", 0.0)),
                "wobble_px": float(scribe.get("wobble_px", 0.0)),
                "num_systems": len(gt.get("systems", [])),
            }
        )

    manifest_path = out_path / "dataset_manifest.json"
    manifest_path.write_text(
        json.dumps({"seed": seed, "jitter": jitter, "count": count, "samples": records}, indent=2),
        encoding="utf-8",
    )
    index_path = _build_html_index(records, out_path)
    print(f"[ok] Generated {count} unseen scribes + Bezier GT -> {index_path.resolve()}")
    return index_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate N unseen procedural scribes + ML ground truth.")
    parser.add_argument("--count", type=int, default=6, help="Number of unseen scribes to synthesize")
    parser.add_argument("--out-dir", default="out/03_infinite_omr", help="Output directory")
    parser.add_argument("--seed", type=int, default=777, help="Deterministic RNG seed")
    parser.add_argument("--jitter", type=float, default=0.25, help="Parameter jitter scale")
    parser.add_argument("--page-w", type=int, default=1200, help="Rendered page width in pixels")
    parser.add_argument("--page-h", type=int, default=1400, help="Rendered page height in pixels")
    args = parser.parse_args(argv)
    run(
        args.out_dir,
        count=args.count,
        seed=args.seed,
        jitter=args.jitter,
        page_w=args.page_w,
        page_h=args.page_h,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
