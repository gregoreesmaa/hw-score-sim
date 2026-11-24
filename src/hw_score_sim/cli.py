"""Command-line interfaces for `hw-score-sim` and `hw-score-gallery`."""

from __future__ import annotations

import argparse
import concurrent.futures
import difflib
import html
import pathlib
import random
import sys
import tempfile
import time
from typing import List, Optional, Sequence, Tuple

import numpy as np

from . import renderer, stroke_engine, structural_glyphs as sg, styles


SHOWCASE_MUSICXML = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 3.1 Partwise//EN" "http://www.musicxml.org/dtds/partwise.dtd">
<score-partwise version="3.1">
  <work><work-title>Handwritten Score Simulator Showcase</work-title></work>
  <part-list>
    <score-part id="P1"><part-name>Showcase</part-name></score-part>
  </part-list>
  <part id="P1">
    <measure number="1">
      <attributes>
        <divisions>8</divisions>
        <key><fifths>0</fifths></key>
        <time><beats>4</beats><beat-type>4</beat-type></time>
        <clef><sign>G</sign><line>2</line></clef>
      </attributes>
      <note><pitch><step>C</step><octave>4</octave></pitch><duration>4</duration><type>eighth</type><stem>up</stem></note>
      <note><rest/><duration>4</duration><type>eighth</type></note>
      <note><pitch><step>E</step><octave>4</octave></pitch><duration>2</duration><type>16th</type><stem>up</stem></note>
      <note><rest/><duration>6</duration><type>eighth</type><dot/></note>
      <note><pitch><step>G</step><octave>4</octave></pitch><duration>1</duration><type>32nd</type><stem>up</stem></note>
      <note><rest/><duration>7</duration><type>eighth</type><dot/></note>
      <note><pitch><step>A</step><octave>5</octave></pitch><duration>4</duration><type>eighth</type><stem>down</stem></note>
      <note><rest/><duration>4</duration><type>eighth</type></note>
    </measure>
    <measure number="2">
      <note><pitch><step>D</step><octave>4</octave></pitch><duration>4</duration><type>eighth</type><stem>up</stem><beam number="1">begin</beam></note>
      <note><pitch><step>F</step><octave>4</octave></pitch><duration>4</duration><type>eighth</type><stem>up</stem><beam number="1">end</beam></note>
      <note><pitch><step>E</step><octave>4</octave></pitch><duration>2</duration><type>16th</type><stem>up</stem><beam number="1">begin</beam><beam number="2">begin</beam></note>
      <note><pitch><step>G</step><octave>4</octave></pitch><duration>2</duration><type>16th</type><stem>up</stem><beam number="1">continue</beam><beam number="2">continue</beam></note>
      <note><pitch><step>A</step><octave>4</octave></pitch><duration>2</duration><type>16th</type><stem>up</stem><beam number="1">continue</beam><beam number="2">continue</beam></note>
      <note><pitch><step>F</step><octave>4</octave></pitch><duration>2</duration><type>16th</type><stem>up</stem><beam number="1">end</beam><beam number="2">end</beam></note>
      <note><pitch><step>G</step><octave>5</octave></pitch><duration>1</duration><type>32nd</type><stem>down</stem><beam number="1">begin</beam><beam number="2">begin</beam><beam number="3">begin</beam></note>
      <note><pitch><step>F</step><octave>5</octave></pitch><duration>1</duration><type>32nd</type><stem>down</stem><beam number="1">continue</beam><beam number="2">continue</beam><beam number="3">continue</beam></note>
      <note><pitch><step>E</step><octave>5</octave></pitch><duration>1</duration><type>32nd</type><stem>down</stem><beam number="1">continue</beam><beam number="2">continue</beam><beam number="3">continue</beam></note>
      <note><pitch><step>D</step><octave>5</octave></pitch><duration>1</duration><type>32nd</type><stem>down</stem><beam number="1">end</beam><beam number="2">end</beam><beam number="3">end</beam></note>
      <note><pitch><step>C</step><octave>6</octave></pitch><duration>6</duration><type>eighth</type><dot/><stem>down</stem><beam number="1">begin</beam></note>
      <note><pitch><step>B</step><octave>5</octave></pitch><duration>2</duration><type>16th</type><stem>down</stem><beam number="1">end</beam><beam number="2">backward hook</beam></note>
    </measure>
    <measure number="3">
      <note><pitch><step>F</step><octave>5</octave></pitch><duration>2</duration><type>16th</type><stem>down</stem></note>
      <note><rest/><duration>6</duration><type>eighth</type><dot/></note>
      <note><pitch><step>D</step><octave>5</octave></pitch><duration>1</duration><type>32nd</type><stem>down</stem></note>
      <note><rest/><duration>7</duration><type>eighth</type><dot/></note>
      <note><pitch><step>A</step><octave>3</octave></pitch><duration>8</duration><type>quarter</type><stem>up</stem></note>
      <note><pitch><step>C</step><octave>6</octave></pitch><duration>8</duration><type>quarter</type><stem>down</stem></note>
    </measure>
  </part>
</score-partwise>
"""


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entrypoint for `hw-score-sim`."""
    parser = argparse.ArgumentParser(
        prog="hw-score-sim",
        description="Render a MusicXML or MuseScore (.mscx/.mscz) file in handwritten styles.",
    )
    parser.add_argument("--input", help="Path to .musicxml, .mxl, .mscx, or .mscz")
    parser.add_argument(
        "--styles",
        default="beethoven_hurried_autograph,debussy_pencil_sketch,student_ballpoint_blue",
        help="Comma-separated style names or 'all' for all registered styles",
    )
    parser.add_argument(
        "--out-dir",
        default="out",
        help="Directory to write rendered PNGs and GT JSON files",
    )
    parser.add_argument(
        "--export-train-dir",
        default=None,
        help="Optional directory to export cropped per-system PNGs and per-system MusicXML targets",
    )
    parser.add_argument("--seed", type=int, default=42, help="Deterministic RNG seed")
    parser.add_argument("--page-w", type=int, default=1650, help="Output page width in pixels")
    parser.add_argument("--page-h", type=int, default=2200, help="Output page height in pixels")
    parser.add_argument(
        "--clean-paper",
        action="store_true",
        help="Render pure handwritten ink on flat white paper, skipping paper aging/textures and wet-ink/wear accidents",
    )
    parser.add_argument(
        "--list-styles",
        action="store_true",
        help="Print all registered handwritten style names and exit",
    )
    args = parser.parse_args(argv)

    if args.list_styles:
        for name in styles.list_styles():
            print(name)
        return 0

    if not args.input:
        parser.error("--input is required unless --list-styles is specified")

    if args.styles.strip().lower() == "all":
        style_names = styles.list_styles()
    else:
        style_names = [s.strip() for s in args.styles.split(",") if s.strip()]

    in_path = pathlib.Path(args.input)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx, sname in enumerate(style_names):
        out_png = out_dir / f"{in_path.stem}__{sname}.png"
        try:
            gt = renderer.render_score_handwritten(
                score_path=in_path,
                style_name=sname,
                out_png=out_png,
                seed=args.seed + idx,
                page_w=args.page_w,
                page_h=args.page_h,
                export_train_dir=args.export_train_dir,
                clean_paper=args.clean_paper,
            )
        except KeyError:
            all_known = styles.list_styles()
            suggestions = difflib.get_close_matches(sname, all_known, n=5, cutoff=0.4)
            msg = f"Error: unknown style {sname!r}."
            if suggestions:
                msg += f" Did you mean: {', '.join(suggestions)}?"
            msg += " Run `hw-score-sim --list-styles` to see all available styles."
            print(msg, file=sys.stderr)
            return 1
        extra = ""
        if args.export_train_dir:
            n_exp = len(gt.get("train_exports", []))
            extra = f", train_strips={n_exp}"
        print(
            f"[ok] {sname}: {out_png} "
            f"({len(gt['systems'])} systems, gt={out_png.with_suffix('.json').name}{extra})"
        )
    return 0


# ---------------------------------------------------------------------------
# Gallery Helpers (`hw-score-gallery`)
# ---------------------------------------------------------------------------


def _render_score_gallery_job(
    job: Tuple[int, str, str, str, int, str, int, int],
) -> Tuple[int, str, str, str, bool, str]:
    i, sc, sname, mode, seed, out_dir_s, page_w, page_h = job
    out_dir = pathlib.Path(out_dir_s)
    sc_stem = pathlib.Path(sc).stem
    if mode == "random_scribe":
        r = np.random.default_rng(seed)
        custom = styles.sample_random_scribe(r, base_style=sname, jitter=0.25)
        reg_name = f"rand_{i:03d}_{sname}"
        styles.register_style(reg_name, custom, overwrite=True)
        use_name = reg_name
        label = f"rand({sname})"
    else:
        use_name = sname
        label = sname
    out_png = out_dir / f"{i + 1:03d}_{sc_stem}__{use_name}.png"
    try:
        renderer.render_score_handwritten(sc, use_name, str(out_png), seed=seed, page_w=page_w, page_h=page_h)
        return (i, out_png.name, sc_stem, label, True, "")
    except Exception as exc:
        return (i, out_png.name, sc_stem, label, False, str(exc))


def _render_stems_flags_job(
    job: Tuple[int, str, str, str, str, int],
) -> Tuple[int, str, str, str, bool, str]:
    idx, category, variant_name, xml_path, out_dir_s, seed = job
    out_dir = pathlib.Path(out_dir_s)
    base = styles.get_style("romantic_copyist_oblong")
    base["wobble_px"] = 0.30
    base["tremor_px"] = 0.04
    base["slant_deg"] = 0.0
    base["staff_curve_px"] = 0.8
    base["staff_slope_deg"] = 0.1
    base["spacing_jitter"] = 0.08
    base["bar_width_var"] = 0.10
    base["sp"] = 22.0
    base["margin_x"] = 48.0
    base["margin_top"] = 85.0
    base["margin_bottom"] = 55.0
    base["stem_archetype"] = "straight_classic"
    base["flag_archetype"] = "classic_droop"
    base["beam_archetype"] = "straight_classic"
    base["ledger_style"] = styles.LEDGER_STYLES[idx % len(styles.LEDGER_STYLES)]

    if category == "stem":
        base["stem_archetype"] = variant_name
    elif category == "flag":
        base["flag_archetype"] = variant_name
    elif category == "beam":
        base["beam_archetype"] = variant_name

    reg_name = f"showcase_{category}_{variant_name}"
    styles.register_style(reg_name, base, overwrite=True)

    fname = f"{idx + 1:02d}_{category}__{variant_name}.png"
    out_png = out_dir / fname
    try:
        renderer.render_score_handwritten(
            score_path=xml_path,
            style_name=reg_name,
            out_png=str(out_png),
            seed=seed,
            page_w=1440,
            page_h=270,
        )
        jpath = out_png.with_suffix(".json")
        if jpath.exists():
            jpath.unlink()
        return (idx, category, variant_name, fname, True, "")
    except Exception as exc:
        return (idx, category, variant_name, fname, False, str(exc))


def _render_structural_gallery(out_dir: pathlib.Path, seed: int) -> int:
    paper_bg = (0.975, 0.955, 0.915)
    staff_ink = (0.38, 0.34, 0.30)
    glyph_ink = (0.09, 0.08, 0.07)
    base_style = {
        "name": "gallery_scribe",
        "era": "romantic",
        "ink_rgb": glyph_ink,
        "staff_ink_rgb": staff_ink,
        "nib_angle_deg": 32.0,
        "nib_min_ratio": 0.32,
        "stroke_scale": 1.0,
        "wobble_px": 0.40,
        "tremor_px": 0.16,
        "ink_dropout": 0.02,
        "grain_strength": 0.05,
        "ink_var": 0.02,
        "ink_pooling": 0.18,
        "slant_deg": 0.0,
        "staff_pre_ruled": True,
    }

    def _draw_five(canvas: stroke_engine.StrokeCanvas, x0: float, x1: float, yc: float, sp: float) -> List[float]:
        y_lines = [yc + (i - 2) * sp for i in range(5)]
        for yl in y_lines:
            stroke_engine.draw_staff_line(
                canvas,
                x0,
                x1,
                lambda x, y=yl: np.full_like(x, y, dtype=np.float64) if isinstance(x, np.ndarray) else y,
                staff_ink,
                1.25,
                base_style,
            )
        return y_lines

    def _clef_card(arch: str, out_png: pathlib.Path, s: int) -> None:
        rng = np.random.default_rng(s)
        w, h, sp = 420, 175, 16.0
        yc = h * 0.50
        c = stroke_engine.StrokeCanvas(w, h, paper_bg, rng=rng)
        _draw_five(c, 18.0, w - 18.0, yc, sp)
        sg.draw_clef_variant(c, 68.0, yc + sp, sp, "G", arch, base_style, rng)
        sg.draw_clef_variant(c, 164.0, yc - sp, sp, "F", arch, base_style, rng)
        sg.draw_clef_variant(c, 258.0, yc, sp, "C", arch, base_style, rng)
        sg.draw_clef_variant(c, 354.0, yc + sp, sp, "G8vb", arch, base_style, rng)
        c.to_pil().save(out_png)

    def _keysig_card(arch: str, out_png: pathlib.Path, s: int) -> None:
        rng = np.random.default_rng(s)
        w, h, sp = 460, 165, 16.0
        yc = h * 0.50
        c = stroke_engine.StrokeCanvas(w, h, paper_bg, rng=rng)
        y_lines = _draw_five(c, 16.0, w - 16.0, yc, sp)
        sg.draw_keysig_variant(c, 34.0, y_lines, sp, 4, "G", arch, base_style, rng)
        sg.draw_keysig_variant(c, 188.0, y_lines, sp, -4, "G", arch, base_style, rng)
        sg.draw_accidental_variant(c, 336.0, yc, sp, "natural", arch, base_style, rng)
        sg.draw_accidental_variant(c, 376.0, yc - sp * 0.5, sp, "double_sharp", arch, base_style, rng)
        sg.draw_accidental_variant(c, 420.0, yc + sp * 0.5, sp, "double_flat", arch, base_style, rng)
        c.to_pil().save(out_png)

    def _barline_card(arch: str, out_png: pathlib.Path, s: int) -> None:
        rng = np.random.default_rng(s)
        w, h, sp = 460, 165, 16.0
        yc = h * 0.50
        c = stroke_engine.StrokeCanvas(w, h, paper_bg, rng=rng)
        y_lines = _draw_five(c, 16.0, w - 16.0, yc, sp)
        y_top, y_bot = y_lines[0], y_lines[-1]
        for bx, bkind in (
            (52.0, "single"),
            (118.0, "double"),
            (188.0, "final"),
            (256.0, "repeat_start"),
            (338.0, "repeat_end"),
            (414.0, "repeat_both"),
        ):
            sg.draw_barline_variant(c, bx, y_top, y_bot, sp, bkind, arch, base_style, rng)
        c.to_pil().save(out_png)

    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    sections = [
        ("1. Clefs (18 Archetypes - G, F, C, G8vb)", "clef", sg.CLEF_ARCHETYPES, _clef_card),
        ("2. Key Signatures & Accidentals (18 Archetypes)", "keysig", sg.KEYSIG_ARCHETYPES, _keysig_card),
        ("3. Barlines (18 Archetypes)", "barline", sg.BARLINE_ARCHETYPES, _barline_card),
    ]
    html_sections: List[str] = []
    total = 0
    for title, prefix, arches, fn in sections:
        cards: List[str] = []
        for idx, arch in enumerate(arches):
            total += 1
            fname = f"{prefix}_{idx + 1:02d}_{arch}.png"
            fn(arch, out_dir / fname, seed + total * 17)
            cards.append(
                f'<div class="card"><a href="{fname}" target="_blank"><img src="{fname}" loading="lazy"/></a>'
                f'<div class="cap"><b>#{idx + 1:02d}</b> <code>{html.escape(arch)}</code></div></div>'
            )
        html_sections.append(f"<h2>{html.escape(title)}</h2><div class='grid'>{''.join(cards)}</div>")

    page_html = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>Structural Glyphs Gallery</title>"
        "<style>body{font-family:system-ui,sans-serif;background:#18181b;color:#f4f4f5;margin:22px}"
        ".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:12px}"
        ".card{background:#27272a;border-radius:8px;overflow:hidden;border:1px solid #3f3f46}"
        ".card img{width:100%;display:block;background:#f9f4ea}.cap{padding:7px 10px;font-size:12px}"
        "</style></head><body><h1>Handwritten Structural Glyphs Gallery</h1>"
        + "".join(html_sections)
        + "</body></html>"
    )
    index_path = out_dir / "index.html"
    index_path.write_text(page_html, encoding="utf-8")
    print(f"Rendered {total} structural cards in {time.time() - t0:.2f}s | gallery={index_path.resolve()}")
    return 0


def gallery_main(argv: Optional[Sequence[str]] = None) -> int:
    """Entrypoint for `hw-score-gallery`."""
    ap = argparse.ArgumentParser(
        prog="hw-score-gallery",
        description="Render HTML eyeball galleries for handwritten styles and glyph archetypes.",
    )
    ap.add_argument(
        "--mode",
        choices=("scores", "stems-flags", "structural"),
        default="scores",
        help="Gallery type to render (default: scores)",
    )
    ap.add_argument(
        "--scores-dir",
        type=str,
        default=None,
        help="Optional directory of .musicxml/.mxl/.mscx/.mscz files for 'scores' mode (defaults to built-in showcase)",
    )
    ap.add_argument("--count", type=int, default=20, help="Number of renders in 'scores' mode")
    ap.add_argument("--out-dir", type=str, default="gallery", help="Output gallery directory")
    ap.add_argument("--workers", type=int, default=4, help="Parallel worker processes")
    ap.add_argument("--page-w", type=int, default=1320)
    ap.add_argument("--page-h", type=int, default=1760)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--prefix", type=str, default="", help="Optional style-name prefix filter (e.g. 'wild_')")
    args = ap.parse_args(argv)

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "structural":
        return _render_structural_gallery(out_dir, args.seed)

    if args.mode == "stems-flags":
        with tempfile.TemporaryDirectory() as tmpdir:
            xml_path = pathlib.Path(tmpdir) / "showcase.musicxml"
            xml_path.write_text(SHOWCASE_MUSICXML, encoding="utf-8")
            jobs = []
            idx = 0
            for name in styles.STEM_ARCHETYPES:
                jobs.append((idx, "stem", name, str(xml_path), str(out_dir), args.seed + idx))
                idx += 1
            for name in styles.FLAG_ARCHETYPES:
                jobs.append((idx, "flag", name, str(xml_path), str(out_dir), args.seed + idx))
                idx += 1
            for name in styles.BEAM_ARCHETYPES:
                jobs.append((idx, "beam", name, str(xml_path), str(out_dir), args.seed + idx))
                idx += 1
            t0 = time.time()
            with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as ex:
                results = list(ex.map(_render_stems_flags_job, jobs))

        ok = [r for r in results if r[4]]
        fail = [r for r in results if not r[4]]
        cards = [
            f'<div class="card"><div class="cap"><b>#{i + 1:02d}</b> <code>{html.escape(v)}</code> [{cat}]</div>'
            f'<a href="{fn}" target="_blank"><img src="{fn}" loading="lazy"/></a></div>'
            for i, cat, v, fn, _, _ in sorted(ok)
        ]
        page_html = (
            "<!doctype html><html><head><meta charset='utf-8'><title>Stems, Flags & Beams Gallery</title>"
            "<style>body{font-family:system-ui,sans-serif;background:#141417;color:#f4f4f5;margin:24px}"
            ".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(640px,1fr));gap:16px}"
            ".card{background:#232329;border-radius:8px;overflow:hidden;border:1px solid #3f3f46}"
            ".card img{width:100%;display:block;background:#fff}.cap{padding:8px 12px;font-size:13px}</style>"
            f"</head><body><h2>Stems, Flags &amp; Beams Gallery ({len(ok)}/{len(jobs)})</h2>"
            f"<div class='grid'>{''.join(cards)}</div></body></html>"
        )
        (out_dir / "index.html").write_text(page_html, encoding="utf-8")
        print(f"Rendered {len(ok)}/{len(jobs)} strips in {time.time() - t0:.1f}s | fails={len(fail)}")
        return 0 if not fail else 1

    # Default mode: 'scores'
    all_styles = styles.list_styles()
    if args.prefix:
        all_styles = [s for s in all_styles if s.startswith(args.prefix)]
    if not all_styles:
        print("No styles matched the given prefix.")
        return 1

    with tempfile.TemporaryDirectory() as tmpdir:
        score_candidates: List[str] = []
        if args.scores_dir:
            sdir = pathlib.Path(args.scores_dir)
            for ext in ("*.mscx", "*.mscz", "*.musicxml", "*.mxl"):
                score_candidates.extend(str(p) for p in sorted(sdir.glob(ext)))
        if not score_candidates:
            default_xml = pathlib.Path(tmpdir) / "showcase.musicxml"
            default_xml.write_text(SHOWCASE_MUSICXML, encoding="utf-8")
            score_candidates.append(str(default_xml))

        rng = random.Random(args.seed)
        jobs = []
        for i in range(args.count):
            sc = score_candidates[i % len(score_candidates)]
            if i < len(all_styles):
                sname = all_styles[i]
                mode = "named"
            else:
                sname = all_styles[rng.randrange(len(all_styles))]
                mode = "random_scribe"
            jobs.append((i, sc, sname, mode, args.seed + i, str(out_dir), args.page_w, args.page_h))

        t0 = time.time()
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as ex:
            results = list(ex.map(_render_score_gallery_job, jobs))

    ok = [r for r in results if r[4]]
    fail = [r for r in results if not r[4]]
    cards = [
        f'<div class="card"><a href="{fn}" target="_blank"><img src="{fn}" loading="lazy"/></a>'
        f'<div class="cap"><b>#{i + 1:03d}</b> {html.escape(lbl)}<br/><span>{html.escape(stem)}</span></div></div>'
        for i, fn, stem, lbl, _, _ in sorted(ok)
    ]
    page_html = (
        "<!doctype html><html><head><meta charset='utf-8'><title>Handwritten Score Gallery</title>"
        "<style>body{font-family:system-ui,sans-serif;background:#18181b;color:#f4f4f5;margin:20px}"
        ".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:14px}"
        ".card{background:#27272a;border-radius:8px;overflow:hidden;border:1px solid #3f3f46}"
        ".card img{width:100%;display:block;background:#fff}.cap{padding:8px 10px;font-size:12px}"
        ".cap span{color:#a1a1aa}</style></head><body>"
        f"<h2>{len(ok)} Handwritten Score Simulations</h2><div class='grid'>{''.join(cards)}</div></body></html>"
    )
    index_path = out_dir / "index.html"
    index_path.write_text(page_html, encoding="utf-8")
    print(f"Rendered {len(ok)}/{args.count} PNGs in {time.time() - t0:.1f}s | fails={len(fail)} | gallery={index_path.resolve()}")
    return 0 if not fail else 1


if __name__ == "__main__":
    raise SystemExit(main())
