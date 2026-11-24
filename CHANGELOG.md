# Changelog

All notable changes to `hw-score-sim` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.1] — 2026-11-24

### Added

- **25 Flagship Hero Scribes (`src/hw_score_sim/style_packs/hero_scribes.json`):**
  - Expanded the registered style catalog from **159 → 184** styles across 11 curated JSON packs + 20 built-in core profiles.
  - Adds 25 deeply researched, paleographically distinct flagship composer & copyist hands spanning five historical eras and traditions:
    - *Renaissance & Baroque:* Palestrina Sistine Choirbook, Monteverdi Vespers Quill, Couperin Versailles Clavecin, Anna Magdalena Bach Notebook, Vivaldi Turin Concerto Rush.
    - *Romantic Fair Copies & Opera:* Mendelssohn Leipzig Fair Copy, Clara Schumann Concert Part, Fanny Hensel Berlin Salon, Rossini Naples Opera Rush, Berlioz Paris Grand Score.
    - *Late-Romantic & Impressionist:* Lili Boulanger Prix de Rome, Ravel Watchmaker Micro-Nib, Scriabin Ecstatic Pencil, Mussorgsky *Pictures* Iron-Gall, Mahler Vienna Blue Crayon.
    - *20th-Century Modernist & Film:* Stravinsky Morges Fine Pen, Messiaen Birdsong Cahier, George Crumb *Makrokosmos* Fineliner, John Williams Hollywood Pencil, Ennio Morricone Cinecittà Ink.
    - *Jazz, Tango & Contemporary:* Thelonious Monk Angular Lead Sheet, Bill Evans Jazz Pencil, Astor Piazzolla Buenos Aires Tango, Blue Note Van Gelder Ozalid, Songwriter Spiral Gel-Pen.
  - Every hero profile passes strict `validate_style()` with zero warnings/clamps and maps cleanly across all 4 axes in `catalog.py`.

---

## [0.1.0] — 2026-11-24

Initial public MIT-licensed release of `hw-score-sim`, a pure-Python procedural handwritten sheet-music rendering engine for MusicXML and MuseScore files.

### Added

- **Symbolic Notation Parser & Exporter (`musescore_compat.py`):**
  - Native parsing for MusicXML 3.1/4.0 (`.musicxml`, `.mxl`) and MuseScore 3/4 (`.mscx`, `.mscz`) with zero external binaries.
  - Full notation coverage: multi-voice polyphony (voices 1–4), opposing stems and collision nudges, ties & slurs (intra- and cross-measure), tuplets, articulations, ornaments, grace notes, dynamics, hairpins, tempo markings, lyrics, mid-measure clef/key/time changes, `8va`/`8vb` clefs, double and courtesy accidentals, multi-bar rests, and volta repeat brackets.
  - `export_score_to_musicxml()` helper for per-system ground-truth targets.
- **159 Handwritten Scribe Styles Across 10 JSON Packs (`styles.py`, `style_packs/*.json`):**
  - 20 built-in core scribe profiles plus 139 styles across 10 auto-discovered JSON packs spanning Renaissance/Baroque quills, Romantic autographs, 20th-century modernist/film-studio hands, 1970s Real Book & jazz copyists, and extreme/degraded everyday pens and pencils.
  - Declarative `"extends"` multi-level JSON inheritance with cycle detection and strict parameter validation.
  - Continuous style interpolation (`interpolate_styles`) and unbounded unseen-scribe generation (`sample_random_scribe`).
- **134 Discrete Glyph Archetypes + 5 Ledger Styles (`glyph_variants.py`, `structural_glyphs.py`):**
  - 18 noteheads (including ultra-slim `needle_slash`, `hairline_dash`, `pinprick_dot`, `vertical_tick`, `slim_oval`, `slim_diamond`), 18 stems, 20 flags, 14 beams, 18 clefs, 18 key-signature/accidental styles, 18 barlines, 10 rests, and 5 ledger-line styles.
- **Calligraphic Stroke & Paper Physics (`stroke_engine.py`, `layout_quirks.py`):**
  - Pure NumPy/Pillow `StrokeCanvas` with broad/pointed nib-angle width modulation, pressure taper, ink pooling, dry-brush dropout, split-nib effects, smooth bow wobble, neuromuscular tremor, and tileable paper-tooth texture.
  - 6 procedural paper stocks (`aged_rag`, `bright_white`, `cream_manuscript`, `spiral_notebook`, `ozalid_blue`, `soviet_green_tint`) with laid-paper grain, foxing/mottling, reverse-side bleed-through, eraser smudges, curved rastrum staves, and end-of-bar cramming.
- **Ground-Truth & ML Dataset Ergonomics (`renderer.py`, Python API):**
  - Schema v1.0 companion JSON ground truth with Bezier staff curves, system/measure bounds, and symbol annotations.
  - Ergonomic Python & OMR dataset helpers (`render_sample`, `ScribeSampler`, `filter_styles`, `to_coco_annotations`, `to_yolo_labels`) plus optional cropped per-system PNG + MusicXML export.
- **Command-Line Interfaces (`cli.py`):**
  - `hw-score-sim` for single/batch rendering, `--list-styles`, and `--export-train-dir`.
  - `hw-score-gallery` for parallel HTML visual galleries (`scores`, `stems-flags`, and `structural` modes).
- **Documentation & Examples:**
  - Comprehensive `README.md`, `ARCHITECTURE.md`, `CONTRIBUTING.md`, `CHANGELOG.md`, and runnable examples (`examples/01_time_machine_strip.py` through `examples/04_custom_scribe_from_json.py`).
