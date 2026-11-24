# AGENTS.md — hw-score-sim

Procedural handwritten sheet-music rendering engine (`.musicxml`, `.mxl`, `.mscx`, `.mscz` → manuscript PNG + pixel-accurate Schema v1.0 ground truth, COCO, YOLO, and per-system MusicXML). Pure Python (`numpy>=1.24`, `pillow>=10.0`, Python `>=3.10`), 100% MIT-licensed, zero external engraving binaries.

## Layout

| Path | Responsibility |
|---|---|
| `src/hw_score_sim/styles.py` | Scribe style registry: 20 built-in core profiles + auto-discovery of `style_packs/*.json` (234 total), `"extends"` inheritance with cycle detection, `REQUIRED_STYLE_KEYS`, `_NUMERIC_CLAMP_BOUNDS`, `validate_style`, `register_style`, `reload_style_packs`, `interpolate_styles`, `sample_random_scribe`, and archetype tuples (`NOTEHEAD_ARCHETYPES`, `STEM_ARCHETYPES`, `FLAG_ARCHETYPES`, `BEAM_ARCHETYPES`, `BEAM_SLOPE_STYLES`, `BEAM_GROUPING_MODES`, `REST_ARCHETYPES`, `LEDGER_STYLES`, `PAPER_TYPES`). |
| `src/hw_score_sim/stroke_engine.py` | Calligraphic vector-stroke rasterizer: `StrokeCanvas` (float32 RGB in `[0, 1]` with tileable 256×256 paper-tooth grain) and primitives (`draw_calligraphic_polyline`, `draw_filled_blob`, `draw_staff_line`, `draw_rastrum_staff`). Models nib angle/contrast, pressure taper, ink pooling, dry-brush dropout, split-nib, bow wobble, and tremor. |
| `src/hw_score_sim/glyph_variants.py` | Procedural vector generators for 18 notehead, 18 stem, 20 flag, 14 beam, 10 legacy quarter-rest, and 5 ledger-line archetypes, plus dots, ties, slurs, articulations, hairpins, dynamics, and correction scratches. |
| `src/hw_score_sim/structural_glyphs.py` | Self-contained registry of 54 structural variants: 18 `CLEF_ARCHETYPES`, 18 `KEYSIG_ARCHETYPES` (key signatures & accidentals), and 18 `BARLINE_ARCHETYPES`, plus `resolve_structural_archetypes`. |
| `src/hw_score_sim/rest_variants.py` | 30 standalone pause/rest archetypes (`NEW_QUARTER_REST_ARCHETYPES` 18 + `FLAGGED_REST_ARCHETYPES` 6 + `BAR_REST_ARCHETYPES` 6 = `ALL_NEW_PAUSE_ARCHETYPES`, zero overlap with legacy `REST_ARCHETYPES`) dispatched via `draw_pause_variant`. |
| `src/hw_score_sim/layout_quirks.py` | Procedural manuscript paper synthesis (`make_manuscript_paper` across 6 `PAPER_TYPES`: grain, foxing/mottling, bleed-through, eraser smudges, stains), curved rastrum staff splines (`build_curved_staff_y`), slope drift, bar-width variance, end-of-bar cramming, spacing jitter, and marginalia. |
| `src/hw_score_sim/musescore_compat.py` | MusicXML 3.1/4.0 (`.musicxml`, `.mxl`) and MuseScore 3/4 (`.mscx`, `.mscz`) parser into typed IR (`ParsedScore`, `PartData`, `MeasureData`, `NoteEvent`, `RestEvent`, etc.) + `export_score_to_musicxml`. |
| `src/hw_score_sim/renderer.py` | Page orchestrator (`layout_score_pages`, `render_score_handwritten` / `render_score`): composites paper, staves, beams/stems/slurs/glyphs onto `StrokeCanvas`, writes PNG + Schema v1.0 GT JSON, and optionally exports cropped per-system strips + MusicXML slices. |
| `src/hw_score_sim/sampler.py` | Stateless curriculum sampler: `derive_seed` (64-bit BLAKE2b), 4 `DIFFICULTY_PRESETS`, 8 `INK_PRESETS`, 6 `PAPER_PRESETS`, `apply_ink_and_paper`, and `ScribeSampler` (`.sample(score_id, variant_idx)`). |
| `src/hw_score_sim/catalog.py` | Faceted 4-axis paleographic index (`era`, `medium`, `implement`, `legibility` 1–5) exposing `filter_styles` and `describe_style` without mutating JSON packs. |
| `src/hw_score_sim/ml_export.py` | In-memory `RenderSample` & `render_sample` (accepts style names or ephemeral dicts without mutating `SCRIBE_STYLES`), `to_coco_annotations`, `to_yolo_labels`, atomic `export_dataset_sample` (`*.tmp` + `os.replace`), `HandwrittenScoreDataset` (lazy `torch`), and `to_hf_generator`. |
| `src/hw_score_sim/cli.py` | Entrypoints `hw-score-sim` (`main`) and `hw-score-gallery` (`gallery_main`). |
| `src/hw_score_sim/style_packs/*.json` | 11 curated JSON packs (214 styles, including 75 flagship `hero_*` scribes) auto-loaded on import via `reload_style_packs()`. |
| `examples/` | 4 runnable recipes (`01_time_machine_strip.py`, `02_scribe_morph.py`, `03_infinite_omr_dataset.py`, `04_custom_scribe_from_json.py`) + shared `_sample_score.py`. |
| `tests/` | 12 `unittest` modules covering styles, pluggable architecture, stroke engine, stroke uniqueness, stems/flags/beams, structural glyphs, rest variants, MuseScore compat, renderer, sampler/catalog, ML export, and example smoke tests. |

## Core Philosophy & Non-Negotiable Invariants

1. **Handwriting never repeats itself:**
   - **Zero glyph caching, zero bitmap/font stamping.** Every single glyph instance (notehead, stem, flag, beam, rest, clef, accidental, barline) MUST be procedurally simulated fresh from Bezier/Catmull-Rom/polyline nib strokes with per-instance motor jitter drawn from the passed `rng`.
   - Enforced as a hard gate by `tests/test_stroke_uniqueness.py` (`TestNoIdenticalGlyphReuse`).
2. **100% MIT / permissive stack:**
   - Runtime dependencies are strictly pure `numpy` + `pillow`.
   - Never introduce MuseScore, LilyPond, Verovio, Ghostscript, or any GPL/AGPL binaries, and never commit or bundle scanned glyph cutouts, traced proprietary fonts, or `CC-BY-NC` assets.
3. **Stateless determinism:**
   - Never use unseeded global `random` or `np.random` state; always thread caller-supplied `numpy.random.Generator` / `random.Random` instances.
   - `derive_seed(master_seed, score_id, variant_idx=0, stage="render")` in `sampler.py` uses length-prefixed 64-bit BLAKE2b (`person=b"hw_score_sim_v1"`), guaranteeing bit-identical pixels and GT JSON across multiprocessing workers (`ProcessPoolExecutor`, `DataLoader`) and dataset shuffle orders.
4. **Test & registry isolation:**
   - `SCRIBE_STYLES` is process-global. Any test or example that calls `register_style(...)` or `reload_style_packs(packs_dir=...)` MUST restore the default registry in `tearDown` / `finally` via `reload_style_packs(reset_to_builtins=True)`.
   - For ephemeral or sampled scribe dicts during rendering, prefer passing the dict directly to `render_sample(score, scribe_dict, seed=...)` (which never mutates `SCRIBE_STYLES`).

## Run & Verification Commands

```bash
# Editable install (with dev tools)
pip install -e ".[dev]"

# Full test suite (CI command, stdlib unittest — zero extra deps)
python -m unittest discover -s tests -p "test_*.py" -v

# Alternative runner + linter
pytest -q
ruff check src tests

# Runnable examples (CI smoke check uses example 04)
python examples/01_time_machine_strip.py
python examples/02_scribe_morph.py
python examples/03_infinite_omr_dataset.py
python examples/04_custom_scribe_from_json.py --out-dir /tmp/hw_score_sim_ci_smoke --page-w 800 --page-h 1000

# CLI: list styles, render scores, export per-system training strips
hw-score-sim --list-styles
hw-score-sim --input score.musicxml --styles bach_leipzig_late_quill,beethoven_late_sketchbook --out-dir out/ --seed 42
hw-score-sim --input score.mscz --styles all --out-dir out/pages --export-train-dir out/train_strips --seed 42

# CLI: HTML visual inspection galleries (modes: scores | stems-flags | structural)
hw-score-gallery --mode scores --count 24 --out-dir gallery/scores
hw-score-gallery --mode stems-flags --out-dir gallery/stems_flags
hw-score-gallery --mode structural --out-dir gallery/structural
```

## Adding Styles & Archetypes Safely

### Adding Scribe Styles (JSON-Only Preferred)
- Place entries in an existing or new `src/hw_score_sim/style_packs/*.json` file (auto-discovered on import).
- **Use `"extends"` inheritance** from any built-in or pack style rather than duplicating all 35 `REQUIRED_STYLE_KEYS`. Multi-level inheritance is resolved topologically with cycle detection; `rest_archetype_weights` merges parent + child keys and renormalizes to `1.0`.
- **Zero name collisions:** style names must be globally unique across `_BUILTIN_STYLES` and all `style_packs/*.json` files (later pack files in sorted filename order overwrite earlier keys in `reload_style_packs`).
- **Validation & clamping (`validate_style` / `_validate_and_normalize_style`):**
  - Checks all `REQUIRED_STYLE_KEYS`, 3-channel RGB tuples in `[0.0, 1.0]`, valid archetype enum strings, and strict hard-error ranges (`stroke_scale > 0`, `nib_min_ratio in (0, 1]`, `notehead_scale in [0.50, 1.45]`, `notehead_aspect in [0.25, 2.20]`, `notehead_fill_density in [0.25, 1.50]`, `stem_length_scale in [0.65, 1.55]`, `beam_thickness_scale in [0.45, 2.10]`, `beam_split_to_flag_prob in [0.0, 1.0]`).
  - Clamps all interpolatable numeric fields to `_NUMERIC_CLAMP_BOUNDS` in `styles.py` (keep any new numeric parameter registered in `_NUMERIC_CLAMP_BOUNDS` and `_OPTIONAL_NUMERIC_DEFAULTS` so `validate_style`, `interpolate_styles`, `sample_random_scribe`, and `ScribeSampler` stay in sync).

### Adding Glyph Archetypes
1. Register the unique archetype name (zero collision across registries) in the corresponding tuple:
   - `styles.py`: `NOTEHEAD_ARCHETYPES`, `STEM_ARCHETYPES`, `FLAG_ARCHETYPES`, `BEAM_ARCHETYPES`, `BEAM_SLOPE_STYLES`, `BEAM_GROUPING_MODES`, `REST_ARCHETYPES`, `LEDGER_STYLES`.
   - `structural_glyphs.py`: `CLEF_ARCHETYPES`, `KEYSIG_ARCHETYPES`, `BARLINE_ARCHETYPES`.
   - `rest_variants.py`: `NEW_QUARTER_REST_ARCHETYPES`, `FLAGGED_REST_ARCHETYPES`, `BAR_REST_ARCHETYPES` (must remain disjoint from legacy `REST_ARCHETYPES`).
2. Implement purely procedural vector geometry via `StrokeCanvas` primitives, scaled relative to staff space `sp`, honoring `style` nib/wobble/tremor params and per-instance `rng` jitter.
3. Verify with `python -m unittest discover -s tests -p "test_*.py" -v` (especially `test_stroke_uniqueness.py`, `test_styles.py`, `test_style_architecture.py`) and inspect via `hw-score-gallery`.

## CI & Release Automation

- **CI (`.github/workflows/ci.yml`):** Runs on `push` and `pull_request` to `main` across `ubuntu-latest` and `windows-latest` × Python `3.10`, `3.11`, `3.12`. Installs `-e . build twine`, runs `python -m unittest discover -s tests -p "test_*.py" -v`, builds sdist/wheel (`python -m build`), and checks metadata (`twine check dist/*`).
- **Release (`.github/workflows/release.yml`):** Triggers on `v*.*.*` tag pushes (and `workflow_dispatch`). Verifies `pyproject.toml` `project.version` matches the tag, runs the full `unittest` suite, builds + verifies `dist/*`, publishes to PyPI via `pypa/gh-action-pypi-publish@release/v1`, and creates a GitHub Release with `dist/*` attached and auto-generated release notes.
- **PyPI auth:** Configured for **OIDC Trusted Publishing** (zero static secrets; register this repo, workflow `release.yml`, and environment `pypi` as a Trusted Publisher on PyPI). Alternatively, supply a `PYPI_API_TOKEN` secret and uncomment `password: ${{ secrets.PYPI_API_TOKEN }}` in `release.yml`.
- **Cutting a release:** Bump `version` in `pyproject.toml`, commit on `main`, then run:
  ```bash
  git tag vX.Y.Z && git push origin vX.Y.Z
  ```
