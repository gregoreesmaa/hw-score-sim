# Contributing to `hw-score-sim`

Thank you for helping grow the open-source handwritten music paleography and OMR ecosystem!

Whether you are a musicologist modeling a 17th-century copyist hand, a jazz arranger capturing a classic studio marker style, or an ML engineer adding new glyph archetypes and dataset exporters, we welcome your contributions.

---

## Ground Rules

- **100% MIT & Procedural-Only:** Every contribution (code, JSON style packs, and glyph generators) is licensed under the [MIT License](LICENSE). Do **not** submit scanned glyph cutouts, traced proprietary fonts, or assets derived from `CC-BY-NC` or `GPL` datasets.
- **Deterministic by Default:** Never rely on unseeded global `random` or `np.random` calls. Always thread the caller-provided `rng` (`numpy.random.Generator` or `random.Random`) through every new function.

---

## Development Setup

```bash
git clone https://github.com/gregoreesmaa/hw-score-sim.git
cd hw-score-sim
pip install -e ".[dev]"
```

Run the full test suite and linter before opening a PR:

```bash
# Standard library unittest runner (zero extra deps)
python -m unittest discover -s tests -v

# Or pytest + ruff
pytest -q
ruff check src tests
```

---

## Contributing a JSON Style Pack (Zero Python Required!)

Adding a new scribe style takes ~30 lines of JSON — no Python edits needed.

### 1. Pick a Pack or Create a New One

All `*.json` files inside `src/hw_score_sim/style_packs/` are auto-discovered on import by `reload_style_packs()`.

- To add a style to an existing era/tradition, append an entry to the matching pack (e.g., `baroque_classical.json`, `romantic_autographs.json`, `jazz_studio_copyist.json`).
- To contribute a themed collection (e.g., 5–15 related scribes), add a new `<pack_name>.json` file in `src/hw_score_sim/style_packs/`.

### 2. Use `"extends"` Inheritance

Instead of repeating all 35 required keys, extend any built-in or pack style and override only the parameters that define your scribe:

```json
{
  "my_custom_scribe": {
    "extends": "baroque_quill_neat",
    "description": "Concise 1-sentence visual summary of the scribe's hand.",
    "era": "baroque",
    "paleography_notes": "Optional historical context: nib cut, paper stock, ink chemistry, and characteristic quirks.",
    "ink_rgb": [0.18, 0.13, 0.09],
    "nib_angle_deg": 42.0,
    "nib_min_ratio": 0.16,
    "slant_deg": 12.0,
    "notehead_archetype": "teardrop",
    "stem_archetype": "tapered_quill",
    "flag_archetype": "baroque_curl",
    "beam_archetype": "calligraphic_ribbon",
    "clef_archetype": "baroque_ornate_scroll",
    "keysig_archetype": "baroque_vertical_flat",
    "barline_archetype": "tapered_quill_bar",
    "paper_type": "aged_rag"
  }
}
```

> **Validation rules:** `styles.validate_style()` automatically checks `REQUIRED_STYLE_KEYS`, RGB bounds (`[0, 1]`), valid archetype names, positive normalized `rest_archetype_weights`, and circular-inheritance protection.

### 3. Visually Inspect Your Style

Render a quick preview gallery to verify readability and paleographic character:

```bash
hw-score-sim --input path/to/sample.musicxml --styles my_custom_scribe --out-dir tmp_preview/
hw-score-gallery --mode scores --prefix my_custom --count 4 --out-dir tmp_gallery/
```

### 4. Verify & Open a PR

Run `python -m unittest discover -s tests -v` to confirm `test_styles.py` and `test_style_architecture.py` pass with your new style, and attach a sample PNG screenshot in your PR description!

---

## Adding a New Glyph Archetype

New glyph archetypes expand the morphological diversity available to both named styles and `sample_random_scribe` / `ScribeSampler`.

1. **Register the archetype name:**
   - Noteheads, stems, flags, beams, rests, ledgers → add to the corresponding tuple in `src/hw_score_sim/styles.py` (`NOTEHEAD_ARCHETYPES`, `STEM_ARCHETYPES`, `FLAG_ARCHETYPES`, `BEAM_ARCHETYPES`, `REST_ARCHETYPES`, `LEDGER_STYLES`).
   - Clefs, key signatures/accidentals, barlines → add to `CLEF_ARCHETYPES`, `KEYSIG_ARCHETYPES`, or `BARLINE_ARCHETYPES` in `src/hw_score_sim/structural_glyphs.py`.
2. **Implement the procedural geometry:**
   - Draw exclusively using `StrokeCanvas` primitives (`draw_calligraphic_polyline`, `draw_filled_blob`) in `glyph_variants.py` or `structural_glyphs.py`.
   - Scale all coordinates relative to `sp` (staff space) so the glyph renders cleanly at any page DPI/resolution.
   - Respect `style["nib_angle_deg"]`, `style["stroke_scale"]`, `style["wobble_px"]`, and `rng`.
3. **Add tests & gallery verification:**
   - Update the relevant test module (`tests/test_stems_flags_beams.py`, `tests/test_structural_glyphs.py`, or `tests/test_renderer.py`) to verify determinism, non-empty ink pixels, and clean bounding behavior.
   - Run `hw-score-gallery --mode stems-flags` or `--mode structural` and inspect the rendered card before submitting.
