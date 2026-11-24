# `hw-score-sim`

**Turn any MusicXML or MuseScore file into authentic, pixel-accurate handwritten sheet music — 234 historical & modern scribe styles (including 75 flagship Hero Scribes), 164 discrete glyph archetypes + 18 beam leveling/grouping modes, infinite interpolated hands, zero MuseScore/LilyPond binaries, 100% MIT-licensed.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-brightgreen.svg)](pyproject.toml)
[![Zero External Binaries](https://img.shields.io/badge/binaries-none%20%28pure%20NumPy%20%2B%20Pillow%29-orange.svg)](ARCHITECTURE.md)

---

## Why `hw-score-sim`?

Optical Music Recognition (OMR) has a dirty secret: **printed sheet music is nearly solved, while real human manuscripts remain brutally hard.** Yet anyone trying to train an open-weights or commercial handwritten OMR model immediately hits two walls:

1. **The `CC-BY-NC` Wall.** Existing handwritten OMR corpora and asset-based synthesizers rely on scanned glyph libraries distributed under non-commercial (`CC-BY-NC`) licenses. If you want to ship an open-source, permissively licensed, or commercial OMR model, you cannot legally train on them.
2. **The "Handwritten Font" Illusion.** Stamping a static jazz or "handwritten" TrueType font onto straight computer-ruled staves fools nobody — least of all a neural network. Real composers and copyists never draw two identical noteheads or stems. A 1735 Leipzig goose quill cuts razor-thin upstrokes and pools oxidized iron-gall ink on downstrokes; a hurried Beethoven pocket sketchbook slashes diagonal noteheads across warping rastrum staves and crams the last two beats into the barline; a 1970s Real Book felt-tip marker bleeds into warm cream stock with bold chisel beams.

**`hw-score-sim` replaces static glyph stamps with procedural paleography.** Every stroke is synthesized from scratch in pure Python (`numpy` + `pillow`) using physical nib geometry, pressure taper, micro-tremor, dry-brush dropout, paper-tooth fiber absorption, curved rastrum staves, and human rhythmic spacing quirks — paired automatically with **exact pixel-accurate ground truth** (Bezier staff curves, symbol bounding boxes, COCO, YOLO, and MusicXML).

---

## 30-Second Quickstart

### Install

```bash
pip install hw-score-sim
```

Or from source for local development:

```bash
git clone https://github.com/gregoreesmaa/hw-score-sim.git
cd hw-score-sim
pip install -e ".[dev]"
```

### Command Line (`hw-score-sim` & `hw-score-gallery`)

Render any `.musicxml`, `.mxl`, `.mscx`, or `.mscz` score across one or more scribe styles — no MuseScore or LilyPond installation required:

```bash
# Render a score in three iconic styles (PNG + companion JSON ground truth)
hw-score-sim \
  --input my_score.musicxml \
  --styles bach_leipzig_late_quill,beethoven_late_sketchbook,real_book_1970s_felt_tip \
  --out-dir out/

# Browse all 234 registered scribe styles
hw-score-sim --list-styles

# Export cropped per-system PNG strips + per-system MusicXML training targets
hw-score-sim \
  --input my_score.mscz \
  --styles all \
  --out-dir out/pages \
  --export-train-dir out/train_strips \
  --seed 42

# Generate an interactive HTML eyeball gallery of styles or glyph archetypes
hw-score-gallery --mode scores --count 24 --out-dir gallery/scores
hw-score-gallery --mode stems-flags --out-dir gallery/stems_flags
hw-score-gallery --mode structural --out-dir gallery/structural
```

### Python API (Single Render + ML Dataset Factory)

```python
from pathlib import Path
import hw_score_sim as hss

# 1. One-liner: render a score and get image + Schema v1.0 ground truth
sample = hss.render_sample(
    "my_score.musicxml",
    "beethoven_late_sketchbook",
    seed=42,
)
sample.image.save("beethoven_page.png")

# 2. Export directly to COCO or YOLO detection formats for OMR training
coco_dict = hss.to_coco_annotations([sample])
yolo_lines = hss.to_yolo_labels(sample)

# 3. Filter styles by era and sample infinite interpolated unseen hands
romantic_styles = hss.filter_styles(era="romantic")
sampler = hss.ScribeSampler(
    preset="wild_archive",
    style_pool=romantic_styles,
    master_seed=2026,
)

Path("out").mkdir(exist_ok=True)
for idx in range(5):
    scribe, sample_seed = sampler.sample("my_score", variant_idx=idx)
    unseen = hss.render_sample("my_score.musicxml", scribe, seed=sample_seed)
    unseen.image.save(f"out/unseen_{idx:03d}.png")
```

---

## Honest Comparison

`hw-score-sim` is built specifically for permissive, reproducible handwritten OMR synthesis. Here is how it compares honestly against the two most common alternatives:

| Dimension | `hw-score-sim` | Smashcima | MuseScore / LilyPond |
|---|---|---|---|
| **License** | **100% MIT** (code + procedural styles) | Engine MIT, **scanned glyph assets `CC-BY-NC`** | **GPL** binaries |
| **Commercial / Open-Weights ML Training** | Unrestricted | Blocked by `CC-BY-NC` assets | Requires isolating GPL pipeline; engraved look only |
| **Visual Authenticity** | Procedural nib physics, 234 styles, 164 glyph archetypes + 18 beam modes, 6 paper stocks | Real scanned handwritten cutouts (e.g., MUSCIMA++ / Mashcima) | Publication-grade typeset engraving (uniform fonts) |
| **Style Diversity** | 234 named styles + **infinite continuous interpolation** | Bound to scanned writer pools | Fixed engraving fonts |
| **Runtime Dependencies** | Pure Python (`numpy`, `pillow`) — **zero external binaries** | Python + external asset downloads | Headless Qt/GUI or LilyPond binary installs |
| **Ground Truth** | Exact Bezier staves, glyph bboxes, COCO, YOLO, MusicXML | Pixel masks & MuNG graphs | External alignment or SVG scraping required |
| **Best Use Case** | Large-scale permissive handwritten OMR pretraining & domain randomization | Academic-only benchmarks matching MUSCIMA++ writers | Typeset sheet music publishing |

---

## The Scribe Universe

### 234 Scribe Styles Across 11 Curated Packs (+ 20 Built-In Core Profiles)

Every style is a rich paleographic profile specifying 35+ physical parameters — nib angle and contrast ratio, stroke scale, bow wobble, neuromuscular tremor, ink pooling, dry-brush dropout, slant, notehead/stem/flag/beam/clef/key-signature/barline/rest archetypes, rastrum curvature, bar-width variance, end-of-bar cramming, paper stock, foxing mottle, reverse-side bleed-through, and marginalia.

| Style Pack (`src/hw_score_sim/style_packs/`) | Count | Historical & Visual Focus |
|---|---|---|
| *(Built-in core profiles in `styles.py`)* | 20 | Foundational Baroque, Classical, Romantic, Jazz, Student, Left-Handed, Tremor, Hectograph, Red-Pencil |
| `hero_scribes.json` | 75 | **Flagship Hero Scribes**: 75 signature paleographic profiles from Hildegard & Palestrina through Baroque, Classical, Romantic, Impressionist, Avant-Garde, Broadway, Jazz, Film/Game/Anime, and Global Folk/Session |
| `baroque_classical.json` | 6 | Bach Leipzig quill, Handel London opera rush, Vivaldi furia, Telemann, Haydn, Mozart Vienna |
| `romantic_autographs.json` | 6 | Beethoven late sketchbook, Schubert Lieder, Schumann diary, Chopin Stichvorlage, Brahms, Liszt |
| `modernist_20c.json` | 6 | Debussy violet ink, Ravel microscopic precision, Stravinsky Stravigor nib, Bartók field notebook, etc. |
| `jazz_studio_copyist.json` | 5 | 1970s Real Book felt-tip, Hollywood Ozalid vellum, Broadway pit copyist chisel, Big Band pencil |
| `everyday_student_folk.json` | 6 | Conservatory exam pencil, amateur gel pen, pub fiddler pocket notebook, church organist hymnal |
| `wild_historical_renaissance_baroque.json` | 22 | Extreme historical hands: Scarlatti needle harpsichord, Couperin pinprick, Frescobaldi lozenge choirbook, Monteverdi |
| `wild_romantic_virtuoso_autographs.json` | 22 | Extreme Romantic autographs: Beethoven Op. 131 blots, Chopin Nohant pinprick, Liszt Weimar red revisions |
| `wild_20c_avantgarde_film_studio.json` | 22 | Stravinsky needle vellum, Prokofiev telegraphic slash, Shostakovich Soviet ballpoint, avant-garde & film scoring |
| `wild_jazz_pop_world_vernacular.json` | 22 | Bebop napkin ballpoint, Ellington/Strayhorn arranger slash, Blue Note hard-bop pencil, vernacular & studio |
| `wild_extreme_degraded_everyday.json` | 22 | 0.25 mm mechanical pencil, dry Bic ballpoint, lefty blue gel smear, arthritis tremor, extreme degradations |
| **Total Registered Styles** | **234** | **Plus infinite continuous interpolation via `interpolate_styles()` & `ScribeSampler`** |

### 164 Discrete Glyph Archetypes + 18 Beam Leveling/Grouping Modes + 5 Ledger Styles + 6 Paper Stocks

| Category | Count | Archetypes |
|---|---|---|
| **Noteheads** | **18** | `oval`, `slash`, `teardrop`, `compact`, `heavy_wedge`, `hollow_diamond`, **ultra-slim:** `slim_oval`, `needle_slash`, `hairline_dash`, `vertical_tick`, `pinprick_dot`, `slim_diamond`, plus `boomerang_hook`, `open_loop`, `zigzag_fill`, `broad_nib_lozenge`, `ink_blot`, `chevron_wedge` |
| **Stems** | **18** | `straight_classic`, `hairline_needle`, `heavy_thick`, `tapered_quill`, `bowed_right`, `bowed_left`, `s_curve_flex`, `short_stubby`, `tall_spindly`, `steep_forward_slant`, `back_slant_lefty`, `detached_gap`, `center_piercing`, `overshoot_notehead`, `tremor_shaky`, `double_stroke_sketch`, `flared_base`, `whip_taper` |
| **Flags** | **20** | `classic_droop`, `baroque_curl`, `sharp_angular_7`, `hurried_slash`, `tiny_flick`, `long_dramatic_s`, `closed_loop_flourish`, `broad_nib_pennant`, `detached_tick`, `wide_horizontal_wing`, `steep_vertical_drop`, `zigzag_frantic`, `hooked_j_tail`, `calligraphic_ribbon`, `stubby_wedge`, `double_back_loop`, `feathered_sweep`, `flat_bar_flag`, `reverse_kick`, `minimal_dot_tail` |
| **Beams** | **14** | `straight_classic`, `thick_marker`, `hairline_thin`, `arched_convex`, `sagged_concave`, `tapered_wedge`, `overhanging_ends`, `short_clipped`, `scribble_double_pass`, `calligraphic_ribbon`, `stepped_wobbly`, `gapped_broken`, `wide_spaced_bars`, `tight_crammed_bars` |
| **Beam Leveling & Grouping** | **18** | **10 slope styles** (`horizontal_flat`, `engraver_gentle`, `steep_contour`, `exaggerated_swoop`, `reverse_quirk`, `concave_valley`, `convex_arch`, `stepped_secondary_tilt`, `high_floating_beam`, `tight_hug_beam`) + **8 grouping modes** (`standard_beat`, `all_flags_vocal`, `mixed_split_flags`, `long_phrase_beam`, `pair_beams_2`, `uneven_3_plus_1`, `feathered_splay`, `french_broken_secondary`) |
| **Clefs** | **18** | `classic_engraved`, `baroque_ornate_scroll`, `hurried_single_loop`, `tall_spindly_g`, `stubby_compact`, `minimal_s_slash`, `figure_8_treble`, `beethoven_wild_slash`, `chopin_delicate_spiral`, `jazz_realbook_bold`, `sharp_angular_modern`, `detached_dot_bass`, `open_c_hook_bass`, `double_bar_c_clef`, `bracket_c_clef`, `calligraphic_broad_nib`, `tremor_shaky_clef`, `backslant_lefty_clef` |
| **Key Sigs & Accidentals** | **18** | `classic_neat`, `tall_narrow_sharp`, `wide_slanted_hash`, `hurried_x_sharp`, `open_loop_flat`, `pointed_teardrop_flat`, `tiny_pinprick_acc`, `oversized_bold_acc`, `crammed_tight_cluster`, `wide_spread_keysig`, `ascending_drift_keysig`, `descending_drift_keysig`, `double_stroke_sketch`, `calligraphic_nib_acc`, `baroque_vertical_flat`, `jazz_heavy_marker_acc`, `hairline_needle_acc`, `jagged_zigzag_acc` |
| **Barlines** | **18** | `crisp_vertical`, `tall_overshoot`, `short_clipped_inside`, `steep_forward_slant`, `back_slant_lefty`, `bowed_curved_bar`, `s_wobble_freehand`, `heavy_thick_marker`, `hairline_faint`, `tapered_quill_bar`, `double_pass_sketch`, `gapped_broken_bar`, `hooked_top_bottom_serif`, `wide_double_gap`, `tight_double_gap`, `oversized_repeat_blobs`, `tiny_pinprick_repeat_dots`, `slashed_repeat_ticks` |
| **Rests / Pauses** | **40** | **10 legacy quarter rests** (`lightning_z`, `backwards_7_hook`, `cursive_3`, `hurried_slash_zig`, `baroque_c_hook`, `calligraphic_s_curl`, `sharp_v_tick`, `baroque_flag_loop`, `modern_tall_z`, `scribble_s`) + **18 new quarter rests** (`baroque_reversed_r`, `beethoven_violent_zig`, `chopin_gossamer_3`, `schubert_vertical_drop`, `brahms_bold_chevron`, `liszt_sweeping_s_hook`, `debussy_floating_tilde`, `stravinsky_geometric_z`, `shostakovich_looped_2`, `realbook_chisel_bolt`, `bebop_napkin_tick`, `studio_copyist_clean_3`, `student_awkward_e`, `arthritis_tremor_zig`, `lefty_backslant_hook`, `hairline_needle_s`, `open_double_loop_r`, `pinprick_minimal_v`) + **6 flagged rests** + **6 bar/fermata rests** |
| **Ledger Lines** | **5** | `crisp`, `wide_sweep`, `slanted`, `short_tick`, `wavy` |
| **Paper Stocks** | **6** | `aged_rag`, `bright_white`, `cream_manuscript`, `spiral_notebook`, `ozalid_blue`, `soviet_green_tint` |

### 75 Flagship Hero Scribes (`hero_scribes.json`)

Expanded in `v0.2.0`, `hero_scribes.json` provides **75 signature paleographic profiles** spanning nine centuries of music manuscripts — from Hildegard of Bingen's parchment neumes and Sistine Chapel choirbooks to Hollywood Ozalid film scores, Tokyo anime/game studio charts, and Abbey Road string sessions:

| Era / Tradition | Count | Flagship Hero Style IDs |
|---|---|---|
| **Medieval & Renaissance** | 6 | `hero_hildegard_bingen_parchment` · `hero_josquin_cambrai_choirbook` · `hero_palestrina_sistine_choirbook` · `hero_byrd_elizabethan_partbook` · `hero_victoria_avila_cathedral` · `hero_gesualdo_ferrara_madrigal` |
| **Baroque Court & Virtuosi** | 9 | `hero_monteverdi_vespers_quill` · `hero_barbara_strozzi_venice_cantata` · `hero_buxtehude_lubeck_organ` · `hero_purcell_westminster_autograph` · `hero_couperin_versailles_clavecin` · `hero_vivaldi_turin_concerto_rush` · `hero_rameau_paris_opera_score` · `hero_scarlatti_madrid_sonata` · `hero_anna_magdalena_bach_notebook` |
| **Classical & Galant** | 5 | `hero_cpe_bach_berlin_empfindsam` · `hero_gluck_vienna_reform_opera` · `hero_haydn_esterhazy_autograph` · `hero_boccherini_madrid_quintet` · `hero_chevalier_saint_georges_paris` |
| **Romantic Fair Copies, Opera & Virtuosi** | 10 | `hero_paganini_genoa_caprice` · `hero_rossini_naples_opera_rush` · `hero_berlioz_paris_grand_score` · `hero_fanny_hensel_berlin_salon` · `hero_mendelssohn_leipzig_fair_copy` · `hero_clara_schumann_concert_part` · `hero_tchaikovsky_klin_symphony` · `hero_dvorak_prague_slavonic` · `hero_grieg_troldhaugen_lyric` · `hero_elgar_malvern_fountain_pen` |
| **Late-Romantic, Impressionist, Iberian, Nordic & Slavic** | 10 | `hero_mussorgsky_pictures_iron_gall` · `hero_janacek_brno_speech_melody` · `hero_mahler_vienna_blue_crayon` · `hero_albeniz_iberia_paris_ink` · `hero_sibelius_ainola_pencil` · `hero_scriabin_ecstatic_pencil` · `hero_rachmaninoff_ivanovka_ink` · `hero_ravel_watchmaker_micro_nib` · `hero_prokofiev_telegraphic_autograph` · `hero_lili_boulanger_prix_de_rome` |
| **Second Viennese, Modernist & Avant-Garde** | 8 | `hero_schoenberg_vienna_dodecaphonic` · `hero_webern_modling_micro_ink` · `hero_berg_alban_fine_pencil` · `hero_stravinsky_morges_fine_pen` · `hero_messiaen_birdsong_cahier` · `hero_boulez_paris_geometric_pen` · `hero_xenakis_architectural_draft` · `hero_crumb_makrokosmos_fineliner` |
| **American Concert & Broadway** | 5 | `hero_ives_danbury_marginalia` · `hero_gershwin_tin_pan_alley` · `hero_copland_brooklyn_pencil` · `hero_bernstein_dakota_conductor` · `hero_sondheim_broadway_legal_pad` |
| **Jazz, Big Band & Tango** | 9 | `hero_ellington_strayhorn_score` · `hero_gil_evans_orchestral_pencil` · `hero_monk_angular_lead_sheet` · `hero_piazzolla_buenos_aires_tango` · `hero_mingus_workshop_ink` · `hero_miles_davis_modal_sketch` · `hero_coltrane_giant_steps_ink` · `hero_bill_evans_jazz_pencil` · `hero_bluenote_van_gelder_ozalid` |
| **Film, Game & Anime** | 7 | `hero_korngold_warner_ozalid` · `hero_herrmann_hitchcock_ink` · `hero_morricone_cinecitta_ink` · `hero_john_williams_hollywood_pencil` · `hero_hisaishi_tokyo_studio_pencil` · `hero_sakamoto_tokyo_fineliner` · `hero_uematsu_jrpg_notebook` |
| **Global Folk, Studio Session & Contemporary** | 6 | `hero_flamenco_jerez_guitar_sketch` · `hero_klezmer_odessa_fiddle_book` · `hero_irish_clare_session_pad` · `hero_choro_rio_notebook_pen` · `hero_abbey_road_string_session` · `hero_songwriter_spiral_gelpen` |

### 8 Iconic Core Styles

| Style ID | Era & Medium | Signature Paleographic Traits |
|---|---|---|
| `bach_leipzig_late_quill` | 1730s Baroque · Broad goose quill & iron-gall | 43° nib contrast, 15.5° forward impetus, teardrop heads, Baroque hooked C-rests, dense contrapuntal cramming |
| `beethoven_late_sketchbook` | 1825 Late Classical · Blunted quill & graphite | Diagonal `slash` noteheads, heavy ink pooling, wild barline overshoot, 14.5 px staff warp, dense marginalia |
| `chopin_delicate_salon` | 1840s Romantic · Fine pointed steel nib | Feather-light hairlines, delicate spiral clefs, graceful slant, airy salon phrasing on cream manuscript stock |
| `debussy_pencil_sketch` | 1900s Impressionist · Soft graphite on textured rag | Dry paper-tooth grain, soft variable pressure, subtle eraser smudges, gentle organic drift |
| `real_book_1970s_felt_tip` | 1970s Jazz · Broad chisel felt-tip marker | Thick marker beams, bold Real Book clefs, wide slanted accidentals, warm cream lead-sheet stock |
| `hollywood_ozalid_vellum` | 1950s Film Studio · Ink on translucent Ozalid copy | Cool `ozalid_blue` tint, crisp studio-copyist verticality, disciplined spacing built for sight-reading |
| `student_ballpoint_blue` | Modern Everyday · Cheap blue ballpoint on spiral pad | Narrow uniform ballpoint width, occasional ink skip/dropout, uneven bar widths, `spiral_notebook` paper |
| `wild_ext_arthritis_tremor` | Stress-Test Extreme · Shaky nib with high tremor | High-frequency neuromuscular tremor, wavering stems & barlines, irregular spacing — OMR robustness gold |

---

## Community Style Packs: 30-Line JSON-Only PR Workflow

You do **not** need to write a single line of Python to contribute a new historical composer hand, regional copyist tradition, or modern studio style. Style packs in `src/hw_score_sim/style_packs/*.json` are auto-discovered on import and support multi-level `"extends"` inheritance:

```json
{
  "my_vienna_coffeehouse_sketch": {
    "extends": "schubert_rapid_lieder_draft",
    "description": "Hasty Viennese coffeehouse pencil sketch on folded cream stock.",
    "era": "early_romantic",
    "ink_rgb": [0.28, 0.27, 0.25],
    "nib_min_ratio": 0.65,
    "grain_strength": 0.34,
    "notehead_archetype": "needle_slash",
    "stem_archetype": "bowed_right",
    "flag_archetype": "hurried_slash",
    "beam_archetype": "arched_convex",
    "paper_type": "cream_manuscript",
    "eraser_smudge_prob": 0.35
  }
}
```

Drop your `.json` file into `src/hw_score_sim/style_packs/`, preview it with `hw-score-gallery`, run `python -m unittest discover -s tests`, and open a PR! See [CONTRIBUTING.md](CONTRIBUTING.md) for the full guide.

---

## Runnable Examples

Explore the `examples/` directory for end-to-end recipes:

- [`examples/01_time_machine_strip.py`](examples/01_time_machine_strip.py) — Render the same musical passage across centuries (Baroque quill → Beethoven sketchbook → 1970s Real Book → modern ballpoint) into a side-by-side comparison strip.
- [`examples/02_scribe_morph.py`](examples/02_scribe_morph.py) — Morph smoothly between two historical scribes (`interpolate_styles`) to visualize continuous handwriting space.
- [`examples/03_infinite_omr_dataset.py`](examples/03_infinite_omr_dataset.py) — Build a deterministic, domain-randomized OMR training set with `ScribeSampler`, COCO JSON, and YOLO labels.
- [`examples/04_custom_scribe_from_json.py`](examples/04_custom_scribe_from_json.py) — Define a custom scribe in a 30-line JSON snippet using `"extends"`, register it at runtime, and render a full page.

---

## Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md) — End-to-end rendering pipeline, module map, determinism contract, and Ground-Truth Schema v1.0.
- [CONTRIBUTING.md](CONTRIBUTING.md) — How to author JSON style packs, add new glyph archetypes, and run the test suite.
- [CHANGELOG.md](CHANGELOG.md) — Release history.

---

## License & Citation

`hw-score-sim` is released under the **[MIT License](LICENSE)** — free for academic, open-source, and commercial use.

If you use `hw-score-sim` in your research or OMR training pipeline, please cite:

```bibtex
@software{eesmaa_hw_score_sim_2026,
  title        = {hw-score-sim: Procedural Handwritten Sheet-Music Rendering Engine},
  author       = {Gregor Eesmaa and hw-score-sim contributors},
  year         = {2026},
  version      = {0.2.0},
  license      = {MIT},
  url          = {https://github.com/gregoreesmaa/hw-score-sim}
}
```
