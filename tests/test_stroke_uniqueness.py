"""Hard invariant test: handwriting never repeats itself.

Verifies that every glyph instance (noteheads, stems+flags, rests, clefs,
accidentals, barlines) is procedurally simulated fresh from strokes with
per-instance motor jitter — never stamped or cached as identical bitmaps.
"""

from __future__ import annotations

import unittest

import numpy as np

from hw_score_sim import (
    FLAG_ARCHETYPES,
    NOTEHEAD_ARCHETYPES,
    STEM_ARCHETYPES,
    get_style,
)
from hw_score_sim.glyph_variants import (
    draw_accidental,
    draw_barline,
    draw_clef,
    draw_notehead,
    draw_rest,
    draw_stem_and_flag,
)
from hw_score_sim.stroke_engine import StrokeCanvas

INK = (0.12, 0.10, 0.08)


def _render_pair(draw_fn, w: int = 120, h: int = 120, style_name: str = "baroque_quill_neat"):
    style = get_style(style_name)
    rng = np.random.default_rng(2026)
    c1 = StrokeCanvas(h, w, (1.0, 1.0, 1.0))
    draw_fn(c1, style, rng)
    c2 = StrokeCanvas(h, w, (1.0, 1.0, 1.0))
    draw_fn(c2, style, rng)
    return c1.to_uint8(), c2.to_uint8()


class TestNoIdenticalGlyphReuse(unittest.TestCase):
    def test_consecutive_noteheads_never_identical(self) -> None:
        for arch in NOTEHEAD_ARCHETYPES:
            style = dict(get_style("beethoven_hurried_autograph"))
            style["notehead_archetype"] = arch
            rng = np.random.default_rng(77)
            c1 = StrokeCanvas(96, 96, (1.0, 1.0, 1.0))
            draw_notehead(c1, "filled", 48.0, 48.0, 16.0, INK, style, rng)
            c2 = StrokeCanvas(96, 96, (1.0, 1.0, 1.0))
            draw_notehead(c2, "filled", 48.0, 48.0, 16.0, INK, style, rng)
            a, b = c1.to_uint8(), c2.to_uint8()
            self.assertGreater(
                int(np.abs(a.astype(np.int16) - b.astype(np.int16)).sum()),
                0,
                f"Notehead archetype {arch!r} produced identical consecutive stamps!",
            )

    def test_consecutive_stems_and_flags_never_identical(self) -> None:
        for stem_arch, flag_arch in zip(STEM_ARCHETYPES[:8], FLAG_ARCHETYPES[:8]):
            style = dict(get_style("chopin_delicate_salon"))
            style["stem_archetype"] = stem_arch
            style["flag_archetype"] = flag_arch
            rng = np.random.default_rng(88)
            c1 = StrokeCanvas(140, 120, (1.0, 1.0, 1.0))
            draw_stem_and_flag(c1, 50.0, 100.0, 44.0, 16.0, 2, True, INK, style, rng)
            c2 = StrokeCanvas(140, 120, (1.0, 1.0, 1.0))
            draw_stem_and_flag(c2, 50.0, 100.0, 44.0, 16.0, 2, True, INK, style, rng)
            a, b = c1.to_uint8(), c2.to_uint8()
            self.assertGreater(
                int(np.abs(a.astype(np.int16) - b.astype(np.int16)).sum()),
                0,
                f"Stem/flag ({stem_arch}, {flag_arch}) produced identical consecutive stamps!",
            )

    def test_consecutive_rests_clefs_accidentals_barlines_never_identical(self) -> None:
        # Quarter rests
        a, b = _render_pair(lambda c, st, r: draw_rest(c, "quarter", 60.0, 60.0, 16.0, INK, st, r))
        self.assertGreater(int(np.abs(a.astype(np.int16) - b.astype(np.int16)).sum()), 0)

        # Clefs (G, F, C)
        for ck in ("G", "F", "C"):
            a, b = _render_pair(lambda c, st, r, _ck=ck: draw_clef(c, _ck, 60.0, 60.0, 16.0, INK, st, r))
            self.assertGreater(int(np.abs(a.astype(np.int16) - b.astype(np.int16)).sum()), 0)

        # Accidentals
        for ak in ("sharp", "flat", "natural"):
            a, b = _render_pair(lambda c, st, r, _ak=ak: draw_accidental(c, _ak, 60.0, 60.0, 16.0, INK, st, r))
            self.assertGreater(int(np.abs(a.astype(np.int16) - b.astype(np.int16)).sum()), 0)

        # Barlines
        a, b = _render_pair(lambda c, st, r: draw_barline(c, 60.0, 20.0, 100.0, 16.0, "single", INK, st, r))
        self.assertGreater(int(np.abs(a.astype(np.int16) - b.astype(np.int16)).sum()), 0)


if __name__ == "__main__":
    unittest.main()
