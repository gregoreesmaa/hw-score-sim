"""Stateless curriculum scribe sampler and composable ink/paper presets.

Provides:
- ``derive_seed``: pure stateless 64-bit BLAKE2b seed derivation so
  multiprocessing ``DataLoader`` workers produce bit-identical pixels
  regardless of worker count or shuffle order.
- ``DIFFICULTY_PRESETS``: 4 curriculum difficulty envelopes
  (``clean_copyist``, ``everyday``, ``wild_archive``, ``extreme_stress``)
  with exact numeric clamping bounds over ``styles.py`` keys.
- ``INK_PRESETS`` (8 named inks) and ``PAPER_PRESETS`` (6 named stocks)
  plus ``apply_ink_and_paper()`` for non-mutating composable overrides.
- ``ScribeSampler``: deterministic, curriculum-aware sampler with
  ``.sample(score_id, variant_idx=0)``, ``.set_preset()``, and
  ``.set_difficulty(0.0..1.0)``.
"""

from __future__ import annotations

import hashlib
from typing import Any, Optional, Sequence

import numpy as np

from .catalog import filter_styles
from .styles import (
    PAPER_TYPES,
    _NUMERIC_CLAMP_BOUNDS,
    _copy_style_dict,
    list_styles,
    sample_random_scribe,
    validate_style,
)

# ---------------------------------------------------------------------------
# Pure stateless 64-bit BLAKE2b seed derivation
# ---------------------------------------------------------------------------

_BLAKE2B_PERSON = b"hw_score_sim_v1"  # <= 16 bytes


def derive_seed(
    master_seed: int,
    score_id: str,
    variant_idx: int = 0,
    stage: str = "render",
) -> int:
    """Derive a deterministic, stateless 64-bit unsigned integer seed via BLAKE2b.

    Independent of process count, worker index, or dataset shuffle order.
    """
    if not isinstance(score_id, str):
        raise TypeError(f"score_id must be str, got {type(score_id).__name__}")
    if not isinstance(stage, str):
        raise TypeError(f"stage must be str, got {type(stage).__name__}")

    # Length-prefixed canonical encoding prevents boundary collisions across fields.
    stage_b = stage.encode("utf-8")
    score_b = score_id.encode("utf-8")
    payload = (
        f"{int(master_seed)}|".encode("ascii")
        + f"{len(stage_b)}:".encode("ascii")
        + stage_b
        + f"|{len(score_b)}:".encode("ascii")
        + score_b
        + f"|{int(variant_idx)}".encode("ascii")
    )
    digest = hashlib.blake2b(payload, digest_size=8, person=_BLAKE2B_PERSON).digest()
    return int.from_bytes(digest, "big")


# ---------------------------------------------------------------------------
# Composable Ink & Paper Presets
# ---------------------------------------------------------------------------

INK_PRESETS: dict[str, dict[str, Any]] = {
    "iron_gall_fresh": {
        "ink_rgb": (0.14, 0.12, 0.11),
        "staff_ink_rgb": (0.26, 0.22, 0.18),
        "ink_dropout": 0.06,
        "grain_strength": 0.10,
        "ink_var": 0.15,
        "ink_pooling": 0.28,
        "ink_feather": 0.14,
        "bleed_through": 0.14,
    },
    "iron_gall_oxidized_sepia": {
        "ink_rgb": (0.36, 0.25, 0.16),
        "staff_ink_rgb": (0.46, 0.36, 0.26),
        "ink_dropout": 0.18,
        "grain_strength": 0.18,
        "ink_var": 0.26,
        "ink_pooling": 0.30,
        "ink_feather": 0.22,
        "bleed_through": 0.20,
    },
    "india_black": {
        "ink_rgb": (0.05, 0.05, 0.06),
        "staff_ink_rgb": (0.20, 0.20, 0.22),
        "ink_dropout": 0.03,
        "grain_strength": 0.05,
        "ink_var": 0.07,
        "ink_pooling": 0.16,
        "ink_feather": 0.04,
        "bleed_through": 0.05,
    },
    "paris_violet": {
        "ink_rgb": (0.31, 0.18, 0.44),
        "staff_ink_rgb": (0.44, 0.33, 0.54),
        "ink_dropout": 0.11,
        "grain_strength": 0.12,
        "ink_var": 0.18,
        "ink_pooling": 0.18,
        "ink_feather": 0.16,
        "bleed_through": 0.11,
    },
    "soviet_ballpoint_blue": {
        "ink_rgb": (0.13, 0.20, 0.55),
        "staff_ink_rgb": (0.32, 0.36, 0.36),
        "nib_min_ratio": 0.90,
        "ink_dropout": 0.16,
        "grain_strength": 0.12,
        "ink_var": 0.18,
        "ink_pooling": 0.14,
        "ink_feather": 0.05,
        "bleed_through": 0.07,
    },
    "mahler_grease_blue": {
        "ink_rgb": (0.11, 0.27, 0.62),
        "staff_ink_rgb": (0.26, 0.25, 0.24),
        "nib_min_ratio": 0.66,
        "ink_dropout": 0.14,
        "grain_strength": 0.44,
        "ink_var": 0.24,
        "ink_pooling": 0.06,
        "ink_feather": 0.08,
        "bleed_through": 0.03,
    },
    "conductor_red": {
        "ink_rgb": (0.72, 0.14, 0.13),
        "staff_ink_rgb": (0.26, 0.26, 0.28),
        "ink_dropout": 0.12,
        "grain_strength": 0.34,
        "ink_var": 0.20,
        "ink_pooling": 0.06,
        "ink_feather": 0.06,
        "bleed_through": 0.04,
    },
    "hb_graphite": {
        "ink_rgb": (0.34, 0.35, 0.37),
        "staff_ink_rgb": (0.40, 0.40, 0.41),
        "nib_min_ratio": 0.80,
        "ink_dropout": 0.20,
        "grain_strength": 0.38,
        "ink_var": 0.25,
        "ink_pooling": 0.03,
        "ink_feather": 0.02,
        "bleed_through": 0.02,
    },
}

PAPER_PRESETS: dict[str, dict[str, Any]] = {
    "aged_rag": {
        "paper_type": "aged_rag",
        "paper_rgb": (0.89, 0.84, 0.72),
        "paper_grain": 0.11,
        "paper_mottle": 0.20,
    },
    "bright_white": {
        "paper_type": "bright_white",
        "paper_rgb": (0.97, 0.97, 0.96),
        "paper_grain": 0.03,
        "paper_mottle": 0.02,
    },
    "cream_manuscript": {
        "paper_type": "cream_manuscript",
        "paper_rgb": (0.94, 0.91, 0.84),
        "paper_grain": 0.065,
        "paper_mottle": 0.09,
    },
    "spiral_notebook": {
        "paper_type": "spiral_notebook",
        "paper_rgb": (0.95, 0.95, 0.92),
        "paper_grain": 0.05,
        "paper_mottle": 0.04,
    },
    "ozalid_blue": {
        "paper_type": "ozalid_blue",
        "paper_rgb": (0.87, 0.91, 0.94),
        "paper_grain": 0.07,
        "paper_mottle": 0.10,
    },
    "soviet_green_tint": {
        "paper_type": "soviet_green_tint",
        "paper_rgb": (0.88, 0.91, 0.85),
        "paper_grain": 0.085,
        "paper_mottle": 0.12,
    },
}

# Sanity check: all PAPER_PRESETS map cleanly onto styles.PAPER_TYPES
assert set(PAPER_PRESETS.keys()) == set(PAPER_TYPES)


def apply_ink_and_paper(
    style: dict[str, Any],
    *,
    ink: Optional[str] = None,
    paper: Optional[str] = None,
) -> dict[str, Any]:
    """Return a new validated style dict with optional ink and/or paper preset overrides.

    Does not mutate the input ``style`` dictionary.
    """
    if not isinstance(style, dict):
        raise TypeError(f"style must be a dict, got {type(style).__name__}")

    out = _copy_style_dict(style)
    suffix_parts: list[str] = []

    if ink is not None:
        if ink not in INK_PRESETS:
            valid = ", ".join(INK_PRESETS.keys())
            raise KeyError(f"Unknown ink preset {ink!r}. Valid presets: {valid}")
        out.update(INK_PRESETS[ink])
        suffix_parts.append(f"ink={ink}")

    if paper is not None:
        if paper not in PAPER_PRESETS:
            valid = ", ".join(PAPER_PRESETS.keys())
            raise KeyError(f"Unknown paper preset {paper!r}. Valid presets: {valid}")
        out.update(PAPER_PRESETS[paper])
        suffix_parts.append(f"paper={paper}")

    base_name = str(out.get("name", "custom_style"))
    new_name = f"{base_name}__{'_'.join(suffix_parts)}" if suffix_parts else base_name
    return validate_style(out, name=new_name)


# ---------------------------------------------------------------------------
# Curriculum Difficulty Presets (Exact Numeric Clamping Envelopes)
# ---------------------------------------------------------------------------

DIFFICULTY_ORDER: tuple[str, ...] = (
    "clean_copyist",
    "everyday",
    "wild_archive",
    "extreme_stress",
)

DIFFICULTY_PRESETS: dict[str, dict[str, Any]] = {
    "clean_copyist": {
        "description": "Fair-copy & publisher copyist regime: tight wobble, crisp ink, neat spacing.",
        "jitter": 0.08,
        "interpolate_prob": 0.35,
        "legibility_tiers": (1, 2),
        "bounds": {
            "stroke_scale": (0.75, 1.25),
            "wobble_px": (0.0, 0.90),
            "tremor_px": (0.0, 0.20),
            "ink_dropout": (0.0, 0.10),
            "grain_strength": (0.0, 0.16),
            "ink_var": (0.0, 0.16),
            "ink_pooling": (0.0, 0.28),
            "slant_deg": (-6.0, 12.0),
            "notehead_scale": (0.85, 1.15),
            "hollow_open_prob": (0.0, 0.22),
            "barline_overshoot_sp": (0.0, 0.28),
            "spacing_jitter": (0.08, 0.18),
            "bar_width_var": (0.10, 0.18),
            "end_bar_cram": (0.0, 0.14),
            "staff_curve_px": (0.0, 3.5),
            "staff_slope_deg": (-0.6, 0.6),
            "sp_jitter_frac": (0.0, 0.028),
            "paper_grain": (0.0, 0.09),
            "paper_mottle": (0.0, 0.12),
            "bleed_through": (0.0, 0.10),
            "eraser_smudge_prob": (0.0, 0.12),
            "margin_annotations_prob": (0.0, 0.35),
            "ink_feather": (0.0, 0.18),
            "split_nib_prob": (0.0, 0.10),
            "rastrum_wobble": (0.0, 1.0),
            "stain_prob": (0.0, 0.08),
            "notehead_aspect": (0.75, 1.30),
            "notehead_fill_density": (0.75, 1.20),
            "stem_length_scale": (0.88, 1.15),
            "beam_thickness_scale": (0.75, 1.30),
        },
    },
    "everyday": {
        "description": "Working musician, student, and rehearsal regime: moderate jitter and everyday wear.",
        "jitter": 0.16,
        "interpolate_prob": 0.65,
        "legibility_tiers": (1, 2, 3),
        "bounds": {
            "stroke_scale": (0.62, 1.50),
            "wobble_px": (0.0, 1.80),
            "tremor_px": (0.0, 0.55),
            "ink_dropout": (0.0, 0.24),
            "grain_strength": (0.0, 0.42),
            "ink_var": (0.0, 0.30),
            "ink_pooling": (0.0, 0.42),
            "slant_deg": (-10.0, 15.0),
            "notehead_scale": (0.72, 1.28),
            "hollow_open_prob": (0.0, 0.50),
            "barline_overshoot_sp": (0.0, 0.55),
            "spacing_jitter": (0.08, 0.32),
            "bar_width_var": (0.10, 0.28),
            "end_bar_cram": (0.0, 0.28),
            "staff_curve_px": (0.0, 7.5),
            "staff_slope_deg": (-1.3, 1.3),
            "sp_jitter_frac": (0.0, 0.055),
            "paper_grain": (0.0, 0.15),
            "paper_mottle": (0.0, 0.22),
            "bleed_through": (0.0, 0.18),
            "eraser_smudge_prob": (0.0, 0.55),
            "margin_annotations_prob": (0.0, 0.70),
            "ink_feather": (0.0, 0.40),
            "split_nib_prob": (0.0, 0.30),
            "rastrum_wobble": (0.0, 2.5),
            "stain_prob": (0.0, 0.30),
            "notehead_aspect": (0.50, 1.65),
            "notehead_fill_density": (0.55, 1.35),
            "stem_length_scale": (0.78, 1.30),
            "beam_thickness_scale": (0.60, 1.60),
        },
    },
    "wild_archive": {
        "description": "Full historical autograph, vernacular, and archival diversity across all 234 styles.",
        "jitter": 0.24,
        "interpolate_prob": 0.85,
        "legibility_tiers": (1, 2, 3, 4, 5),
        "bounds": {
            "stroke_scale": (0.50, 1.95),
            "wobble_px": (0.0, 3.10),
            "tremor_px": (0.0, 2.20),
            "ink_dropout": (0.0, 0.42),
            "grain_strength": (0.0, 0.62),
            "ink_var": (0.0, 0.45),
            "ink_pooling": (0.0, 0.62),
            "slant_deg": (-14.0, 18.0),
            "notehead_scale": (0.55, 1.40),
            "hollow_open_prob": (0.0, 0.85),
            "barline_overshoot_sp": (0.0, 0.85),
            "spacing_jitter": (0.08, 0.42),
            "bar_width_var": (0.10, 0.34),
            "end_bar_cram": (0.0, 0.38),
            "staff_curve_px": (0.0, 13.0),
            "staff_slope_deg": (-2.1, 2.1),
            "sp_jitter_frac": (0.0, 0.085),
            "paper_grain": (0.0, 0.22),
            "paper_mottle": (0.0, 0.32),
            "bleed_through": (0.0, 0.24),
            "eraser_smudge_prob": (0.0, 0.85),
            "margin_annotations_prob": (0.0, 0.90),
            "ink_feather": (0.0, 0.78),
            "split_nib_prob": (0.0, 0.75),
            "rastrum_wobble": (0.0, 4.8),
            "stain_prob": (0.0, 0.75),
            "notehead_aspect": (0.30, 2.00),
            "notehead_fill_density": (0.30, 1.45),
            "stem_length_scale": (0.68, 1.48),
            "beam_thickness_scale": (0.50, 1.95),
        },
    },
    "extreme_stress": {
        "description": "Adversarial OMR stress-test regime: full physics bounds, heavy jitter, degraded manuscripts.",
        "jitter": 0.35,
        "interpolate_prob": 0.95,
        "legibility_tiers": (3, 4, 5),
        "bounds": {
            "stroke_scale": (0.45, 2.20),
            "wobble_px": (0.60, 4.00),
            "tremor_px": (0.15, 3.50),
            "ink_dropout": (0.08, 0.55),
            "grain_strength": (0.10, 0.75),
            "ink_var": (0.12, 0.55),
            "ink_pooling": (0.05, 0.75),
            "slant_deg": (-14.0, 18.0),
            "notehead_scale": (0.50, 1.45),
            "hollow_open_prob": (0.15, 1.00),
            "barline_overshoot_sp": (0.20, 1.00),
            "spacing_jitter": (0.18, 0.45),
            "bar_width_var": (0.16, 0.35),
            "end_bar_cram": (0.12, 0.40),
            "staff_curve_px": (2.5, 16.0),
            "staff_slope_deg": (-2.5, 2.5),
            "sp_jitter_frac": (0.02, 0.10),
            "paper_grain": (0.05, 0.28),
            "paper_mottle": (0.08, 0.38),
            "bleed_through": (0.04, 0.25),
            "eraser_smudge_prob": (0.10, 1.00),
            "margin_annotations_prob": (0.20, 1.00),
            "ink_feather": (0.10, 1.00),
            "split_nib_prob": (0.05, 1.00),
            "rastrum_wobble": (0.8, 6.0),
            "stain_prob": (0.08, 1.00),
            "notehead_aspect": (0.25, 2.20),
            "notehead_fill_density": (0.25, 1.50),
            "stem_length_scale": (0.65, 1.55),
            "beam_thickness_scale": (0.45, 2.10),
        },
    },
}


def _interpolate_envelope(difficulty: float) -> tuple[float, float, dict[str, tuple[float, float]], tuple[int, ...]]:
    """Interpolate jitter, interpolate_prob, bounds, and legibility tiers for ``difficulty`` in [0, 1]."""
    d = float(difficulty)
    if not (0.0 <= d <= 1.0):
        raise ValueError(f"difficulty must be in [0.0, 1.0], got {difficulty}")

    scaled = d * (len(DIFFICULTY_ORDER) - 1)
    idx_lo = int(np.floor(scaled))
    idx_hi = min(len(DIFFICULTY_ORDER) - 1, idx_lo + 1)
    frac = scaled - idx_lo

    p_lo = DIFFICULTY_PRESETS[DIFFICULTY_ORDER[idx_lo]]
    p_hi = DIFFICULTY_PRESETS[DIFFICULTY_ORDER[idx_hi]]

    jitter = (1.0 - frac) * float(p_lo["jitter"]) + frac * float(p_hi["jitter"])
    interp_prob = (1.0 - frac) * float(p_lo["interpolate_prob"]) + frac * float(p_hi["interpolate_prob"])

    bounds: dict[str, tuple[float, float]] = {}
    for key, (lo_a, hi_a) in p_lo["bounds"].items():
        lo_b, hi_b = p_hi["bounds"][key]
        g_lo, g_hi = _NUMERIC_CLAMP_BOUNDS[key]
        lo = float(np.clip((1.0 - frac) * lo_a + frac * lo_b, g_lo, g_hi))
        hi = float(np.clip((1.0 - frac) * hi_a + frac * hi_b, lo, g_hi))
        bounds[key] = (lo, hi)

    tiers = p_hi["legibility_tiers"] if frac >= 0.5 else p_lo["legibility_tiers"]
    return jitter, interp_prob, bounds, tiers


def _clamp_style_to_bounds(style: dict[str, Any], bounds: dict[str, tuple[float, float]]) -> dict[str, Any]:
    """Clamp all present numeric style keys into ``bounds`` and validate."""
    out = _copy_style_dict(style)
    for key, (lo, hi) in bounds.items():
        if key in out and out[key] is not None:
            v = float(out[key])
            out[key] = lo if v < lo else (hi if v > hi else v)
    return validate_style(out, name=str(out.get("name", "sampled_style")))


class ScribeSampler:
    """Stateless, curriculum-aware procedural scribe sampler for OMR training.

    Uses ``derive_seed`` (BLAKE2b 64-bit) so ``.sample(score_id, variant_idx)``
    is strictly independent of call history, worker count, or shuffle order.
    """

    def __init__(
        self,
        preset: str = "wild_archive",
        master_seed: int = 42,
        difficulty: Optional[float] = None,
        *,
        style_pool: Optional[Sequence[str]] = None,
        ink: Optional[str] = None,
        paper: Optional[str] = None,
        historical_clean: bool = False,
    ) -> None:
        if preset not in DIFFICULTY_PRESETS:
            valid = ", ".join(DIFFICULTY_ORDER)
            raise KeyError(f"Unknown difficulty preset {preset!r}. Valid presets: {valid}")
        if difficulty is not None and not (0.0 <= float(difficulty) <= 1.0):
            raise ValueError(f"difficulty must be in [0.0, 1.0], got {difficulty}")
        if ink is not None and ink not in INK_PRESETS:
            valid = ", ".join(INK_PRESETS.keys())
            raise KeyError(f"Unknown ink preset {ink!r}. Valid presets: {valid}")
        if paper is not None and paper not in PAPER_PRESETS:
            valid = ", ".join(PAPER_PRESETS.keys())
            raise KeyError(f"Unknown paper preset {paper!r}. Valid presets: {valid}")

        self.preset: str = preset
        self.master_seed: int = int(master_seed)
        self.difficulty: Optional[float] = float(difficulty) if difficulty is not None else None
        self._explicit_pool: Optional[tuple[str, ...]] = tuple(style_pool) if style_pool is not None else None
        if self._explicit_pool is not None and len(self._explicit_pool) == 0:
            raise ValueError("style_pool must be non-empty when provided")
        self.ink: Optional[str] = ink
        self.paper: Optional[str] = paper
        self.historical_clean: bool = bool(historical_clean)

    def set_preset(self, preset: str) -> ScribeSampler:
        """Switch to a named difficulty preset and clear any continuous difficulty override."""
        if preset not in DIFFICULTY_PRESETS:
            valid = ", ".join(DIFFICULTY_ORDER)
            raise KeyError(f"Unknown difficulty preset {preset!r}. Valid presets: {valid}")
        self.preset = preset
        self.difficulty = None
        return self

    def set_difficulty(self, difficulty: float) -> ScribeSampler:
        """Set a continuous curriculum difficulty in ``[0.0, 1.0]``."""
        d = float(difficulty)
        if not (0.0 <= d <= 1.0):
            raise ValueError(f"difficulty must be in [0.0, 1.0], got {difficulty}")
        self.difficulty = d
        return self

    def _active_envelope(self) -> tuple[float, float, dict[str, tuple[float, float]], tuple[int, ...]]:
        if self.difficulty is not None:
            return _interpolate_envelope(self.difficulty)
        spec = DIFFICULTY_PRESETS[self.preset]
        return (
            float(spec["jitter"]),
            float(spec["interpolate_prob"]),
            spec["bounds"],
            tuple(spec["legibility_tiers"]),
        )

    def _resolve_pool(self, legibility_tiers: tuple[int, ...]) -> list[str]:
        if self._explicit_pool is not None:
            return list(self._explicit_pool)
        pool = filter_styles(legibility=legibility_tiers)
        return pool if pool else list_styles()

    def sample(
        self,
        score_id: str,
        variant_idx: int = 0,
    ) -> tuple[dict[str, Any], int]:
        """Deterministically sample ``(resolved_style, sample_seed)`` for ``(score_id, variant_idx)``."""
        scribe_seed = derive_seed(self.master_seed, score_id, variant_idx, stage="scribe")
        render_seed = derive_seed(self.master_seed, score_id, variant_idx, stage="render")
        rng = np.random.default_rng(scribe_seed)

        jitter, interp_prob, bounds, tiers = self._active_envelope()
        pool = self._resolve_pool(tiers)

        if len(pool) >= 2 and float(rng.random()) < interp_prob:
            # Sample two styles from the curriculum pool and interpolate + jitter.
            from .styles import interpolate_styles

            idx_a = int(rng.integers(0, len(pool)))
            idx_b = int(rng.integers(0, len(pool)))
            alpha = float(rng.uniform(0.0, 1.0))
            base = interpolate_styles(pool[idx_a], pool[idx_b], alpha, rng=rng)
            scribe = sample_random_scribe(rng, base_style=base, jitter=jitter)
        else:
            idx = int(rng.integers(0, len(pool)))
            scribe = sample_random_scribe(rng, base_style=pool[idx], jitter=jitter)

        if self.ink is not None or self.paper is not None:
            scribe = apply_ink_and_paper(scribe, ink=self.ink, paper=self.paper)

        resolved = _clamp_style_to_bounds(scribe, bounds)
        # Attach a deterministic readable name tied to the stateless seed.
        resolved["name"] = f"scribe_{scribe_seed & 0xFFFFFFFF:08x}"
        if self.historical_clean:
            from .profile_resolver import sanitize_historical_clean_style

            resolved = sanitize_historical_clean_style(resolved)
        return resolved, render_seed


__all__ = [
    "derive_seed",
    "DIFFICULTY_ORDER",
    "DIFFICULTY_PRESETS",
    "INK_PRESETS",
    "PAPER_PRESETS",
    "apply_ink_and_paper",
    "ScribeSampler",
]
