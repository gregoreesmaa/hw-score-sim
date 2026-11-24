"""Tests for hw_score_sim.sampler and hw_score_sim.catalog."""

from __future__ import annotations

import copy
import unittest

from hw_score_sim.catalog import (
    CANONICAL_ERAS,
    CANONICAL_IMPLEMENTS,
    CANONICAL_MEDIA,
    LEGIBILITY_LABELS,
    compute_legibility_score,
    describe_style,
    filter_styles,
    score_legibility,
)
from hw_score_sim.sampler import (
    DIFFICULTY_ORDER,
    DIFFICULTY_PRESETS,
    INK_PRESETS,
    PAPER_PRESETS,
    ScribeSampler,
    apply_ink_and_paper,
    derive_seed,
)
from hw_score_sim.styles import (
    NOTEHEAD_ARCHETYPES,
    PAPER_TYPES,
    SCRIBE_STYLES,
    _NUMERIC_CLAMP_BOUNDS,
    get_style,
    list_styles,
    validate_style,
)


class TestDeriveSeed(unittest.TestCase):
    def test_stateless_determinism_and_64bit_range(self) -> None:
        s1 = derive_seed(42, "bach_bwv_846", variant_idx=0, stage="render")
        s2 = derive_seed(42, "bach_bwv_846", variant_idx=0, stage="render")
        self.assertEqual(s1, s2)
        self.assertIsInstance(s1, int)
        self.assertGreaterEqual(s1, 0)
        self.assertLess(s1, 1 << 64)

    def test_distinct_across_all_axes(self) -> None:
        base = derive_seed(42, "score_001", variant_idx=0, stage="render")
        self.assertNotEqual(base, derive_seed(43, "score_001", variant_idx=0, stage="render"))
        self.assertNotEqual(base, derive_seed(42, "score_002", variant_idx=0, stage="render"))
        self.assertNotEqual(base, derive_seed(42, "score_001", variant_idx=1, stage="render"))
        self.assertNotEqual(base, derive_seed(42, "score_001", variant_idx=0, stage="scribe"))

    def test_no_delimiter_collision(self) -> None:
        a = derive_seed(7, "a|b", variant_idx=0, stage="c")
        b = derive_seed(7, "a", variant_idx=0, stage="b|c")
        self.assertNotEqual(a, b)

    def test_type_validation(self) -> None:
        with self.assertRaises(TypeError):
            derive_seed(42, 123, 0)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            derive_seed(42, "ok", 0, stage=99)  # type: ignore[arg-type]


class TestInkAndPaperPresets(unittest.TestCase):
    EXPECTED_INKS = (
        "iron_gall_fresh",
        "iron_gall_oxidized_sepia",
        "india_black",
        "paris_violet",
        "soviet_ballpoint_blue",
        "mahler_grease_blue",
        "conductor_red",
        "hb_graphite",
    )

    def test_ink_and_paper_preset_names(self) -> None:
        self.assertEqual(set(INK_PRESETS.keys()), set(self.EXPECTED_INKS))
        self.assertEqual(len(INK_PRESETS), 8)
        self.assertEqual(set(PAPER_PRESETS.keys()), set(PAPER_TYPES))
        self.assertEqual(len(PAPER_PRESETS), 6)

        for name, spec in PAPER_PRESETS.items():
            self.assertEqual(spec["paper_type"], name)
            for key in ("paper_rgb", "paper_grain", "paper_mottle"):
                self.assertIn(key, spec)

    def test_apply_ink_and_paper_non_mutating_and_valid(self) -> None:
        base = get_style("baroque_quill_neat")
        snapshot = copy.deepcopy(base)

        for ink_name in self.EXPECTED_INKS:
            out = apply_ink_and_paper(base, ink=ink_name)
            self.assertEqual(out["ink_rgb"], INK_PRESETS[ink_name]["ink_rgb"])
            validate_style(out)

        for paper_name in PAPER_TYPES:
            out = apply_ink_and_paper(base, paper=paper_name)
            self.assertEqual(out["paper_type"], paper_name)
            self.assertEqual(out["paper_rgb"], PAPER_PRESETS[paper_name]["paper_rgb"])
            validate_style(out)

        combined = apply_ink_and_paper(base, ink="conductor_red", paper="ozalid_blue")
        self.assertEqual(combined["ink_rgb"], INK_PRESETS["conductor_red"]["ink_rgb"])
        self.assertEqual(combined["paper_type"], "ozalid_blue")
        validate_style(combined)

        # Input dict must remain completely untouched.
        self.assertEqual(base, snapshot)

    def test_apply_ink_and_paper_errors(self) -> None:
        base = get_style("baroque_quill_neat")
        with self.assertRaises(KeyError):
            apply_ink_and_paper(base, ink="nonexistent_ink")
        with self.assertRaises(KeyError):
            apply_ink_and_paper(base, paper="nonexistent_paper")


class TestDifficultyPresetsAndSampler(unittest.TestCase):
    def test_all_four_difficulty_presets_within_global_bounds(self) -> None:
        self.assertEqual(
            DIFFICULTY_ORDER,
            ("clean_copyist", "everyday", "wild_archive", "extreme_stress"),
        )
        self.assertEqual(set(DIFFICULTY_PRESETS.keys()), set(DIFFICULTY_ORDER))

        for preset_name, spec in DIFFICULTY_PRESETS.items():
            bounds = spec["bounds"]
            self.assertGreater(len(bounds), 15)
            for key, (lo, hi) in bounds.items():
                self.assertIn(key, _NUMERIC_CLAMP_BOUNDS, f"{preset_name}.{key} not in global bounds")
                g_lo, g_hi = _NUMERIC_CLAMP_BOUNDS[key]
                self.assertLessEqual(lo, hi, f"{preset_name}.{key} lo > hi")
                self.assertGreaterEqual(lo, g_lo, f"{preset_name}.{key} lo below global bound")
                self.assertLessEqual(hi, g_hi, f"{preset_name}.{key} hi above global bound")

        # Upper bounds widen monotonically across presets for key chaos knobs.
        for key in ("wobble_px", "tremor_px", "ink_dropout", "spacing_jitter", "staff_curve_px"):
            his = [DIFFICULTY_PRESETS[p]["bounds"][key][1] for p in DIFFICULTY_ORDER]
            self.assertEqual(his, sorted(his), f"Non-monotonic upper bounds for {key}: {his}")

    def test_sampler_stateless_order_independence(self) -> None:
        s_a = ScribeSampler(preset="wild_archive", master_seed=1234)
        s_b = ScribeSampler(preset="wild_archive", master_seed=1234)

        # Sample in opposite orders to simulate shuffled DataLoader workers.
        out_a1, seed_a1 = s_a.sample("piece_alpha", variant_idx=0)
        out_a2, seed_a2 = s_a.sample("piece_beta", variant_idx=3)

        out_b2, seed_b2 = s_b.sample("piece_beta", variant_idx=3)
        out_b1, seed_b1 = s_b.sample("piece_alpha", variant_idx=0)

        self.assertEqual(seed_a1, seed_b1)
        self.assertEqual(seed_a2, seed_b2)
        self.assertEqual(out_a1, out_b1)
        self.assertEqual(out_a2, out_b2)

        # Mutating returned dict does not pollute future samples.
        out_a1["wobble_px"] = 999.0
        out_a1_again, seed_a1_again = s_a.sample("piece_alpha", variant_idx=0)
        self.assertEqual(seed_a1_again, seed_b1)
        self.assertEqual(out_a1_again, out_b1)

    def test_sampler_enforces_preset_envelopes(self) -> None:
        sampler = ScribeSampler(master_seed=99)
        for preset in DIFFICULTY_ORDER:
            sampler.set_preset(preset)
            bounds = DIFFICULTY_PRESETS[preset]["bounds"]
            for idx in range(10):
                style, seed = sampler.sample(f"score_{idx}", variant_idx=idx)
                validate_style(style)
                self.assertIsInstance(seed, int)
                for key, (lo, hi) in bounds.items():
                    if key in style and style[key] is not None:
                        self.assertGreaterEqual(
                            style[key], lo - 1e-9, f"{preset} {key}={style[key]} < {lo}"
                        )
                        self.assertLessEqual(
                            style[key], hi + 1e-9, f"{preset} {key}={style[key]} > {hi}"
                        )

    def test_sampler_continuous_difficulty_curriculum(self) -> None:
        sampler = ScribeSampler(master_seed=2026)

        sampler.set_difficulty(0.0)
        easy_wobbles = [sampler.sample(f"s_{i}")[0]["wobble_px"] for i in range(25)]

        sampler.set_difficulty(1.0)
        hard_wobbles = [sampler.sample(f"s_{i}")[0]["wobble_px"] for i in range(25)]

        self.assertLess(sum(easy_wobbles) / len(easy_wobbles), sum(hard_wobbles) / len(hard_wobbles))

        # Midpoint difficulty validates cleanly.
        sampler.set_difficulty(0.5)
        mid_style, _ = sampler.sample("mid_score", variant_idx=2)
        validate_style(mid_style)

        with self.assertRaises(ValueError):
            sampler.set_difficulty(-0.1)
        with self.assertRaises(ValueError):
            sampler.set_difficulty(1.1)
        with self.assertRaises(KeyError):
            sampler.set_preset("not_a_preset")


class TestCatalogAndFilterStyles(unittest.TestCase):
    def test_all_234_styles_mapped_cleanly(self) -> None:
        all_names = list_styles()
        self.assertEqual(len(all_names), 234)
        self.assertEqual(filter_styles(), all_names)

        seen_eras: set[str] = set()
        seen_media: set[str] = set()
        seen_implements: set[str] = set()
        seen_legibilities: set[int] = set()
        seen_papers: set[str] = set()

        for name in all_names:
            desc = describe_style(name)
            self.assertEqual(desc["name"], name)
            self.assertIn(desc["era"], CANONICAL_ERAS)
            self.assertIn(desc["medium"], CANONICAL_MEDIA)
            self.assertIn(desc["implement"], CANONICAL_IMPLEMENTS)
            self.assertIn(desc["legibility"], LEGIBILITY_LABELS)
            self.assertEqual(desc["legibility_label"], LEGIBILITY_LABELS[desc["legibility"]])
            self.assertGreaterEqual(desc["legibility_score"], 0.0)
            self.assertLessEqual(desc["legibility_score"], 1.0)
            self.assertIn(desc["paper_type"], PAPER_TYPES)
            self.assertIn(desc["notehead_archetype"], NOTEHEAD_ARCHETYPES)

            seen_eras.add(desc["era"])
            seen_media.add(desc["medium"])
            seen_implements.add(desc["implement"])
            seen_legibilities.add(desc["legibility"])
            seen_papers.add(desc["paper_type"])

        # Every canonical value across all axes must be covered by the 234-style registry.
        self.assertEqual(seen_eras, set(CANONICAL_ERAS))
        self.assertEqual(seen_media, set(CANONICAL_MEDIA))
        self.assertEqual(seen_implements, set(CANONICAL_IMPLEMENTS))
        self.assertEqual(seen_legibilities, {1, 2, 3, 4, 5})
        self.assertEqual(seen_papers, set(PAPER_TYPES))

        # Verify all 75 flagship hero_* scribes exist, validate cleanly, and map across all 4 axes.
        hero_names = [n for n in all_names if n.startswith("hero_")]
        self.assertEqual(len(hero_names), 75)
        for hname in hero_names:
            validate_style(get_style(hname), name=hname)
            hdesc = describe_style(hname)
            self.assertIn(hdesc["era"], CANONICAL_ERAS)
            self.assertIn(hdesc["medium"], CANONICAL_MEDIA)
            self.assertIn(hdesc["implement"], CANONICAL_IMPLEMENTS)
            self.assertIn(hdesc["legibility"], LEGIBILITY_LABELS)

    def test_physics_legibility_ordering(self) -> None:
        neat_score = compute_legibility_score(SCRIBE_STYLES["romantic_copyist_oblong"])
        wild_score = compute_legibility_score(SCRIBE_STYLES["beethoven_hurried_autograph"])
        tremor_score = compute_legibility_score(SCRIBE_STYLES["tremor_elderly_hand"])

        self.assertLess(neat_score, wild_score)
        self.assertLess(neat_score, tremor_score)
        self.assertEqual(score_legibility(SCRIBE_STYLES["romantic_copyist_oblong"]), 1)
        self.assertGreaterEqual(score_legibility(SCRIBE_STYLES["beethoven_hurried_autograph"]), 4)
        self.assertGreaterEqual(score_legibility(SCRIBE_STYLES["tremor_elderly_hand"]), 4)

    def test_filter_styles_combinations_and_validation(self) -> None:
        baroque_quills = filter_styles(era="baroque", implement="quill")
        self.assertGreater(len(baroque_quills), 0)
        for name in baroque_quills:
            d = describe_style(name)
            self.assertEqual(d["era"], "baroque")
            self.assertEqual(d["implement"], "quill")

        # Iterable OR within a facet + legibility by label & int.
        neat_by_int = filter_styles(legibility=(1, 2))
        neat_by_lbl = filter_styles(legibility=("engraver", "neat"))
        self.assertEqual(neat_by_int, neat_by_lbl)
        self.assertGreater(len(neat_by_int), 0)

        # Paper type, notehead, medium, and tag filters.
        ozalid = filter_styles(paper_type="ozalid_blue")
        self.assertGreater(len(ozalid), 0)
        slashes = filter_styles(notehead="slash")
        self.assertGreater(len(slashes), 0)
        jazz = filter_styles(medium="jazz_lead_sheet")
        self.assertGreater(len(jazz), 0)
        tagged = filter_styles(tag="mahler")
        self.assertIn("wild_rom_mahler_conductor_blue", tagged)

        # Invalid enum values raise ValueError.
        with self.assertRaises(ValueError):
            filter_styles(era="jurassic")
        with self.assertRaises(ValueError):
            filter_styles(medium="unknown_medium")
        with self.assertRaises(ValueError):
            filter_styles(implement="laser")
        with self.assertRaises(ValueError):
            filter_styles(legibility=6)
        with self.assertRaises(ValueError):
            filter_styles(legibility="impossible")
        with self.assertRaises(ValueError):
            filter_styles(paper_type="papyrus")
        with self.assertRaises(ValueError):
            filter_styles(notehead="triangle")


if __name__ == "__main__":
    unittest.main()
