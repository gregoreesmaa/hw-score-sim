"""Manuscript paper synthesis, human-like horizontal spacing, and curved-staff geometry.

Pure NumPy implementation driven exclusively by a caller-provided
``numpy.random.Generator`` so every page is reproducible from its seed.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence

import numpy as np
from PIL import Image

PAPER_TYPES: tuple[str, ...] = (
    "aged_rag",
    "bright_white",
    "cream_manuscript",
    "spiral_notebook",
    "ozalid_blue",
    "soviet_green_tint",
)

_PAPER_TYPE_TINTS: dict[str, tuple[float, float, float]] = {
    "aged_rag": (0.88, 0.82, 0.70),
    "bright_white": (0.98, 0.98, 0.98),
    "cream_manuscript": (0.95, 0.92, 0.84),
    "spiral_notebook": (0.97, 0.98, 0.99),
    "ozalid_blue": (0.86, 0.90, 0.96),
    "soviet_green_tint": (0.86, 0.93, 0.86),
}


def _bilinear_upsample(grid: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Bilinearly upsample a 2D float32 array to (out_h, out_w) via Pillow float32 resize."""
    in_h, in_w = grid.shape
    if in_h == 1 and in_w == 1:
        return np.full((out_h, out_w), float(grid[0, 0]), dtype=np.float32)
    pil_in = Image.fromarray(np.ascontiguousarray(grid, dtype=np.float32), mode="F")
    pil_out = pil_in.resize((int(out_w), int(out_h)), Image.BILINEAR)
    return np.asarray(pil_out, dtype=np.float32)


def make_manuscript_paper(
    width: int,
    height: int,
    style: dict[str, Any],
    rng: np.random.Generator,
) -> np.ndarray:
    """Synthesize realistic float32 RGB manuscript paper in ``[0.0, 1.0]``.

    Includes laid-paper horizontal grain, multi-scale foxing/mottling,
    aged edge darkening, subtle reverse-side bleed-through ghosting,
    and optional faint eraser smudges.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"Paper dimensions must be positive, got ({width}, {height})")

    base_rgb = np.asarray(style.get("paper_rgb", (0.94, 0.91, 0.85)), dtype=np.float32)
    ink_rgb = np.asarray(style.get("ink_rgb", (0.15, 0.13, 0.11)), dtype=np.float32)
    paper_grain = float(style.get("paper_grain", 0.06))
    paper_mottle = float(style.get("paper_mottle", 0.08))
    bleed_through = float(style.get("bleed_through", 0.06))
    eraser_smudge_prob = float(style.get("eraser_smudge_prob", 0.05))
    paper_type = style.get("paper_type")
    stain_prob = float(style.get("stain_prob", 0.0))

    if paper_type in _PAPER_TYPE_TINTS:
        tint = np.asarray(_PAPER_TYPE_TINTS[str(paper_type)], dtype=np.float32)
        # Blend strongly toward the paper_type base tint while respecting subtle style hue.
        base_rgb = np.clip(0.80 * tint + 0.20 * base_rgb, 0.05, 0.99)
        if paper_type == "aged_rag":
            paper_grain = max(paper_grain, 0.10)
            paper_mottle = max(paper_mottle, 0.18)
        elif paper_type == "bright_white":
            paper_mottle = min(paper_mottle, 0.03)

    # Subtle per-page paper tone drift
    tone_shift = rng.normal(0.0, 0.012, size=3).astype(np.float32)
    page_rgb = np.clip(base_rgb + tone_shift, 0.05, 0.99)
    sepia_chroma_cfg = style.get("sepia_paper_chroma")
    if sepia_chroma_cfg is not None and float(sepia_chroma_cfg) > 0.0:
        # Enforce warm aged-sepia paper tint (R > G > B) with controlled chroma (R - B)
        c_tgt = float(np.clip(sepia_chroma_cfg, 0.06, 0.24))
        r_s = float(np.clip(page_rgb[0], 0.80, 0.96))
        b_s = float(np.clip(r_s - c_tgt, 0.56, 0.88))
        r_s = float(np.clip(b_s + c_tgt, 0.72, 0.96))
        g_s = float(b_s + 0.62 * (r_s - b_s))
        page_rgb = np.array([r_s, g_s, b_s], dtype=np.float32)
    img = np.empty((height, width, 3), dtype=np.float32)
    img[:, :, 0] = page_rgb[0]
    img[:, :, 1] = page_rgb[1]
    img[:, :, 2] = page_rgb[2]

    # 1. Laid paper grain (fine fiber noise + subtle horizontal laid lines + faint chain lines)
    if paper_grain > 0.0:
        fine_noise = rng.standard_normal((height, width), dtype=np.float32)
        fine_noise *= np.float32(paper_grain * 0.36)
        # Subtle high-frequency horizontal laid-wire texture
        laid_period = float(rng.uniform(4.5, 8.0))
        y_idx = np.arange(height, dtype=np.float32)
        laid_wave = np.sin(2.0 * np.pi * y_idx / laid_period + float(rng.uniform(0.0, 2.0 * np.pi)))
        laid_1d = (laid_wave * (paper_grain * 0.14)).astype(np.float32)
        fine_noise += laid_1d[:, None]
        # Subtle vertical rag-paper fiber streaks
        fiber_h = max(16, height // 12)
        fiber_w = max(32, width // 6)
        fiber = _bilinear_upsample(
            rng.normal(0.0, 1.0, size=(fiber_h, fiber_w)).astype(np.float32),
            height,
            width,
        )
        fine_noise += fiber * np.float32(paper_grain * 0.14)
        img += fine_noise[:, :, None]

    # 2. Subtle archival paper aging & fine-grain rag texture (replaces blobby low-freq cloud mottle)
    if paper_mottle > 0.0:
        # Very gentle large-scale page warmth gradient (never heavy brown camouflage blotches)
        grad_h = max(3, min(6, height // 256 + 3))
        grad_w = max(3, min(6, width // 256 + 3))
        fine_h = max(24, min(128, height // 16 + 8))
        fine_w = max(24, min(128, width // 16 + 8))

        gentle_tone = _bilinear_upsample(
            rng.normal(0.0, 1.0, size=(grad_h, grad_w)).astype(np.float32),
            height,
            width,
        )
        fine_rag = _bilinear_upsample(
            rng.normal(0.0, 1.0, size=(fine_h, fine_w)).astype(np.float32),
            height,
            width,
        )
        mottle = (0.45 * gentle_tone + 0.55 * fine_rag) * np.float32(min(paper_mottle, 0.25) * 0.11)
        # Subtle warm sepia aging tint
        img[:, :, 0] += mottle * 0.82
        img[:, :, 1] += mottle * 0.96
        img[:, :, 2] += mottle * 1.15

        # Occasional small, delicate warm-brown foxing specks on aged paper
        if paper_mottle >= 0.08:
            n_spots = int(rng.integers(2, max(3, int(14 * min(paper_mottle, 0.25)) + 3)))
            yy, xx = np.ogrid[:height, :width]
            for _ in range(n_spots):
                cy = float(rng.uniform(0.0, height))
                cx = float(rng.uniform(0.0, width))
                rad = float(rng.uniform(2.0, max(3.5, min(height, width) * 0.010)))
                y0 = max(0, int(cy - rad * 2.5))
                y1 = min(height, int(cy + rad * 2.5) + 1)
                x0 = max(0, int(cx - rad * 2.5))
                x1 = min(width, int(cx + rad * 2.5) + 1)
                if y1 <= y0 or x1 <= x0:
                    continue
                dy = (yy[y0:y1, :] - cy).astype(np.float32)
                dx = (xx[:, x0:x1] - cx).astype(np.float32)
                dist2 = (dy * dy + dx * dx) / (rad * rad + 1e-6)
                spot = np.exp(-0.5 * dist2) * float(rng.uniform(0.02, 0.07) * min(paper_mottle, 0.25) * 2.2)
                img[y0:y1, x0:x1, 0] -= spot * 0.55
                img[y0:y1, x0:x1, 1] -= spot * 0.85
                img[y0:y1, x0:x1, 2] -= spot * 1.15

    # 3. Aged sepia edge darkening (subtle vignette)
    edge_strength = float(0.015 + 0.16 * min(paper_mottle, 0.25))
    yn = np.linspace(-1.0, 1.0, height, dtype=np.float32)[:, None]
    xn = np.linspace(-1.0, 1.0, width, dtype=np.float32)[None, :]
    r2 = np.clip(0.5 * (xn * xn + yn * yn), 0.0, 1.0)
    vignette = (r2 * r2) * edge_strength
    img[:, :, 0] -= vignette * 0.72
    img[:, :, 1] -= vignette * 0.92
    img[:, :, 2] -= vignette * 1.18

    # 4. Reverse-side bleed-through ghosting (horizontally-mirrored [and optionally inverted] verso staves, noteheads, stems, beams & barlines)
    verso_alpha_cfg = style.get("verso_bleed_alpha")
    verso_invert = bool(style.get("verso_bleed_invert", False))
    verso_staves_cfg = style.get("verso_bleed_staves")
    has_explicit_verso = verso_alpha_cfg is not None and float(verso_alpha_cfg) > 0.005
    if bleed_through > 0.01 or has_explicit_verso:
        sh, sw = max(64, height // 2), max(64, width // 2)
        ghost_half = np.zeros((sh, sw), dtype=np.float32)
        if verso_staves_cfg is not None:
            n_staves = int(np.clip(int(verso_staves_cfg), 3, 14))
        elif bool(style.get("dense_landscape", False)):
            n_staves = int(np.clip(int(style.get("target_staves_per_page", 12)), 4, 14))
        else:
            n_staves = max(3, min(10, sh // 95))
        sp_ghost = max(2.8, sh / (n_staves * (9.0 if n_staves >= 10 else 13.5)))
        margin_x = int(sw * 0.075)
        span_x = max(16, sw - 2 * margin_x)
        yy_g, xx_g = np.ogrid[:sh, :sw]
        for s in range(n_staves):
            y_base = (s + 0.62) * (sh / (n_staves + 0.30)) + float(rng.uniform(-sp_ghost * 0.8, sp_ghost * 0.8))
            # Faint verso 5-line staff
            for li in range(-2, 3):
                yl = int(round(y_base + li * sp_ghost))
                if 1 <= yl < sh - 1:
                    ghost_half[yl, margin_x : sw - margin_x] = np.maximum(
                        ghost_half[yl, margin_x : sw - margin_x], 0.38
                    )
            # Faint verso barlines (5-7 bars per system)
            n_bars = int(rng.integers(5, 8))
            for bx in np.linspace(margin_x, sw - margin_x, n_bars + 1):
                ix = int(np.clip(round(bx), 1, sw - 2))
                y0_b = max(0, int(round(y_base - 2.0 * sp_ghost)))
                y1_b = min(sh, int(round(y_base + 2.0 * sp_ghost)) + 1)
                ghost_half[y0_b:y1_b, ix] = np.maximum(ghost_half[y0_b:y1_b, ix], 0.50)
            # Elliptical verso noteheads + thin stems + occasional beams
            n_notes = int(rng.integers(14, 28))
            note_xs = np.sort(rng.uniform(margin_x + sp_ghost * 1.5, sw - margin_x - sp_ghost * 1.5, size=n_notes))
            rx = max(1.4, sp_ghost * 0.52)
            ry = max(1.0, sp_ghost * 0.36)
            stem_h = max(4, int(round(sp_ghost * 2.9)))
            prev_tip: Optional[tuple[int, int]] = None
            for ni, nxf in enumerate(note_xs):
                nyf = float(np.clip(y_base + rng.uniform(-2.4, 2.4) * sp_ghost, 4.0, sh - 5.0))
                nx_i, ny_i = int(round(nxf)), int(round(nyf))
                y0_n, y1_n = max(0, int(nyf - ry - 1)), min(sh, int(nyf + ry + 2))
                x0_n, x1_n = max(0, int(nxf - rx - 1)), min(sw, int(nxf + rx + 2))
                if y1_n > y0_n and x1_n > x0_n:
                    d2 = ((yy_g[y0_n:y1_n, :] - nyf) / (ry + 1e-6)) ** 2 + ((xx_g[:, x0_n:x1_n] - nxf) / (rx + 1e-6)) ** 2
                    blob = np.clip(1.2 - d2, 0.0, 1.0).astype(np.float32) * 0.85
                    ghost_half[y0_n:y1_n, x0_n:x1_n] = np.maximum(ghost_half[y0_n:y1_n, x0_n:x1_n], blob)
                if nyf > y_base:
                    sx_i = min(sw - 1, nx_i + int(round(rx * 0.8)))
                    ty_i = max(0, ny_i - stem_h)
                    ghost_half[ty_i:ny_i, sx_i] = np.maximum(ghost_half[ty_i:ny_i, sx_i], 0.65)
                else:
                    sx_i = max(0, nx_i - int(round(rx * 0.8)))
                    ty_i = min(sh - 1, ny_i + stem_h)
                    ghost_half[ny_i:ty_i, sx_i] = np.maximum(ghost_half[ny_i:ty_i, sx_i], 0.65)
                # Connect pairs into faint verso beams
                if prev_tip is not None and (ni % 2 == 1) and abs(sx_i - prev_tip[0]) < span_x * 0.12 and abs(ty_i - prev_tip[1]) < sp_ghost * 2.5:
                    x_a, y_a = prev_tip
                    n_steps = max(2, abs(sx_i - x_a) + 1)
                    bxs = np.linspace(x_a, sx_i, n_steps).astype(np.int32)
                    bys = np.linspace(y_a, ty_i, n_steps).astype(np.int32)
                    for bx_i, by_i in zip(bxs, bys):
                        if 1 <= by_i < sh - 1 and 0 <= bx_i < sw:
                            ghost_half[by_i - 1 : by_i + 1, bx_i] = np.maximum(ghost_half[by_i - 1 : by_i + 1, bx_i], 0.72)
                    prev_tip = None
                else:
                    prev_tip = (sx_i, ty_i)
        # Mirror horizontally (verso side shows through reversed!) and optionally invert vertically
        ghost_flipped = ghost_half[::-1, ::-1] if verso_invert else ghost_half[:, ::-1]
        if verso_invert:
            # Subtle vertical phase shift so verso staves sit slightly offset from recto staves
            v_shift = int(round(float(rng.uniform(0.30, 0.55)) * (sh / max(1, n_staves))))
            if 0 < v_shift < sh:
                ghost_flipped = np.roll(ghost_flipped, shift=v_shift, axis=0)
        ghost_u8 = np.ascontiguousarray(np.clip(ghost_flipped, 0.0, 1.0) * 255.0, dtype=np.uint8)
        from PIL import ImageFilter
        pil_g = Image.fromarray(ghost_u8, mode="L").filter(ImageFilter.BoxBlur(1.2))
        ghost = np.asarray(pil_g.resize((int(width), int(height)), Image.BILINEAR), dtype=np.float32) * np.float32(1.0 / 255.0)
        verso_rgb = np.clip(ink_rgb * 0.65 + np.array([0.24, 0.20, 0.15], dtype=np.float32), 0.0, 1.0)
        if has_explicit_verso:
            eff_alpha = float(np.clip(verso_alpha_cfg, 0.0, 0.30))
        elif bool(style.get("historical_era", False)):
            eff_alpha = min(bleed_through, 0.28)
        else:
            eff_alpha = min(bleed_through, 0.25) * 0.28
        alpha = (np.clip(ghost, 0.0, 1.0) * eff_alpha)[:, :, None]
        img = img * (1.0 - alpha) + verso_rgb[None, None, :] * alpha

    # 5. Optional faint eraser smudges
    if eraser_smudge_prob > 0.0 and float(rng.random()) < eraser_smudge_prob:
        n_smudges = int(rng.integers(1, 5))
        yy, xx = np.ogrid[:height, :width]
        for _ in range(n_smudges):
            cy = float(rng.uniform(height * 0.12, height * 0.88))
            cx = float(rng.uniform(width * 0.10, width * 0.90))
            ry = float(rng.uniform(max(8.0, height * 0.015), max(16.0, height * 0.045)))
            rx = float(rng.uniform(max(18.0, width * 0.03), max(36.0, width * 0.10)))
            y0 = max(0, int(cy - ry * 2.5))
            y1 = min(height, int(cy + ry * 2.5) + 1)
            x0 = max(0, int(cx - rx * 2.5))
            x1 = min(width, int(cx + rx * 2.5) + 1)
            if y1 <= y0 or x1 <= x0:
                continue
            dy = (yy[y0:y1, :] - cy).astype(np.float32) / (ry + 1e-6)
            dx = (xx[:, x0:x1] - cx).astype(np.float32) / (rx + 1e-6)
            r2_s = dx * dx + dy * dy
            # Lightened abraded core + faint smudged graphite/ink halo
            core = np.exp(-1.4 * r2_s) * float(rng.uniform(0.025, 0.065))
            halo = np.exp(-0.45 * r2_s) * float(rng.uniform(0.02, 0.055))
            img[y0:y1, x0:x1, :] += core[:, :, None]
            img[y0:y1, x0:x1, :] -= halo[:, :, None] * (1.0 - ink_rgb[None, None, :]) * 0.55

    # 6. Optional spiral-notebook ruled lines & left margin line
    if paper_type == "spiral_notebook":
        rule_spacing = max(14, int(round(height / 32.0)))
        rule_rgb = np.array([0.62, 0.78, 0.94], dtype=np.float32)
        for yr in range(rule_spacing, height - rule_spacing // 2, rule_spacing):
            img[yr : min(height, yr + 1), :, :] = (
                img[yr : min(height, yr + 1), :, :] * 0.72 + rule_rgb[None, None, :] * 0.28
            )
        margin_x = max(8, int(round(width * 0.095)))
        if margin_x + 1 < width:
            margin_rgb = np.array([0.92, 0.56, 0.58], dtype=np.float32)
            img[:, margin_x : margin_x + 2, :] = (
                img[:, margin_x : margin_x + 2, :] * 0.68 + margin_rgb[None, None, :] * 0.32
            )
        # Subtle spiral-binding punch holes along far-left edge
        # (skip when paper_binding_mode == 'wire_spiral_left' draws the wire-spiral holes below)
        if str(style.get("paper_binding_mode") or "none").strip().lower() != "wire_spiral_left":
            hole_x = max(4, int(round(width * 0.025)))
            hole_r = max(2.5, min(width, height) * 0.012)
            hole_step = max(18, int(round(height / 18.0)))
            yy_h, xx_h = np.ogrid[:height, :width]
            for yh in range(hole_step // 2, height - hole_step // 2, hole_step):
                y0 = max(0, int(yh - hole_r - 2))
                y1 = min(height, int(yh + hole_r + 3))
                x0 = max(0, int(hole_x - hole_r - 2))
                x1 = min(width, int(hole_x + hole_r + 3))
                if y1 > y0 and x1 > x0:
                    d = np.hypot((yy_h[y0:y1, :] - yh).astype(np.float32), (xx_h[:, x0:x1] - hole_x).astype(np.float32))
                    mask = np.clip(hole_r + 0.5 - d, 0.0, 1.0)[..., None]
                    img[y0:y1, x0:x1, :] = img[y0:y1, x0:x1, :] * (1.0 - 0.72 * mask) + np.float32(0.22) * (0.72 * mask)

    # 7. Optional coffee / water ring stain (capillary darker rim + faint tinted interior)
    if stain_prob > 0.0 and float(rng.random()) < stain_prob:
        n_stains = int(rng.integers(1, 3))
        yy_s, xx_s = np.ogrid[:height, :width]
        for _ in range(n_stains):
            cy = float(rng.uniform(height * 0.08, height * 0.92))
            cx = float(rng.uniform(width * 0.08, width * 0.92))
            rad = float(rng.uniform(max(14.0, min(height, width) * 0.045), max(28.0, min(height, width) * 0.14)))
            ring_w = max(2.0, rad * float(rng.uniform(0.08, 0.16)))
            y0 = max(0, int(cy - rad * 1.45))
            y1 = min(height, int(cy + rad * 1.45) + 1)
            x0 = max(0, int(cx - rad * 1.45))
            x1 = min(width, int(cx + rad * 1.45) + 1)
            if y1 <= y0 or x1 <= x0:
                continue
            dy = (yy_s[y0:y1, :] - cy).astype(np.float32)
            dx = (xx_s[:, x0:x1] - cx).astype(np.float32)
            r_px = np.hypot(dx, dy)
            phi = np.arctan2(dy, dx)
            # Organic uneven coffee-cup perimeter
            p1, p2 = float(rng.uniform(0, 2 * np.pi)), float(rng.uniform(0, 2 * np.pi))
            r_eff = rad * (1.0 + 0.06 * np.cos(2.0 * phi + p1) + 0.04 * np.cos(3.0 * phi + p2))
            # Capillary rim peak + soft interior wash
            rim = np.exp(-0.5 * ((r_px - r_eff) / (ring_w + 1e-6)) ** 2)
            interior = np.clip((r_eff - r_px) / max(ring_w * 1.5, 1.0), 0.0, 1.0) * 0.25
            strength = float(rng.uniform(0.14, 0.34))
            alpha = np.clip((rim + interior) * strength, 0.0, 0.55)[..., None]
            # Coffee brown vs fainter water-ring sepia
            is_coffee = float(rng.random()) < 0.70
            stain_rgb = np.array([0.46, 0.30, 0.15] if is_coffee else [0.68, 0.62, 0.50], dtype=np.float32)
            img[y0:y1, x0:x1, :] = img[y0:y1, x0:x1, :] * (1.0 - alpha) + stain_rgb[None, None, :] * alpha

    # 8. Extended v0.3.0 substrate structures: background grids, borders, binding hardware
    grid_mode = str(style.get("paper_background_grid") or "none").strip().lower()
    if grid_mode == "heavy_laid_chainlines":
        chain_step = int(round(rng.uniform(90.0, 130.0)))
        for xc in range(chain_step // 2, width, chain_step):
            img[:, xc : min(width, xc + 2), :] *= np.float32(0.965)
    elif grid_mode == "graph_5mm":
        step = max(14, int(round(min(width, height) / 48.0)))
        g_rgb = np.array([0.74, 0.88, 0.82], dtype=np.float32)
        for i_y, yr in enumerate(range(0, height, step)):
            a = np.float32(0.42 if i_y % 5 == 0 else 0.22)
            img[yr : min(height, yr + 1), :, :] = img[yr : min(height, yr + 1), :, :] * (1.0 - a) + g_rgb * a
        for i_x, xr in enumerate(range(0, width, step)):
            a = np.float32(0.42 if i_x % 5 == 0 else 0.22)
            img[:, xr : min(width, xr + 1), :] = img[:, xr : min(width, xr + 1), :] * (1.0 - a) + g_rgb * a
    elif grid_mode == "yellow_legal_pad":
        legal_tint = np.array([0.97, 0.93, 0.68], dtype=np.float32)
        img = img * np.float32(0.25) + legal_tint[None, None, :] * np.float32(0.75)
        step = max(18, int(round(height / 34.0)))
        rule_rgb = np.array([0.62, 0.78, 0.90], dtype=np.float32)
        for yr in range(step * 2, height - step, step):
            img[yr : min(height, yr + 1), :, :] = img[yr : min(height, yr + 1), :, :] * 0.72 + rule_rgb * 0.28
        mx = max(10, int(round(width * 0.095)))
        red_rgb = np.array([0.90, 0.48, 0.50], dtype=np.float32)
        for off in (0, 5):
            if mx + off + 1 < width:
                img[:, mx + off : mx + off + 2, :] = img[:, mx + off : mx + off + 2, :] * 0.62 + red_rgb * 0.38
    elif grid_mode == "ozalid_cyan_streaks":
        s_noise = rng.normal(0.0, 0.018, size=(height, 1)).astype(np.float32)
        img[:, :, 0] -= s_noise * 0.4
        img[:, :, 2] += s_noise * 0.8

    border_mode = str(style.get("paper_border_mode") or "none").strip().lower()
    if border_mode in ("single_rule_frame", "double_engraved_frame"):
        bx0 = max(12, int(round(width * 0.032)))
        bx1 = min(width - 12, int(round(width * 0.968)))
        by0 = max(12, int(round(height * 0.028)))
        by1 = min(height - 12, int(round(height * 0.972)))
        frame_rgb = ink_rgb * np.float32(0.65) + np.float32(0.25)
        img[by0 : by0 + 2, bx0:bx1, :] = img[by0 : by0 + 2, bx0:bx1, :] * 0.45 + frame_rgb * 0.55
        img[by1 - 2 : by1, bx0:bx1, :] = img[by1 - 2 : by1, bx0:bx1, :] * 0.45 + frame_rgb * 0.55
        img[by0:by1, bx0 : bx0 + 2, :] = img[by0:by1, bx0 : bx0 + 2, :] * 0.45 + frame_rgb * 0.55
        img[by0:by1, bx1 - 2 : bx1, :] = img[by0:by1, bx1 - 2 : bx1, :] * 0.45 + frame_rgb * 0.55
        if border_mode == "double_engraved_frame":
            pad = 7
            img[by0 + pad : by0 + pad + 1, bx0 + pad : bx1 - pad, :] = img[by0 + pad : by0 + pad + 1, bx0 + pad : bx1 - pad, :] * 0.55 + frame_rgb * 0.45
            img[by1 - pad - 1 : by1 - pad, bx0 + pad : bx1 - pad, :] = img[by1 - pad - 1 : by1 - pad, bx0 + pad : bx1 - pad, :] * 0.55 + frame_rgb * 0.45
            img[by0 + pad : by1 - pad, bx0 + pad : bx0 + pad + 1, :] = img[by0 + pad : by1 - pad, bx0 + pad : bx0 + pad + 1, :] * 0.55 + frame_rgb * 0.45
            img[by0 + pad : by1 - pad, bx1 - pad - 1 : bx1 - pad, :] = img[by0 + pad : by1 - pad, bx1 - pad - 1 : bx1 - pad, :] * 0.55 + frame_rgb * 0.45
    elif border_mode == "pre_ruled_header_box":
        hy = max(24, int(round(height * 0.095)))
        hx0, hx1 = int(width * 0.05), int(width * 0.95)
        img[hy : hy + 2, hx0:hx1, :] = img[hy : hy + 2, hx0:hx1, :] * 0.65 + ink_rgb * 0.35

    binding_mode = str(style.get("paper_binding_mode") or "none").strip().lower()
    if binding_mode in ("three_hole_punch", "two_hole_euro"):
        fracs = (0.14, 0.50, 0.86) if binding_mode == "three_hole_punch" else (0.35, 0.65)
        hx = max(12.0, width * 0.032)
        hr = max(6.0, min(width, height) * 0.016)
        yy_b, xx_b = np.ogrid[:height, :width]
        for yf in fracs:
            yh = height * yf
            y0, y1 = max(0, int(yh - hr * 1.6)), min(height, int(yh + hr * 1.6) + 1)
            x0, x1 = max(0, int(hx - hr * 1.6)), min(width, int(hx + hr * 1.6) + 1)
            if y1 <= y0 or x1 <= x0:
                continue
            d = np.hypot((yy_b[y0:y1, :] - yh).astype(np.float32), (xx_b[:, x0:x1] - hx).astype(np.float32))
            hole = np.clip(hr + 0.5 - d, 0.0, 1.0)[..., None]
            rim = np.exp(-0.5 * ((d - hr) / max(1.5, hr * 0.14)) ** 2)[..., None] * np.float32(0.28)
            img[y0:y1, x0:x1, :] = img[y0:y1, x0:x1, :] * (1.0 - rim) * (1.0 - hole) + np.float32(0.18) * hole
    elif binding_mode == "wire_spiral_left":
        hx = max(10.0, width * 0.030)
        hr = max(4.0, min(width, height) * 0.011)
        step = max(24, int(round(height / 18.0)))
        yy_b, xx_b = np.ogrid[:height, :width]
        for yh in range(step // 2, height - step // 2, step):
            y0, y1 = max(0, int(yh - hr * 1.5)), min(height, int(yh + hr * 1.5) + 1)
            x0, x1 = 0, min(width, int(hx + hr * 1.6) + 1)
            if y1 <= y0 or x1 <= x0:
                continue
            d = np.hypot((yy_b[y0:y1, :] - yh).astype(np.float32), (xx_b[:, x0:x1] - hx).astype(np.float32))
            hole = np.clip(hr + 0.5 - d, 0.0, 1.0)[..., None]
            img[y0:y1, x0:x1, :] = img[y0:y1, x0:x1, :] * (1.0 - 0.82 * hole) + np.float32(0.14) * (0.82 * hole)
            # Metallic wire loop from page edge into hole
            wy0, wy1 = max(0, int(yh - hr * 0.45)), min(height, int(yh + hr * 0.45) + 1)
            wx1 = min(width, int(hx + 2))
            if wy1 > wy0 and wx1 > 0:
                img[wy0:wy1, 0:wx1, :] = np.float32(0.32)
    elif binding_mode in ("book_gutter_shadow_left", "book_gutter_shadow_right"):
        xs_g = np.arange(width, dtype=np.float32)
        dist_edge = xs_g if binding_mode == "book_gutter_shadow_left" else (width - 1.0 - xs_g)
        shadow = np.exp(-dist_edge / max(24.0, width * 0.065)) * np.float32(0.30)
        img *= (1.0 - shadow[None, :, None])

    # 9. Optional dark archival scanner-bed margins (1-4% border with subtle paper-edge shadow)
    scan_frac = float(np.clip(style.get("scanner_margin_frac", 0.0), 0.0, 0.05))
    if scan_frac >= 0.005:
        mx_l = max(2, int(round(width * scan_frac * float(rng.uniform(0.70, 1.25)))))
        mx_r = max(2, int(round(width * scan_frac * float(rng.uniform(0.70, 1.25)))))
        my_t = max(2, int(round(height * scan_frac * float(rng.uniform(0.70, 1.25)))))
        my_b = max(2, int(round(height * scan_frac * float(rng.uniform(0.70, 1.25)))))
        bed_val = float(rng.uniform(0.08, 0.14))
        bed_rgb = np.array([bed_val, bed_val * 0.98, bed_val * 0.95], dtype=np.float32)
        # Soft paper-edge drop shadow (3-6px inside the sheet)
        sh_w = max(3, int(round(min(width, height) * 0.004)))
        if my_t + sh_w < height - my_b and mx_l + sh_w < width - mx_r:
            for k in range(sh_w):
                fade = float(np.exp(-0.7 * (k + 1))) * 0.22
                img[my_t + k, mx_l : width - mx_r, :] *= np.float32(1.0 - fade)
                img[height - my_b - 1 - k, mx_l : width - mx_r, :] *= np.float32(1.0 - fade)
                img[my_t : height - my_b, mx_l + k, :] *= np.float32(1.0 - fade)
                img[my_t : height - my_b, width - mx_r - 1 - k, :] *= np.float32(1.0 - fade)
        # Fill outer scanner-bed margins with dark matte texture
        bed_noise_t = rng.normal(0.0, 0.012, size=(my_t, width, 1)).astype(np.float32)
        bed_noise_b = rng.normal(0.0, 0.012, size=(my_b, width, 1)).astype(np.float32)
        bed_noise_l = rng.normal(0.0, 0.012, size=(height, mx_l, 1)).astype(np.float32)
        bed_noise_r = rng.normal(0.0, 0.012, size=(height, mx_r, 1)).astype(np.float32)
        img[:my_t, :, :] = bed_rgb[None, None, :] + bed_noise_t
        img[height - my_b :, :, :] = bed_rgb[None, None, :] + bed_noise_b
        img[:, :mx_l, :] = bed_rgb[None, None, :] + bed_noise_l
        img[:, width - mx_r :, :] = bed_rgb[None, None, :] + bed_noise_r

    return np.clip(img, 0.0, 1.0).astype(np.float32, copy=False)


def draw_margin_annotations(
    canvas: Any,
    style: dict[str, Any],
    rng: np.random.Generator,
    staff_ys: Optional[Sequence[float]] = None,
    sp: float = 12.0,
) -> list[dict[str, Any]]:
    """Draw optional handwritten rehearsal marks (boxed letters) and margin scratches.

    Returns a list of annotation metadata dicts describing what was drawn.
    """
    from .stroke_engine import draw_calligraphic_polyline

    w = int(canvas.width)
    h = int(canvas.height)
    ink_rgb = style.get("ink_rgb", (0.15, 0.13, 0.11))
    stroke_w = max(0.95, float(sp) * 0.14)
    drawn: list[dict[str, Any]] = []

    if staff_ys is not None and len(staff_ys) > 0:
        anchor_ys = [float(y) for y in staff_ys]
    else:
        anchor_ys = [float(h * frac) for frac in (0.16, 0.42, 0.68)]

    # 1. Boxed rehearsal letter mark (e.g. [A], [B], [C]) above a staff or in upper margin.
    y_ref = anchor_ys[0]
    rx = float(rng.uniform(max(18.0, w * 0.06), max(28.0, min(w - 28.0, w * 0.24))))
    ry = float(np.clip(y_ref - 2.6 * sp, 16.0, max(18.0, h - 20.0)))
    box_r = max(6.0, sp * 0.95)
    box = np.array([
        [rx - box_r, ry - box_r],
        [rx + box_r, ry - box_r],
        [rx + box_r, ry + box_r],
        [rx - box_r, ry + box_r],
        [rx - box_r, ry - box_r],
    ], dtype=np.float64) + rng.normal(0.0, 0.06 * sp, size=(5, 2))
    draw_calligraphic_polyline(canvas, box, ink_rgb, stroke_w * 0.95, style, kind="annotation")
    # Stylized 'A' or 'B' rehearsal glyph inside the box
    letter = str(rng.choice(["A", "B", "C"]))
    s = box_r * 0.58
    if letter == "A":
        draw_calligraphic_polyline(canvas, np.array([[rx - s, ry + s], [rx, ry - s], [rx + s, ry + s]]), ink_rgb, stroke_w, style, kind="annotation")
        draw_calligraphic_polyline(canvas, np.array([[rx - 0.55 * s, ry + 0.15 * s], [rx + 0.55 * s, ry + 0.10 * s]]), ink_rgb, stroke_w * 0.85, style, kind="annotation")
    elif letter == "B":
        draw_calligraphic_polyline(canvas, np.array([[rx - 0.5 * s, ry - s], [rx - 0.5 * s, ry + s]]), ink_rgb, stroke_w, style, kind="annotation")
        draw_calligraphic_polyline(canvas, np.array([[rx - 0.5 * s, ry - s], [rx + 0.5 * s, ry - 0.55 * s], [rx - 0.45 * s, ry], [rx + 0.6 * s, ry + 0.55 * s], [rx - 0.5 * s, ry + s]]), ink_rgb, stroke_w * 0.95, style, kind="annotation")
    else:
        draw_calligraphic_polyline(canvas, np.array([[rx + 0.6 * s, ry - 0.7 * s], [rx - 0.1 * s, ry - s], [rx - 0.6 * s, ry], [rx - 0.1 * s, ry + s], [rx + 0.6 * s, ry + 0.7 * s]]), ink_rgb, stroke_w, style, kind="annotation")
    drawn.append({"kind": "rehearsal_box", "label": letter, "x": rx, "y": ry})

    # 2. Quick conductor/copyist margin cue (check mark, asterisk, or marginal slash) in left/right margin.
    y_cue = anchor_ys[min(1, len(anchor_ys) - 1)]
    on_left = bool(rng.random() < 0.6)
    cx = float(rng.uniform(8.0, max(12.0, w * 0.07))) if on_left else float(rng.uniform(min(w - 12.0, w * 0.93), max(14.0, w - 8.0)))
    cy = float(np.clip(y_cue + rng.normal(0.0, sp), 12.0, max(14.0, h - 12.0)))
    cue_kind = str(rng.choice(["check", "asterisk", "slashes"]))
    if cue_kind == "check":
        pts = np.array([[cx - 0.5 * sp, cy], [cx - 0.1 * sp, cy + 0.45 * sp], [cx + 0.7 * sp, cy - 0.6 * sp]], dtype=np.float64)
        draw_calligraphic_polyline(canvas, pts, ink_rgb, stroke_w * 1.1, style, kind="annotation")
    elif cue_kind == "asterisk":
        r_a = 0.55 * sp
        for ang_deg in (0.0, 60.0, 120.0):
            rad = np.deg2rad(ang_deg)
            p0 = np.array([cx - r_a * np.cos(rad), cy - r_a * np.sin(rad)])
            p1 = np.array([cx + r_a * np.cos(rad), cy + r_a * np.sin(rad)])
            draw_calligraphic_polyline(canvas, np.vstack((p0, p1)), ink_rgb, stroke_w * 0.9, style, kind="annotation")
    else:
        for k_s in range(2):
            off = (k_s - 0.5) * 0.35 * sp
            pts = np.array([[cx + off - 0.3 * sp, cy + 0.5 * sp], [cx + off + 0.3 * sp, cy - 0.5 * sp]], dtype=np.float64)
            draw_calligraphic_polyline(canvas, pts, ink_rgb, stroke_w * 1.05, style, kind="annotation")
    drawn.append({"kind": f"margin_{cue_kind}", "x": cx, "y": cy})

    return drawn


def compute_quirky_note_positions(
    beat_offsets: Sequence[float],
    bar_x0: float,
    bar_x1: float,
    sp: float,
    style: dict[str, Any],
    rng: np.random.Generator,
    durations: Optional[Sequence[float]] = None,
) -> list[float]:
    """Compute human-like non-uniform horizontal note positions within a measure.

    Applies duration-power spacing (``duration_spacing_gamma``), post-downbeat breath,
    organic beat stretching/compression, right-barline cramming (``end_bar_cram``),
    piecewise late-bar panic elbow (``bar_end_panic_strength``), and per-note jitter
    while strictly preserving monotonic order (``x[i] + min_gap <= x[i+1]``) inside
    ``[bar_x0, bar_x1]``.
    """
    from .styles import resolve_ductus_param

    n = len(beat_offsets)
    if n == 0:
        return []
    if bar_x1 <= bar_x0:
        raise ValueError(f"bar_x1 ({bar_x1}) must be greater than bar_x0 ({bar_x0})")

    bar_w = float(bar_x1 - bar_x0)
    spacing_jitter = float(np.clip(style.get("spacing_jitter", 0.20), 0.0, 0.80))
    end_bar_cram = float(np.clip(style.get("end_bar_cram", 0.15), 0.0, 0.80))
    gamma = float(np.clip(resolve_ductus_param(style, "duration_spacing_gamma", 1.0), 0.12, 1.35))
    panic_start = float(np.clip(resolve_ductus_param(style, "bar_end_panic_start", 0.68), 0.45, 0.90))
    panic_str = float(np.clip(resolve_ductus_param(style, "bar_end_panic_strength", 0.0), 0.0, 0.85))
    downbeat_boost = float(np.clip(resolve_ductus_param(style, "downbeat_weight_boost", 0.0), 0.0, 0.42))

    left_pad = min(max(0.8 * sp, 1.0), 0.14 * bar_w)
    right_pad = min(max(0.7 * sp, 1.0), 0.12 * bar_w)
    x_min = float(bar_x0 + left_pad)
    x_max = float(bar_x1 - right_pad)
    usable_w = max(x_max - x_min, 1e-3)

    if n == 1:
        center = x_min + usable_w * float(np.clip(0.22 + rng.normal(0.0, 0.08 + 0.12 * spacing_jitter), 0.05, 0.65))
        return [float(np.clip(center, x_min, x_max))]

    beats = np.asarray(beat_offsets, dtype=np.float64)
    b_min = float(beats[0])
    b_max = float(beats[-1])
    span = b_max - b_min

    if span <= 1e-9:
        u = np.linspace(0.0, 1.0, n, dtype=np.float64)
    elif abs(gamma - 1.0) > 1e-3 or downbeat_boost > 1e-4 or (durations is not None and len(durations) == n):
        dt = np.maximum(np.diff(beats), 1e-4)
        if durations is not None and len(durations) == n:
            dur_arr = np.maximum(np.asarray(durations[:-1], dtype=np.float64), 1e-3)
            dt = 0.5 * dt + 0.5 * dur_arr
        du_raw = np.power(dt, gamma)
        if downbeat_boost > 0.0 and len(du_raw) >= 1 and beats[0] <= 0.125:
            du_raw[0] *= (1.0 + 0.55 * downbeat_boost)
        u = np.concatenate(([0.0], np.cumsum(du_raw)))
        u /= max(u[-1], 1e-9)
    else:
        u = (beats - b_min) / span
        for i in range(1, n):
            if u[i] <= u[i - 1]:
                u[i] = u[i - 1] + 1e-4
        u = (u - u[0]) / max(u[-1] - u[0], 1e-9)

    # 1. Right-barline cramming + piecewise late-bar panic elbow
    if end_bar_cram > 0.0:
        cram_amt = float(rng.uniform(0.35, 1.0)) * end_bar_cram
        power = 1.0 / (1.0 + 1.8 * cram_amt)
        u_warped = np.power(np.clip(u, 0.0, 1.0), power)
    else:
        u_warped = u.copy()

    if panic_str > 0.0:
        p_amt = float(rng.uniform(0.45, 1.0)) * panic_str
        tail = np.maximum(0.0, u_warped - panic_start) / max(1.0 - panic_start, 1e-4)
        elbow_compress = 1.0 - 0.62 * p_amt
        u_warped = np.where(
            u_warped <= panic_start,
            u_warped * (1.0 + 0.25 * p_amt),
            panic_start * (1.0 + 0.25 * p_amt)
            + (1.0 - panic_start) * np.power(tail, 1.0 / (1.0 + 2.4 * p_amt)) * elbow_compress,
        )
        u_warped = (u_warped - u_warped[0]) / max(u_warped[-1] - u_warped[0], 1e-9)

    # 2. Organic per-interval stretch/compression
    du = np.diff(u_warped)
    du = np.maximum(du, 1e-4)
    if spacing_jitter > 0.0:
        log_scale = rng.normal(0.0, spacing_jitter, size=n - 1)
        scale = np.clip(np.exp(log_scale), 0.35, 2.50)
        du = du * scale

    cum = np.concatenate(([0.0], np.cumsum(du)))
    cum /= max(cum[-1], 1e-9)

    # Slight random left/right margin utilization inside the bar
    start_frac = float(np.clip(rng.uniform(0.0, 0.06 + 0.06 * spacing_jitter), 0.0, 0.15))
    end_frac = float(np.clip(1.0 - rng.uniform(0.0, 0.06 + 0.08 * (1.0 - end_bar_cram)), 0.78, 1.0))
    xs = x_min + (start_frac + cum * (end_frac - start_frac)) * usable_w

    # 3. Small per-note horizontal jitter
    if spacing_jitter > 0.0:
        avg_gap = usable_w / max(n - 1, 1)
        xs = xs + rng.normal(0.0, 0.14 * spacing_jitter * avg_gap, size=n)

    # 4. Enforce strict monotonicity with non-overlapping min_gap inside [x_min, x_max]
    max_feasible_gap = (usable_w * 0.92) / max(n - 1, 1)
    min_gap = float(min(max(0.85 * sp, 1e-3), max_feasible_gap))

    xs[0] = max(xs[0], x_min)
    for i in range(1, n):
        if xs[i] < xs[i - 1] + min_gap:
            xs[i] = xs[i - 1] + min_gap

    xs[-1] = min(xs[-1], x_max)
    for i in range(n - 2, -1, -1):
        if xs[i] > xs[i + 1] - min_gap:
            xs[i] = xs[i + 1] - min_gap

    # Final safety clamp if left edge was pushed slightly below x_min
    if xs[0] < x_min:
        xs = np.linspace(x_min, x_max, n, dtype=np.float64)

    return [float(v) for v in xs]


def build_curved_staff_y(
    x0: float,
    x1: float,
    y_center: float,
    sp: float,
    style: dict[str, Any],
    rng: np.random.Generator,
) -> tuple[Callable[[float | np.ndarray, int], float | np.ndarray], np.ndarray, float]:
    """Build a hand-drawn curved staff y-evaluator and exact cubic Bezier centerline.

    Returns ``(y_fn, ctrl_pts, sp_out)`` where:
      - ``y_fn(x, line_idx)`` evaluates the y-coordinate at horizontal position ``x``
        for staff line ``line_idx`` in ``[-2, -1, 0, 1, 2]`` (and supports ledger lines
        outside that range). For ``line_idx == 0``, ``y_fn(x, 0)`` matches the cubic
        Bezier centerline defined by ``ctrl_pts`` exactly.
      - ``ctrl_pts`` is a ``(4, 2)`` float64 array of exact cubic Bezier control points
        ``[P0, P1, P2, P3]`` for Bezier staff-curve ground truth.
      - ``sp_out`` is the effective inter-line staff spacing ``sp``.
    """
    if x1 <= x0:
        raise ValueError(f"x1 ({x1}) must be greater than x0 ({x0})")
    if sp <= 0.0:
        raise ValueError(f"sp must be positive, got {sp}")

    span_x = float(x1 - x0)
    curve_px = abs(float(style.get("staff_curve_px", 3.5)))
    slope_cfg = float(style.get("staff_slope_deg", 0.6))
    sp_jitter_frac = float(np.clip(style.get("sp_jitter_frac", 0.025), 0.0, 0.15))

    # Global slight staff slope preserving configured sign (e.g. left-handed downward slope)
    if slope_cfg != 0.0:
        mean_slope = slope_cfg * 0.65
        spread = abs(slope_cfg) * 0.35
        slope_deg = float(np.clip(rng.uniform(mean_slope - spread, mean_slope + spread), -2.5, 2.5))
    else:
        slope_deg = 0.0
    dy_slope = span_x * float(np.tan(np.deg2rad(slope_deg)))

    y0 = float(y_center - 0.5 * dy_slope)
    y3 = float(y_center + 0.5 * dy_slope)

    # Smooth quadratic arch + cubic S-wave hand-drawn curve
    arch = float(rng.uniform(-curve_px, curve_px)) if curve_px > 0.0 else 0.0
    scurve = float(rng.uniform(-0.65 * curve_px, 0.65 * curve_px)) if curve_px > 0.0 else 0.0

    y1 = y0 + dy_slope / 3.0 + arch + scurve
    y2 = y0 + 2.0 * dy_slope / 3.0 + arch - scurve

    # Uniformly spaced x control points make B_x(t) = x0 + t * (x1 - x0) exact
    ctrl_pts = np.array(
        [
            [float(x0), y0],
            [float(x0 + span_x / 3.0), y1],
            [float(x0 + 2.0 * span_x / 3.0), y2],
            [float(x1), y3],
        ],
        dtype=np.float64,
    )

    # Subtle per-line sp jitter for the 5 staff lines [-2, -1, 0, 1, 2]
    # Center line (0) has zero offset so y_fn(x, 0) == exact Bezier centerline.
    max_line_shift = 0.22 * sp
    line_offsets: dict[int, float] = {0: 0.0}
    line_drifts: dict[int, float] = {0: 0.0}
    for k in (-2, -1, 1, 2):
        raw_off = float(rng.normal(0.0, sp * sp_jitter_frac))
        raw_drift = float(rng.normal(0.0, sp * sp_jitter_frac * 0.6))
        line_offsets[k] = float(np.clip(raw_off, -max_line_shift, max_line_shift))
        line_drifts[k] = float(np.clip(raw_drift, -0.5 * max_line_shift, 0.5 * max_line_shift))

    cy0, cy1, cy2, cy3 = float(y0), float(y1), float(y2), float(y3)
    inv_span = 1.0 / span_x
    sp_out = float(sp)

    def y_fn(x: float | np.ndarray, line_idx: int = 0) -> float | np.ndarray:
        is_scalar = np.ndim(x) == 0
        x_arr = np.asarray(x, dtype=np.float64)
        t = (x_arr - x0) * inv_span
        omt = 1.0 - t
        y_mid = (
            (omt * omt * omt) * cy0
            + 3.0 * (omt * omt) * t * cy1
            + 3.0 * omt * (t * t) * cy2
            + (t * t * t) * cy3
        )
        idx = int(line_idx)
        offset = line_offsets.get(idx, 0.0)
        drift = line_drifts.get(idx, 0.0)
        y_val = y_mid + idx * sp_out + offset + drift * (t - 0.5)
        if is_scalar:
            return float(y_val)
        return y_val

    return y_fn, ctrl_pts, sp_out
