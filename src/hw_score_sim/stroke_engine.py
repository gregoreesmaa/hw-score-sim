"""Core vector-stroke calligraphy, nib physics, and organic ink rasterizer.

Pure-Python + NumPy + Pillow implementation for realistic handwritten music
strokes (broad-nib quill, fountain pen, ballpoint, graphite pencil, felt-tip).
Every stroke has unique smooth bow, micro-tremor, nib-angle width modulation,
pressure taper/pooling, dry-brush dropout, paper-tooth grain, and ink variation.
"""

from __future__ import annotations

import math
from typing import Callable, Optional, Sequence, Union

import numpy as np
from PIL import Image

from . import _substrate_cal


RGBLike = Union[np.ndarray, Sequence[float]]


def _normalize_rgb(rgb: RGBLike) -> np.ndarray:
    """Convert an RGB sequence/array (0..1 or 0..255) to float32 in [0, 1]."""
    arr = np.asarray(rgb, dtype=np.float32).reshape(3)
    if float(np.max(arr)) > 1.0:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0)


def _get_nib_angle_rad(style: dict) -> float:
    """Resolve nib angle in radians from style dict."""
    if "nib_angle_deg" in style:
        return math.radians(float(style["nib_angle_deg"]))
    if "nib_angle" in style:
        val = float(style["nib_angle"])
        # Heuristic: values outside [-2*pi, 2*pi] are degrees.
        if abs(val) > 2.0 * math.pi + 1e-6:
            return math.radians(val)
        return val
    return math.radians(35.0)


def _get_min_ratio(style: dict) -> float:
    for key in ("nib_min_ratio", "min_width_ratio", "min_ratio"):
        if key in style:
            return float(np.clip(style[key], 0.05, 1.0))
    return 0.35


def _generate_paper_tooth(rng: np.random.Generator, tile_size: int = 256) -> np.ndarray:
    """Generate a seamless tileable paper-tooth texture in [0, 1]."""
    # Combine coarse (64x64) and fine (256x256) noise for realistic paper fiber.
    coarse_n = tile_size // 4
    coarse = rng.random((coarse_n, coarse_n), dtype=np.float32)
    # Bilinear-ish upsample via periodic interpolation.
    xs = np.linspace(0, coarse_n, tile_size, endpoint=False, dtype=np.float32)
    x0 = np.floor(xs).astype(np.int32) % coarse_n
    x1 = (x0 + 1) % coarse_n
    fx = xs - np.floor(xs)
    c_horiz = coarse[:, x0] * (1.0 - fx[None, :]) + coarse[:, x1] * fx[None, :]
    c_full = c_horiz[x0, :] * (1.0 - fx[:, None]) + c_horiz[x1, :] * fx[:, None]

    fine = rng.random((tile_size, tile_size), dtype=np.float32)
    tooth = 0.55 * c_full + 0.45 * fine
    return np.clip(tooth, 0.0, 1.0).astype(np.float32)


class StrokeCanvas:
    """Float32 RGB canvas (0..1) with coherent paper-tooth texture and ink blending."""

    def __init__(
        self,
        width: int,
        height: int,
        bg_rgb: RGBLike,
        rng: Optional[np.random.Generator] = None,
        init_buf: Optional[np.ndarray] = None,
    ) -> None:
        self.width = int(width)
        self.height = int(height)
        self.bg_rgb = _normalize_rgb(bg_rgb)
        self.rng = rng if rng is not None else np.random.default_rng()
        if (
            init_buf is not None
            and isinstance(init_buf, np.ndarray)
            and init_buf.shape == (self.height, self.width, 3)
            and init_buf.dtype == np.float32
        ):
            self.buf = init_buf
        else:
            self.buf = np.empty((self.height, self.width, 3), dtype=np.float32)
            self.buf[:, :, 0] = self.bg_rgb[0]
            self.buf[:, :, 1] = self.bg_rgb[1]
            self.buf[:, :, 2] = self.bg_rgb[2]
        self._tooth_tile: Optional[np.ndarray] = None
        self._tooth_full: Optional[np.ndarray] = None
        self._stroke_count: int = 0

    def advance_ink_dip(self, style: dict) -> tuple[float, float]:
        """Advance stateful quill/dip-nib ink depletion sawtooth cycle.

        Returns ``(opacity_mult, extra_dropout)``. When ``ink_dip_period`` is
        ``0`` (default), returns ``(1.0, 0.0)`` with zero effect.
        """
        period = int(round(float(style.get("ink_dip_period", 0))))
        if period <= 0:
            return 1.0, 0.0
        pos = self._stroke_count % period
        self._stroke_count += 1
        frac = pos / float(max(period - 1, 1))
        # Fresh-dip surge -> smooth depletion -> abrupt re-dip sawtooth
        if frac < 0.15:
            opacity_mult = 1.08 - 0.53 * frac
        else:
            opacity_mult = 1.0 - 0.34 * ((frac - 0.15) / 0.85)
        extra_dropout = 0.0 if frac < 0.35 else 0.22 * ((frac - 0.35) / 0.65)
        return float(opacity_mult), float(extra_dropout)

    def get_tooth_patch(self, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
        """Return paper-tooth texture values in [0, 1] for [y0:y1, x0:x1]."""
        if self._tooth_full is None:
            if self._tooth_tile is None:
                self._tooth_tile = _generate_paper_tooth(self.rng, tile_size=256)
            tile = self._tooth_tile
            th, tw = tile.shape
            reps_y = (self.height + th - 1) // th
            reps_x = (self.width + tw - 1) // tw
            self._tooth_full = np.ascontiguousarray(
                np.tile(tile, (reps_y, reps_x))[: self.height, : self.width]
            )
        return self._tooth_full[y0:y1, x0:x1]

    def composite_patch(
        self,
        x0: int,
        y0: int,
        x1: int,
        y1: int,
        alpha: np.ndarray,
        ink_rgb: np.ndarray,
    ) -> None:
        """Alpha-composite an ink patch onto the canvas buffer."""
        if x1 <= x0 or y1 <= y0:
            return
        a = alpha if alpha.dtype == np.float32 else alpha.astype(np.float32)
        a3 = np.clip(a, np.float32(0.0), np.float32(1.0))[..., None]
        dst = self.buf[y0:y1, x0:x1]
        # In-place over-blend: dst += (ink_rgb - dst) * a3 (1 temp buffer vs 5-6)
        diff = ink_rgb[None, None, :] - dst
        diff *= a3
        dst += diff

    def to_uint8(self) -> np.ndarray:
        """Convert buffer to uint8 RGB ndarray (H, W, 3)."""
        return np.clip(self.buf * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8)

    def to_pil(self) -> Image.Image:
        """Convert buffer to a Pillow RGB Image."""
        return Image.fromarray(self.to_uint8(), mode="RGB")


def _smooth_1d_noise(rng: np.random.Generator, n_samples: int, n_knots: int) -> np.ndarray:
    """Generate smooth zero-mean unit-scale 1D noise of length n_samples."""
    if n_samples <= 1:
        return np.zeros(n_samples, dtype=np.float64)
    n_knots = max(2, min(n_knots, n_samples))
    knots = rng.standard_normal(n_knots)
    # Direct uniform-grid indexing (avoids np.searchsorted).
    pos = np.linspace(0.0, float(n_knots - 1), n_samples)
    idx = np.minimum(pos.astype(np.intp), n_knots - 2)
    local_t = pos - idx
    mu = 0.5 * (1.0 - np.cos(math.pi * local_t))
    return knots[idx] * (1.0 - mu) + knots[idx + 1] * mu


def _smooth_1d_noise_pair(
    rng: np.random.Generator, n_samples: int, n_knots: int
) -> tuple[np.ndarray, np.ndarray]:
    """Generate two independent smooth 1D noise arrays sharing uniform-grid idx/mu."""
    if n_samples <= 1:
        z = np.zeros(n_samples, dtype=np.float64)
        return z, z.copy()
    n_knots = max(2, min(n_knots, n_samples))
    knots_a = rng.standard_normal(n_knots)
    knots_b = rng.standard_normal(n_knots)
    pos = np.linspace(0.0, float(n_knots - 1), n_samples)
    idx = np.minimum(pos.astype(np.intp), n_knots - 2)
    local_t = pos - idx
    mu = 0.5 * (1.0 - np.cos(math.pi * local_t))
    omt = 1.0 - mu
    return knots_a[idx] * omt + knots_a[idx + 1] * mu, knots_b[idx] * omt + knots_b[idx + 1] * mu


def _polyline_grad(pts: np.ndarray) -> np.ndarray:
    """Fast central-difference gradient along axis 0 for (M, 2) polylines."""
    n = len(pts)
    g = np.empty_like(pts)
    if n <= 2:
        d = pts[-1] - pts[0]
        g[:] = d
    else:
        g[0] = pts[1] - pts[0]
        g[-1] = pts[-1] - pts[-2]
        g[1:-1] = 0.5 * (pts[2:] - pts[:-2])
    return g


def _resample_polyline(pts: np.ndarray, step_px: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Resample a polyline at approximately uniform arc-length intervals.

    Returns (resampled_pts (M, 2), t_norm (M,) in [0, 1], total_len).
    """
    diffs = np.diff(pts, axis=0)
    seg_lens = np.hypot(diffs[:, 0], diffs[:, 1])
    cum = np.concatenate(([0.0], np.cumsum(seg_lens)))
    total_len = float(cum[-1])
    if total_len < 1e-6:
        return pts[:1].copy(), np.zeros(1, dtype=np.float64), 0.0

    # Remove zero-length consecutive duplicates for clean interpolation.
    keep = np.concatenate(([True], seg_lens > 1e-7))
    pts_clean = pts[keep]
    cum_clean = cum[keep]
    if len(pts_clean) < 2:
        return pts[:1].copy(), np.zeros(1, dtype=np.float64), 0.0

    n_steps = max(2, int(math.ceil(total_len / max(0.35, step_px))) + 1)
    s_eval = np.linspace(0.0, total_len, n_steps)
    rx = np.interp(s_eval, cum_clean, pts_clean[:, 0])
    ry = np.interp(s_eval, cum_clean, pts_clean[:, 1])
    return np.column_stack((rx, ry)), s_eval / total_len, total_len


def _perturb_ink_color(
    color_rgb: RGBLike, style: dict, rng: np.random.Generator
) -> tuple[np.ndarray, float]:
    """Return slightly varied (ink_rgb, stroke_opacity) for a single stroke."""
    base_rgb = _normalize_rgb(color_rgb)
    ink_var = float(style.get("ink_var", 0.06))
    base_opacity = float(style.get("stroke_opacity", style.get("opacity", 0.92)))

    if ink_var > 0.0:
        # Correlated luminance shift + ultra-subtle chromatic variation (no rainbow tint).
        lum_shift = float(rng.normal(0.0, ink_var * 0.45))
        chroma_shift = rng.normal(0.0, ink_var * 0.015, size=3).astype(np.float32)
        ink_rgb = np.clip(base_rgb + lum_shift + chroma_shift, 0.0, 1.0).astype(np.float32)
        opacity = float(np.clip(base_opacity * (1.0 + rng.normal(0.0, ink_var)), 0.12, 1.0))
    else:
        ink_rgb = base_rgb
        opacity = float(np.clip(base_opacity, 0.12, 1.0))
    return ink_rgb, opacity


def _soft_blur_2d(arr: np.ndarray, radius: int) -> np.ndarray:
    """Fast pure-NumPy 2D separable box blur (2 passes for smooth feather halo)."""
    r = int(radius)
    if r <= 0 or arr.size == 0:
        return arr.copy()
    out = arr.astype(np.float32, copy=True)
    if r == 1:
        inv3 = np.float32(1.0 / 3.0)
        for _ in range(2):
            p = np.pad(out, ((0, 0), (1, 1)), mode="edge")
            acc = p[:, :-2].copy()
            acc += p[:, 1:-1]
            acc += p[:, 2:]
            acc *= inv3
            p = np.pad(acc, ((1, 1), (0, 0)), mode="edge")
            out = p[:-2, :].copy()
            out += p[1:-1, :]
            out += p[2:, :]
            out *= inv3
        return out
    if r == 2:
        inv5 = np.float32(1.0 / 5.0)
        for _ in range(2):
            p = np.pad(out, ((0, 0), (2, 2)), mode="edge")
            acc = p[:, :-4].copy()
            acc += p[:, 1:-3]
            acc += p[:, 2:-2]
            acc += p[:, 3:-1]
            acc += p[:, 4:]
            acc *= inv5
            p = np.pad(acc, ((2, 2), (0, 0)), mode="edge")
            out = p[:-4, :].copy()
            out += p[1:-3, :]
            out += p[2:-2, :]
            out += p[3:-1, :]
            out += p[4:, :]
            out *= inv5
        return out
    if r == 3:
        inv7 = np.float32(1.0 / 7.0)
        for _ in range(2):
            p = np.pad(out, ((0, 0), (3, 3)), mode="edge")
            acc = p[:, :-6].copy()
            acc += p[:, 1:-5]
            acc += p[:, 2:-4]
            acc += p[:, 3:-3]
            acc += p[:, 4:-2]
            acc += p[:, 5:-1]
            acc += p[:, 6:]
            acc *= inv7
            p = np.pad(acc, ((3, 3), (0, 0)), mode="edge")
            out = p[:-6, :].copy()
            out += p[1:-5, :]
            out += p[2:-4, :]
            out += p[3:-3, :]
            out += p[4:-2, :]
            out += p[5:-1, :]
            out += p[6:, :]
            out *= inv7
        return out
    k = 2 * r + 1
    inv_k = np.float32(1.0 / k)
    for _ in range(2):
        # Horizontal pass
        padded = np.pad(out, ((0, 0), (r, r)), mode="edge")
        cs = np.cumsum(padded, axis=1, dtype=np.float32)
        cs = np.pad(cs, ((0, 0), (1, 0)), mode="constant", constant_values=0.0)
        out = (cs[:, k:] - cs[:, :-k]) * inv_k
        # Vertical pass
        padded = np.pad(out, ((r, r), (0, 0)), mode="edge")
        cs = np.cumsum(padded, axis=0, dtype=np.float32)
        cs = np.pad(cs, ((1, 0), (0, 0)), mode="constant", constant_values=0.0)
        out = (cs[k:, :] - cs[:-k, :]) * inv_k
    return out


def _apply_ink_feather(
    patch: np.ndarray,
    canvas: StrokeCanvas,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    ink_feather: float,
    tooth: Optional[np.ndarray] = None,
    edge_crispness: float = 0.0,
) -> np.ndarray:
    """Add an organic porous-paper wet-ink feathering/bleed halo around a stroke patch."""
    f = float(np.clip(ink_feather, 0.0, 1.0))
    if f <= 1e-4 or patch.size == 0:
        return patch
    crisp = float(np.clip(edge_crispness, 0.0, 1.0))
    radius = max(1, int(round(1.0 + 2.5 * f * (1.0 - 0.35 * crisp))))
    halo = _soft_blur_2d(patch, radius)
    if tooth is None:
        tooth = canvas.get_tooth_patch(x0, y0, x1, y1)
    # Porous fibers wick ink unevenly outward; crisp iron-gall attenuates soft outer halo.
    fiber_mod = np.float32(0.50) + np.float32(0.85) * tooth
    halo_gain = np.float32(0.55 * f * (1.0 - 0.50 * crisp))
    feathered = np.clip(halo * fiber_mod * halo_gain, np.float32(0.0), np.float32(0.85))
    return np.maximum(patch, feathered)


def _stamp_discs_to_canvas(
    canvas: StrokeCanvas,
    xs: np.ndarray,
    ys: np.ndarray,
    radii: np.ndarray,
    alphas: np.ndarray,
    ink_rgb: np.ndarray,
    stroke_opacity: float,
    grain_strength: float,
    ink_feather: float = 0.0,
    pool_strengths: Optional[np.ndarray] = None,
    edge_crispness: float = 0.0,
    nib_anisotropy: float = 0.0,
    nib_angle_rad: float = 0.0,
    ink_mass: float = 0.0,
) -> None:
    """Rasterize variable-radius anti-aliased discs (or chisel-nib footprints) into a local bbox and blend."""
    if len(xs) == 0:
        return
    valid = (alphas > 1e-4) & (radii > 1e-3)
    if not np.any(valid):
        return
    if not np.all(valid):
        xs = xs[valid]
        ys = ys[valid]
        radii = radii[valid]
        alphas = alphas[valid]
        if pool_strengths is not None:
            pool_strengths = pool_strengths[valid]

    n_discs = len(xs)
    max_r = float(np.max(radii))
    feather_f = 0.0 if ink_feather <= 0.0 else (1.0 if ink_feather >= 1.0 else float(ink_feather))
    crisp_f = 0.0 if edge_crispness <= 0.0 else (1.0 if edge_crispness >= 1.0 else float(edge_crispness))
    aniso_f = 0.0 if nib_anisotropy <= 0.0 else (1.0 if nib_anisotropy >= 1.0 else float(nib_anisotropy))
    mass_f = 0.0 if ink_mass <= 0.0 else (1.0 if ink_mass >= 1.0 else float(ink_mass))
    feather_pad = int(math.ceil(feather_f * 4.0)) + 1 if feather_f > 1e-4 else 0
    pad = int(math.ceil(max_r + 1.5)) + feather_pad
    x0 = max(0, int(math.floor(float(np.min(xs)))) - pad)
    y0 = max(0, int(math.floor(float(np.min(ys)))) - pad)
    x1 = min(canvas.width, int(math.ceil(float(np.max(xs)))) + pad + 1)
    y1 = min(canvas.height, int(math.ceil(float(np.max(ys)))) + pad + 1)
    if x1 <= x0 or y1 <= y0:
        return

    pw = x1 - x0
    ph = y1 - y0
    rad_i = int(math.ceil(max_r + 1.0))
    win_w = 2 * rad_i + 1

    # Chunk sparse long bboxes (e.g. slurs, beams, braces) so we don't allocate/blur giant empty patches.
    if n_discs > 32 and pw * ph > 24000 and pw * ph > n_discs * win_w * win_w * 3:
        chunk_sz = 32
        for start in range(0, n_discs, chunk_sz):
            end = start + chunk_sz
            ps_chunk = pool_strengths[start:end] if pool_strengths is not None else None
            _stamp_discs_to_canvas(
                canvas,
                xs[start:end],
                ys[start:end],
                radii[start:end],
                alphas[start:end],
                ink_rgb,
                stroke_opacity,
                grain_strength,
                ink_feather=feather_f,
                pool_strengths=ps_chunk,
                edge_crispness=crisp_f,
                nib_anisotropy=aniso_f,
                nib_angle_rad=nib_angle_rad,
                ink_mass=mass_f,
            )
        return

    lx = (xs - np.float64(x0)).astype(np.float32)
    ly = (ys - np.float64(y0)).astype(np.float32)
    r_f32 = radii.astype(np.float32, copy=False)
    a_f32 = alphas.astype(np.float32, copy=False)

    cx_i = np.floor(lx).astype(np.int32)
    cy_i = np.floor(ly).astype(np.int32)
    fx = lx - cx_i
    fy = ly - cy_i

    # Vectorized (B, W, W) disc / angled chisel-nib coverage computation.
    offsets = np.arange(-rad_i, rad_i + 1, dtype=np.float32)
    gx = offsets[None, None, :] - fx[:, None, None]  # (B, 1, W)
    gy = offsets[None, :, None] - fy[:, None, None]  # (B, W, 1)
    if aniso_f > 1e-4:
        cos_a = np.float32(math.cos(nib_angle_rad))
        sin_a = np.float32(math.sin(nib_angle_rad))
        u_nib = gx * cos_a + gy * sin_a
        v_nib = -gx * sin_a + gy * cos_a
        inv_minor = np.float32(1.0 / max(0.38, 1.0 - 0.55 * aniso_f))
        dist = np.sqrt(u_nib * u_nib + (v_nib * inv_minor) ** 2)
    else:
        dist = np.sqrt(gx * gx + gy * gy)  # (B, W, W)

    if crisp_f > 1e-4:
        edge_gain = np.float32(1.0 + 1.5 * crisp_f)
        cov = np.clip((r_f32[:, None, None] - dist) * edge_gain + np.float32(0.5), np.float32(0.0), np.float32(1.0))
    else:
        cov = np.clip((r_f32[:, None, None] + np.float32(0.5)) - dist, np.float32(0.0), np.float32(1.0))
    cov *= a_f32[:, None, None]

    use_pool = pool_strengths is not None and float(np.max(pool_strengths)) > 1e-4

    # Check if all (W x W) windows fit strictly inside [0, ph) x [0, pw).
    bx0 = cx_i - rad_i
    by0 = cy_i - rad_i
    touches_edge = (
        int(bx0.min()) < 0
        or int(by0.min()) < 0
        or int(bx0.max()) + win_w > pw
        or int(by0.max()) + win_w > ph
    )

    if touches_edge:
        bpad = rad_i + 1
        patch_buf = np.zeros((ph + 2 * bpad, pw + 2 * bpad), dtype=np.float32)
        pool_buf = np.zeros((ph + 2 * bpad, pw + 2 * bpad), dtype=np.float32) if use_pool else None
        bx0_s = (bx0 + bpad).tolist()
        by0_s = (by0 + bpad).tolist()
        max_y_lim = ph + 2 * bpad - win_w
        max_x_lim = pw + 2 * bpad - win_w
        if pool_buf is not None and pool_strengths is not None:
            ps_f32 = pool_strengths.astype(np.float32, copy=False)
            p_cov = cov * ps_f32[:, None, None]
            for i in range(n_discs):
                xi, yi = bx0_s[i], by0_s[i]
                if 0 <= xi <= max_x_lim and 0 <= yi <= max_y_lim:
                    sub = patch_buf[yi : yi + win_w, xi : xi + win_w]
                    np.maximum(sub, cov[i], out=sub)
                    if ps_f32[i] > 1e-4:
                        psub = pool_buf[yi : yi + win_w, xi : xi + win_w]
                        np.maximum(psub, p_cov[i], out=psub)
        else:
            for i in range(n_discs):
                xi, yi = bx0_s[i], by0_s[i]
                if 0 <= xi <= max_x_lim and 0 <= yi <= max_y_lim:
                    sub = patch_buf[yi : yi + win_w, xi : xi + win_w]
                    np.maximum(sub, cov[i], out=sub)
        patch = patch_buf[bpad : bpad + ph, bpad : bpad + pw]
        pool_patch = pool_buf[bpad : bpad + ph, bpad : bpad + pw] if pool_buf is not None else None
    else:
        patch = np.zeros((ph, pw), dtype=np.float32)
        pool_patch = np.zeros((ph, pw), dtype=np.float32) if use_pool else None
        bx0_l = bx0.tolist()
        by0_l = by0.tolist()
        if pool_patch is not None and pool_strengths is not None:
            ps_f32 = pool_strengths.astype(np.float32, copy=False)
            p_cov = cov * ps_f32[:, None, None]
            for i in range(n_discs):
                xi, yi = bx0_l[i], by0_l[i]
                sub = patch[yi : yi + win_w, xi : xi + win_w]
                np.maximum(sub, cov[i], out=sub)
                if ps_f32[i] > 1e-4:
                    psub = pool_patch[yi : yi + win_w, xi : xi + win_w]
                    np.maximum(psub, p_cov[i], out=psub)
        else:
            for i in range(n_discs):
                xi, yi = bx0_l[i], by0_l[i]
                sub = patch[yi : yi + win_w, xi : xi + win_w]
                np.maximum(sub, cov[i], out=sub)

    tooth: Optional[np.ndarray] = None
    if feather_f > 1e-4 or grain_strength > 0.0:
        tooth = canvas.get_tooth_patch(x0, y0, x1, y1)

    if feather_f > 1e-4:
        patch = _apply_ink_feather(
            patch, canvas, x0, y0, x1, y1, feather_f, tooth=tooth, edge_crispness=crisp_f
        )

    if grain_strength > 0.0 and tooth is not None:
        # Paper tooth attenuates ink more strongly in lighter/drier areas.
        grain_mod = np.clip(
            np.float32(1.0) - np.float32(grain_strength * 1.25) * (np.float32(1.0) - tooth),
            np.float32(0.08),
            np.float32(1.0),
        )
        patch *= grain_mod

    if pool_patch is not None or mass_f > 1e-4:
        # Iron-gall capillary rim darkening + endpoint/junction ink_mass pooling.
        eff_pool = pool_patch if pool_patch is not None else np.zeros_like(patch)
        if mass_f > 1e-4:
            # Edge perimeter rim where wet iron-gall concentrates as it dries.
            rim = np.clip(patch * (np.float32(1.0) - patch) * np.float32(4.0), np.float32(0.0), np.float32(1.0))
            eff_pool = np.clip(eff_pool + rim * np.float32(0.42 * mass_f), np.float32(0.0), np.float32(1.0))
        eff_alpha = np.clip(
            patch * np.float32(stroke_opacity) + eff_pool * np.float32(0.20),
            np.float32(0.0),
            np.float32(1.0),
        )
        darken = np.clip(
            np.float32(1.0) - np.float32(0.35) * eff_pool,
            np.float32(0.50),
            np.float32(1.0),
        )[..., None]
        local_ink = np.clip(ink_rgb[None, None, :] * darken, np.float32(0.0), np.float32(1.0))
        dst = canvas.buf[y0:y1, x0:x1]
        diff = local_ink - dst
        diff *= eff_alpha[..., None]
        dst += diff
    else:
        patch *= np.float32(stroke_opacity)
        canvas.composite_patch(x0, y0, x1, y1, patch, ink_rgb)


def draw_calligraphic_polyline(
    canvas: StrokeCanvas,
    pts: np.ndarray,
    color_rgb: RGBLike,
    base_width: float,
    style: dict,
    kind: str = "generic",
) -> None:
    """Render a variable-width, unique-per-stroke calligraphic/handwritten path.

    Applies per-stroke smooth bow, micro-tremor, broad-nib angle width physics,
    pressure taper + optional endpoint ink pooling, dry-brush/ink-skip dropout,
    paper-tooth grain, and per-stroke ink/opacity variation.
    """
    pts_arr = np.asarray(pts, dtype=np.float64)
    if pts_arr.ndim != 2 or pts_arr.shape[1] != 2 or len(pts_arr) == 0:
        return

    rng = canvas.rng
    width_scale = float(style.get("stroke_scale", style.get("nib_width_scale", 1.0)))
    bw = max(0.35, float(base_width) * width_scale)
    grain_strength = float(np.clip(style.get("grain_strength", 0.10), 0.0, 1.0))
    ink_rgb, stroke_opacity = _perturb_ink_color(color_rgb, style, rng)
    dip_op_mult, dip_extra_drop = canvas.advance_ink_dip(style)
    stroke_opacity = float(np.clip(stroke_opacity * dip_op_mult, 0.12, 1.0))

    if len(pts_arr) == 1:
        r = np.array([max(0.5, bw * 0.5)], dtype=np.float64)
        a = np.array([1.0], dtype=np.float64)
        _stamp_discs_to_canvas(
            canvas, pts_arr[:, 0], pts_arr[:, 1], r, a, ink_rgb, stroke_opacity, grain_strength
        )
        return

    step_px = max(0.45, min(1.15, bw * 0.35))
    res_pts, t, total_len = _resample_polyline(pts_arr, step_px)
    if total_len < 1e-4 or len(res_pts) < 2:
        r = np.array([max(0.5, bw * 0.5)], dtype=np.float64)
        a = np.array([1.0], dtype=np.float64)
        _stamp_discs_to_canvas(
            canvas, pts_arr[:1, 0], pts_arr[:1, 1], r, a, ink_rgb, stroke_opacity, grain_strength
        )
        return

    # Tangents and normals along resampled path.
    grad = _polyline_grad(res_pts)
    gnorm = np.hypot(grad[:, 0], grad[:, 1])
    gnorm = np.where(gnorm > 1e-7, gnorm, 1.0)
    tx = grad[:, 0] / gnorm
    ty = grad[:, 1] / gnorm
    nx = -ty
    ny = tx

    # 1. Per-stroke smooth quadratic/cubic bow + micro-tremor.
    wobble_px = float(style.get("wobble_px", 0.7))
    tremor_px = float(style.get("tremor_px", 0.25))
    if kind == "beam":
        wobble_px *= 0.65
        tremor_px *= 0.6
    elif kind == "stem":
        wobble_px *= 0.85

    len_scale = min(1.0, total_len / 14.0)
    if wobble_px > 0.0:
        b1 = float(rng.normal(0.0, wobble_px * 0.55 * len_scale))
        b2 = float(rng.normal(0.0, wobble_px * 0.45 * len_scale))
        bow = b1 * (4.0 * t * (1.0 - t)) + b2 * (4.0 * t * (1.0 - t) * (2.0 * t - 1.0))
        res_pts[:, 0] += nx * bow
        res_pts[:, 1] += ny * bow

    if tremor_px > 0.0:
        n_knots = max(3, int(total_len / 6.0) + 2)
        tn_raw, tt_raw = _smooth_1d_noise_pair(rng, len(res_pts), n_knots)
        trem_n = tn_raw * (tremor_px * len_scale)
        trem_t = tt_raw * (tremor_px * 0.35 * len_scale)
        res_pts[:, 0] += nx * trem_n + tx * trem_t
        res_pts[:, 1] += ny * trem_n + ty * trem_t

    # Recompute tangents after perturbation for accurate nib angle.
    grad2 = _polyline_grad(res_pts)
    theta = np.arctan2(grad2[:, 1], grad2[:, 0])

    # 2. Broad-nib / italic angle width physics.
    nib_angle = _get_nib_angle_rad(style)
    min_ratio = _get_min_ratio(style)
    if kind == "beam":
        # Beams are dragged broadly; keep thickness fuller.
        min_ratio = max(min_ratio, 0.72)
    nib_factor = min_ratio + (1.0 - min_ratio) * np.abs(np.sin(theta - nib_angle))

    # 3. Pressure envelope (tapered start/end + mid-stroke undulation + ink-pool dots).
    taper_start = float(style.get("taper_start", 0.14))
    taper_end = float(style.get("taper_end", 0.18))
    pressure_var = float(style.get("pressure_var", 0.14))

    env = np.ones_like(t)
    if taper_start > 1e-4:
        mask_s = t < taper_start
        env[mask_s] *= 0.45 + 0.55 * np.sin(0.5 * math.pi * (t[mask_s] / taper_start))
    if taper_end > 1e-4:
        mask_e = t > (1.0 - taper_end)
        env[mask_e] *= 0.35 + 0.65 * np.sin(0.5 * math.pi * ((1.0 - t[mask_e]) / taper_end))

    if pressure_var > 0.0:
        p_wave = _smooth_1d_noise(rng, len(res_pts), max(3, int(total_len / 20.0) + 2))
        env *= np.clip(1.0 + pressure_var * 0.45 * p_wave, 0.55, 1.45)

    widths = bw * nib_factor * env
    alphas = np.ones_like(t)

    # Optional ink-pool swell at endpoints (fountain pen / quill landing & lift).
    ink_pool = float(style.get("ink_pooling", style.get("ink_pool", 0.25)))
    if ink_pool > 0.0 and total_len > 3.0:
        if rng.random() < min(0.85, 0.35 + ink_pool):
            pool_s = float(rng.uniform(0.10, 0.35)) * ink_pool
            sigma_s = max(0.03, min(0.10, 2.5 / max(total_len, 1.0)))
            g_start = np.exp(-0.5 * (t / sigma_s) ** 2)
            widths += bw * pool_s * g_start
            alphas = np.minimum(1.0, alphas + 0.2 * g_start)
        if rng.random() < min(0.75, 0.25 + ink_pool * 0.8):
            pool_e = float(rng.uniform(0.08, 0.30)) * ink_pool
            sigma_e = max(0.03, min(0.10, 2.5 / max(total_len, 1.0)))
            g_end = np.exp(-0.5 * ((1.0 - t) / sigma_e) ** 2)
            widths += bw * pool_e * g_end
            alphas = np.minimum(1.0, alphas + 0.15 * g_end)

    # 4. Speed-dependent ink pooling at sharp curvature / turns + ink_mass endpoint/junction pooling.
    ink_mass = float(np.clip(style.get("ink_mass", 0.0), 0.0, 1.0))
    edge_crispness = float(np.clip(style.get("edge_crispness", 0.0), 0.0, 1.0))
    nib_anisotropy = float(np.clip(style.get("nib_anisotropy", 0.0), 0.0, 1.0))
    pool_strengths: Optional[np.ndarray] = None
    if ink_pool > 0.0 and len(res_pts) >= 4:
        unwrapped = np.unwrap(theta)
        dtheta = np.abs(np.diff(unwrapped))
        curv_raw = np.zeros(len(res_pts), dtype=np.float64)
        curv_raw[:-1] += 0.5 * dtheta
        curv_raw[1:] += 0.5 * dtheta
        # Ignore tiny numeric jitter; respond strongly to real corners/turns.
        curv = np.clip((curv_raw - 0.04) / 0.30, 0.0, 1.0)
        if float(np.max(curv)) > 1e-4:
            widths += bw * (0.20 * ink_pool) * curv
            alphas = np.minimum(1.0, alphas + 0.25 * ink_pool * curv)
            pool_strengths = np.clip(curv * min(1.0, ink_pool * 1.4), 0.0, 1.0).astype(np.float32)

    if ink_mass > 1e-4 and total_len > 2.5:
        sigma_m = max(0.03, min(0.09, 2.2 / max(total_len, 1.0)))
        end_mass = (
            np.exp(-0.5 * (t / sigma_m) ** 2) + np.exp(-0.5 * ((1.0 - t) / sigma_m) ** 2)
        ) * (0.75 * ink_mass)
        if pool_strengths is None:
            pool_strengths = np.clip(end_mass, 0.0, 1.0).astype(np.float32)
        else:
            pool_strengths = np.clip(pool_strengths + end_mass.astype(np.float32), 0.0, 1.0)

    # 5. Dry-brush / ink-skip dropout along stroke.
    ink_dropout = float(np.clip(float(style.get("ink_dropout", 0.05)) + dip_extra_drop, 0.0, 0.95))
    if ink_dropout > 1e-4:
        n_drop_knots = max(4, int(total_len / 5.5) + 2)
        d_noise = _smooth_1d_noise(rng, len(res_pts), n_drop_knots)
        # Convert normal noise to uniform-like [0, 1] via smooth sigmoid.
        u_noise = 1.0 / (1.0 + np.exp(-1.5 * d_noise))
        # Attenuate where u_noise < ink_dropout.
        drop_scale = np.clip((u_noise - ink_dropout * 0.65) / max(0.12, ink_dropout * 0.55 + 0.15), 0.0, 1.0)
        alphas *= (1.0 - ink_dropout * 0.85) + (ink_dropout * 0.85) * drop_scale

    ink_feather = float(np.clip(style.get("ink_feather", 0.0), 0.0, 1.0))
    radii = np.maximum(0.28, 0.5 * widths)

    # 6. Optional pressure-gated split-nib double-edge (parted quill/fountain-pen tines).
    split_nib_prob = float(np.clip(style.get("split_nib_prob", 0.0), 0.0, 1.0))
    if split_nib_prob > 0.0 and bw >= 1.2 and total_len > 4.0 and float(rng.random()) < split_nib_prob:
        g2norm = np.hypot(grad2[:, 0], grad2[:, 1])
        g2norm = np.where(g2norm > 1e-7, g2norm, 1.0)
        nx2 = -grad2[:, 1] / g2norm
        ny2 = grad2[:, 0] / g2norm
        # Gate railroading by downstroke pressure (nib_factor * env > 0.55) so
        # tapered entry/exit remain unified while flexed bellies railroad naturally.
        pressure_gate = np.clip((nib_factor * env - 0.55) / 0.30, 0.0, 1.0)
        sep = np.maximum(0.45, radii * (0.45 + 0.25 * env)) * pressure_gate
        r_hi = np.maximum(0.25, radii * (1.0 - 0.44 * pressure_gate))
        r_lo = np.maximum(0.25, radii * (1.0 - 0.52 * pressure_gate))
        ps_both = np.concatenate((pool_strengths, pool_strengths)) if pool_strengths is not None else None
        _stamp_discs_to_canvas(
            canvas,
            np.concatenate((res_pts[:, 0] + nx2 * sep, res_pts[:, 0] - nx2 * sep)),
            np.concatenate((res_pts[:, 1] + ny2 * sep, res_pts[:, 1] - ny2 * sep)),
            np.concatenate((r_hi, r_lo)),
            np.concatenate((alphas * (1.0 - 0.08 * pressure_gate), alphas * (1.0 - 0.22 * pressure_gate))),
            ink_rgb,
            stroke_opacity,
            grain_strength,
            ink_feather=ink_feather,
            pool_strengths=ps_both,
            edge_crispness=edge_crispness,
            nib_anisotropy=nib_anisotropy,
            nib_angle_rad=nib_angle,
            ink_mass=ink_mass,
        )
        return

    _stamp_discs_to_canvas(
        canvas,
        res_pts[:, 0],
        res_pts[:, 1],
        radii,
        alphas,
        ink_rgb,
        stroke_opacity,
        grain_strength,
        ink_feather=ink_feather,
        pool_strengths=pool_strengths,
        edge_crispness=edge_crispness,
        nib_anisotropy=nib_anisotropy,
        nib_angle_rad=nib_angle,
        ink_mass=ink_mass,
    )


def draw_staff_line(
    canvas: StrokeCanvas,
    x0: float,
    x1: float,
    y_fn: Callable[[Union[float, np.ndarray]], Union[float, np.ndarray]],
    color_rgb: RGBLike,
    base_width: float,
    style: dict,
    shared_wave: Optional[np.ndarray] = None,
) -> None:
    """Draw a hand-drawn or pre-ruled staff line with subtle waviness & nib variation."""
    if x1 < x0:
        x0, x1 = x1, x0
    ix0 = max(0, int(math.floor(x0)))
    ix1 = min(canvas.width, int(math.ceil(x1)) + 1)
    if ix1 <= ix0:
        return

    rng = canvas.rng
    xs = np.arange(ix0, ix1, dtype=np.float64)
    n = len(xs)
    if n == 0:
        return

    try:
        ys = np.asarray(y_fn(xs), dtype=np.float64)
        if ys.shape != xs.shape:
            ys = np.array([float(y_fn(float(x))) for x in xs], dtype=np.float64)
    except Exception:
        ys = np.array([float(y_fn(float(x))) for x in xs], dtype=np.float64)

    pre_ruled = bool(style.get("staff_pre_ruled", False))
    wobble_px = float(style.get("staff_wobble_px", style.get("wobble_px", 0.6) * (0.25 if pre_ruled else 0.75)))
    tremor_px = float(style.get("staff_tremor_px", style.get("tremor_px", 0.25) * (0.2 if pre_ruled else 0.7)))

    t = np.linspace(0.0, 1.0, n)
    if shared_wave is not None:
        sw = np.asarray(shared_wave, dtype=np.float64)
        if len(sw) == n:
            ys = ys + sw
        elif len(sw) >= 2:
            ys = ys + np.interp(t, np.linspace(0.0, 1.0, len(sw)), sw)
    else:
        rastrum_wobble = float(style.get("rastrum_wobble", 0.0))
        if rastrum_wobble > 0.0:
            n_rknots = max(4, int(n / 90.0) + 3)
            ys = ys + _smooth_1d_noise(rng, n, n_rknots) * rastrum_wobble
        sp_est = float(style.get("sp", style.get("staff_space", max(8.0, float(base_width) / 0.085))))
        ys = ys + _substrate_cal.rastrum_micro_wave(xs, sp_est)

    if wobble_px > 0.0:
        b1 = float(rng.normal(0.0, wobble_px * 0.5))
        b2 = float(rng.normal(0.0, wobble_px * 0.35))
        ys = ys + b1 * (4.0 * t * (1.0 - t)) + b2 * (4.0 * t * (1.0 - t) * (2.0 * t - 1.0))
    if tremor_px > 0.0:
        n_knots = max(4, int(n / 18.0) + 2)
        ys = ys + _smooth_1d_noise(rng, n, n_knots) * tremor_px

    # Width variation along the line.
    bw = max(0.4, float(base_width) * float(style.get("staff_width_scale", 1.0)))
    w_var = 0.08 if pre_ruled else float(style.get("pressure_var", 0.16))
    w_noise = _smooth_1d_noise(rng, n, max(4, int(n / 35.0) + 2))
    radii = np.maximum(0.3, 0.5 * bw * np.clip(1.0 + w_var * 0.5 * w_noise, 0.6, 1.45))

    # Taper gently at ends.
    end_px = min(10, n // 8)
    if end_px >= 2:
        ramp = np.linspace(0.55, 1.0, end_px)
        radii[:end_px] *= ramp
        radii[-end_px:] *= ramp[::-1]

    # Ink dropout along staff line + optional L->R fade + dry-nib skips.
    ink_dropout = float(np.clip(style.get("ink_dropout", 0.05) * (0.35 if pre_ruled else 0.85), 0.0, 0.9))
    col_alpha = np.ones(n, dtype=np.float32)
    if ink_dropout > 1e-4:
        d_noise = _smooth_1d_noise(rng, n, max(5, int(n / 12.0) + 2))
        u_noise = 1.0 / (1.0 + np.exp(-1.5 * d_noise))
        drop_scale = np.clip((u_noise - ink_dropout * 0.65) / max(0.15, ink_dropout * 0.55 + 0.15), 0.0, 1.0)
        col_alpha = ((1.0 - ink_dropout * 0.75) + (ink_dropout * 0.75) * drop_scale).astype(np.float32)

    lr_fade = float(np.clip(style.get("staff_lr_fade", 0.0), -0.35, 0.55))
    if abs(lr_fade) > 1e-4:
        col_alpha *= np.clip(1.0 - lr_fade * t, 0.22, 1.15).astype(np.float32)

    dry_skip_prob = float(np.clip(style.get("staff_dry_skip_prob", 0.0), 0.0, 0.65))
    if dry_skip_prob > 0.0 and n > 60 and float(rng.random()) < dry_skip_prob:
        n_skips = int(rng.integers(1, 3))
        for _ in range(n_skips):
            c_idx = float(rng.uniform(0.12 * n, 0.88 * n))
            s_w = float(rng.uniform(8.0, min(55.0, n * 0.08)))
            idx_arr = np.arange(n, dtype=np.float32)
            notch = np.exp(-0.5 * ((idx_arr - c_idx) / max(3.0, s_w)) ** 2)
            col_alpha *= (1.0 - 0.88 * notch).astype(np.float32)

    ink_feather = float(np.clip(style.get("ink_feather", 0.0) * (0.4 if pre_ruled else 0.8), 0.0, 1.0))
    feather_pad = int(math.ceil(ink_feather * 3.0)) + 1 if ink_feather > 1e-4 else 0
    pad = int(math.ceil(float(np.max(radii)) + 1.5)) + feather_pad
    iy0 = max(0, int(math.floor(float(np.min(ys)))) - pad)
    iy1 = min(canvas.height, int(math.ceil(float(np.max(ys)))) + pad + 1)
    if iy1 <= iy0:
        return

    yy = np.arange(iy0, iy1, dtype=np.float32)[:, None]
    dist = np.abs(yy - ys.astype(np.float32)[None, :])
    patch = np.clip(radii.astype(np.float32)[None, :] + np.float32(0.5) - dist, 0.0, 1.0)
    patch *= col_alpha[None, :]

    grain_strength = float(np.clip(style.get("grain_strength", 0.10) * (0.5 if pre_ruled else 1.0), 0.0, 1.0))
    tooth: Optional[np.ndarray] = None
    if ink_feather > 1e-4 or grain_strength > 0.0:
        tooth = canvas.get_tooth_patch(ix0, iy0, ix1, iy1)

    if ink_feather > 1e-4:
        patch = _apply_ink_feather(patch, canvas, ix0, iy0, ix1, iy1, ink_feather, tooth=tooth)

    if grain_strength > 0.0 and tooth is not None:
        patch *= np.clip(1.0 - grain_strength * (1.0 - tooth) * 1.15, 0.1, 1.0)

    ink_rgb, stroke_opacity = _perturb_ink_color(color_rgb, style, rng)
    staff_opacity = float(style.get("staff_opacity", stroke_opacity * (0.82 if pre_ruled else 0.94)))
    patch *= np.float32(np.clip(staff_opacity, 0.15, 1.0))
    canvas.composite_patch(ix0, iy0, ix1, iy1, patch, ink_rgb)


def draw_rastrum_staff(
    canvas: StrokeCanvas,
    x0: float,
    x1: float,
    y_fn: Callable[[Union[float, np.ndarray], int], Union[float, np.ndarray]],
    color_rgb: RGBLike,
    base_width: float,
    style: dict,
    line_indices: Sequence[int] = (-2, -1, 0, 1, 2),
) -> np.ndarray:
    """Draw a 5-line staff with coherent rastrum parallel hand-drawn wobble.

    All staff lines share a single smooth hand-drawn wave of amplitude
    ``style.get("rastrum_wobble", 0.0)`` (plus each line's own subtle nib/tremor
    micro-variation), mimicking a traditional 5-nibbed rastrum pen.

    Returns the shared 1D wave array (in pixels) applied across the staff lines.
    """
    if x1 < x0:
        x0, x1 = x1, x0
    ix0 = max(0, int(math.floor(x0)))
    ix1 = min(canvas.width, int(math.ceil(x1)) + 1)
    n = max(0, ix1 - ix0)
    if n == 0:
        return np.zeros(0, dtype=np.float64)

    rastrum_amp = float(style.get("rastrum_wobble", 0.0))
    if rastrum_amp > 0.0:
        n_knots = max(4, int(n / 85.0) + 3)
        shared_wave = _smooth_1d_noise(canvas.rng, n, n_knots) * rastrum_amp
    else:
        shared_wave = np.zeros(n, dtype=np.float64)

    # Coherent 5-line rastrum harmonic micro-profile (scale-invariant in staff-space units)
    xs_r = np.arange(ix0, ix1, dtype=np.float64)
    sp_r = float(style.get("sp", style.get("staff_space", 16.0)))
    if len(line_indices) >= 2:
        try:
            xm = 0.5 * (x0 + x1)
            dy_sp = abs(float(y_fn(xm, line_indices[1])) - float(y_fn(xm, line_indices[0])))
            if 4.0 <= dy_sp <= 48.0:
                sp_r = dy_sp
        except Exception:
            pass
    shared_wave = shared_wave + _substrate_cal.rastrum_micro_wave(xs_r, sp_r)

    # Line-weight profile multipliers across the 5 rastrum nibs (-2..+2)
    profile = str(style.get("staff_line_weight_profile", "uniform")).strip().lower()
    profile_map = {
        "uniform": (1.0, 1.0, 1.0, 1.0, 1.0),
        "outer_heavy": (1.32, 0.92, 0.90, 0.92, 1.32),
        "bottom_heavy": (0.85, 0.92, 1.00, 1.12, 1.28),
        "taper_top_to_bot": (1.26, 1.12, 1.00, 0.90, 0.82),
        "alternating_worn_rastrum": (1.14, 0.82, 1.22, 0.88, 1.18),
    }
    w_mults = profile_map.get(profile, (1.0, 1.0, 1.0, 1.0, 1.0))

    stagger_sp = float(np.clip(style.get("staff_end_stagger_sp", 0.0), 0.0, 1.4))
    stagger_px = stagger_sp * max(8.0, float(base_width) / 0.085)
    mid_lift_prob = float(np.clip(style.get("staff_mid_lift_prob", 0.0), 0.0, 0.50))
    do_splice = mid_lift_prob > 0.0 and (x1 - x0) > 220.0 and float(canvas.rng.random()) < mid_lift_prob
    x_split = x0 + (x1 - x0) * float(canvas.rng.uniform(0.42, 0.62)) if do_splice else 0.0
    splice_dy = float(canvas.rng.uniform(-2.0, 2.0)) if do_splice else 0.0

    # Slight independent line micro-wobble on top of the shared rastrum motion.
    local_style = dict(style)
    local_style["rastrum_wobble"] = 0.0
    for k_i, idx in enumerate(line_indices):
        wm = w_mults[k_i % len(w_mults)]
        dx0 = float(canvas.rng.uniform(-stagger_px, stagger_px)) if stagger_px > 0.0 else 0.0
        dx1 = float(canvas.rng.uniform(-stagger_px, stagger_px)) if stagger_px > 0.0 else 0.0
        lx0 = max(0.0, x0 + dx0)
        lx1 = min(float(canvas.width - 1), x1 + dx1)
        if lx1 <= lx0 + 2.0:
            lx0, lx1 = x0, x1
        if do_splice:
            overlap = 10.0
            draw_staff_line(
                canvas, lx0, min(lx1, x_split + overlap),
                lambda x, li=idx: y_fn(x, li), color_rgb, base_width * wm, local_style, shared_wave=shared_wave,
            )
            draw_staff_line(
                canvas, max(lx0, x_split - overlap * 0.5), lx1,
                lambda x, li=idx, dy=splice_dy: np.asarray(y_fn(x, li)) + dy,
                color_rgb, base_width * wm, local_style, shared_wave=shared_wave,
            )
        else:
            draw_staff_line(
                canvas,
                lx0,
                lx1,
                lambda x, li=idx: y_fn(x, li),
                color_rgb,
                base_width * wm,
                local_style,
                shared_wave=shared_wave,
            )
    return shared_wave


def draw_filled_blob(
    canvas: StrokeCanvas,
    cx: float,
    cy: float,
    rx: float,
    ry: float,
    angle_deg: float,
    color_rgb: RGBLike,
    style: dict,
    hollow: bool = False,
    open_gap: float = 0.0,
) -> None:
    """Render an organic handwritten notehead (filled blob or hollow calligraphic loop).

    Args:
        canvas: Target StrokeCanvas.
        cx, cy: Center coordinates in pixels.
        rx, ry: Semi-axes in pixels.
        angle_deg: Rotation angle in degrees (negative = tilted upward to right).
        color_rgb: Base ink color.
        style: Style parameter dict.
        hollow: If True, draws an organic hollow (half/whole) notehead loop.
        open_gap: For hollow noteheads, fraction of circumference [0, 0.45] left unclosed
                  (values > 1.0 are interpreted as degrees).
    """
    rng = canvas.rng
    rx = max(0.8, float(rx))
    ry = max(0.8, float(ry))
    theta0 = math.radians(float(angle_deg))

    if hollow:
        # Convert open_gap to radians.
        gap_rad = math.radians(open_gap) if open_gap > 1.0 else float(np.clip(open_gap, 0.0, 0.45)) * 2.0 * math.pi
        # Start near top-right or bottom-left (typical handwritten pen landing).
        start_phi = float(rng.uniform(-0.35, 0.45))
        if gap_rad > 1e-4:
            span = 2.0 * math.pi - gap_rad
        else:
            # Slight authentic overlap when closed!
            span = 2.0 * math.pi + float(rng.uniform(0.08, 0.38))

        n_pts = 28
        phis = np.linspace(start_phi, start_phi + span, n_pts)
        # Organic radial asymmetry so no two hollow ovals are identical.
        c2 = float(rng.normal(0.0, 0.06))
        p2 = float(rng.uniform(0.0, 2.0 * math.pi))
        c3 = float(rng.normal(0.0, 0.045))
        p3 = float(rng.uniform(0.0, 2.0 * math.pi))
        rmod = 1.0 + c2 * np.cos(2.0 * phis + p2) + c3 * np.cos(3.0 * phis + p3)

        lx = rx * rmod * np.cos(phis)
        ly = ry * rmod * np.sin(phis)
        cos_t, sin_t = math.cos(theta0), math.sin(theta0)
        pts = np.column_stack((cx + lx * cos_t - ly * sin_t, cy + lx * sin_t + ly * cos_t))

        # Hollow notehead stroke width scales with minor axis.
        stroke_w = max(1.0, min(rx, ry) * float(style.get("hollow_stroke_ratio", 0.44)))
        # Keep wobble proportional to notehead size so tiny notes stay legible.
        local_style = dict(style)
        local_style["wobble_px"] = min(float(style.get("wobble_px", 0.7)), min(rx, ry) * 0.28)
        local_style["tremor_px"] = min(float(style.get("tremor_px", 0.25)), min(rx, ry) * 0.14)
        local_style["taper_start"] = 0.08
        local_style["taper_end"] = 0.10
        draw_calligraphic_polyline(canvas, pts, color_rgb, stroke_w, local_style, kind="hollow_notehead")
        return

    # Filled organic blob: evaluate perturbed signed-distance field on local bbox.
    wobble_px = float(style.get("wobble_px", 0.7))
    irreg = float(np.clip(wobble_px / max(rx, ry, 4.0), 0.02, 0.18))
    c1 = float(rng.normal(0.0, irreg * 0.45))
    p1 = float(rng.uniform(0.0, 2.0 * math.pi))
    c2 = float(rng.normal(0.0, irreg * 0.65))
    p2 = float(rng.uniform(0.0, 2.0 * math.pi))
    c3 = float(rng.normal(0.0, irreg * 0.50))
    p3 = float(rng.uniform(0.0, 2.0 * math.pi))

    ink_feather = float(np.clip(style.get("ink_feather", 0.0), 0.0, 1.0))
    feather_pad = int(math.ceil(ink_feather * 4.0)) + 1 if ink_feather > 1e-4 else 0
    max_r = max(rx, ry) * (1.0 + abs(c1) + abs(c2) + abs(c3)) + 2.0 + feather_pad
    x0 = max(0, int(math.floor(cx - max_r)))
    y0 = max(0, int(math.floor(cy - max_r)))
    x1 = min(canvas.width, int(math.ceil(cx + max_r)) + 1)
    y1 = min(canvas.height, int(math.ceil(cy + max_r)) + 1)
    if x1 <= x0 or y1 <= y0:
        return

    gx = np.arange(x0, x1, dtype=np.float32) - np.float32(cx)
    gy = np.arange(y0, y1, dtype=np.float32) - np.float32(cy)
    cos_t = np.float32(math.cos(-theta0))
    sin_t = np.float32(math.sin(-theta0))

    ux = gx[None, :] * cos_t - gy[:, None] * sin_t
    uy = gx[None, :] * sin_t + gy[:, None] * cos_t
    nx = ux / np.float32(rx)
    ny = uy / np.float32(ry)
    norm_r = np.sqrt(nx * nx + ny * ny)
    # Chebyshev trig identities avoid arctan2 + 3x array np.cos().
    inv_r = np.float32(1.0) / np.maximum(norm_r, np.float32(1e-6))
    c = nx * inv_r
    s = ny * inv_r
    cc = c * c
    c2_phi = np.float32(2.0) * cc - np.float32(1.0)
    s2_phi = np.float32(2.0) * s * c
    c3_phi = (np.float32(4.0) * cc - np.float32(3.0)) * c
    s3_phi = (np.float32(3.0) - np.float32(4.0) * s * s) * s

    edge_crispness = float(np.clip(style.get("edge_crispness", 0.0), 0.0, 1.0))
    nib_anisotropy = float(np.clip(style.get("nib_anisotropy", 0.0), 0.0, 1.0))
    ink_mass = float(np.clip(style.get("ink_mass", 0.0), 0.0, 1.0))

    boundary = (
        np.float32(1.0)
        + np.float32(c1 * math.cos(p1)) * c
        - np.float32(c1 * math.sin(p1)) * s
        + np.float32(c2 * math.cos(p2)) * c2_phi
        - np.float32(c2 * math.sin(p2)) * s2_phi
        + np.float32(c3 * math.cos(p3)) * c3_phi
        - np.float32(c3 * math.sin(p3)) * s3_phi
    )
    if nib_anisotropy > 1e-4:
        # Subtle broad-nib quill-dab 4-fold chisel squaring aligned to nib angle.
        c4_phi = np.float32(2.0) * c2_phi * c2_phi - np.float32(1.0)
        s4_phi = np.float32(2.0) * s2_phi * c2_phi
        nib_rel = _get_nib_angle_rad(style) - theta0
        dab_amp = np.float32(0.065 * nib_anisotropy)
        boundary = boundary + dab_amp * (
            np.float32(math.cos(4.0 * nib_rel)) * c4_phi
            + np.float32(math.sin(4.0 * nib_rel)) * s4_phi
        )

    # Anti-aliased edge in pixel units (approximate local scale by harmonic mean of rx, ry).
    scale_px = np.float32(max(1.0, 2.0 * rx * ry / (rx + ry)))
    signed_px = (boundary - norm_r) * scale_px
    edge_gain = np.float32(1.0 + 1.35 * edge_crispness)
    patch = np.clip(signed_px * edge_gain + np.float32(0.5), np.float32(0.0), np.float32(1.0))

    # Subtle internal dry-brush / pencil shading streak when grain or dropout is high.
    dip_op_mult, dip_extra_drop = canvas.advance_ink_dip(style)
    grain_strength = float(np.clip(style.get("grain_strength", 0.10), 0.0, 1.0))
    ink_dropout = float(np.clip(float(style.get("ink_dropout", 0.05)) + dip_extra_drop, 0.0, 0.95))

    tooth: Optional[np.ndarray] = None
    if ink_feather > 1e-4 or grain_strength > 0.0:
        tooth = canvas.get_tooth_patch(x0, y0, x1, y1)

    if ink_feather > 1e-4:
        patch = _apply_ink_feather(
            patch, canvas, x0, y0, x1, y1, ink_feather, tooth=tooth, edge_crispness=edge_crispness
        )

    if grain_strength > 0.0 and tooth is not None:
        patch *= np.clip(1.0 - grain_strength * (1.0 - tooth) * 1.15, 0.12, 1.0)
    if ink_dropout > 0.08:
        # Occasional subtle dry streak across the center of a filled notehead.
        streak_dir = np.float32(rng.uniform(-0.4, 0.4))
        streak = 0.5 + 0.5 * np.sin((ux * streak_dir + uy) * np.float32(0.8) + np.float32(rng.uniform(0, 6.28)))
        patch *= np.float32(1.0 - 0.35 * ink_dropout) + np.float32(0.35 * ink_dropout) * streak

    ink_rgb, stroke_opacity = _perturb_ink_color(color_rgb, style, rng)
    stroke_opacity = float(np.clip(stroke_opacity * dip_op_mult, 0.12, 1.0))
    if ink_mass > 1e-4:
        # Iron-gall capillary rim pooling: slightly darker ink near outer oval boundary.
        rim_zone = np.exp(-0.5 * ((signed_px - np.float32(0.9)) / np.float32(0.85)) ** 2) * patch
        rim_strength = np.clip(rim_zone * np.float32(ink_mass), np.float32(0.0), np.float32(1.0))
        eff_alpha = np.clip(patch * np.float32(stroke_opacity) + rim_strength * np.float32(0.14), np.float32(0.0), np.float32(1.0))
        darken = np.clip(np.float32(1.0) - np.float32(0.30) * rim_strength, np.float32(0.55), np.float32(1.0))[..., None]
        local_ink = np.clip(ink_rgb[None, None, :] * darken, np.float32(0.0), np.float32(1.0))
        dst = canvas.buf[y0:y1, x0:x1]
        diff = local_ink - dst
        diff *= eff_alpha[..., None]
        dst += diff
    else:
        patch *= np.float32(stroke_opacity)
        canvas.composite_patch(x0, y0, x1, y1, patch, ink_rgb)
