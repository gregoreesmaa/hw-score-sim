"""3-Probe Adversarial Sim-vs-Real Realism Classifier & Neural Network Forensics Suite.

Evaluates how distinguishable synthetic handwritten scores are from real archival
manuscripts using three complementary probes and statistical controls:

- **Probe A (Scale-Normalized Multi-View Notation-Patch CNN + Bilateral Grad-CAM):**
  Trains a dilated-receptive-field GroupNorm CNN on interior, edge-gated,
  staff-spacing-normalized notation patches across an orthogonal 6-view ablation
  ladder (``rgb``, ``gray``, ``gray_norm``, ``lowfreq_gray``, ``highfreq_gray``,
  ``binary``) with strict manuscript-level group cross-validation and bilateral
  Grad-CAM attribution (sim-fake ``+1`` and real-cue ``-1``).
- **Probe B (CoordConv + Spatial-Pyramid Full-Page Macro-Layout CNN + Grad-CAM):**
  Trains a GroupNorm CoordConv CNN with a 2x2+1x1 spatial pyramid head on
  aspect-preserving letterboxed full pages to detect macro-layout, margin, and
  density shortcuts.
- **Probe C (Frozen Feature Linear Probe + Standardized FD + Coverage/Diversity):**
  Fits an L2-regularized logistic regression probe (pure NumPy Newton-IRLS with
  Armijo backtracking and Woodbury acceleration), computes raw and standardized
  shrinkage Frechet distance, k-NN Precision/Recall/Density/Coverage, variance
  ratio, top uncovered real pages, and per-group feature attribution.
- **Controls & Diagnostics:**
  Group-clustered bootstrap 95% confidence intervals, real-vs-real 50/50 null
  control, domain telemetry (staff spacing, aspect ratio, readability SNR), and
  an actionable ``subsystem_scorecard``.

``torch`` is imported lazily so the core ``hw-score-sim`` package has no hard
PyTorch dependency.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from PIL import Image, ImageFilter, ImageOps

IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"})

PATCH_VIEW_MODES: tuple[str, ...] = (
    "rgb",
    "gray",
    "gray_norm",
    "lowfreq_gray",
    "highfreq_gray",
    "binary",
)

_VIEW_ALIASES: dict[str, str] = {
    "lowfreq": "lowfreq_gray",
    "highfreq": "highfreq_gray",
}

DEFAULT_EMBEDDING_FEATURE_SPEC: tuple[tuple[str, str], ...] = (
    # color_moments (19)
    ("color_r_mean", "color_moments"),
    ("color_r_std", "color_moments"),
    ("color_r_q10", "color_moments"),
    ("color_r_q50", "color_moments"),
    ("color_r_q90", "color_moments"),
    ("color_g_mean", "color_moments"),
    ("color_g_std", "color_moments"),
    ("color_g_q10", "color_moments"),
    ("color_g_q50", "color_moments"),
    ("color_g_q90", "color_moments"),
    ("color_b_mean", "color_moments"),
    ("color_b_std", "color_moments"),
    ("color_b_q10", "color_moments"),
    ("color_b_q50", "color_moments"),
    ("color_b_q90", "color_moments"),
    ("color_rg_mean", "color_moments"),
    ("color_rg_std", "color_moments"),
    ("color_yb_mean", "color_moments"),
    ("color_yb_std", "color_moments"),
    # gradient_orientation (12)
    ("grad_mag_mean", "gradient_orientation"),
    ("grad_mag_std", "gradient_orientation"),
    ("grad_mag_q75", "gradient_orientation"),
    ("grad_mag_q95", "gradient_orientation"),
    ("grad_ori_bin0", "gradient_orientation"),
    ("grad_ori_bin1", "gradient_orientation"),
    ("grad_ori_bin2", "gradient_orientation"),
    ("grad_ori_bin3", "gradient_orientation"),
    ("grad_ori_bin4", "gradient_orientation"),
    ("grad_ori_bin5", "gradient_orientation"),
    ("grad_ori_bin6", "gradient_orientation"),
    ("grad_ori_bin7", "gradient_orientation"),
    # radial_fft_8rings (8)
    ("fft_ring0", "radial_fft_8rings"),
    ("fft_ring1", "radial_fft_8rings"),
    ("fft_ring2", "radial_fft_8rings"),
    ("fft_ring3", "radial_fft_8rings"),
    ("fft_ring4", "radial_fft_8rings"),
    ("fft_ring5", "radial_fft_8rings"),
    ("fft_ring6", "radial_fft_8rings"),
    ("fft_ring7", "radial_fft_8rings"),
    # directional_gabor_2x4 (16)
    ("gabor_s0_o0_mean", "directional_gabor_2x4"),
    ("gabor_s0_o0_std", "directional_gabor_2x4"),
    ("gabor_s0_o1_mean", "directional_gabor_2x4"),
    ("gabor_s0_o1_std", "directional_gabor_2x4"),
    ("gabor_s0_o2_mean", "directional_gabor_2x4"),
    ("gabor_s0_o2_std", "directional_gabor_2x4"),
    ("gabor_s0_o3_mean", "directional_gabor_2x4"),
    ("gabor_s0_o3_std", "directional_gabor_2x4"),
    ("gabor_s1_o0_mean", "directional_gabor_2x4"),
    ("gabor_s1_o0_std", "directional_gabor_2x4"),
    ("gabor_s1_o1_mean", "directional_gabor_2x4"),
    ("gabor_s1_o1_std", "directional_gabor_2x4"),
    ("gabor_s1_o2_mean", "directional_gabor_2x4"),
    ("gabor_s1_o2_std", "directional_gabor_2x4"),
    ("gabor_s1_o3_mean", "directional_gabor_2x4"),
    ("gabor_s1_o3_std", "directional_gabor_2x4"),
)


def _require_torch():
    """Import PyTorch lazily so core ``hw_score_sim`` stays torch-free."""
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for CNN probes in `hw_score_sim.realism_bench`. "
            "Install torch or run the embedding probe standalone."
        ) from exc
    return torch, nn, F


# ---------------------------------------------------------------------------
# 1. Safe Image I/O, Anti-Shortcut Canonicalizer & photo-degrade Pre-Pass
# ---------------------------------------------------------------------------


def _safe_exif_transpose(img: Image.Image) -> Image.Image:
    """Apply EXIF orientation safely without failing on malformed EXIF blocks."""
    try:
        return ImageOps.exif_transpose(img)
    except (OSError, ValueError, SyntaxError, TypeError, KeyError, ZeroDivisionError):
        return img


def _to_rgb_safe(img: Image.Image) -> Image.Image:
    """Convert any PIL image mode to RGB without alpha-blackening or 16-bit clipping.

    - Composites ``RGBA`` / ``LA`` / ``PA`` / palette-with-transparency over opaque
      white ``(255, 255, 255, 255)`` before RGB conversion.
    - Rescales 16-bit / 32-bit integer / float images (``I;16``, ``I;16B``,
      ``I;16L``, ``I;16N``, ``I``, ``F``) into ``[0, 255]`` ``uint8`` before RGB
      conversion.
    """
    mode = img.mode
    if mode == "RGB":
        return img

    if mode in ("RGBA", "LA", "PA") or (mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(bg, rgba).convert("RGB")

    if mode.startswith("I;16") or mode in ("I", "F"):
        arr = np.asarray(img, dtype=np.float64)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        if arr.size == 0:
            return img.convert("RGB")
        lo = float(arr.min())
        hi = float(arr.max())
        if mode.startswith("I;16"):
            scaled = arr * (255.0 / 65535.0) if hi > 255.0 else arr
        elif mode == "F" and lo >= 0.0 and hi <= 1.0:
            scaled = arr * 255.0
        elif lo >= 0.0 and hi <= 255.0:
            scaled = arr
        elif lo >= 0.0 and hi <= 65535.0:
            scaled = arr * (255.0 / 65535.0)
        elif hi > lo:
            scaled = (arr - lo) / (hi - lo) * 255.0
        else:
            scaled = np.zeros_like(arr)
        u8 = np.clip(np.round(scaled), 0.0, 255.0).astype(np.uint8)
        return Image.fromarray(u8, mode="L").convert("RGB")

    return img.convert("RGB")


def canonicalize_for_probe(
    img: Image.Image | str | Path,
    max_dim: int = 1024,
    jpeg_quality: int = 75,
    grayscale: bool = False,
    pre_blur_radius: float = 0.6,
) -> Image.Image:
    """Normalize real and synthetic images through an identical anti-shortcut pipeline.

    Applies safe EXIF orientation normalization, safe RGB conversion, aspect-preserving
    resize to ``max_dim``, mild shared anti-aliasing blur (``pre_blur_radius``),
    deterministic JPEG round-trip at ``jpeg_quality`` (so the shared bottleneck
    dominates over prior PNG-vs-JPEG generation history and metadata), and optional
    grayscale conversion.
    """
    if isinstance(img, (str, Path)):
        with Image.open(img) as opened:
            pil_img = _to_rgb_safe(_safe_exif_transpose(opened))
            pil_img.load()
            if pil_img is opened:
                pil_img = pil_img.copy()
    else:
        pil_img = _to_rgb_safe(_safe_exif_transpose(img))

    w, h = pil_img.size
    if w <= 0 or h <= 0:
        raise ValueError(f"Invalid image dimensions: {(w, h)}")

    if max_dim > 0:
        longest = max(w, h)
        if longest != max_dim:
            scale = float(max_dim) / float(longest)
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))
            pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    if pre_blur_radius > 0.0:
        pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=float(pre_blur_radius)))

    # Shared JPEG bottleneck erases PNG-vs-JPEG, ICC profile, and EXIF shortcuts.
    buf = io.BytesIO()
    pil_img.save(
        buf,
        format="JPEG",
        quality=int(jpeg_quality),
        subsampling=2,
        optimize=False,
    )
    buf.seek(0)
    with Image.open(buf) as reloaded:
        canonical = _to_rgb_safe(reloaded)
        canonical.load()

    if grayscale:
        return canonical.convert("L")
    return canonical


def predegrade_sim_images(
    sim_paths: Sequence[str | Path],
    photo_degrade_bin: str | Path,
    cache_dir: str | Path,
    max_width: int = 1536,
    seed: int = 42,
) -> list[Path]:
    """Run an external ``photo-degrade`` binary over synthetic pages into ``cache_dir``.

    Cleans and recreates staging/output folders, converts non-PNG/JPEG inputs to PNG,
    invokes ``photo-degrade`` with ``--suffix _degraded``, and strictly raises
    ``RuntimeError`` if any staged stem is missing from output.
    """
    if not sim_paths:
        return []
    seed_int = int(seed)
    if seed_int < 0:
        raise ValueError(f"seed must be >= 0, got {seed_int}")

    cache_path = Path(cache_dir)
    stage_dir = cache_path / "_staging"
    out_dir = cache_path / "degraded"
    shutil.rmtree(stage_dir, ignore_errors=True)
    shutil.rmtree(out_dir, ignore_errors=True)
    stage_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    staged_stems: list[str] = []
    for idx, raw_src in enumerate(sim_paths):
        src = Path(raw_src)
        stem = f"sim_{idx:05d}_{src.stem}"
        staged_stems.append(stem)
        ext = src.suffix.lower()
        if ext in {".png", ".jpg", ".jpeg"}:
            dst = stage_dir / f"{stem}{ext}"
            dst.write_bytes(src.read_bytes())
        else:
            dst = stage_dir / f"{stem}.png"
            with Image.open(src) as opened:
                rgb = _to_rgb_safe(_safe_exif_transpose(opened))
                rgb.save(dst, format="PNG")

    cmd = [
        str(photo_degrade_bin),
        "--input",
        str(stage_dir),
        "--out-dir",
        str(out_dir),
        "--format",
        "jpeg",
        "--max-width",
        str(int(max_width)),
        "--seed",
        str(seed_int),
        "--suffix",
        "_degraded",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"photo-degrade failed (exit {proc.returncode}): {proc.stderr.strip() or proc.stdout.strip()}"
        )

    out_files = sorted(
        p for p in out_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    by_stem: dict[str, Path] = {}
    for p in out_files:
        for stem in staged_stems:
            if p.stem == f"{stem}_degraded" or p.stem == stem or p.stem.startswith(f"{stem}_"):
                by_stem.setdefault(stem, p)
                break

    missing = [stem for stem in staged_stems if stem not in by_stem]
    if missing:
        raise RuntimeError(
            f"photo-degrade did not produce outputs for {len(missing)}/{len(staged_stems)} staged pages: "
            f"{missing[:5]}"
        )
    return [by_stem[stem] for stem in staged_stems]


# ---------------------------------------------------------------------------
# 2. Staff-Spacing Telemetry, Domain Stats, Patch Sampler & Letterbox
# ---------------------------------------------------------------------------


def estimate_staff_spacing_px(gray: np.ndarray, n_strips: int = 6) -> tuple[float | None, float]:
    """Estimate interline staff spacing in pixels via strip-averaged vertical-projection ACF.

    Splits the interior crop into ``n_strips`` vertical column strips, computes the
    detrended vertical-projection autocorrelation per strip, averages across strips
    (skew-invariant up to ~5 deg), and applies a sub-harmonic guard to avoid ``2*sp``
    harmonic lock. Returns ``(sp_px, peak_prominence)`` where ``sp_px`` is ``None`` if
    no reliable periodic staff peak is detected in ``[4, 40]`` px.
    """
    arr = np.asarray(gray, dtype=np.float64)
    if arr.ndim == 3:
        arr = arr.mean(axis=2)
    if arr.ndim != 2 or arr.shape[0] < 24 or arr.shape[1] < 16:
        return None, 0.0
    if float(arr.max()) > 1.5:
        arr = arr / 255.0

    h, w = arr.shape
    if h >= 64 and w >= 64:
        my = int(round(h * 0.06))
        mx = int(round(w * 0.06))
        crop = arr[my : h - my, mx : w - mx]
    else:
        crop = arr

    min_sp = 4
    max_lag = min(24, crop.shape[0] // 3)
    if max_lag < min_sp:
        return None, 0.0

    # Include lag = min_sp - 1 as a left-neighbor guard so lag == min_sp must be a true local peak.
    all_lags = np.arange(min_sp - 1, max_lag + 1, dtype=np.int64)
    lags = all_lags[1:]
    eff_strips = max(1, min(int(n_strips), max(1, crop.shape[1] // 8)))
    edges = np.linspace(0, crop.shape[1], num=eff_strips + 1, dtype=np.int64)
    win = min(19, max(5, (crop.shape[0] // 10) | 1))
    half_w = win // 2
    kernel = np.ones(win, dtype=np.float64) / float(win)

    strip_acfs: list[np.ndarray] = []
    for s_idx in range(eff_strips):
        x0, x1 = int(edges[s_idx]), int(edges[s_idx + 1])
        if x1 <= x0:
            continue
        row_dark = 1.0 - crop[:, x0:x1].mean(axis=1)
        if float(np.std(row_dark)) < 1e-4:
            continue
        padded = np.pad(row_dark, (half_w, half_w), mode="edge")
        trend = np.convolve(padded, kernel, mode="valid")
        sig = row_dark - trend
        denom = float(np.dot(sig, sig))
        if denom < 1e-8:
            continue
        s_acf = np.empty(len(all_lags), dtype=np.float64)
        for i, lag in enumerate(all_lags):
            s_acf[i] = float(np.dot(sig[:-lag], sig[lag:])) / denom
        strip_acfs.append(s_acf)

    if not strip_acfs:
        return None, 0.0

    acf_full = np.mean(np.stack(strip_acfs, axis=0), axis=0)
    acf = acf_full[1:]
    best_idx = int(np.argmax(acf))
    best_val = float(acf[best_idx])
    baseline = float(np.median(acf))
    prominence = max(0.0, best_val - baseline)

    if best_val < 0.12 or prominence < 0.08:
        return None, prominence

    # Guard against 2*sp harmonic lock: prefer the smallest true local peak >= 0.75 * best_val.
    chosen_idx = best_idx
    n_lags = len(acf)
    for i in range(n_lags):
        left_val = float(acf_full[i])
        left_ok = float(acf[i]) >= left_val
        right_ok = i == n_lags - 1 or float(acf[i]) >= float(acf[i + 1])
        if left_ok and right_ok and float(acf[i]) >= 0.75 * best_val:
            chosen_idx = i
            break

    chosen_lag = int(lags[chosen_idx])
    half_lag = chosen_lag // 2
    if half_lag >= min_sp:
        half_idx = half_lag - min_sp
        if float(acf[half_idx]) >= 0.70 * best_val:
            chosen_lag = half_lag

    return float(chosen_lag), prominence


def compute_domain_stats(
    pages: Sequence[Image.Image],
    patch_fallback_rate: float = 0.0,
    border_margin_frac: float = 0.10,
) -> dict[str, Any]:
    """Compute scale, aspect-ratio, ink-density, and notation-readability SNR telemetry."""
    sp_vals: list[float] = []
    prom_vals: list[float] = []
    aspect_ratios: list[float] = []
    landscape_flags: list[float] = []
    ink_fracs: list[float] = []
    contrasts: list[float] = []

    margin_f = max(0.0, min(0.4, float(border_margin_frac)))
    for im in pages:
        w, h = im.size
        aspect_ratios.append(float(w) / max(float(h), 1.0))
        landscape_flags.append(1.0 if w > h else 0.0)
        gray = np.asarray(_to_rgb_safe(im).convert("L"), dtype=np.float32) / 255.0
        sp_px, prom = estimate_staff_spacing_px(gray)
        if sp_px is not None:
            sp_vals.append(sp_px)
        prom_vals.append(prom)

        # Crop outer border margins before luminance/contrast telemetry.
        gh, gw = gray.shape
        my = int(round(gh * margin_f))
        mx = int(round(gw * margin_f))
        if gh - 2 * my >= 8 and gw - 2 * mx >= 8:
            inner = gray[my : gh - my, mx : gw - mx]
        else:
            inner = gray

        paper_est = float(np.percentile(inner, 85))
        thresh = min(0.75, max(0.20, paper_est - 0.15))
        ink_mask = inner < thresh
        ink_pixels = inner[ink_mask]
        ink_est = float(np.median(ink_pixels)) if ink_pixels.size > 0 else float(np.percentile(inner, 1))
        ink_fracs.append(float(np.mean(ink_mask)))
        contrasts.append((paper_est - ink_est) / max(paper_est + ink_est, 1e-6))

    ar_np = np.asarray(aspect_ratios, dtype=np.float64) if aspect_ratios else np.array([1.0])
    ink_np = np.asarray(ink_fracs, dtype=np.float64) if ink_fracs else np.array([0.0])
    c_np = np.asarray(contrasts, dtype=np.float64) if contrasts else np.array([0.0])
    p_np = np.asarray(prom_vals, dtype=np.float64) if prom_vals else np.array([0.0])

    if sp_vals:
        sp_np = np.asarray(sp_vals, dtype=np.float64)
        sp_p10, sp_p50, sp_p90 = (
            float(np.percentile(sp_np, 10)),
            float(np.percentile(sp_np, 50)),
            float(np.percentile(sp_np, 90)),
        )
    else:
        sp_p10 = sp_p50 = sp_p90 = None

    return {
        "n_pages": int(len(pages)),
        "sp_px_detected_frac": float(len(sp_vals) / max(len(pages), 1)),
        "sp_px_p10": sp_p10,
        "sp_px_p50": sp_p50,
        "sp_px_p90": sp_p90,
        "aspect_ratio_p10": float(np.percentile(ar_np, 10)),
        "aspect_ratio_p50": float(np.percentile(ar_np, 50)),
        "aspect_ratio_p90": float(np.percentile(ar_np, 90)),
        "landscape_frac": float(np.mean(landscape_flags)) if landscape_flags else 0.0,
        "ink_fraction_mean": float(ink_np.mean()),
        "ink_fraction_std": float(ink_np.std()),
        "patch_fallback_rate": float(patch_fallback_rate),
        "michelson_contrast_mean": float(c_np.mean()),
        "michelson_contrast_p10": float(np.percentile(c_np, 10)),
        "staff_peak_prominence_mean": float(p_np.mean()),
    }


def _median_border_rgb(arr: np.ndarray) -> tuple[int, int, int]:
    """Compute median RGB color along the 1px outer border of an HxWx3 uint8 array."""
    if arr.ndim != 3 or arr.shape[2] < 3 or arr.shape[0] == 0 or arr.shape[1] == 0:
        return (128, 128, 128)
    border_pixels = np.concatenate(
        [
            arr[0, :, :3],
            arr[-1, :, :3],
            arr[:, 0, :3],
            arr[:, -1, :3],
        ],
        axis=0,
    )
    med = np.median(border_pixels, axis=0)
    return (int(round(float(med[0]))), int(round(float(med[1]))), int(round(float(med[2]))))


def letterbox_page(
    img: Image.Image,
    size: int = 512,
    interior_crop_frac: float = 0.0,
    pad_mode: str = "reflect",
) -> Image.Image:
    """Aspect-preserving resize + center pad to ``size x size``.

    Defaults to ``pad_mode="reflect"`` (edge-reflected padding) so there is no
    artificial 0-variance step edge at the letterbox boundary.
    """
    rgb = _to_rgb_safe(img)
    w, h = rgb.size
    if w <= 0 or h <= 0:
        raise ValueError(f"Invalid image dimensions: {(w, h)}")
    if interior_crop_frac > 0.0:
        frac = min(0.45, float(interior_crop_frac))
        cx = int(round(w * frac))
        cy = int(round(h * frac))
        if w - 2 * cx >= 16 and h - 2 * cy >= 16:
            rgb = rgb.crop((cx, cy, w - cx, h - cy))
            w, h = rgb.size

    scale = float(size) / float(max(w, h))
    new_w = max(1, min(size, int(round(w * scale))))
    new_h = max(1, min(size, int(round(h * scale))))
    resized = rgb.resize((new_w, new_h), Image.Resampling.LANCZOS)
    ox = (size - new_w) // 2
    oy = (size - new_h) // 2
    pad_left = ox
    pad_right = size - new_w - ox
    pad_top = oy
    pad_bottom = size - new_h - oy

    arr = np.asarray(resized, dtype=np.uint8)
    if pad_mode in ("reflect", "edge"):
        eff_mode = "edge" if (pad_mode == "reflect" and (new_h < 2 or new_w < 2)) else pad_mode
        padded = np.pad(
            arr,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode=eff_mode,
        )
        return Image.fromarray(padded, mode="RGB")
    if pad_mode == "constant":
        fill_rgb = _median_border_rgb(arr)
        canvas = Image.new("RGB", (size, size), color=fill_rgb)
        canvas.paste(resized, (ox, oy))
        return canvas
    raise ValueError(f"Unknown pad_mode: {pad_mode!r}")


def sample_ink_patches(
    img: Image.Image,
    num_patches: int = 8,
    patch_size: int = 256,
    min_ink_fraction: float = 0.04,
    max_ink_fraction: float = 0.35,
    min_edge_density: float = 0.015,
    border_margin_frac: float = 0.10,
    scale_normalize: bool = True,
    target_sp_px: float = 10.0,
    ink_threshold: float | None = None,
    seed: int = 0,
    max_attempts_per_patch: int = 48,
    post_blur_radius: float = 0.6,
    post_jpeg_quality: int = 75,
    return_stats: bool = False,
) -> list[Image.Image] | tuple[list[Image.Image], dict[str, Any]]:
    """Sample ``num_patches`` interior, edge-gated, scale-normalized ink patches.

    Rejects outer border margins (``border_margin_frac``) and flat dark scanner
    gutters (via ``max_ink_fraction`` and ``min_edge_density`` in both passes).
    When ``scale_normalize=True`` and a reliable staff spacing ``sp_px`` is detected,
    scales the crop window proportionally to ``sp_px / target_sp_px`` (clamped to
    ``[0.5 * patch_size, min(h, w, 2.5 * patch_size)]``) before resizing to
    ``patch_size x patch_size``. Applies a shared post-resize ``GaussianBlur`` and
    JPEG round-trip so upscaled/downscaled patches share identical JPEG-grid period
    and optical MTF.
    """
    if num_patches <= 0:
        empty: list[Image.Image] = []
        if return_stats:
            return empty, {"fallback_count": 0, "total": 0, "crop_size": int(patch_size), "sp_px": None}
        return empty

    rgb = _to_rgb_safe(img)
    w, h = rgb.size
    gray_orig = np.asarray(rgb.convert("L"), dtype=np.float32) / 255.0

    sp_px: float | None = None
    crop_size = int(patch_size)
    if scale_normalize:
        sp_px, _ = estimate_staff_spacing_px(gray_orig)
        if sp_px is not None and sp_px > 0.0:
            desired = int(round(float(patch_size) * (sp_px / float(target_sp_px))))
            lo_sz = max(16, int(round(0.5 * patch_size)))
            hi_sz = max(lo_sz, min(h, w, int(round(2.5 * patch_size))))
            crop_size = max(lo_sz, min(hi_sz, desired))

    if w < crop_size or h < crop_size:
        arr_orig = np.asarray(rgb, dtype=np.uint8)
        fill_rgb = _median_border_rgb(arr_orig)
        pad_w = max(w, crop_size)
        pad_h = max(h, crop_size)
        canvas = Image.new("RGB", (pad_w, pad_h), color=fill_rgb)
        canvas.paste(rgb, ((pad_w - w) // 2, (pad_h - h) // 2))
        rgb = canvas
        w, h = rgb.size

    gray = np.asarray(rgb.convert("L"), dtype=np.float32) / 255.0
    if ink_threshold is None:
        paper_est = float(np.percentile(gray, 85))
        thresh = min(0.75, max(0.20, paper_est - 0.15))
    else:
        thresh = float(ink_threshold)

    dark_mask = (gray < thresh).astype(np.int32)
    integ = np.zeros((h + 1, w + 1), dtype=np.int64)
    integ[1:, 1:] = np.cumsum(np.cumsum(dark_mask, axis=0), axis=1)

    gx = np.zeros_like(gray, dtype=np.float64)
    gy = np.zeros_like(gray, dtype=np.float64)
    gx[:, 1:] = np.abs(gray[:, 1:] - gray[:, :-1])
    gy[1:, :] = np.abs(gray[1:, :] - gray[:-1, :])
    edge_map = 0.5 * (gx + gy)
    edge_integ = np.zeros((h + 1, w + 1), dtype=np.float64)
    edge_integ[1:, 1:] = np.cumsum(np.cumsum(edge_map, axis=0), axis=1)

    area = float(crop_size * crop_size)

    def _ink_frac(x0: int, y0: int) -> float:
        x1 = x0 + crop_size
        y1 = y0 + crop_size
        total = integ[y1, x1] - integ[y0, x1] - integ[y1, x0] + integ[y0, x0]
        return float(total) / area

    def _edge_density(x0: int, y0: int) -> float:
        x1 = x0 + crop_size
        y1 = y0 + crop_size
        total = edge_integ[y1, x1] - edge_integ[y0, x1] - edge_integ[y1, x0] + edge_integ[y0, x0]
        return float(total) / area

    # Restrict sampling to interior safe box.
    mx = int(round(w * max(0.0, float(border_margin_frac))))
    my = int(round(h * max(0.0, float(border_margin_frac))))
    if w - 2 * mx < crop_size:
        mx = max(0, (w - crop_size) // 2)
    if h - 2 * my < crop_size:
        my = max(0, (h - crop_size) // 2)
    min_x, max_x = mx, max(mx, w - mx - crop_size)
    min_y, max_y = my, max(my, h - my - crop_size)

    rng = np.random.default_rng(int(seed))
    coords: list[tuple[int, int]] = []
    min_sep = max(16, crop_size // 4)
    total_attempts = max(num_patches * max_attempts_per_patch, 64)

    # Pass 1: spatially separated, ink-bounded, edge-gated interior patches.
    for _ in range(total_attempts):
        if len(coords) >= num_patches:
            break
        x0 = int(rng.integers(min_x, max_x + 1)) if max_x > min_x else min_x
        y0 = int(rng.integers(min_y, max_y + 1)) if max_y > min_y else min_y
        frac = _ink_frac(x0, y0)
        if min_ink_fraction <= frac <= max_ink_fraction and _edge_density(x0, y0) >= min_edge_density:
            if all(abs(x0 - cx) + abs(y0 - cy) >= min_sep for cx, cy in coords):
                coords.append((x0, y0))

    # Pass 2: relax spatial separation, keep BOTH ink bounds and edge density gate.
    if len(coords) < num_patches:
        for _ in range(total_attempts // 2):
            if len(coords) >= num_patches:
                break
            x0 = int(rng.integers(min_x, max_x + 1)) if max_x > min_x else min_x
            y0 = int(rng.integers(min_y, max_y + 1)) if max_y > min_y else min_y
            frac = _ink_frac(x0, y0)
            if min_ink_fraction <= frac <= max_ink_fraction and _edge_density(x0, y0) >= min_edge_density:
                coords.append((x0, y0))

    # Deterministic interior center-biased fallback.
    fallback_count = 0
    cx_base = (min_x + max_x) // 2
    cy_base = (min_y + max_y) // 2
    span_x = max_x - min_x
    span_y = max_y - min_y
    while len(coords) < num_patches:
        jitter_x = int(rng.integers(-span_x // 4, span_x // 4 + 1)) if span_x >= 4 else 0
        jitter_y = int(rng.integers(-span_y // 4, span_y // 4 + 1)) if span_y >= 4 else 0
        x0 = min(max(min_x, cx_base + jitter_x), max_x)
        y0 = min(max(min_y, cy_base + jitter_y), max_y)
        coords.append((x0, y0))
        fallback_count += 1

    patches: list[Image.Image] = []
    do_post_blur = float(post_blur_radius) > 0.0
    do_post_jpeg = int(post_jpeg_quality) > 0
    for x0, y0 in coords:
        crop = rgb.crop((x0, y0, x0 + crop_size, y0 + crop_size))
        if crop_size != patch_size:
            crop = crop.resize((patch_size, patch_size), Image.Resampling.LANCZOS)
        if do_post_blur:
            crop = crop.filter(ImageFilter.GaussianBlur(radius=float(post_blur_radius)))
        if do_post_jpeg:
            buf = io.BytesIO()
            crop.save(
                buf,
                format="JPEG",
                quality=int(post_jpeg_quality),
                subsampling=2,
                optimize=False,
            )
            buf.seek(0)
            with Image.open(buf) as reloaded:
                crop = _to_rgb_safe(reloaded)
                crop.load()
                if crop is reloaded:
                    crop = crop.copy()
        patches.append(crop)

    if return_stats:
        return patches, {
            "fallback_count": int(fallback_count),
            "total": int(num_patches),
            "crop_size": int(crop_size),
            "sp_px": sp_px,
        }
    return patches


# ---------------------------------------------------------------------------
# 3. Manuscript-Level Group Split, Exact Pure-NumPy ROC-AUC & Bootstrap CI
# ---------------------------------------------------------------------------


_PAGE_SUFFIX_RE = re.compile(
    r"(?:_page\d+|_pg\d+|_p\d+|_f\d+[rv]?|_fol\d+[rv]?|_\d{1,4}|-\d{1,4})$",
    flags=re.IGNORECASE,
)


def infer_manuscript_group(
    path: str | Path,
    root: str | Path | None = None,
    domain_prefix: str = "",
    *,
    use_parent_dir: bool = True,
) -> str:
    """Infer a manuscript/work-level CV group ID so sibling pages stay in one fold.

    - If ``use_parent_dir=True`` and ``path`` is nested inside a subdirectory of
      ``root``, uses the parent subdirectory relative to ``root`` as the manuscript
      group.
    - Otherwise strips trailing page/folio suffixes (``_page\\d+``, ``_pg\\d+``,
      ``_p\\d+``, ``_f\\d+[rv]?``, ``_fol\\d+[rv]?``, ``_\\d{1,4}``, ``-\\d{1,4}``)
      from the filename stem.
    """
    p = Path(path)
    group_key = ""
    if use_parent_dir and root is not None:
        try:
            rel = p.relative_to(Path(root))
            if len(rel.parts) > 1:
                group_key = rel.parent.as_posix()
        except ValueError:
            group_key = ""

    if not group_key:
        stem = p.stem
        stripped = _PAGE_SUFFIX_RE.sub("", stem)
        group_key = stripped if stripped else stem

    return f"{domain_prefix}::{group_key}" if domain_prefix else group_key


def infer_manuscript_groups(
    paths: Sequence[str | Path],
    root: str | Path | None = None,
    domain_prefix: str = "",
) -> list[str]:
    """Infer manuscript-level CV group IDs across a batch of paths.

    Only uses parent subdirectories relative to ``root`` when ``root`` contains
    at least 2 distinct image-bearing subdirectories, preventing a single wrapper
    subfolder (e.g. ``root/rism_hw/*.jpg``) from collapsing an entire domain into
    1 group.
    """
    use_parent_dir = False
    if root is not None:
        root_path = Path(root)
        distinct_parents: set[str] = set()
        for raw_p in paths:
            try:
                rel = Path(raw_p).relative_to(root_path)
                if len(rel.parts) > 1:
                    distinct_parents.add(rel.parent.as_posix())
            except ValueError:
                continue
        use_parent_dir = len(distinct_parents) >= 2
    return [
        infer_manuscript_group(p, root=root, domain_prefix=domain_prefix, use_parent_dir=use_parent_dir)
        for p in paths
    ]


def group_kfold_indices(
    groups: Sequence[Any],
    labels: Sequence[int] | None = None,
    n_splits: int = 5,
    seed: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create strict group K-fold splits with ZERO group leakage.

    When ``labels`` are provided, performs stratified group K-fold using a running
    fold offset across classes so fold sizes differ by at most 1. Raises
    ``ValueError`` if ``min_class_count < 2``.
    """
    n_samples = len(groups)
    if n_samples == 0:
        raise ValueError("Cannot split empty dataset.")
    if labels is not None and len(labels) != n_samples:
        raise ValueError("`groups` and `labels` must have identical lengths.")
    if int(n_splits) < 2:
        raise ValueError(f"n_splits must be >= 2, got {n_splits}")

    group_to_indices: dict[Any, list[int]] = {}
    group_to_label: dict[Any, int] = {}
    for idx, g in enumerate(groups):
        group_to_indices.setdefault(g, []).append(idx)
        if labels is not None:
            group_to_label.setdefault(g, int(labels[idx]))

    unique_groups = list(group_to_indices.keys())
    if len(unique_groups) < 2:
        raise ValueError(f"Need at least 2 unique groups for K-fold split, got {len(unique_groups)}.")

    rng = np.random.default_rng(int(seed))

    if labels is not None:
        class_groups: dict[int, list[Any]] = {}
        for g in unique_groups:
            class_groups.setdefault(group_to_label[g], []).append(g)
        min_class_count = min(len(gs) for gs in class_groups.values())
        if min_class_count < 2:
            raise ValueError(
                f"Stratified group K-fold requires at least 2 unique groups per class, "
                f"got min_class_count={min_class_count}."
            )
        k = max(2, min(int(n_splits), min_class_count))
        fold_groups: list[list[Any]] = [[] for _ in range(k)]
        offset = 0
        for cls in sorted(class_groups.keys()):
            cls_list = list(class_groups[cls])
            order = rng.permutation(len(cls_list))
            for pos, idx_g in enumerate(order):
                fold_groups[(offset + pos) % k].append(cls_list[int(idx_g)])
            offset = (offset + len(cls_list)) % k
    else:
        k = max(2, min(int(n_splits), len(unique_groups)))
        fold_groups = [[] for _ in range(k)]
        order = rng.permutation(len(unique_groups))
        for pos, idx_g in enumerate(order):
            fold_groups[pos % k].append(unique_groups[int(idx_g)])

    all_indices = np.arange(n_samples, dtype=np.int64)
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for fold_idx in range(k):
        val_list: list[int] = []
        for g in fold_groups[fold_idx]:
            val_list.extend(group_to_indices[g])
        val_idx = np.sort(np.asarray(val_list, dtype=np.int64))
        val_mask = np.zeros(n_samples, dtype=bool)
        val_mask[val_idx] = True
        train_idx = all_indices[~val_mask]
        splits.append((train_idx, val_idx))

    return splits


def compute_roc_auc(
    y_true: Sequence[int] | np.ndarray,
    y_score: Sequence[float] | np.ndarray,
) -> float:
    """Compute exact ROC-AUC via Mann-Whitney U with average-rank tie handling.

    Pure NumPy implementation (zero scikit-learn dependency), numerically
    identical to ``sklearn.metrics.roc_auc_score``.
    """
    yt_raw = np.asarray(y_true, dtype=np.float64).ravel()
    ys = np.asarray(y_score, dtype=np.float64).ravel()
    if yt_raw.shape[0] != ys.shape[0] or yt_raw.shape[0] == 0:
        raise ValueError("`y_true` and `y_score` must be non-empty arrays of equal length.")
    if not np.all(np.isfinite(yt_raw)):
        raise ValueError("`y_true` contains non-finite values.")
    if not np.all((yt_raw == 0.0) | (yt_raw == 1.0)):
        raise ValueError("`y_true` must contain only binary {0, 1} labels.")
    if not np.all(np.isfinite(ys)):
        raise ValueError("`y_score` contains non-finite values.")

    yt = yt_raw.astype(np.int64)
    pos_mask = yt == 1
    n_pos = int(np.sum(pos_mask))
    n_neg = int(yt.shape[0] - n_pos)
    if n_pos == 0 or n_neg == 0:
        return 0.5

    order = np.argsort(ys, kind="mergesort")
    sorted_scores = ys[order]
    n = sorted_scores.shape[0]

    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_scores[j] == sorted_scores[i]:
            j += 1
        avg_rank = 0.5 * ((i + 1) + j)
        ranks[i:j] = avg_rank
        i = j

    inv_ranks = np.empty(n, dtype=np.float64)
    inv_ranks[order] = ranks
    rank_sum_pos = float(np.sum(inv_ranks[pos_mask]))
    u_stat = rank_sum_pos - (n_pos * (n_pos + 1)) / 2.0
    return float(np.clip(u_stat / (n_pos * n_neg), 0.0, 1.0))


def bootstrap_group_auc_ci(
    y_true: Sequence[int] | np.ndarray,
    y_score: Sequence[float] | np.ndarray,
    groups: Sequence[Any],
    n_boot: int = 500,
    seed: int = 0,
) -> tuple[float, float]:
    """Compute a stratified group-clustered 95% bootstrap confidence interval for ROC-AUC."""
    base = compute_roc_auc(y_true, y_score)
    yt = np.asarray(y_true, dtype=np.int64).ravel()
    ys = np.asarray(y_score, dtype=np.float64).ravel()
    if len(groups) != len(yt):
        raise ValueError("`groups` must match `y_true` length.")

    neg_clusters: list[np.ndarray] = []
    pos_clusters: list[np.ndarray] = []
    by_group: dict[Any, list[int]] = {}
    for idx, g in enumerate(groups):
        by_group.setdefault(g, []).append(idx)
    for g, idxs in by_group.items():
        arr_i = np.asarray(idxs, dtype=np.int64)
        if int(yt[arr_i[0]]) == 1:
            pos_clusters.append(arr_i)
        else:
            neg_clusters.append(arr_i)

    if not neg_clusters or not pos_clusters:
        return base, base

    rng = np.random.default_rng(int(seed))
    n_neg_g = len(neg_clusters)
    n_pos_g = len(pos_clusters)
    boot_aucs = np.empty(max(1, int(n_boot)), dtype=np.float64)

    for b in range(len(boot_aucs)):
        neg_pick = rng.integers(0, n_neg_g, size=n_neg_g)
        pos_pick = rng.integers(0, n_pos_g, size=n_pos_g)
        idx_list = [neg_clusters[int(i)] for i in neg_pick] + [pos_clusters[int(i)] for i in pos_pick]
        b_idx = np.concatenate(idx_list)
        boot_aucs[b] = compute_roc_auc(yt[b_idx], ys[b_idx])

    lo = float(np.percentile(boot_aucs, 2.5))
    hi = float(np.percentile(boot_aucs, 97.5))
    return float(np.clip(lo, 0.0, 1.0)), float(np.clip(hi, 0.0, 1.0))


# ---------------------------------------------------------------------------
# 4. Orthogonal 6-View Patch Ablation Ladder & Grad-CAM Utilities
# ---------------------------------------------------------------------------


def _normalize_gray_robust(gray: np.ndarray) -> np.ndarray:
    """Percentile-normalize a 2D grayscale array in ``[0, 1]`` (``p1 -> 0``, ``p90 -> 1``)."""
    p1 = float(np.percentile(gray, 1))
    p90 = float(np.percentile(gray, 90))
    if p90 - p1 < 1e-5:
        return np.full_like(gray, 0.5, dtype=np.float32)
    return np.clip((gray - p1) / (p90 - p1), 0.0, 1.0).astype(np.float32)


def _sauvola_binarize(gray: np.ndarray, window: int = 25, k: float = 0.2, r: float = 0.5) -> np.ndarray:
    """Fast pure-NumPy Sauvola local adaptive binarization via integral images.

    Returns a float32 array with ``0.0`` for ink pixels and ``1.0`` for paper.
    """
    h, w = gray.shape
    half = max(1, int(window) // 2)
    pad = np.pad(gray.astype(np.float64), ((half, half), (half, half)), mode="reflect")
    pad_sq = pad * pad

    integ = np.zeros((pad.shape[0] + 1, pad.shape[1] + 1), dtype=np.float64)
    integ[1:, 1:] = np.cumsum(np.cumsum(pad, axis=0), axis=1)
    integ_sq = np.zeros_like(integ)
    integ_sq[1:, 1:] = np.cumsum(np.cumsum(pad_sq, axis=0), axis=1)

    win_sz = 2 * half + 1
    area = float(win_sz * win_sz)
    y0 = np.arange(h)[:, None]
    y1 = y0 + win_sz
    x0 = np.arange(w)[None, :]
    x1 = x0 + win_sz

    s1 = integ[y1, x1] - integ[y0, x1] - integ[y1, x0] + integ[y0, x0]
    s2 = integ_sq[y1, x1] - integ_sq[y0, x1] - integ_sq[y1, x0] + integ_sq[y0, x0]
    mean = s1 / area
    var = np.maximum(s2 / area - mean * mean, 0.0)
    std = np.sqrt(var)

    thresh = mean * (1.0 + float(k) * (std / max(float(r), 1e-6) - 1.0))
    return (gray.astype(np.float64) >= thresh).astype(np.float32)


def transform_patch_view(patch: Image.Image, mode: str = "rgb") -> np.ndarray:
    """Convert a PIL patch into a ``(3, H, W)`` float32 array in ``[-1, 1]`` for an ablation view.

    Orthogonal 6-view ladder (plus backward-compatible ``lowfreq``/``highfreq`` aliases):
    - ``rgb``: full RGB patch
    - ``gray``: luminance-only (replicated to 3 channels)
    - ``gray_norm``: per-patch robust percentile-normalized grayscale (``p1->0, p90->1``)
    - ``lowfreq_gray`` (alias ``lowfreq``): Gaussian blur (``sigma=2.5``) of ``gray_norm``
    - ``highfreq_gray`` (alias ``highfreq``): DoG band-pass (``blur(0.7) - blur(2.5)``) of ``gray_norm``
    - ``binary``: identical Sauvola local adaptive binarization of ``gray_norm``
    """
    canonical_mode = _VIEW_ALIASES.get(mode, mode)
    rgb = _to_rgb_safe(patch)

    if canonical_mode == "rgb":
        arr = np.asarray(rgb, dtype=np.float32) / 255.0
    elif canonical_mode == "gray":
        g = np.asarray(rgb.convert("L"), dtype=np.float32) / 255.0
        arr = np.stack([g, g, g], axis=-1)
    elif canonical_mode == "gray_norm":
        g = np.asarray(rgb.convert("L"), dtype=np.float32) / 255.0
        gn = _normalize_gray_robust(g)
        arr = np.stack([gn, gn, gn], axis=-1)
    elif canonical_mode == "lowfreq_gray":
        g = np.asarray(rgb.convert("L"), dtype=np.float32) / 255.0
        gn = _normalize_gray_robust(g)
        gn_img = Image.fromarray((gn * 255.0).round().astype(np.uint8), mode="L")
        blurred = np.asarray(gn_img.filter(ImageFilter.GaussianBlur(radius=2.5)), dtype=np.float32) / 255.0
        arr = np.stack([blurred, blurred, blurred], axis=-1)
    elif canonical_mode == "highfreq_gray":
        g = np.asarray(rgb.convert("L"), dtype=np.float32) / 255.0
        gn = _normalize_gray_robust(g)
        gn_img = Image.fromarray((gn * 255.0).round().astype(np.uint8), mode="L")
        b_small = np.asarray(gn_img.filter(ImageFilter.GaussianBlur(radius=0.7)), dtype=np.float32) / 255.0
        b_large = np.asarray(gn_img.filter(ImageFilter.GaussianBlur(radius=2.5)), dtype=np.float32) / 255.0
        dog = np.clip(0.5 + (b_small - b_large) * 2.5, 0.0, 1.0)
        arr = np.stack([dog, dog, dog], axis=-1)
    elif canonical_mode == "binary":
        g = np.asarray(rgb.convert("L"), dtype=np.float32) / 255.0
        gn = _normalize_gray_robust(g)
        bin_map = _sauvola_binarize(gn, window=25, k=0.2, r=0.5)
        arr = np.stack([bin_map, bin_map, bin_map], axis=-1)
    else:
        raise ValueError(f"Unknown patch view mode: {mode!r}")

    arr = (arr - 0.5) * 2.0
    return np.ascontiguousarray(arr.transpose(2, 0, 1), dtype=np.float32)


def _jet_colormap(heatmap: np.ndarray) -> np.ndarray:
    """Map a 2D ``[0, 1]`` array to an RGB uint8 heatmap without matplotlib."""
    v = np.clip(heatmap, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * v - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * v - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * v - 1.0), 0.0, 1.0)
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255.0).round().astype(np.uint8)


def save_gradcam_overlay(
    image: Image.Image,
    heatmap: np.ndarray,
    out_path: str | Path,
    alpha: float = 0.45,
) -> Path:
    """Save a side-by-side ``[original | Grad-CAM overlay]`` PNG image atomically."""
    rgb = _to_rgb_safe(image)
    w, h = rgb.size
    hm = np.asarray(heatmap, dtype=np.float32)
    if hm.ndim > 2:
        hm = np.squeeze(hm)
    if hm.ndim != 2:
        raise ValueError(f"Expected 2D heatmap for Grad-CAM overlay, got shape {hm.shape}")
    hm = np.nan_to_num(hm, nan=0.0, posinf=1.0, neginf=0.0)
    if hm.shape != (h, w):
        hm_img = Image.fromarray((np.clip(hm, 0.0, 1.0) * 255.0).round().astype(np.uint8), mode="L")
        hm_img = hm_img.resize((w, h), Image.Resampling.BILINEAR)
        hm = np.asarray(hm_img, dtype=np.float32) / 255.0

    base = np.asarray(rgb, dtype=np.float32)
    color = _jet_colormap(hm).astype(np.float32)
    blended = np.clip((1.0 - alpha) * base + alpha * color, 0.0, 255.0).round().astype(np.uint8)

    sep = np.full((h, 4, 3), 240, dtype=np.uint8)
    side_by_side = np.concatenate([base.astype(np.uint8), sep, blended], axis=1)
    dest = Path(out_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest.with_name(dest.name + ".tmp")
    Image.fromarray(side_by_side, mode="RGB").save(tmp_path, format="PNG")
    os.replace(tmp_path, dest)
    return dest


# ---------------------------------------------------------------------------
# 5. Probe A & Probe B Neural Architectures (Lazy PyTorch)
# ---------------------------------------------------------------------------


def build_patch_forensic_cnn(in_channels: int = 3):
    """Instantiate a dilated GroupNorm ``PatchForensicCNN`` with RF >= 128 px."""
    torch, nn, _ = _require_torch()

    class PatchForensicCNN(nn.Module):
        def __init__(self, ch_in: int = 3) -> None:
            super().__init__()
            self.block1 = nn.Sequential(
                nn.Conv2d(ch_in, 24, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(8, 24),
                nn.GELU(),
                nn.MaxPool2d(2),
            )
            self.block2 = nn.Sequential(
                nn.Conv2d(24, 48, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(8, 48),
                nn.GELU(),
                nn.MaxPool2d(2),
            )
            self.block3 = nn.Sequential(
                nn.Conv2d(48, 96, kernel_size=3, padding=2, dilation=2, bias=False),
                nn.GroupNorm(8, 96),
                nn.GELU(),
                nn.Conv2d(96, 96, kernel_size=3, padding=2, dilation=2, bias=False),
                nn.GroupNorm(8, 96),
                nn.GELU(),
                nn.MaxPool2d(2),
            )
            # Last conv block kept at 32x32 (for 256x256 input) with dilation=4 for RF >= 128 px.
            self.last_conv = nn.Sequential(
                nn.Conv2d(96, 128, kernel_size=3, padding=4, dilation=4, bias=False),
                nn.GroupNorm(8, 128),
                nn.GELU(),
                nn.Conv2d(128, 128, kernel_size=3, padding=2, dilation=2, bias=False),
                nn.GroupNorm(8, 128),
                nn.GELU(),
            )
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.head = nn.Sequential(
                nn.Dropout(p=0.2),
                nn.Linear(128, 1),
            )

        def forward(self, x):
            x = self.block1(x)
            x = self.block2(x)
            x = self.block3(x)
            x = self.last_conv(x)
            x = self.pool(x).flatten(1)
            return self.head(x).squeeze(-1)

    return PatchForensicCNN(ch_in=in_channels)


def build_fullpage_forensic_cnn(in_channels: int = 3):
    """Instantiate a CoordConv + GroupNorm + Spatial Pyramid ``FullPageForensicCNN``."""
    torch, nn, F = _require_torch()

    class FullPageForensicCNN(nn.Module):
        def __init__(self, ch_in: int = 3) -> None:
            super().__init__()
            # +2 channels for normalized (x, y) CoordConv planes in [-1, 1].
            self.block1 = nn.Sequential(
                nn.Conv2d(ch_in + 2, 16, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(8, 16),
                nn.GELU(),
                nn.MaxPool2d(2),
            )
            self.block2 = nn.Sequential(
                nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(8, 32),
                nn.GELU(),
                nn.MaxPool2d(2),
            )
            self.block3 = nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(8, 64),
                nn.GELU(),
                nn.MaxPool2d(2),
            )
            self.block4 = nn.Sequential(
                nn.Conv2d(64, 96, kernel_size=3, padding=2, dilation=2, bias=False),
                nn.GroupNorm(8, 96),
                nn.GELU(),
                nn.MaxPool2d(2),
            )
            self.last_conv = nn.Sequential(
                nn.Conv2d(96, 128, kernel_size=3, padding=2, dilation=2, bias=False),
                nn.GroupNorm(8, 128),
                nn.GELU(),
            )
            # 1x1 (128) + 2x2 (512) = 640-dim spatial pyramid head.
            self.head = nn.Sequential(
                nn.Dropout(p=0.2),
                nn.Linear(128 * 5, 1),
            )

        def forward(self, x):
            b, _, h, w = x.shape
            yy = torch.linspace(-1.0, 1.0, steps=h, device=x.device, dtype=x.dtype)
            xx = torch.linspace(-1.0, 1.0, steps=w, device=x.device, dtype=x.dtype)
            grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
            coords = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0).expand(b, -1, -1, -1)
            x = torch.cat([x, coords], dim=1)
            x = self.block1(x)
            x = self.block2(x)
            x = self.block3(x)
            x = self.block4(x)
            feat = self.last_conv(x)
            p1 = F.adaptive_avg_pool2d(feat, 1).flatten(1)
            p2 = F.adaptive_avg_pool2d(feat, 2).flatten(1)
            pooled = torch.cat([p1, p2], dim=1)
            return self.head(pooled).squeeze(-1)

    return FullPageForensicCNN(ch_in=in_channels)


def PatchForensicCNN(in_channels: int = 3):
    """Construct a ``PatchForensicCNN`` instance (imports ``torch`` lazily)."""
    return build_patch_forensic_cnn(in_channels=in_channels)


def FullPageForensicCNN(in_channels: int = 3):
    """Construct a ``FullPageForensicCNN`` instance (imports ``torch`` lazily)."""
    return build_fullpage_forensic_cnn(in_channels=in_channels)


def compute_gradcam(
    model,
    x,
    target_layer=None,
    target_sign: float = 1.0,
) -> np.ndarray:
    """Compute bilateral Grad-CAM heatmaps in ``[0, 1]`` matching input ``(H, W)``.

    ``target_sign=+1.0`` highlights evidence pushing toward ``sim`` (fake);
    ``target_sign=-1.0`` highlights evidence pushing toward ``real``.
    Returns a float32 NumPy array of shape ``(B, H, W)``.
    """
    torch, _, F = _require_torch()
    layer = target_layer if target_layer is not None else model.last_conv
    activations_holder: list[Any] = []

    def _fwd_hook(_module, _inp, output):
        output.retain_grad()
        activations_holder.append(output)

    first_param = next(model.parameters(), None)
    model_dev = first_param.device if first_param is not None else x.device
    was_training = model.training
    model.eval()
    handle = layer.register_forward_hook(_fwd_hook)
    try:
        with torch.enable_grad():
            x_in = x.detach().to(model_dev).requires_grad_(True)
            model.zero_grad(set_to_none=True)
            logits = model(x_in)
            target = (float(target_sign) * logits).sum()
            target.backward()

            if not activations_holder:
                raise RuntimeError("Grad-CAM forward hook did not capture any activations.")
            acts = activations_holder[0]
            grads = acts.grad
            if grads is None:
                raise RuntimeError("Grad-CAM captured activations have no gradient.")
            # Element-wise HiResCAM: identical to Grad-CAM for GAP heads and faithful for 2x2 pyramid heads.
            cam = F.relu((grads * acts).sum(dim=1, keepdim=True))
            cam = F.interpolate(
                cam,
                size=(x_in.shape[2], x_in.shape[3]),
                mode="bilinear",
                align_corners=False,
            )
            cam_np = cam.squeeze(1).detach().cpu().numpy().astype(np.float32)
    finally:
        handle.remove()
        activations_holder.clear()
        model.zero_grad(set_to_none=True)
        model.train(was_training)

    cam_np = np.nan_to_num(cam_np, nan=0.0, posinf=1.0, neginf=0.0)
    out = np.zeros_like(cam_np, dtype=np.float32)
    for i in range(cam_np.shape[0]):
        c = cam_np[i]
        c_min = float(c.min())
        c_max = float(c.max())
        if c_max - c_min > 1e-8:
            out[i] = (c - c_min) / (c_max - c_min)
        else:
            out[i] = 0.0
    return out


def _resolve_device(device: str):
    torch, _, _ = _require_torch()
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _augment_batch_tensor(xb, rng: np.random.Generator, allow_hflip: bool = True, max_shift: int = 4):
    """Apply lightweight deterministic spatial augmentation (replicate shift + optional h-flip)."""
    torch, _, F = _require_torch()
    out = xb
    if allow_hflip and bool(rng.integers(0, 2)):
        out = torch.flip(out, dims=[3])
    if max_shift > 0:
        sy = int(rng.integers(-max_shift, max_shift + 1))
        sx = int(rng.integers(-max_shift, max_shift + 1))
        if sy != 0 or sx != 0:
            _, _, h, w = out.shape
            padded = F.pad(out, (max_shift, max_shift, max_shift, max_shift), mode="replicate")
            y0 = max_shift - sy
            x0 = max_shift - sx
            out = padded[:, :, y0 : y0 + h, x0 : x0 + w]
    return out


def _train_and_eval_cnn_kfold(
    tensors: np.ndarray,
    labels: np.ndarray,
    groups: Sequence[Any],
    model_factory: Callable[[], Any],
    *,
    folds: int = 5,
    epochs: int = 8,
    batch_size: int = 16,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str = "auto",
    seed: int = 42,
    no_cudnn: bool = False,
    allow_hflip: bool = True,
    top_k_gradcam: int = 0,
    source_images: Sequence[Image.Image] | None = None,
    gradcam_dir: Path | None = None,
    gradcam_prefix: str = "sample",
) -> dict[str, Any]:
    """Train a binary CNN with strict group K-fold CV, telemetry, and bilateral Grad-CAM."""
    torch, nn, _ = _require_torch()
    prev_cudnn = bool(torch.backends.cudnn.enabled)
    if no_cudnn:
        torch.backends.cudnn.enabled = False

    try:
        dev = _resolve_device(device)
        splits = group_kfold_indices(groups=groups, labels=labels, n_splits=folds, seed=seed)
        oof_logits = np.zeros(len(labels), dtype=np.float64)
        oof_probs = np.zeros(len(labels), dtype=np.float64)
        val_models_by_idx: dict[int, tuple[int, Any]] = {}
        fold_metrics: list[dict[str, Any]] = []

        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            fold_seed = int(seed) + fold_idx * 97
            torch.manual_seed(fold_seed)
            if dev.type == "cuda":
                torch.cuda.manual_seed_all(fold_seed)

            model = model_factory().to(dev)
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
            n_epochs = max(1, int(epochs))
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

            y_train = labels[train_idx].astype(np.float32)
            n_pos = float(np.sum(y_train == 1.0))
            n_neg = float(np.sum(y_train == 0.0))
            pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], dtype=torch.float32, device=dev)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

            rng = np.random.default_rng(int(seed) + fold_idx * 131)
            bs = max(1, int(batch_size))
            last_train_loss = 0.0
            for _epoch in range(n_epochs):
                model.train()
                perm = train_idx[rng.permutation(len(train_idx))]
                epoch_loss_sum = 0.0
                epoch_count = 0
                for start in range(0, len(perm), bs):
                    batch_ids = perm[start : start + bs]
                    xb = torch.from_numpy(tensors[batch_ids]).to(dev)
                    xb = _augment_batch_tensor(xb, rng, allow_hflip=allow_hflip, max_shift=4)
                    yb = torch.from_numpy(labels[batch_ids].astype(np.float32)).to(dev)
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(xb)
                    loss = criterion(logits, yb)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(
                            f"Non-finite training loss in fold {fold_idx} epoch {_epoch}: {float(loss.detach().cpu().item())}"
                        )
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    optimizer.step()
                    epoch_loss_sum += float(loss.detach().cpu().item()) * len(batch_ids)
                    epoch_count += len(batch_ids)
                scheduler.step()
                last_train_loss = epoch_loss_sum / max(epoch_count, 1)

            model.eval()
            train_logits_fold = np.zeros(len(train_idx), dtype=np.float64)
            val_logits_fold = np.zeros(len(val_idx), dtype=np.float64)
            val_loss_sum = 0.0
            with torch.no_grad():
                for start in range(0, len(train_idx), bs):
                    batch_ids = train_idx[start : start + bs]
                    xb = torch.from_numpy(tensors[batch_ids]).to(dev)
                    lg = model(xb).detach().cpu().numpy()
                    train_logits_fold[start : start + len(batch_ids)] = lg

                for start in range(0, len(val_idx), bs):
                    batch_ids = val_idx[start : start + bs]
                    xb = torch.from_numpy(tensors[batch_ids]).to(dev)
                    yb = torch.from_numpy(labels[batch_ids].astype(np.float32)).to(dev)
                    lg_t = model(xb)
                    val_loss_t = criterion(lg_t, yb)
                    if not torch.isfinite(lg_t).all() or not torch.isfinite(val_loss_t):
                        raise FloatingPointError(
                            f"Non-finite validation logits or loss in fold {fold_idx}."
                        )
                    val_loss_sum += float(val_loss_t.detach().cpu().item()) * len(batch_ids)
                    lg_np = lg_t.detach().cpu().numpy()
                    val_logits_fold[start : start + len(batch_ids)] = lg_np
                    oof_logits[batch_ids] = lg_np
                    oof_probs[batch_ids] = torch.sigmoid(lg_t).detach().cpu().numpy()

            fold_train_auc = compute_roc_auc(labels[train_idx], train_logits_fold)
            fold_val_auc = compute_roc_auc(labels[val_idx], val_logits_fold)
            fold_val_loss = val_loss_sum / max(len(val_idx), 1)
            fold_metrics.append(
                {
                    "fold": int(fold_idx),
                    "train_auc": float(fold_train_auc),
                    "val_auc": float(fold_val_auc),
                    "train_loss": float(last_train_loss),
                    "val_loss": float(fold_val_loss),
                }
            )

            if top_k_gradcam > 0 and source_images is not None and gradcam_dir is not None:
                model_cpu = model.to("cpu")
                for idx in val_idx:
                    val_models_by_idx[int(idx)] = (fold_idx, model_cpu)

            del optimizer, criterion, pos_weight, model
            if dev.type == "cuda":
                torch.cuda.empty_cache()

        pooled_auc = compute_roc_auc(labels, oof_logits)
        ci_low, ci_high = bootstrap_group_auc_ci(labels, oof_logits, groups, n_boot=500, seed=seed)
        val_aucs = [m["val_auc"] for m in fold_metrics]
        train_aucs = [m["train_auc"] for m in fold_metrics]
        auc_mean = float(np.mean(val_aucs)) if val_aucs else pooled_auc
        auc_std = float(np.std(val_aucs)) if val_aucs else 0.0
        mean_train_auc = float(np.mean(train_aucs)) if train_aucs else 0.5

        underfit_warning = bool(mean_train_auc < 0.60 and auc_mean < 0.58)
        overfit_warning = bool((mean_train_auc - auc_mean) > 0.20)

        gradcam_files: list[dict[str, Any]] = []
        gradcam_real_files: list[dict[str, Any]] = []

        if top_k_gradcam > 0 and source_images is not None and gradcam_dir is not None:
            # Sim "fake" top-K (+1.0 target_sign)
            sim_indices = np.where(labels == 1)[0]
            ranked_sim = sim_indices[np.argsort(-oof_logits[sim_indices])]
            for rank, sample_idx in enumerate(ranked_sim[: int(top_k_gradcam)]):
                sample_i = int(sample_idx)
                _, model_cpu = val_models_by_idx[sample_i]
                x_single = torch.from_numpy(tensors[sample_i : sample_i + 1])
                heatmap = compute_gradcam(model_cpu, x_single, target_sign=1.0)[0]
                out_png = gradcam_dir / f"{gradcam_prefix}_sim_top{rank + 1:02d}_p{oof_probs[sample_i]:.3f}.png"
                save_gradcam_overlay(source_images[sample_i], heatmap, out_png)
                gradcam_files.append(
                    {
                        "rank": rank + 1,
                        "direction": "sim_fake",
                        "sample_index": sample_i,
                        "group": str(groups[sample_i]),
                        "sim_prob": float(oof_probs[sample_i]),
                        "path": str(out_png),
                    }
                )

            # Real "missing real cues" top-K (-1.0 target_sign)
            real_indices = np.where(labels == 0)[0]
            ranked_real = real_indices[np.argsort(oof_logits[real_indices])]
            for rank, sample_idx in enumerate(ranked_real[: int(top_k_gradcam)]):
                sample_i = int(sample_idx)
                _, model_cpu = val_models_by_idx[sample_i]
                x_single = torch.from_numpy(tensors[sample_i : sample_i + 1])
                heatmap = compute_gradcam(model_cpu, x_single, target_sign=-1.0)[0]
                out_png = gradcam_dir / f"{gradcam_prefix}_real_top{rank + 1:02d}_p{oof_probs[sample_i]:.3f}.png"
                save_gradcam_overlay(source_images[sample_i], heatmap, out_png)
                gradcam_real_files.append(
                    {
                        "rank": rank + 1,
                        "direction": "real_cue",
                        "sample_index": sample_i,
                        "group": str(groups[sample_i]),
                        "sim_prob": float(oof_probs[sample_i]),
                        "path": str(out_png),
                    }
                )

        return {
            "roc_auc": float(pooled_auc),
            "pooled_oof_auc": float(pooled_auc),
            "auc_mean": float(auc_mean),
            "auc_std": float(auc_std),
            "auc_ci_low": float(ci_low),
            "auc_ci_high": float(ci_high),
            "n_samples": int(len(labels)),
            "n_folds": int(len(splits)),
            "mean_prob_real": float(oof_probs[labels == 0].mean()) if np.any(labels == 0) else 0.5,
            "mean_prob_sim": float(oof_probs[labels == 1].mean()) if np.any(labels == 1) else 0.5,
            "fold_metrics": fold_metrics,
            "underfit_warning": underfit_warning,
            "overfit_warning": overfit_warning,
            "gradcam_files": gradcam_files,
            "gradcam_real_files": gradcam_real_files,
        }
    finally:
        torch.backends.cudnn.enabled = prev_cudnn


# ---------------------------------------------------------------------------
# 6. Probe C — Frozen Feature Linear Probe, Standardized FD & Diversity Metrics
# ---------------------------------------------------------------------------


def extract_default_embedding(pil_img: Image.Image) -> np.ndarray:
    """Pure-NumPy multi-scale Gabor/gradient/spectral/color-moment descriptor (55 dims).

    Uses ``norm="ortho"`` for the 2D FFT so radial spectral log-power is O(1)
    commensurate with color, gradient, and Gabor features.
    """
    rgb_256 = _to_rgb_safe(pil_img).resize((256, 256), Image.Resampling.BILINEAR)
    arr = np.asarray(rgb_256, dtype=np.float64) / 255.0

    feats: list[float] = []

    # 1. Color moments (19 dims)
    for c in range(3):
        ch = arr[:, :, c]
        feats.extend(
            [
                float(ch.mean()),
                float(ch.std()),
                float(np.percentile(ch, 10)),
                float(np.percentile(ch, 50)),
                float(np.percentile(ch, 90)),
            ]
        )
    rg = arr[:, :, 0] - arr[:, :, 1]
    yb = 0.5 * (arr[:, :, 0] + arr[:, :, 1]) - arr[:, :, 2]
    feats.extend([float(rg.mean()), float(rg.std()), float(yb.mean()), float(yb.std())])

    # 2. Gradient magnitude & 8-bin orientation histogram (12 dims)
    gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    gx = gray[:, 2:] - gray[:, :-2]
    gy = gray[2:, :] - gray[:-2, :]
    gx = gx[1:-1, :]
    gy = gy[:, 1:-1]
    mag = np.hypot(gx, gy)
    ang = np.mod(np.arctan2(gy, gx), math.pi)
    feats.extend(
        [
            float(mag.mean()),
            float(mag.std()),
            float(np.percentile(mag, 75)),
            float(np.percentile(mag, 95)),
        ]
    )
    hist, _ = np.histogram(ang, bins=8, range=(0.0, math.pi), weights=mag)
    hist_sum = float(hist.sum()) + 1e-8
    feats.extend((hist / hist_sum).tolist())

    # 3. Radial FFT log-power spectrum (8 log-spaced rings) with ortho normalization
    gray_128 = (
        np.asarray(
            rgb_256.convert("L").resize((128, 128), Image.Resampling.BILINEAR),
            dtype=np.float64,
        )
        / 255.0
    )
    g_zero = gray_128 - gray_128.mean()
    fft2 = np.fft.fftshift(np.fft.fft2(g_zero, norm="ortho"))
    power = np.log1p(np.abs(fft2) ** 2)

    yy, xx = np.mgrid[-64:64, -64:64]
    rr = np.hypot(xx, yy)
    ring_edges = np.geomspace(1.0, 64.0, num=9)
    for r0, r1 in zip(ring_edges[:-1], ring_edges[1:]):
        mask = (rr >= r0) & (rr < r1)
        feats.append(float(power[mask].mean()) if np.any(mask) else 0.0)

    # 4. Frequency-domain Gabor-like directional filter bank (2 scales x 4 orientations = 16 dims)
    theta = np.arctan2(yy, xx)
    for freq, sigma_r in ((10.0, 4.0), (26.0, 8.0)):
        radial = np.exp(-0.5 * ((rr - freq) / sigma_r) ** 2)
        for k in range(4):
            ang0 = k * (math.pi / 4.0)
            d_ang = np.minimum(
                np.abs(np.mod(theta - ang0 + math.pi / 2.0, math.pi) - math.pi / 2.0),
                math.pi / 2.0,
            )
            angular = np.exp(-0.5 * (d_ang / 0.35) ** 2)
            resp = np.fft.ifft2(np.fft.ifftshift(fft2 * radial * angular), norm="ortho")
            resp_mag = np.abs(resp)
            feats.extend([float(resp_mag.mean()), float(resp_mag.std())])

    vec = np.asarray(feats, dtype=np.float64)
    return np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)


def compute_frechet_distance(
    emb_real: np.ndarray,
    emb_sim: np.ndarray,
    eps: float = 1e-5,
    shrinkage: float = 0.0,
) -> float:
    """Compute regularized Frechet Distance purely in NumPy with optional diagonal shrinkage."""
    xr = np.asarray(emb_real, dtype=np.float64)
    xs = np.asarray(emb_sim, dtype=np.float64)
    if xr.ndim != 2 or xs.ndim != 2:
        raise ValueError(f"Expected 2D embedding matrices, got shapes {xr.shape} and {xs.shape}.")
    if xr.shape[0] == 0 or xs.shape[0] == 0 or xr.shape[1] == 0:
        raise ValueError("Embedding matrices must be non-empty.")
    if xr.shape[1] != xs.shape[1]:
        raise ValueError(f"Embedding dimension mismatch: {xr.shape[1]} vs {xs.shape[1]}")
    if not np.all(np.isfinite(xr)) or not np.all(np.isfinite(xs)):
        raise ValueError("Embedding matrices contain non-finite values.")
    if not math.isfinite(float(eps)) or float(eps) < 0.0:
        raise ValueError(f"eps must be finite and >= 0.0, got {eps}")
    if not (0.0 <= float(shrinkage) <= 1.0):
        raise ValueError(f"shrinkage must be in [0, 1], got {shrinkage}")

    d = xr.shape[1]
    mu_r = xr.mean(axis=0)
    mu_s = xs.mean(axis=0)

    eye = np.eye(d, dtype=np.float64)
    cov_r = np.cov(xr, rowvar=False) if xr.shape[0] > 1 else np.zeros((d, d), dtype=np.float64)
    cov_s = np.cov(xs, rowvar=False) if xs.shape[0] > 1 else np.zeros((d, d), dtype=np.float64)
    cov_r = np.atleast_2d(cov_r)
    cov_s = np.atleast_2d(cov_s)

    s = float(shrinkage)
    if s > 0.0:
        cov_r = (1.0 - s) * cov_r + s * (float(np.trace(cov_r)) / d) * eye
        cov_s = (1.0 - s) * cov_s + s * (float(np.trace(cov_s)) / d) * eye

    cov_r = cov_r + float(eps) * eye
    cov_s = cov_s + float(eps) * eye
    cov_r = 0.5 * (cov_r + cov_r.T)
    cov_s = 0.5 * (cov_s + cov_s.T)

    vals_r, vecs_r = np.linalg.eigh(cov_r)
    sqrt_r = (vecs_r * np.sqrt(np.maximum(vals_r, 0.0))) @ vecs_r.T
    prod = sqrt_r @ cov_s @ sqrt_r
    prod = 0.5 * (prod + prod.T)
    vals_prod = np.linalg.eigvalsh(prod)
    tr_sqrt = float(np.sum(np.sqrt(np.maximum(vals_prod, 0.0))))

    diff = mu_r - mu_s
    fd = float(np.dot(diff, diff) + np.trace(cov_r) + np.trace(cov_s) - 2.0 * tr_sqrt)
    if not math.isfinite(fd):
        raise FloatingPointError(f"Non-finite Frechet distance: {fd}")
    return max(0.0, fd)


def _fit_logistic_irls(
    x_train: np.ndarray,
    y_train: np.ndarray,
    l2_reg: float = 1.0,
    max_iter: int = 30,
    tol: float = 1e-6,
) -> tuple[np.ndarray, float]:
    """Fit L2-regularized binary logistic regression via Newton-IRLS in pure NumPy.

    Includes Armijo backtracking line search and Woodbury identity acceleration
    when ``d > n`` and ``l2_reg > 0`` for O(N^2 D + N^3) scaling.
    """
    reg_val = float(l2_reg)
    if reg_val < 0.0:
        raise ValueError(f"l2_reg must be >= 0.0, got {reg_val}")

    n, d = x_train.shape
    xb = np.hstack([x_train, np.ones((n, 1), dtype=np.float64)])
    w = np.zeros(d + 1, dtype=np.float64)
    reg_diag = np.full(d + 1, reg_val, dtype=np.float64)
    reg_diag[d] = 1e-6  # Do not strongly penalize intercept.

    def _objective(w_vec: np.ndarray) -> float:
        z_v = xb @ w_vec
        nll = float(np.sum(np.logaddexp(0.0, z_v) - y_train * z_v))
        pen = 0.5 * float(np.sum(reg_diag * (w_vec * w_vec)))
        return nll + pen

    use_woodbury = d > n and reg_val > 0.0
    d_inv = (1.0 / reg_diag) if use_woodbury else None

    for _ in range(max_iter):
        z = np.clip(xb @ w, -35.0, 35.0)
        p = 1.0 / (1.0 + np.exp(-z))
        weights = np.maximum(p * (1.0 - p), 1e-6)
        grad = xb.T @ (p - y_train) + reg_diag * w

        if use_woodbury and d_inv is not None:
            v = d_inv * grad
            x_dinv = xb * d_inv[None, :]
            m_mat = np.diag(1.0 / weights) + x_dinv @ xb.T
            rhs = xb @ v
            try:
                u = np.linalg.solve(m_mat, rhs)
            except np.linalg.LinAlgError:
                u = np.linalg.lstsq(m_mat, rhs, rcond=None)[0]
            step = v - (x_dinv.T @ u)
        else:
            hess = (xb.T * weights) @ xb + np.diag(reg_diag)
            try:
                step = np.linalg.solve(hess, grad)
            except np.linalg.LinAlgError:
                step = np.linalg.lstsq(hess, grad, rcond=None)[0]

        # Armijo backtracking line search
        obj_cur = _objective(w)
        dir_deriv = -float(np.dot(grad, step))
        alpha = 1.0
        w_new = w
        if dir_deriv < 0.0:
            while alpha >= 1e-4:
                w_try = w - alpha * step
                if _objective(w_try) <= obj_cur + 1e-4 * alpha * dir_deriv:
                    w_new = w_try
                    break
                alpha *= 0.5

        delta = float(np.max(np.abs(w_new - w)))
        w = w_new
        if delta < tol:
            break

    return w[:-1], float(w[-1])


def _pairwise_euclidean(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute exact pairwise Euclidean distance matrix between rows of ``a`` and ``b``."""
    a2 = np.sum(a * a, axis=1, keepdims=True)
    b2 = np.sum(b * b, axis=1, keepdims=True).T
    sq = np.maximum(a2 + b2 - 2.0 * (a @ b.T), 0.0)
    return np.sqrt(sq)


def compute_knn_prdc(
    emb_real: np.ndarray,
    emb_sim: np.ndarray,
    k: int = 5,
    real_groups: Sequence[Any] | None = None,
    real_paths: Sequence[str | Path] | None = None,
    top_k_uncovered: int = 5,
) -> dict[str, Any]:
    """Compute k-NN Precision, Recall, Density, Coverage, variance ratio & uncovered real pages."""
    n_r = emb_real.shape[0]
    n_s = emb_sim.shape[0]
    if n_r < 2 or n_s < 2:
        return {
            "k": 1,
            "precision": 0.0,
            "recall": 0.0,
            "density": 0.0,
            "coverage": 0.0,
            "variance_ratio": 1.0,
            "top_uncovered_real": [],
        }

    k_eff = max(1, min(int(k), n_r - 1, n_s - 1))
    d_rr = _pairwise_euclidean(emb_real, emb_real)
    np.fill_diagonal(d_rr, np.inf)
    r_real = np.partition(d_rr, k_eff - 1, axis=1)[:, k_eff - 1]

    d_ss = _pairwise_euclidean(emb_sim, emb_sim)
    np.fill_diagonal(d_ss, np.inf)
    r_sim = np.partition(d_ss, k_eff - 1, axis=1)[:, k_eff - 1]

    d_sr = _pairwise_euclidean(emb_sim, emb_real)  # (n_s, n_r)
    d_rs = d_sr.T  # (n_r, n_s)

    in_real_ball = d_sr <= r_real[None, :]
    in_sim_ball = d_rs <= r_sim[None, :]

    precision = float(np.mean(np.any(in_real_ball, axis=1)))
    recall = float(np.mean(np.any(in_sim_ball, axis=1)))
    density = float(np.sum(in_real_ball) / max(float(k_eff * n_s), 1.0))
    nn_real_to_sim = np.min(d_rs, axis=1)
    coverage = float(np.mean(nn_real_to_sim <= r_real))

    tr_r = float(np.sum(np.var(emb_real, axis=0, ddof=1)))
    tr_s = float(np.sum(np.var(emb_sim, axis=0, ddof=1)))
    variance_ratio = float(tr_s / max(tr_r, 1e-8))

    uncovered_order = np.argsort(-nn_real_to_sim)
    g_r = list(real_groups) if real_groups is not None else [f"real_{i}" for i in range(n_r)]
    p_r = [str(p) for p in real_paths] if real_paths is not None else None
    top_uncovered = [
        {
            "index": int(idx),
            "group": str(g_r[int(idx)]),
            "path": p_r[int(idx)] if p_r is not None and int(idx) < len(p_r) else None,
            "nn_dist": float(nn_real_to_sim[int(idx)]),
            "r_real": float(r_real[int(idx)]),
            "covered": bool(nn_real_to_sim[int(idx)] <= r_real[int(idx)]),
        }
        for idx in uncovered_order[: max(0, int(top_k_uncovered))]
    ]

    return {
        "k": int(k_eff),
        "precision": precision,
        "recall": recall,
        "density": density,
        "coverage": coverage,
        "variance_ratio": variance_ratio,
        "top_uncovered_real": top_uncovered,
    }


def run_embedding_probe(
    real_images: Sequence[Image.Image] | None = None,
    sim_images: Sequence[Image.Image] | None = None,
    *,
    embed_fn: Callable[[Image.Image], np.ndarray] | None = None,
    real_embeddings: np.ndarray | None = None,
    sim_embeddings: np.ndarray | None = None,
    real_groups: Sequence[Any] | None = None,
    sim_groups: Sequence[Any] | None = None,
    real_paths: Sequence[str | Path] | None = None,
    feature_spec: Sequence[tuple[str, str]] | None = None,
    folds: int = 5,
    l2_reg: float = 1.0,
    fd_eps: float = 1e-5,
    knn_k: int = 5,
    seed: int = 42,
) -> dict[str, Any]:
    """Run Probe C: L2 logistic linear probe, raw/standardized FD, PRDC & feature attribution."""
    used_default = False
    if real_embeddings is None or sim_embeddings is None:
        if real_images is None or sim_images is None:
            raise ValueError("Provide either (real_images, sim_images) or (real_embeddings, sim_embeddings).")
        used_default = embed_fn is None
        fn = embed_fn if embed_fn is not None else extract_default_embedding
        emb_r = np.stack([np.asarray(fn(im), dtype=np.float64).ravel() for im in real_images], axis=0)
        emb_s = np.stack([np.asarray(fn(im), dtype=np.float64).ravel() for im in sim_images], axis=0)
    else:
        emb_r = np.asarray(real_embeddings, dtype=np.float64)
        emb_s = np.asarray(sim_embeddings, dtype=np.float64)

    if emb_r.ndim != 2 or emb_s.ndim != 2:
        raise ValueError("Embeddings must be 2D matrices.")
    n_r, d = emb_r.shape
    n_s = emb_s.shape[0]
    if n_r == 0 or n_s == 0:
        raise ValueError("Both real and sim embedding sets must be non-empty.")

    x_all = np.vstack([emb_r, emb_s])
    y_all = np.concatenate([np.zeros(n_r, dtype=np.int64), np.ones(n_s, dtype=np.int64)])
    g_r = list(real_groups) if real_groups is not None else [f"real_{i}" for i in range(n_r)]
    g_s = list(sim_groups) if sim_groups is not None else [f"sim_{i}" for i in range(n_s)]
    groups_all = g_r + g_s

    splits = group_kfold_indices(groups=groups_all, labels=y_all, n_splits=folds, seed=seed)
    oof_logits = np.zeros(len(y_all), dtype=np.float64)
    fold_aucs: list[float] = []
    fold_weights: list[np.ndarray] = []

    for train_idx, val_idx in splits:
        x_tr = x_all[train_idx]
        y_tr = y_all[train_idx].astype(np.float64)
        x_va = x_all[val_idx]

        mean = x_tr.mean(axis=0)
        std = x_tr.std(axis=0)
        std = np.where(std > 1e-8, std, 1.0)
        x_tr_norm = (x_tr - mean) / std
        x_va_norm = (x_va - mean) / std

        w, b = _fit_logistic_irls(x_tr_norm, y_tr, l2_reg=l2_reg)
        logits = x_va_norm @ w + b
        oof_logits[val_idx] = logits
        fold_aucs.append(compute_roc_auc(y_all[val_idx], logits))
        fold_weights.append(w)

    # Score raw OOF logits directly (avoids +/-30 sigmoid saturation ties).
    pooled_auc = compute_roc_auc(y_all, oof_logits)
    ci_low, ci_high = bootstrap_group_auc_ci(y_all, oof_logits, groups_all, n_boot=500, seed=seed)
    raw_fd = compute_frechet_distance(emb_r, emb_s, eps=fd_eps, shrinkage=0.0)

    # Standardized space (by real mean/std) with auto diagonal shrinkage when N < 5*D.
    mu_r = emb_r.mean(axis=0)
    std_r = emb_r.std(axis=0)
    std_r = np.where(std_r > 1e-8, std_r, 1.0)
    emb_r_std = (emb_r - mu_r) / std_r
    emb_s_std = (emb_s - mu_r) / std_r

    min_n = min(n_r, n_s)
    auto_shrinkage = min(0.5, max(0.1, 1.0 - float(min_n) / float(5 * d))) if min_n < 5 * d else 0.0
    std_fd = compute_frechet_distance(emb_r_std, emb_s_std, eps=fd_eps, shrinkage=auto_shrinkage)
    prdc = compute_knn_prdc(emb_r_std, emb_s_std, k=knn_k, real_groups=g_r, real_paths=real_paths)

    # Feature attribution
    w_stack = np.stack(fold_weights, axis=0)
    mean_w = w_stack.mean(axis=0)
    mean_abs_w = np.abs(w_stack).mean(axis=0)
    std_shift = emb_s_std.mean(axis=0) - emb_r_std.mean(axis=0)

    spec: Sequence[tuple[str, str]]
    if feature_spec is not None and len(feature_spec) == d:
        spec = feature_spec
    elif (used_default or feature_spec is None) and d == len(DEFAULT_EMBEDDING_FEATURE_SPEC):
        spec = DEFAULT_EMBEDDING_FEATURE_SPEC
    else:
        spec = [(f"feat_{i}", "embedding") for i in range(d)]

    group_buckets: dict[str, list[int]] = {}
    for idx, (_fname, gname) in enumerate(spec):
        group_buckets.setdefault(gname, []).append(idx)

    group_attribution: dict[str, dict[str, float]] = {}
    for gname, idxs in group_buckets.items():
        idx_arr = np.asarray(idxs, dtype=np.int64)
        group_attribution[gname] = {
            "mean_abs_weight": float(mean_abs_w[idx_arr].mean()),
            "max_abs_weight": float(mean_abs_w[idx_arr].max()),
            "mean_std_shift": float(std_shift[idx_arr].mean()),
            "mean_abs_std_shift": float(np.abs(std_shift[idx_arr]).mean()),
        }

    top_order = np.argsort(-mean_abs_w)[:10]
    top_features = [
        {
            "name": spec[int(i)][0],
            "group": spec[int(i)][1],
            "mean_weight": float(mean_w[int(i)]),
            "mean_abs_weight": float(mean_abs_w[int(i)]),
            "std_mean_shift": float(std_shift[int(i)]),
        }
        for i in top_order
    ]

    return {
        "roc_auc": float(pooled_auc),
        "pooled_oof_auc": float(pooled_auc),
        "auc_mean": float(np.mean(fold_aucs)) if fold_aucs else pooled_auc,
        "auc_std": float(np.std(fold_aucs)) if fold_aucs else 0.0,
        "auc_ci_low": float(ci_low),
        "auc_ci_high": float(ci_high),
        "frechet_distance": float(raw_fd),
        "raw_fd": float(raw_fd),
        "standardized_fd": float(std_fd),
        "fd_shrinkage": float(auto_shrinkage),
        "precision": float(prdc["precision"]),
        "recall": float(prdc["recall"]),
        "density": float(prdc["density"]),
        "coverage": float(prdc["coverage"]),
        "variance_ratio": float(prdc["variance_ratio"]),
        "top_uncovered_real": prdc["top_uncovered_real"],
        "feature_attribution": {
            "groups": group_attribution,
            "top_features": top_features,
        },
        "embedding_dim": int(d),
        "n_real": int(n_r),
        "n_sim": int(n_s),
        "n_folds": int(len(splits)),
    }


def run_null_control(
    real_embeddings: np.ndarray,
    real_groups: Sequence[Any],
    folds: int = 5,
    seed: int = 42,
    n_repeats: int = 5,
) -> dict[str, Any]:
    """Run a repeated Real-vs-Real 50/50 manuscript-grouped split null control on Probe C."""
    by_group: dict[Any, list[int]] = {}
    for idx, g in enumerate(real_groups):
        by_group.setdefault(g, []).append(idx)
    unique_groups = sorted(by_group.keys(), key=str)
    if len(unique_groups) < 4:
        return {
            "skipped": True,
            "passed": True,
            "reason": f"Need >= 4 unique real groups for 50/50 null control, got {len(unique_groups)}.",
            "n_repeats": 0,
            "repeat_aucs": [],
            "null_auc": 0.5,
            "null_auc_ci_low": 0.0,
            "null_auc_ci_high": 1.0,
            "null_standardized_fd": 0.0,
            "null_coverage": 1.0,
        }

    repeats = max(1, int(n_repeats))
    null_aucs: list[float] = []
    ci_lows: list[float] = []
    ci_highs: list[float] = []
    std_fds: list[float] = []
    covs: list[float] = []

    half = len(unique_groups) // 2
    for rep in range(repeats):
        rng = np.random.default_rng(int(seed) + 777 + rep * 31)
        perm = rng.permutation(len(unique_groups))
        g_a = [unique_groups[int(i)] for i in perm[:half]]
        g_b = [unique_groups[int(i)] for i in perm[half:]]

        idx_a = [i for g in g_a for i in by_group[g]]
        idx_b = [i for g in g_b for i in by_group[g]]
        groups_a = [f"nullA::{real_groups[i]}" for i in idx_a]
        groups_b = [f"nullB::{real_groups[i]}" for i in idx_b]

        res = run_embedding_probe(
            real_embeddings=real_embeddings[idx_a],
            sim_embeddings=real_embeddings[idx_b],
            real_groups=groups_a,
            sim_groups=groups_b,
            folds=min(int(folds), len(g_a), len(g_b)),
            seed=int(seed) + 778 + rep * 31,
        )
        null_aucs.append(float(res["pooled_oof_auc"]))
        ci_lows.append(float(res["auc_ci_low"]))
        ci_highs.append(float(res["auc_ci_high"]))
        std_fds.append(float(res["standardized_fd"]))
        covs.append(float(res["coverage"]))

    mean_null_auc = float(np.mean(null_aucs))
    mean_ci_low = float(np.mean(ci_lows))
    mean_ci_high = float(np.mean(ci_highs))
    passed = bool(
        abs(mean_null_auc - 0.5) <= 0.12
        and mean_ci_low <= 0.60
        and mean_ci_high >= 0.40
    )
    return {
        "skipped": False,
        "passed": passed,
        "n_repeats": int(repeats),
        "repeat_aucs": null_aucs,
        "null_auc": mean_null_auc,
        "null_auc_ci_low": mean_ci_low,
        "null_auc_ci_high": mean_ci_high,
        "null_standardized_fd": float(np.mean(std_fds)),
        "null_coverage": float(np.mean(covs)),
    }


# ---------------------------------------------------------------------------
# 7. Forensic Diagnosis Generator & Actionable Subsystem Scorecard
# ---------------------------------------------------------------------------


def build_subsystem_scorecard(
    *,
    auc_patch_rgb: float,
    auc_patch_gray: float,
    auc_patch_gray_norm: float,
    auc_patch_lowfreq_gray: float,
    auc_patch_highfreq_gray: float,
    auc_patch_binary: float,
    auc_fullpage: float,
    coverage: float,
    variance_ratio: float,
    embed_group_attribution: dict[str, Any] | None = None,
    domain_stats: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Map each orthogonal realism axis to concrete ``hw-score-sim`` / ``photo-degrade`` modules."""
    color_delta = float(auc_patch_rgb - auc_patch_gray)
    exposure_delta = float(auc_patch_gray - auc_patch_gray_norm)

    def _status_delta(val: float, warn: float = 0.05, fail: float = 0.10) -> str:
        return "fail" if val > fail else ("warn" if val > warn else "pass")

    def _status_auc(val: float, warn: float = 0.62, fail: float = 0.72) -> str:
        return "fail" if val > fail else ("warn" if val > warn else "pass")

    # Handle ceiling-saturation masking on color_tint and exposure_contrast deltas.
    if float(auc_patch_gray) >= 0.88 and color_delta <= 0.05:
        cm_shift = (embed_group_attribution or {}).get("color_moments", {}).get("mean_abs_std_shift")
        if cm_shift is not None and float(cm_shift) > 1.25:
            color_status = "fail"
        elif cm_shift is not None and float(cm_shift) > 0.75:
            color_status = "warn"
        else:
            color_status = "masked"
    else:
        color_status = _status_delta(color_delta)

    if float(auc_patch_gray_norm) >= 0.88 and exposure_delta <= 0.05:
        r_c = (domain_stats or {}).get("real", {}).get("michelson_contrast_mean")
        s_c = (domain_stats or {}).get("sim", {}).get("michelson_contrast_mean")
        if r_c is not None and s_c is not None and abs(float(r_c) - float(s_c)) > 0.15:
            exposure_status = "warn"
        else:
            exposure_status = "masked"
    else:
        exposure_status = _status_delta(exposure_delta)

    cov_status = "fail" if (coverage < 0.55 or variance_ratio < 0.35) else (
        "warn" if (coverage < 0.70 or variance_ratio < 0.50) else "pass"
    )

    return [
        {
            "axis": "color_tint",
            "metric_name": "delta_rgb_minus_gray",
            "metric_value": color_delta,
            "status": color_status,
            "target_modules": ["styles.py", "renderer.py", "photo-degrade"],
            "action": "Broaden paper chromaticity, sepia balance, and ink color jitter ranges.",
        },
        {
            "axis": "exposure_contrast",
            "metric_name": "delta_gray_minus_gray_norm",
            "metric_value": exposure_delta,
            "status": exposure_status,
            "target_modules": ["styles.py", "renderer.py", "photo-degrade"],
            "action": "Match global luminance, ink-to-paper contrast ratio, and exposure curves.",
        },
        {
            "axis": "illumination_mottling",
            "metric_name": "auc_patch_lowfreq_gray",
            "metric_value": float(auc_patch_lowfreq_gray),
            "status": _status_auc(auc_patch_lowfreq_gray),
            "target_modules": ["styles.py", "renderer.py", "photo-degrade"],
            "action": "Align low-frequency paper mottling, show-through bleed, and lighting gradients.",
        },
        {
            "axis": "nib_texture_grain",
            "metric_name": "auc_patch_highfreq_gray",
            "metric_value": float(auc_patch_highfreq_gray),
            "status": _status_auc(auc_patch_highfreq_gray),
            "target_modules": ["styles.py", "stroke_engine.py", "renderer.py", "photo-degrade"],
            "action": "Tune nib edge feathering, ink absorption micro-roughness, and paper grain.",
        },
        {
            "axis": "glyph_stroke_morphology",
            "metric_name": "auc_patch_binary",
            "metric_value": float(auc_patch_binary),
            "status": _status_auc(auc_patch_binary),
            "target_modules": [
                "styles.py",
                "stroke_engine.py",
                "glyph_variants.py",
                "expressive_glyphs.py",
                "structural_glyphs.py",
                "rest_variants.py",
                "renderer.py",
            ],
            "action": "Diversify glyph ductus archetypes, staff-line wobble, and organic spacing.",
        },
        {
            "axis": "macro_page_layout",
            "metric_name": "auc_fullpage",
            "metric_value": float(auc_fullpage),
            "status": _status_auc(auc_fullpage, warn=0.65, fail=0.75),
            "target_modules": ["renderer.py", "layout_quirks.py", "profile_resolver.py", "manuscript_marks.py"],
            "action": "Match margin ratios, system counts, page aspect ratios, and archival annotations.",
        },
        {
            "axis": "diversity_coverage",
            "metric_name": "knn_coverage",
            "metric_value": float(coverage),
            "status": cov_status,
            "target_modules": ["styles.py", "sampler.py", "profile_resolver.py", "photo-degrade"],
            "action": "Inspect top uncovered real pages and expand scribe/paper/layout sampling diversity.",
        },
    ]


def diagnose_realism_gaps(
    auc_patch_rgb: float,
    auc_patch_gray: float,
    auc_patch_lowfreq: float,
    auc_patch_highfreq: float,
    auc_fullpage: float,
    auc_embed: float,
    frechet_distance: float,
    *,
    auc_patch_gray_norm: float | None = None,
    auc_patch_binary: float | None = None,
    coverage: float | None = None,
    variance_ratio: float | None = None,
    domain_stats: dict[str, Any] | None = None,
    null_control: dict[str, Any] | None = None,
    underfit_probes: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Translate probe metrics, controls, and domain stats into ranked actionable diagnostics."""
    findings: list[dict[str, Any]] = []

    # 1. Statistical control & confounder checks
    if underfit_probes:
        probe_list = ", ".join(str(p) for p in underfit_probes)
        findings.append(
            {
                "code": "PROBE_UNDERFIT_WARNING",
                "severity": "medium",
                "score": float(len(underfit_probes)),
                "message": (
                    f"CNN probe(s) showed signs of underfitting (train AUC < 0.60 and val AUC < 0.58): "
                    f"{probe_list}."
                ),
                "recommendation": "Increase training epochs or dataset size before concluding indistinguishability.",
            }
        )

    if null_control is not None and not null_control.get("passed", True):
        findings.append(
            {
                "code": "NULL_CONTROL_FAILED",
                "severity": "high",
                "score": float(null_control.get("null_auc", 0.5)),
                "message": (
                    f"Real-vs-Real 50/50 null control failed (null AUC={null_control.get('null_auc', 0.5):.3f}, "
                    f"95% CI=[{null_control.get('null_auc_ci_low', 0.0):.3f}, {null_control.get('null_auc_ci_high', 1.0):.3f}])."
                ),
                "recommendation": "Inspect real dataset heterogeneity or group splitting before interpreting probe AUCs.",
            }
        )

    if domain_stats is not None:
        r_st = domain_stats.get("real") or {}
        s_st = domain_stats.get("sim") or {}
        r_sp = r_st.get("sp_px_p50")
        s_sp = s_st.get("sp_px_p50")
        if r_sp and s_sp and r_sp > 0 and s_sp > 0:
            ratio = float(s_sp) / float(r_sp)
            if abs(math.log(ratio)) > math.log(1.25):
                findings.append(
                    {
                        "code": "CONFOUNDER_SCALE_MISMATCH",
                        "severity": "high",
                        "score": float(abs(math.log(ratio))),
                        "message": (
                            f"Median staff spacing differs significantly (real={r_sp:.1f}px vs sim={s_sp:.1f}px, "
                            f"ratio={ratio:.2f}x)."
                        ),
                        "recommendation": "Align rendered staff-spacing scale in renderer.py / profile_resolver.py to real scans.",
                    }
                )
        r_ar = r_st.get("aspect_ratio_p50")
        s_ar = s_st.get("aspect_ratio_p50")
        r_land = r_st.get("landscape_frac", 0.0)
        s_land = s_st.get("landscape_frac", 0.0)
        if r_ar is not None and s_ar is not None:
            ar_diff = abs(float(s_ar) - float(r_ar))
            land_diff = abs(float(s_land) - float(r_land))
            if ar_diff > 0.20 or land_diff > 0.25:
                findings.append(
                    {
                        "code": "CONFOUNDER_ASPECT_RATIO",
                        "severity": "medium",
                        "score": float(max(ar_diff, land_diff)),
                        "message": (
                            f"Page aspect-ratio/orientation distribution differs (real AR={r_ar:.2f}, "
                            f"landscape={r_land:.2f} vs sim AR={s_ar:.2f}, landscape={s_land:.2f})."
                        ),
                        "recommendation": "Match portrait/landscape page proportions in profile_resolver.py.",
                    }
                )
        s_contrast = s_st.get("michelson_contrast_mean")
        s_prom = s_st.get("staff_peak_prominence_mean")
        if (s_contrast is not None and s_contrast < 0.25) or (s_prom is not None and s_prom < 0.08):
            findings.append(
                {
                    "code": "LEGIBILITY_SNR_WARNING",
                    "severity": "medium",
                    "score": float(1.0 - (s_contrast if s_contrast is not None else 0.5)),
                    "message": (
                        f"Synthetic notation readability SNR is low (Michelson contrast={s_contrast}, "
                        f"staff prominence={s_prom})."
                    ),
                    "recommendation": "Reduce excessive degradation/fade in styles.py, renderer.py, or photo-degrade to preserve OMR legibility.",
                }
            )

    if (coverage is not None and coverage < 0.70) or (variance_ratio is not None and variance_ratio < 0.40):
        cov_val = float(coverage) if coverage is not None else 1.0
        vr_val = float(variance_ratio) if variance_ratio is not None else 1.0
        findings.append(
            {
                "code": "MODE_COLLAPSE_RISK",
                "severity": "high" if (cov_val < 0.50 or vr_val < 0.30) else "medium",
                "score": float(max(1.0 - cov_val, 1.0 - vr_val)),
                "message": (
                    f"Synthetic distribution under-covers real manuscripts (k-NN coverage={cov_val:.3f}, "
                    f"variance_ratio={vr_val:.3f})."
                ),
                "recommendation": "Expand scribe, paper, and layout diversity in styles.py and profile_resolver.py.",
            }
        )

    all_aucs = [
        auc_patch_rgb,
        auc_patch_gray,
        auc_patch_lowfreq,
        auc_patch_highfreq,
        auc_fullpage,
        auc_embed,
    ]
    if auc_patch_gray_norm is not None:
        all_aucs.append(auc_patch_gray_norm)
    if auc_patch_binary is not None:
        all_aucs.append(auc_patch_binary)

    if max(all_aucs) <= 0.58 and not findings:
        return [
            {
                "code": "INDISTINGUISHABLE",
                "severity": "pass",
                "score": float(max(all_aucs)),
                "message": (
                    f"All probes are near chance (max AUC={max(all_aucs):.3f}, "
                    f"FD={frechet_distance:.3f}) with healthy coverage; synthetic pages align with real manuscripts."
                ),
                "recommendation": "No major realism gap detected; lock current simulator profile as baseline.",
            }
        ]

    # 2. Orthogonal visual gap findings
    color_delta = float(auc_patch_rgb - auc_patch_gray)
    if color_delta > 0.08:
        findings.append(
            {
                "code": "COLOR_TONE_GAP",
                "severity": "high" if color_delta > 0.15 else "medium",
                "score": color_delta,
                "message": (
                    f"RGB patch AUC ({auc_patch_rgb:.3f}) exceeds grayscale ({auc_patch_gray:.3f}) "
                    f"by {color_delta:.3f}: paper tint or ink chromaticity is a primary giveaway."
                ),
                "recommendation": (
                    "Broaden paper white-balance/sepia chromaticity and ink color jitter in styles.py, "
                    "renderer.py, and photo-degrade."
                ),
            }
        )

    if auc_patch_gray_norm is not None:
        exp_delta = float(auc_patch_gray - auc_patch_gray_norm)
        if exp_delta > 0.08:
            findings.append(
                {
                    "code": "EXPOSURE_CONTRAST_GAP",
                    "severity": "high" if exp_delta > 0.15 else "medium",
                    "score": exp_delta,
                    "message": (
                        f"Grayscale AUC ({auc_patch_gray:.3f}) exceeds percentile-normalized grayscale "
                        f"({auc_patch_gray_norm:.3f}) by {exp_delta:.3f}: exposure/contrast level separates domains."
                    ),
                    "recommendation": "Align brightness/contrast curves in styles.py, renderer.py, and photo-degrade.",
                }
            )

    if auc_patch_highfreq > 0.70:
        findings.append(
            {
                "code": "NIB_STROKE_GAP",
                "severity": "high" if auc_patch_highfreq > 0.85 else "medium",
                "score": float(auc_patch_highfreq),
                "message": (
                    f"High-frequency band-pass patch AUC={auc_patch_highfreq:.3f}: nib edge sharpness, "
                    "ink absorption roughness, or fine paper fiber grain separates sim from real."
                ),
                "recommendation": (
                    "Inspect highfreq_gray Grad-CAM overlays; tune nib edge feathering and paper grain "
                    "in styles.py, stroke_engine.py, renderer.py, and photo-degrade."
                ),
            }
        )

    if auc_patch_lowfreq > 0.70:
        findings.append(
            {
                "code": "MACRO_TONE_BLEED_GAP",
                "severity": "high" if auc_patch_lowfreq > 0.85 else "medium",
                "score": float(auc_patch_lowfreq),
                "message": (
                    f"Low-frequency patch AUC={auc_patch_lowfreq:.3f}: local illumination gradients, "
                    "paper mottling, or show-through bleed differs from real pages."
                ),
                "recommendation": (
                    "Match paper aging gradients, mottling, and show-through profiles in styles.py, renderer.py, and photo-degrade."
                ),
            }
        )

    morph_auc = float(auc_patch_binary) if auc_patch_binary is not None else float(auc_patch_gray)
    if morph_auc > 0.70:
        findings.append(
            {
                "code": "STROKE_GEOMETRY_GAP",
                "severity": "high" if morph_auc > 0.85 else "medium",
                "score": morph_auc,
                "message": (
                    f"Binarized/grayscale stroke morphology AUC={morph_auc:.3f}: stroke geometry, "
                    "staff-line wobble, or glyph morphology remains distinguishable."
                ),
                "recommendation": (
                    "Inspect binary Grad-CAM overlays; diversify glyph ductus, staff wobble, and spacing "
                    "in styles.py, stroke_engine.py, glyph_variants.py, expressive_glyphs.py, "
                    "structural_glyphs.py, rest_variants.py, and renderer.py."
                ),
            }
        )

    if auc_fullpage > 0.75:
        findings.append(
            {
                "code": "MACRO_LAYOUT_GAP",
                "severity": "high" if auc_fullpage > 0.88 else "medium",
                "score": float(auc_fullpage),
                "message": (
                    f"Full-page macro-layout AUC={auc_fullpage:.3f}: page framing, margin ratios, "
                    "system density, or borders separate synthetic layouts from real manuscripts."
                ),
                "recommendation": (
                    "Inspect Full-Page Grad-CAM overlays; diversify margins, system counts, and "
                    "annotations in renderer.py, layout_quirks.py, profile_resolver.py, and manuscript_marks.py."
                ),
            }
        )

    if auc_embed > 0.75:
        findings.append(
            {
                "code": "EMBEDDING_DISTRIBUTION_GAP",
                "severity": "high" if auc_embed > 0.88 else "medium",
                "score": float(auc_embed),
                "message": (
                    f"Frozen feature linear probe AUC={auc_embed:.3f} (Frechet Distance={frechet_distance:.3f}): "
                    "global spectral/orientation/color feature distributions diverge."
                ),
                "recommendation": (
                    "Inspect Probe C feature_attribution groups and align spectral/orientation/tone statistics "
                    "in styles.py, stroke_engine.py, renderer.py, and photo-degrade."
                ),
            }
        )

    if not findings:
        probe_scores = {
            "patch_rgb": auc_patch_rgb,
            "patch_gray": auc_patch_gray,
            "patch_lowfreq": auc_patch_lowfreq,
            "patch_highfreq": auc_patch_highfreq,
            "fullpage": auc_fullpage,
            "embedding": auc_embed,
        }
        if auc_patch_gray_norm is not None:
            probe_scores["patch_gray_norm"] = auc_patch_gray_norm
        if auc_patch_binary is not None:
            probe_scores["patch_binary"] = auc_patch_binary
        top_probe, top_auc = max(probe_scores.items(), key=lambda kv: kv[1])
        findings.append(
            {
                "code": "MILD_REALISM_GAP",
                "severity": "low",
                "score": float(top_auc),
                "message": (
                    f"Mild residual separability (highest probe: {top_probe} AUC={top_auc:.3f}, "
                    f"FD={frechet_distance:.3f})."
                ),
                "recommendation": f"Inspect {top_probe} Grad-CAMs/features for subtle residual cues.",
            }
        )

    severity_rank = {"high": 3, "medium": 2, "low": 1, "pass": 0}
    findings.sort(key=lambda item: (severity_rank.get(item["severity"], 0), item["score"]), reverse=True)
    return findings


# ---------------------------------------------------------------------------
# 8. End-to-End Benchmark Runner & CLI
# ---------------------------------------------------------------------------


def _collect_image_paths(directory: str | Path, max_pages: int | None, seed: int) -> list[Path]:
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {root}")
    paths = sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not paths:
        raise ValueError(f"No images found in {root}")
    if max_pages is not None and len(paths) > int(max_pages):
        rng = np.random.default_rng(int(seed))
        chosen = np.sort(rng.choice(len(paths), size=int(max_pages), replace=False))
        paths = [paths[int(i)] for i in chosen]
    return paths


def _load_valid_pages(
    paths: Sequence[Path],
    root: Path,
    domain_prefix: str,
    max_dim: int,
    jpeg_quality: int,
    pre_blur_radius: float,
) -> tuple[list[Image.Image], list[str], list[Path], int]:
    """Canonicalize pages, skipping corrupt/unreadable files cleanly."""
    pages: list[Image.Image] = []
    valid_paths: list[Path] = []
    skipped = 0
    for p in paths:
        try:
            im = canonicalize_for_probe(
                p,
                max_dim=max_dim,
                jpeg_quality=jpeg_quality,
                pre_blur_radius=pre_blur_radius,
            )
        except (OSError, ValueError, SyntaxError, EOFError, Image.DecompressionBombError):
            skipped += 1
            continue
        pages.append(im)
        valid_paths.append(p)
    groups = infer_manuscript_groups(valid_paths, root=root, domain_prefix=domain_prefix)
    return pages, groups, valid_paths, skipped


def run_realism_benchmark(
    real_dir: str | Path,
    sim_dir: str | Path,
    out_dir: str | Path,
    *,
    max_pages: int | None = 100,
    patches_per_page: int = 6,
    patch_size: int = 256,
    fullpage_size: int = 512,
    max_dim: int = 1024,
    jpeg_quality: int = 75,
    pre_blur_radius: float = 0.6,
    interior_crop_frac: float = 0.0,
    epochs: int = 8,
    folds: int = 5,
    batch_size: int = 16,
    device: str = "auto",
    photo_degrade_bin: str | Path | None = None,
    embed_fn: Callable[[Image.Image], np.ndarray] | None = None,
    top_k_gradcam: int = 6,
    seed: int = 42,
    no_cudnn: bool = False,
) -> dict[str, Any]:
    """Run the complete 3-probe adversarial realism suite and write ``report.json`` + Grad-CAMs."""
    # Validate numeric arguments up front.
    if max_pages is not None and int(max_pages) < 1:
        raise ValueError(f"max_pages must be >= 1 or None, got {max_pages}")
    if int(patches_per_page) < 1:
        raise ValueError(f"patches_per_page must be >= 1, got {patches_per_page}")
    if int(patch_size) < 16:
        raise ValueError(f"patch_size must be >= 16, got {patch_size}")
    if int(fullpage_size) < 32:
        raise ValueError(f"fullpage_size must be >= 32, got {fullpage_size}")
    if int(max_dim) < 0:
        raise ValueError(f"max_dim must be >= 0, got {max_dim}")
    if not (1 <= int(jpeg_quality) <= 100):
        raise ValueError(f"jpeg_quality must be in [1, 100], got {jpeg_quality}")
    if not math.isfinite(float(pre_blur_radius)) or float(pre_blur_radius) < 0.0:
        raise ValueError(f"pre_blur_radius must be finite and >= 0.0, got {pre_blur_radius}")
    if not math.isfinite(float(interior_crop_frac)) or not (0.0 <= float(interior_crop_frac) <= 0.45):
        raise ValueError(f"interior_crop_frac must be finite and in [0.0, 0.45], got {interior_crop_frac}")
    if int(epochs) < 1:
        raise ValueError(f"epochs must be >= 1, got {epochs}")
    if int(folds) < 2:
        raise ValueError(f"folds must be >= 2, got {folds}")
    if int(batch_size) < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if int(top_k_gradcam) < 0:
        raise ValueError(f"top_k_gradcam must be >= 0, got {top_k_gradcam}")
    if int(seed) < 0:
        raise ValueError(f"seed must be >= 0, got {seed}")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    real_root = Path(real_dir)
    sim_root = Path(sim_dir)

    real_paths = _collect_image_paths(real_root, max_pages=max_pages, seed=seed)
    sim_paths = _collect_image_paths(sim_root, max_pages=max_pages, seed=seed + 1)
    sim_orig_paths = list(sim_paths)

    if photo_degrade_bin is not None:
        sim_paths = predegrade_sim_images(
            sim_paths=sim_paths,
            photo_degrade_bin=photo_degrade_bin,
            cache_dir=out_path / "sim_degraded_cache",
            max_width=1536,
            seed=seed,
        )

    # Canonicalize every page identically; skip unreadable files cleanly.
    real_pages, real_groups, valid_real_paths, skipped_real = _load_valid_pages(
        real_paths,
        root=real_root,
        domain_prefix="real",
        max_dim=max_dim,
        jpeg_quality=jpeg_quality,
        pre_blur_radius=pre_blur_radius,
    )
    sim_pages: list[Image.Image] = []
    valid_sim_orig_paths: list[Path] = []
    skipped_sim = 0
    for p, orig_p in zip(sim_paths, sim_orig_paths):
        try:
            im = canonicalize_for_probe(
                p,
                max_dim=max_dim,
                jpeg_quality=jpeg_quality,
                pre_blur_radius=pre_blur_radius,
            )
        except (OSError, ValueError, SyntaxError, EOFError, Image.DecompressionBombError):
            skipped_sim += 1
            continue
        sim_pages.append(im)
        valid_sim_orig_paths.append(orig_p)
    sim_groups = infer_manuscript_groups(valid_sim_orig_paths, root=sim_root, domain_prefix="sim")

    if len(real_pages) < 2 or len(sim_pages) < 2:
        raise ValueError(
            f"Insufficient valid pages after loading (real={len(real_pages)}, sim={len(sim_pages)}; "
            f"skipped_real={skipped_real}, skipped_sim={skipped_sim}). Need at least 2 per domain."
        )
    if len(set(real_groups)) < 2 or len(set(sim_groups)) < 2:
        raise ValueError(
            f"Insufficient unique manuscript groups (real={len(set(real_groups))}, "
            f"sim={len(set(sim_groups))}). Need at least 2 unique manuscript groups per domain."
        )

    # --- Probe A: Scale-normalized interior ink patches + orthogonal 6-view ablation ladder ---
    patch_images: list[Image.Image] = []
    patch_labels: list[int] = []
    patch_groups: list[str] = []
    real_page_patches: list[list[Image.Image]] = []
    sim_page_patches: list[list[Image.Image]] = []
    real_fb_total = 0
    sim_fb_total = 0

    for idx, (im, gid) in enumerate(zip(real_pages, real_groups)):
        crops, st = sample_ink_patches(
            im,
            num_patches=patches_per_page,
            patch_size=patch_size,
            seed=seed + idx,
            post_blur_radius=pre_blur_radius,
            post_jpeg_quality=jpeg_quality,
            return_stats=True,
        )
        real_page_patches.append(crops)
        real_fb_total += int(st["fallback_count"])
        patch_images.extend(crops)
        patch_labels.extend([0] * len(crops))
        patch_groups.extend([gid] * len(crops))

    for idx, (im, gid) in enumerate(zip(sim_pages, sim_groups)):
        crops, st = sample_ink_patches(
            im,
            num_patches=patches_per_page,
            patch_size=patch_size,
            seed=seed + 100_000 + idx,
            post_blur_radius=pre_blur_radius,
            post_jpeg_quality=jpeg_quality,
            return_stats=True,
        )
        sim_page_patches.append(crops)
        sim_fb_total += int(st["fallback_count"])
        patch_images.extend(crops)
        patch_labels.extend([1] * len(crops))
        patch_groups.extend([gid] * len(crops))

    domain_stats = {
        "real": compute_domain_stats(
            real_pages,
            patch_fallback_rate=float(real_fb_total) / max(len(real_pages) * patches_per_page, 1),
        ),
        "sim": compute_domain_stats(
            sim_pages,
            patch_fallback_rate=float(sim_fb_total) / max(len(sim_pages) * patches_per_page, 1),
        ),
    }

    patch_labels_np = np.asarray(patch_labels, dtype=np.int64)
    gradcam_patch_views = {"rgb", "binary", "highfreq_gray"}
    patch_results: dict[str, dict[str, Any]] = {}
    for view_idx, mode in enumerate(PATCH_VIEW_MODES):
        view_tensors = np.stack([transform_patch_view(im, mode=mode) for im in patch_images], axis=0)
        do_cam = top_k_gradcam > 0 and mode in gradcam_patch_views
        res = _train_and_eval_cnn_kfold(
            tensors=view_tensors,
            labels=patch_labels_np,
            groups=patch_groups,
            model_factory=lambda: build_patch_forensic_cnn(in_channels=3),
            folds=folds,
            epochs=epochs,
            batch_size=batch_size,
            device=device,
            seed=seed + view_idx * 1000,
            no_cudnn=no_cudnn,
            allow_hflip=True,
            top_k_gradcam=top_k_gradcam if do_cam else 0,
            source_images=patch_images if do_cam else None,
            gradcam_dir=(out_path / f"gradcam_patch_{mode}") if do_cam else None,
            gradcam_prefix=f"patch_{mode}",
        )
        patch_results[f"patch_{mode}"] = res

    # --- Probe B: CoordConv + Spatial Pyramid full-page macro-layout CNN (gray_norm) ---
    fp_images = [
        letterbox_page(im, size=fullpage_size, interior_crop_frac=interior_crop_frac) for im in real_pages
    ] + [
        letterbox_page(im, size=fullpage_size, interior_crop_frac=interior_crop_frac) for im in sim_pages
    ]
    fp_tensors = np.stack([transform_patch_view(im, mode="gray_norm") for im in fp_images], axis=0)
    fp_labels_np = np.concatenate(
        [np.zeros(len(real_pages), dtype=np.int64), np.ones(len(sim_pages), dtype=np.int64)]
    )
    fp_groups = real_groups + sim_groups
    fullpage_result = _train_and_eval_cnn_kfold(
        tensors=fp_tensors,
        labels=fp_labels_np,
        groups=fp_groups,
        model_factory=lambda: build_fullpage_forensic_cnn(in_channels=3),
        folds=folds,
        epochs=epochs,
        batch_size=max(2, batch_size // 2),
        device=device,
        seed=seed + len(PATCH_VIEW_MODES) * 1000,
        no_cudnn=no_cudnn,
        allow_hflip=False,
        top_k_gradcam=top_k_gradcam,
        source_images=fp_images,
        gradcam_dir=out_path / "gradcam_fullpage",
        gradcam_prefix="fullpage",
    )

    # --- Probe C: Patch-averaged default embeddings (or custom embed_fn) + PRDC + Attribution ---
    if embed_fn is None:
        real_embs = np.stack(
            [
                np.mean([extract_default_embedding(p) for p in page_crops], axis=0)
                for page_crops in real_page_patches
            ],
            axis=0,
        )
        sim_embs = np.stack(
            [
                np.mean([extract_default_embedding(p) for p in page_crops], axis=0)
                for page_crops in sim_page_patches
            ],
            axis=0,
        )
        embed_result = run_embedding_probe(
            real_embeddings=real_embs,
            sim_embeddings=sim_embs,
            real_groups=real_groups,
            sim_groups=sim_groups,
            real_paths=valid_real_paths,
            feature_spec=DEFAULT_EMBEDDING_FEATURE_SPEC,
            folds=folds,
            seed=seed + 9000,
        )
    else:
        real_embs = np.stack([np.asarray(embed_fn(im), dtype=np.float64).ravel() for im in real_pages], axis=0)
        sim_embs = np.stack([np.asarray(embed_fn(im), dtype=np.float64).ravel() for im in sim_pages], axis=0)
        embed_result = run_embedding_probe(
            real_embeddings=real_embs,
            sim_embeddings=sim_embs,
            real_groups=real_groups,
            sim_groups=sim_groups,
            real_paths=valid_real_paths,
            folds=folds,
            seed=seed + 9000,
        )

    # --- Real-vs-Real 50/50 Null Control ---
    null_control = run_null_control(real_embs, real_groups, folds=folds, seed=seed)

    # --- Forensic Diagnostics & Subsystem Scorecard ---
    auc_rgb = patch_results["patch_rgb"]["roc_auc"]
    auc_gray = patch_results["patch_gray"]["roc_auc"]
    auc_gray_norm = patch_results["patch_gray_norm"]["roc_auc"]
    auc_lf = patch_results["patch_lowfreq_gray"]["roc_auc"]
    auc_hf = patch_results["patch_highfreq_gray"]["roc_auc"]
    auc_bin = patch_results["patch_binary"]["roc_auc"]
    auc_fp = fullpage_result["roc_auc"]
    auc_emb = embed_result["roc_auc"]

    underfit_probes = [
        name
        for name, res in {**patch_results, "fullpage": fullpage_result}.items()
        if res.get("underfit_warning", False)
    ]

    diagnostics = diagnose_realism_gaps(
        auc_patch_rgb=auc_rgb,
        auc_patch_gray=auc_gray,
        auc_patch_lowfreq=auc_lf,
        auc_patch_highfreq=auc_hf,
        auc_fullpage=auc_fp,
        auc_embed=auc_emb,
        frechet_distance=embed_result["frechet_distance"],
        auc_patch_gray_norm=auc_gray_norm,
        auc_patch_binary=auc_bin,
        coverage=embed_result["coverage"],
        variance_ratio=embed_result["variance_ratio"],
        domain_stats=domain_stats,
        null_control=null_control,
        underfit_probes=underfit_probes,
    )

    subsystem_scorecard = build_subsystem_scorecard(
        auc_patch_rgb=auc_rgb,
        auc_patch_gray=auc_gray,
        auc_patch_gray_norm=auc_gray_norm,
        auc_patch_lowfreq_gray=auc_lf,
        auc_patch_highfreq_gray=auc_hf,
        auc_patch_binary=auc_bin,
        auc_fullpage=auc_fp,
        coverage=embed_result["coverage"],
        variance_ratio=embed_result["variance_ratio"],
        embed_group_attribution=embed_result["feature_attribution"]["groups"],
        domain_stats=domain_stats,
    )

    report: dict[str, Any] = {
        "config": {
            "real_dir": str(real_dir),
            "sim_dir": str(sim_dir),
            "out_dir": str(out_path),
            "n_real_pages": len(real_pages),
            "n_sim_pages": len(sim_pages),
            "n_real_groups": len(set(real_groups)),
            "n_sim_groups": len(set(sim_groups)),
            "skipped_real": int(skipped_real),
            "skipped_sim": int(skipped_sim),
            "patches_per_page": int(patches_per_page),
            "patch_size": int(patch_size),
            "fullpage_size": int(fullpage_size),
            "max_dim": int(max_dim),
            "jpeg_quality": int(jpeg_quality),
            "pre_blur_radius": float(pre_blur_radius),
            "interior_crop_frac": float(interior_crop_frac),
            "epochs": int(epochs),
            "folds": int(folds),
            "batch_size": int(batch_size),
            "device": str(device),
            "photo_degrade_bin": str(photo_degrade_bin) if photo_degrade_bin is not None else None,
            "seed": int(seed),
            "no_cudnn": bool(no_cudnn),
        },
        "domain_stats": domain_stats,
        "null_control": null_control,
        "probes": {
            **patch_results,
            "fullpage": fullpage_result,
            "embedding": embed_result,
        },
        "summary": {
            "auc_patch_rgb": auc_rgb,
            "auc_patch_gray": auc_gray,
            "auc_patch_gray_norm": auc_gray_norm,
            "auc_patch_lowfreq_gray": auc_lf,
            "auc_patch_highfreq_gray": auc_hf,
            "auc_patch_binary": auc_bin,
            # Backward-compatible summary keys:
            "auc_patch_lowfreq": auc_lf,
            "auc_patch_highfreq": auc_hf,
            "auc_fullpage": auc_fp,
            "auc_embed": auc_emb,
            "frechet_distance": embed_result["frechet_distance"],
            "standardized_fd": embed_result["standardized_fd"],
            "knn_precision": embed_result["precision"],
            "knn_recall": embed_result["recall"],
            "knn_density": embed_result["density"],
            "knn_coverage": embed_result["coverage"],
            "variance_ratio": embed_result["variance_ratio"],
        },
        "diagnostics": diagnostics,
        "subsystem_scorecard": subsystem_scorecard,
    }

    report_path = out_path / "report.json"
    tmp_report = report_path.with_name(report_path.name + ".tmp")
    tmp_report.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(tmp_report, report_path)
    return report


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m hw_score_sim.realism_bench",
        description="3-Probe Adversarial Sim-vs-Real Realism Classifier & Neural Forensics Suite",
    )
    parser.add_argument("--real-dir", required=True, help="Directory of real archival manuscript images.")
    parser.add_argument("--sim-dir", required=True, help="Directory of synthetic handwritten score images.")
    parser.add_argument("--out-dir", required=True, help="Output directory for report.json and Grad-CAM overlays.")
    parser.add_argument("--max-pages", type=int, default=100, help="Maximum pages sampled per domain (default: 100).")
    parser.add_argument("--patches-per-page", type=int, default=6, help="Ink-bearing patches per page (default: 6).")
    parser.add_argument("--epochs", type=int, default=8, help="Training epochs per CV fold (default: 8).")
    parser.add_argument("--folds", type=int, default=5, help="Manuscript-level group K-fold splits (default: 5).")
    parser.add_argument("--device", default="auto", help="PyTorch device: auto, cpu, cuda (default: auto).")
    parser.add_argument(
        "--photo-degrade-bin",
        default=None,
        help="Optional path to photo-degrade binary to pre-degrade sim images into a temp cache.",
    )
    parser.add_argument(
        "--interior-crop-frac",
        type=float,
        default=0.0,
        help="Optional symmetric outer border crop fraction [0.0, 0.45] for Probe B full-page views (default: 0.0).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Deterministic RNG seed (default: 42).")
    parser.add_argument("--no-cudnn", action="store_true", help="Disable cuDNN (recommended on Windows).")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    report = run_realism_benchmark(
        real_dir=args.real_dir,
        sim_dir=args.sim_dir,
        out_dir=args.out_dir,
        max_pages=args.max_pages,
        patches_per_page=args.patches_per_page,
        interior_crop_frac=args.interior_crop_frac,
        epochs=args.epochs,
        folds=args.folds,
        device=args.device,
        photo_degrade_bin=args.photo_degrade_bin,
        seed=args.seed,
        no_cudnn=args.no_cudnn,
    )
    summary = report["summary"]
    print(
        "Realism Benchmark Summary: "
        f"rgb={summary['auc_patch_rgb']:.4f} "
        f"gray={summary['auc_patch_gray']:.4f} "
        f"gray_norm={summary['auc_patch_gray_norm']:.4f} "
        f"lf={summary['auc_patch_lowfreq_gray']:.4f} "
        f"hf={summary['auc_patch_highfreq_gray']:.4f} "
        f"bin={summary['auc_patch_binary']:.4f} "
        f"fullpage={summary['auc_fullpage']:.4f} "
        f"embed={summary['auc_embed']:.4f} "
        f"stdFD={summary['standardized_fd']:.4f} "
        f"cov={summary['knn_coverage']:.3f}"
    )
    for diag in report["diagnostics"]:
        print(f"  [{diag['severity'].upper()}] {diag['code']}: {diag['message']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
