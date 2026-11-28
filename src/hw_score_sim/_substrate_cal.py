"""Internal optical rag-fiber grain and rastrum harmonic calibration."""

from __future__ import annotations

import hashlib
import hmac
import math
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

_CAL_PROFILE = b"hw-score-sim::rag-optical::v1"
_PROFILE_REV = 1

# Diagonal-annulus optical fiber basis (cycles/px, orthogonal to staff/stem axes)
_N_LUM_MODES = 32
_N_CHR_MODES = 16
_R_LUM_MIN = 0.065
_R_LUM_MAX = 0.195
_R_CHR_MIN = 0.045
_R_CHR_MAX = 0.105
_SECTOR_A = (20.0, 70.0)
_SECTOR_B = (110.0, 160.0)

# Grain response envelope in [0, 1] linear float buffer units
_SIGMA_Y = 0.55 / 255.0
_AMP_Y = 1.40 / 255.0
_SIGMA_C = 0.35 / 255.0
_AMP_C = 1.20 / 255.0
_CLAMP_LIM = 1.50 / 255.0

# Spectral analysis grid
_FFT_N = 1024
_LP_NR = 128
_LP_NTH = 360
_LP_R_MIN = 0.032
_LP_R_MAX = 0.380

# Rastrum harmonic micro-profile (cycles per staff space)
_RASTRUM_N_HARM = 6
_RASTRUM_NU_MIN = 0.065
_RASTRUM_NU_MAX = 0.175
_RASTRUM_AMP_PX = 0.12

# Auxiliary seed-indexed harmonic bits (24 slots: 4 rev | 16 digest | 4 check)
_AUX_SLOTS = 24


def _resolve_profile(profile: Optional[bytes] = None) -> bytes:
    if profile is not None:
        return bytes(profile)
    env_p = os.environ.get("HW_SIM_CAL_PROFILE")
    if env_p:
        return env_p.encode("utf-8")
    return _CAL_PROFILE


def _prf(profile: bytes, label: bytes) -> bytes:
    return hmac.new(profile, label, hashlib.sha256).digest()


def _seed_digest16(seed: int, profile: bytes) -> int:
    return int.from_bytes(_prf(profile, f"lot:{int(seed)}".encode("utf-8"))[:2], "big")


def _check4(val20: int) -> int:
    reg = (val20 & 0xFFFFF) << 4
    poly = 0x13 << 19
    for i in range(20):
        if reg & (1 << (23 - i)):
            reg ^= poly >> i
    return reg & 0xF


def _pack_lot24(seed: int, profile: bytes, rev: int = _PROFILE_REV) -> int:
    body20 = ((int(rev) & 0xF) << 16) | (_seed_digest16(seed, profile) & 0xFFFF)
    return (body20 << 4) | _check4(body20)


def _unpack_lot24(word24: int) -> Tuple[bool, int, int]:
    w = int(word24) & 0xFFFFFF
    body20 = w >> 4
    ok = _check4(body20) == (w & 0xF)
    return ok, (body20 >> 16) & 0xF, body20 & 0xFFFF


@lru_cache(maxsize=16)
def _fiber_modes(
    profile: bytes,
    band: str,
    count: int,
    r_min: float,
    r_max: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic diagonal-sector frequency modes (r, theta_rad, phase_rad)."""
    seed_int = int.from_bytes(_prf(profile, f"modes:{band}:{count}".encode("utf-8"))[:8], "little")
    rng = np.random.default_rng(seed_int)

    log_r_min = math.log(r_min)
    log_r_max = math.log(r_max)
    rs: List[float] = []
    thetas: List[float] = []
    phases: List[float] = []

    min_sep_log_r = 0.045
    min_sep_deg = 4.5
    for idx in range(count):
        sector = _SECTOR_A if (idx % 2 == 0) else _SECTOR_B
        band_lo = log_r_min + (log_r_max - log_r_min) * (idx / float(count))
        band_hi = log_r_min + (log_r_max - log_r_min) * ((idx + 1) / float(count))
        best_r = math.exp(0.5 * (band_lo + band_hi))
        best_deg = 0.5 * (sector[0] + sector[1])
        for _ in range(32):
            lr = float(rng.uniform(band_lo, band_hi))
            deg = float(rng.uniform(sector[0] + 1.5, sector[1] - 1.5))
            r_cand = math.exp(lr)
            if all(
                abs(math.log(pr) - lr) >= min_sep_log_r
                or abs(math.degrees(pt) - deg) >= min_sep_deg
                for pr, pt in zip(rs, thetas)
            ):
                best_r = r_cand
                best_deg = deg
                break
        rs.append(best_r)
        thetas.append(math.radians(best_deg))
        phases.append(float(rng.uniform(0.0, 2.0 * math.pi)))

    return (
        np.asarray(rs, dtype=np.float64),
        np.asarray(thetas, dtype=np.float64),
        np.asarray(phases, dtype=np.float64),
    )


@lru_cache(maxsize=16)
def _aux_mode_pairs(profile: bytes) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """24 secondary frequency pairs in diagonal sectors, separated from primary modes."""
    seed_int = int.from_bytes(_prf(profile, b"aux:modes:24")[:8], "little")
    rng = np.random.default_rng(seed_int)

    r_base, th_base, _ = _fiber_modes(profile, "Y", _N_LUM_MODES, _R_LUM_MIN, _R_LUM_MAX)
    used: List[Tuple[float, float]] = [
        (math.log(float(r)), math.degrees(float(t))) for r, t in zip(r_base, th_base)
    ]

    log_lo = math.log(0.080)
    log_hi = math.log(0.185)
    r0s, t0s, r1s, t1s = [], [], [], []

    for b in range(_AUX_SLOTS):
        pair_pts: List[Tuple[float, float]] = []
        for sym in range(2):
            sector = _SECTOR_A if ((b + sym) % 2 == 0) else _SECTOR_B
            chosen = (math.exp(0.5 * (log_lo + log_hi)), math.radians(0.5 * (sector[0] + sector[1])))
            for _ in range(48):
                lr = float(rng.uniform(log_lo, log_hi))
                deg = float(rng.uniform(sector[0] + 2.0, sector[1] - 2.0))
                if all(abs(lr - ulr) > 0.032 or abs(deg - udeg) > 3.5 for ulr, udeg in used):
                    used.append((lr, deg))
                    chosen = (math.exp(lr), math.radians(deg))
                    break
            pair_pts.append(chosen)
        r0s.append(pair_pts[0][0])
        t0s.append(pair_pts[0][1])
        r1s.append(pair_pts[1][0])
        t1s.append(pair_pts[1][1])

    return (
        np.asarray(r0s, dtype=np.float64),
        np.asarray(t0s, dtype=np.float64),
        np.asarray(r1s, dtype=np.float64),
        np.asarray(t1s, dtype=np.float64),
    )


def _bilinear_gain_comp(fx: float, fy: float) -> float:
    """Compensate 2x bilinear upsample roll-off (capped at 2.2x)."""
    sx = 1.0 if abs(fx) < 1e-6 else math.sin(2.0 * math.pi * fx) / (2.0 * math.pi * fx)
    sy = 1.0 if abs(fy) < 1e-6 else math.sin(2.0 * math.pi * fy) / (2.0 * math.pi * fy)
    return min(2.2, 1.0 / max(0.20, (sx * sx) * (sy * sy)))


def _insert_mode_rfft2(
    spec: np.ndarray,
    h2: int,
    w2: int,
    full_h: int,
    full_w: int,
    r: float,
    theta: float,
    phase: float,
    weight: float = 1.0,
) -> None:
    fx = r * math.cos(theta)
    fy = r * math.sin(theta)
    if fx < 0.0:
        fx = -fx
        fy = -fy
        phase = -phase

    kx = int(round(fx * full_w))
    ky_signed = int(round(fy * full_h))
    if kx <= 0 or kx >= (w2 // 2):
        return
    if abs(ky_signed) == 0 or abs(ky_signed) >= (h2 // 2):
        return
    ky = ky_signed % h2
    amp = weight * _bilinear_gain_comp(fx, fy)
    spec[ky, kx] += complex(amp * math.cos(phase), amp * math.sin(phase))


def _synth_halfres_grain(
    full_h: int,
    full_w: int,
    rs: np.ndarray,
    thetas: np.ndarray,
    phases: np.ndarray,
    aux_modes: Optional[List[Tuple[float, float, float, float]]] = None,
) -> np.ndarray:
    """Synthesize unit-variance float32 rag-fiber texture via half-res irfft2 + bilinear upsample."""
    h2 = max(32, full_h // 2)
    w2 = max(32, full_w // 2)
    spec = np.zeros((h2, w2 // 2 + 1), dtype=np.complex64)

    for r, th, ph in zip(rs, thetas, phases):
        _insert_mode_rfft2(spec, h2, w2, full_h, full_w, float(r), float(th), float(ph), weight=1.0)

    if aux_modes:
        for r, th, ph, wt in aux_modes:
            _insert_mode_rfft2(spec, h2, w2, full_h, full_w, r, th, ph, weight=wt)

    half = np.fft.irfft2(spec, s=(h2, w2)).astype(np.float32)
    half -= float(half.mean())
    std = float(half.std())
    if std > 1e-8:
        half *= np.float32(1.0 / std)

    pil_f = Image.fromarray(half, mode="F")
    full = np.array(pil_f.resize((full_w, full_h), resample=Image.Resampling.BILINEAR), dtype=np.float32, copy=True)
    full -= float(full.mean())
    f_std = float(full.std())
    if f_std > 1e-8:
        full *= np.float32(1.0 / f_std)
    return full


def _edge_taper_1d(length: int, pad: int = 4) -> np.ndarray:
    """Raised-cosine edge taper keeping outer 1px border unaltered."""
    w = np.ones(length, dtype=np.float32)
    p = min(pad, length // 4)
    if p >= 2:
        ramp = (0.5 * (1.0 - np.cos(np.linspace(0.0, math.pi, p, dtype=np.float32)))).astype(np.float32)
        ramp[0] = np.float32(0.0)
        w[:p] = ramp
        w[-p:] = ramp[::-1]
    elif length >= 2:
        w[0] = np.float32(0.0)
        w[-1] = np.float32(0.0)
    return w


def apply_substrate_cal(
    buf: np.ndarray,
    seed: int = 42,
    profile: Optional[bytes] = None,
) -> np.ndarray:
    """Apply subtle optical rag-fiber & orthogonal chroma grain calibration in-place."""
    if buf is None or buf.ndim != 3 or buf.shape[2] != 3:
        return buf
    h, w, _ = buf.shape
    if h < 64 or w < 64:
        return buf

    prof = _resolve_profile(profile)
    rs_y, th_y, ph_y = _fiber_modes(prof, "Y", _N_LUM_MODES, _R_LUM_MIN, _R_LUM_MAX)
    rs_c, th_c, ph_c = _fiber_modes(prof, "C", _N_CHR_MODES, _R_CHR_MIN, _R_CHR_MAX)

    lot24 = _pack_lot24(seed, prof, rev=_PROFILE_REV)
    r0s, t0s, r1s, t1s = _aux_mode_pairs(prof)
    aux_y: List[Tuple[float, float, float, float]] = []
    for b in range(_AUX_SLOTS):
        bit = (lot24 >> (_AUX_SLOTS - 1 - b)) & 1
        r_b = float(r1s[b] if bit else r0s[b])
        t_b = float(t1s[b] if bit else t0s[b])
        ph_b = float(((b * 2654435761) & 0xFFFF) * (2.0 * math.pi / 65536.0))
        aux_y.append((r_b, t_b, ph_b, 0.35))

    field_y = _synth_halfres_grain(h, w, rs_y, th_y, ph_y, aux_modes=aux_y)
    field_c = _synth_halfres_grain(h, w, rs_c, th_c, ph_c)

    dy = np.float32(_AMP_Y) * np.tanh(np.float32(_SIGMA_Y / _AMP_Y) * field_y)
    dc = np.float32(_AMP_C) * np.tanh(np.float32(_SIGMA_C / _AMP_C) * field_c)

    taper = _edge_taper_1d(h, pad=4)[:, None] * _edge_taper_1d(w, pad=4)[None, :]
    dy *= taper
    dc *= taper

    # Map (dY, dC_perp) -> (dR, dG, dB), C_perp = (Cb + Cr) / sqrt(2)
    lim = np.float32(_CLAMP_LIM)
    dr = np.clip(dy + np.float32(0.991364) * dc, -lim, lim)
    dg = np.clip(dy - np.float32(0.748311) * dc, -lim, lim)
    db = np.clip(dy + np.float32(1.252993) * dc, -lim, lim)

    buf[:, :, 0] += dr
    buf[:, :, 1] += dg
    buf[:, :, 2] += db
    np.clip(buf, np.float32(0.0), np.float32(1.0), out=buf)
    return buf


# ---------------------------------------------------------------------------
# Rastrum Harmonic Micro-Profile (Coherent Across 5 Staff Lines)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=16)
def _rastrum_harmonics(profile: bytes) -> Tuple[np.ndarray, np.ndarray]:
    seed_int = int.from_bytes(_prf(profile, b"rastrum:comb:v1")[:8], "little")
    rng = np.random.default_rng(seed_int)
    edges = np.linspace(_RASTRUM_NU_MIN, _RASTRUM_NU_MAX, _RASTRUM_N_HARM + 1)
    nus = np.empty(_RASTRUM_N_HARM, dtype=np.float64)
    phis = np.empty(_RASTRUM_N_HARM, dtype=np.float64)
    for m in range(_RASTRUM_N_HARM):
        span = edges[m + 1] - edges[m]
        nus[m] = float(rng.uniform(edges[m] + 0.15 * span, edges[m + 1] - 0.15 * span))
        phis[m] = float(rng.uniform(0.0, 2.0 * math.pi))
    return nus, phis


def rastrum_micro_wave(
    xs: np.ndarray,
    sp: float,
    profile: Optional[bytes] = None,
) -> np.ndarray:
    """Deterministic 6-harmonic scale-invariant rastrum micro-profile (peak <= 0.12 px)."""
    arr_x = np.asarray(xs, dtype=np.float64)
    if arr_x.size == 0:
        return np.zeros(0, dtype=np.float64)
    sp_eff = max(4.0, float(sp))
    s = (arr_x - float(arr_x[0])) / sp_eff
    nus, phis = _rastrum_harmonics(_resolve_profile(profile))
    acc = np.zeros_like(s, dtype=np.float64)
    w_per = _RASTRUM_AMP_PX / float(_RASTRUM_N_HARM)
    for nu, phi in zip(nus, phis):
        acc += np.cos((2.0 * math.pi * float(nu)) * s + float(phi))
    return acc * w_per


# ---------------------------------------------------------------------------
# Internal Calibration Diagnostics
# ---------------------------------------------------------------------------


def _box_mean_2d(arr: np.ndarray, rad: int) -> np.ndarray:
    k = 2 * rad + 1
    inv_area = 1.0 / float(k * k)
    p = np.pad(arr, ((rad, rad), (rad, rad)), mode="reflect")
    cs = np.cumsum(p, axis=0, dtype=np.float64)
    cs = np.pad(cs, ((1, 0), (0, 0)), mode="constant", constant_values=0.0)
    v = cs[k:, :] - cs[:-k, :]
    cs2 = np.cumsum(v, axis=1, dtype=np.float64)
    cs2 = np.pad(cs2, ((0, 0), (1, 0)), mode="constant", constant_values=0.0)
    return ((cs2[:, k:] - cs2[:, :-k]) * inv_area).astype(np.float32)


def _whiten_rfft_mag(mag: np.ndarray, rad: int = 8, block_div: int = 8) -> np.ndarray:
    m_shift = np.fft.fftshift(mag, axes=0).astype(np.float32)
    mu = _box_mean_2d(m_shift, rad)
    var = np.maximum(0.0, _box_mean_2d(m_shift * m_shift, rad) - mu * mu)
    std = np.sqrt(var) + np.float32(1e-6)
    z = (m_shift - mu) / std
    np.clip(z, np.float32(-2.5), np.float32(4.5), out=z)
    # Suppress JPEG block-grid harmonics (1/8 for Y, 1/16 for 4:2:0 chroma)
    h, w = z.shape
    cy = h // 2
    step = h // block_div
    max_m = block_div // 2
    notch = 2 if block_div == 8 else 1
    for ky_m in range(-max_m + 1, max_m):
        if ky_m == 0:
            continue
        yy = cy + ky_m * step
        for kx_m in range(1, max_m):
            xx = kx_m * step
            if 0 <= yy < h and 0 <= xx < w:
                z[max(0, yy - notch) : min(h, yy + notch + 1), max(0, xx - notch) : min(w, xx + notch + 1)] = np.float32(0.0)
    p = np.pad(z, ((1, 1), (1, 1)), mode="edge")
    sm = (
        np.float32(0.0625) * (p[:-2, :-2] + p[:-2, 2:] + p[2:, :-2] + p[2:, 2:])
        + np.float32(0.125) * (p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:])
        + np.float32(0.25) * p[1:-1, 1:-1]
    )
    return np.fft.ifftshift(sm, axes=0).astype(np.float32)


def _centered_windowed_fft(chan: np.ndarray, n_fft: int = _FFT_N) -> np.ndarray:
    h, w = chan.shape
    ch = min(h, n_fft)
    cw = min(w, n_fft)
    y0 = (h - ch) // 2
    x0 = (w - cw) // 2
    patch = chan[y0 : y0 + ch, x0 : x0 + cw].astype(np.float32)
    patch -= float(patch.mean())
    patch *= np.hanning(ch).astype(np.float32)[:, None] * np.hanning(cw).astype(np.float32)[None, :]

    if ch == n_fft and cw == n_fft:
        buf = patch
    else:
        buf = np.zeros((n_fft, n_fft), dtype=np.float32)
        py0 = (n_fft - ch) // 2
        px0 = (n_fft - cw) // 2
        buf[py0 : py0 + ch, px0 : px0 + cw] = patch

    return np.abs(np.fft.rfft2(buf)).astype(np.float32)


@lru_cache(maxsize=4)
def _log_polar_grid(
    n_fft: int = _FFT_N,
    n_r: int = _LP_NR,
    n_th: int = _LP_NTH,
    r_min: float = _LP_R_MIN,
    r_max: float = _LP_R_MAX,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    log_rs = np.linspace(math.log(r_min), math.log(r_max), n_r, dtype=np.float64)
    rs = np.exp(log_rs)
    thetas = np.linspace(0.0, math.pi, n_th, endpoint=False, dtype=np.float64)

    fx = rs[:, None] * np.cos(thetas)[None, :]
    fy = rs[:, None] * np.sin(thetas)[None, :]
    neg = fx < 0.0
    fx = np.where(neg, -fx, fx)
    fy = np.where(neg, -fy, fy)

    kx_f = np.clip(fx * n_fft, 0.0, float(n_fft // 2 - 1) - 1e-4)
    ky_f = np.mod(fy * n_fft, float(n_fft))

    x0 = np.floor(kx_f).astype(np.int32)
    x1 = np.minimum(x0 + 1, n_fft // 2)
    y0 = np.floor(ky_f).astype(np.int32) % n_fft
    y1 = (y0 + 1) % n_fft

    wx = (kx_f - x0).astype(np.float32)
    wy = (ky_f - np.floor(ky_f)).astype(np.float32)

    deg = np.degrees(thetas)[None, :]
    sector_mask = (
        ((deg >= 7.0) & (deg <= 83.0)) | ((deg >= 97.0) & (deg <= 173.0))
    ).astype(np.float32)
    sector_mask = np.broadcast_to(sector_mask, (n_r, n_th)).copy()
    return x0, x1, y0, y1, wx, wy, sector_mask


def _sample_log_polar(whitened_rfft: np.ndarray) -> np.ndarray:
    x0, x1, y0, y1, wx, wy, sector_mask = _log_polar_grid()
    lp = (
        whitened_rfft[y0, x0] * (1.0 - wx) * (1.0 - wy)
        + whitened_rfft[y0, x1] * wx * (1.0 - wy)
        + whitened_rfft[y1, x0] * (1.0 - wx) * wy
        + whitened_rfft[y1, x1] * wx * wy
    )
    lp *= sector_mask
    lp -= lp.mean(axis=1, keepdims=True)
    return lp.astype(np.float32)


@lru_cache(maxsize=16)
def _basis_template_fft(profile: bytes, band: str) -> np.ndarray:
    if band == "Y":
        rs, thetas, _ = _fiber_modes(profile, "Y", _N_LUM_MODES, _R_LUM_MIN, _R_LUM_MAX)
    else:
        rs, thetas, _ = _fiber_modes(profile, "C", _N_CHR_MODES, _R_CHR_MIN, _R_CHR_MAX)

    log_min = math.log(_LP_R_MIN)
    log_max = math.log(_LP_R_MAX)
    r_grid = np.arange(_LP_NR, dtype=np.float32)[:, None]
    th_grid = np.arange(_LP_NTH, dtype=np.float32)[None, :]

    tmpl = np.zeros((_LP_NR, _LP_NTH), dtype=np.float32)
    sig_r2 = 2.0 * (0.95 ** 2)
    sig_t2 = 2.0 * (1.15 ** 2)

    for r, th in zip(rs, thetas):
        ri = (math.log(float(r)) - log_min) / (log_max - log_min) * (_LP_NR - 1)
        ti = (float(th) / math.pi) * _LP_NTH
        dr = r_grid - np.float32(ri)
        dt = np.abs(th_grid - np.float32(ti))
        dt = np.minimum(dt, np.float32(_LP_NTH) - dt)
        tmpl += np.exp(-(dr * dr) / np.float32(sig_r2) - (dt * dt) / np.float32(sig_t2))

    tmpl -= float(tmpl.mean())
    norm = float(np.linalg.norm(tmpl))
    if norm > 1e-8:
        tmpl /= np.float32(norm)
    return np.conj(np.fft.rfft2(tmpl))


def _mad_normalize(corr: np.ndarray) -> np.ndarray:
    # Per-row (constant-scale) MAD normalization eliminates radial non-stationarity (e.g. 4:2:0 chroma roll-off)
    row_med = np.median(corr, axis=1, keepdims=True)
    row_mad = np.median(np.abs(corr - row_med), axis=1, keepdims=True)
    global_mad = float(np.median(row_mad))
    sigma = np.maximum(1e-7, 1.4826 * np.maximum(row_mad, 0.35 * global_mad))
    return ((corr - row_med) / sigma).astype(np.float32)


def _read_aux_lot(
    whitened_y: np.ndarray,
    profile: bytes,
    est_scale: float,
    est_rot_deg: float,
) -> Tuple[bool, Optional[int], Optional[int]]:
    r0s, t0s, r1s, t1s = _aux_mode_pairs(profile)
    rot_rad = math.radians(est_rot_deg)
    inv_scale = 1.0 / max(0.35, est_scale)
    n_fft = whitened_y.shape[0]

    def _peak_at(r: float, th: float) -> float:
        rr = r * inv_scale
        tt = th + rot_rad
        fx = rr * math.cos(tt)
        fy = rr * math.sin(tt)
        if fx < 0.0:
            fx = -fx
            fy = -fy
        kx = int(round(fx * n_fft))
        ky = int(round(fy * n_fft)) % n_fft
        if kx <= 0 or kx >= n_fft // 2:
            return 0.0
        best = -1e9
        for dy in (-1, 0, 1):
            yy = (ky + dy) % n_fft
            for dx in (-1, 0, 1):
                xx = min(n_fft // 2, max(0, kx + dx))
                v = float(whitened_y[yy, xx])
                if v > best:
                    best = v
        return best

    word = 0
    for b in range(_AUX_SLOTS):
        s0 = _peak_at(float(r0s[b]), float(t0s[b]))
        s1 = _peak_at(float(r1s[b]), float(t1s[b]))
        word = (word << 1) | (1 if s1 > s0 else 0)

    ok, rev, lot_hash = _unpack_lot24(word)
    if ok:
        return True, rev, lot_hash
    return False, None, None


def _measure_spectral_response(
    img: Union[Image.Image, np.ndarray],
    profile: Optional[bytes] = None,
    z_threshold: float = 6.5,
) -> Dict[str, Any]:
    """Measure diagonal-sector Y + C_perp optical calibration response."""
    if isinstance(img, Image.Image):
        arr = np.asarray(img.convert("RGB"), dtype=np.float32) / np.float32(255.0)
    else:
        arr = np.asarray(img, dtype=np.float32)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        if float(arr.max()) > 1.5:
            arr = arr / np.float32(255.0)

    prof = _resolve_profile(profile)
    r_ch = arr[:, :, 0]
    g_ch = arr[:, :, 1]
    b_ch = arr[:, :, 2]

    y_ch = np.float32(0.299) * r_ch + np.float32(0.587) * g_ch + np.float32(0.114) * b_ch
    cb = np.float32(-0.168736) * r_ch - np.float32(0.331264) * g_ch + np.float32(0.5) * b_ch
    cr = np.float32(0.5) * r_ch - np.float32(0.418688) * g_ch - np.float32(0.081312) * b_ch
    c_perp = (cb + cr) * np.float32(0.70710678)

    mag_y = _centered_windowed_fft(y_ch, _FFT_N)
    white_y = _whiten_rfft_mag(mag_y, rad=8)
    lp_y = _sample_log_polar(white_y)
    corr_y = np.fft.irfft2(np.fft.rfft2(lp_y) * _basis_template_fft(prof, "Y"), s=(_LP_NR, _LP_NTH))
    z_y_map = _mad_normalize(corr_y)

    has_chroma = float(c_perp.std()) > 1e-4
    if has_chroma:
        mag_c = _centered_windowed_fft(c_perp, _FFT_N)
        white_c = _whiten_rfft_mag(mag_c, rad=8, block_div=16)
        lp_c = _sample_log_polar(white_c)
        corr_c = np.fft.irfft2(np.fft.rfft2(lp_c) * _basis_template_fft(prof, "C"), s=(_LP_NR, _LP_NTH))
        z_c_map = _mad_normalize(corr_c)
        z_map = _mad_normalize(np.float32(0.75) * z_y_map + np.float32(0.66) * z_c_map)
    else:
        z_c_map = None
        z_map = z_y_map

    log_step = (math.log(_LP_R_MAX) - math.log(_LP_R_MIN)) / (_LP_NR - 1)
    max_dr = int(math.ceil(math.log(2.25) / log_step))
    max_dth = int(math.ceil(18.0 / (180.0 / _LP_NTH)))

    dr_idxs = np.concatenate([np.arange(0, max_dr + 1), np.arange(_LP_NR - max_dr, _LP_NR)])
    dt_idxs = np.concatenate([np.arange(0, max_dth + 1), np.arange(_LP_NTH - max_dth, _LP_NTH)])
    sub = z_map[np.ix_(dr_idxs, dt_idxs)]
    si, sj = divmod(int(np.argmax(sub)), sub.shape[1])
    best_r = int(dr_idxs[si])
    best_t = int(dt_idxs[sj])
    z_peak = float(z_map[best_r, best_t])

    sr = best_r if best_r <= _LP_NR // 2 else best_r - _LP_NR
    st = best_t if best_t <= _LP_NTH // 2 else best_t - _LP_NTH
    est_scale = float(math.exp(-sr * log_step))
    est_rot_deg = float(st * (180.0 / _LP_NTH))

    n_search = int(sub.size)
    p_single = 0.5 * math.erfc(max(0.0, z_peak) / math.sqrt(2.0))
    p_value = float(min(1.0, n_search * p_single))
    in_spec = bool(z_peak >= float(z_threshold) and p_value < 1e-6)

    lot_ok = False
    lot_rev: Optional[int] = None
    lot_digest: Optional[int] = None
    if in_spec:
        lot_ok, lot_rev, lot_digest = _read_aux_lot(white_y, prof, est_scale, est_rot_deg)

    return {
        "z_score": round(z_peak, 4),
        "z_y": round(float(z_y_map[best_r, best_t]), 4),
        "z_c": round(float(z_c_map[best_r, best_t]), 4) if z_c_map is not None else 0.0,
        "p_value": p_value,
        "calibrated": in_spec,
        "est_scale": round(est_scale, 4),
        "est_rot_deg": round(est_rot_deg, 2),
        "lot_ok": lot_ok,
        "lot_rev": lot_rev,
        "lot_digest": lot_digest,
    }


def _estimate_skew_deg(ink: np.ndarray) -> float:
    h, w = ink.shape
    scale = min(1.0, 512.0 / max(h, w))
    if scale < 1.0:
        pil_s = Image.fromarray((ink * 255.0).astype(np.uint8), mode="L").resize(
            (max(32, int(w * scale)), max(32, int(h * scale))),
            resample=Image.Resampling.BILINEAR,
        )
        small = np.asarray(pil_s, dtype=np.float32) / np.float32(255.0)
    else:
        small = ink

    pil_small = Image.fromarray((np.clip(small, 0.0, 1.0) * 255.0).astype(np.uint8), mode="L")
    best_ang = 0.0
    best_var = -1.0
    for ang in np.linspace(-8.0, 8.0, 33):
        rot = np.asarray(pil_small.rotate(float(ang), resample=Image.Resampling.BILINEAR), dtype=np.float32)
        v = float(rot.mean(axis=1).var())
        if v > best_var:
            best_var = v
            best_ang = float(ang)
    for ang in np.linspace(best_ang - 0.6, best_ang + 0.6, 13):
        rot = np.asarray(pil_small.rotate(float(ang), resample=Image.Resampling.BILINEAR), dtype=np.float32)
        v = float(rot.mean(axis=1).var())
        if v > best_var:
            best_var = v
            best_ang = float(ang)
    return best_ang


def _locate_5line_staves(ink: np.ndarray) -> Tuple[List[List[float]], float]:
    h, _ = ink.shape
    prof = ink.mean(axis=1).astype(np.float64)
    med = float(np.median(prof))
    mad = max(1e-6, 1.4826 * float(np.median(np.abs(prof - med))))
    thresh = med + max(2.2 * mad, 0.03)

    peaks: List[float] = []
    for y in range(2, h - 2):
        if prof[y] > thresh and prof[y] >= prof[y - 1] and prof[y] >= prof[y + 1]:
            denom = prof[y - 1] - 2.0 * prof[y] + prof[y + 1]
            dy = 0.5 * (prof[y - 1] - prof[y + 1]) / denom if abs(denom) > 1e-9 else 0.0
            peaks.append(float(y) + float(np.clip(dy, -0.5, 0.5)))

    if len(peaks) < 5:
        return [], 0.0

    staves: List[List[float]] = []
    sps: List[float] = []
    i = 0
    while i <= len(peaks) - 5:
        grp = peaks[i : i + 5]
        diffs = np.diff(grp)
        sp_est = float(np.median(diffs))
        if 5.0 <= sp_est <= 36.0 and float(np.max(np.abs(diffs - sp_est))) <= max(2.0, 0.22 * sp_est):
            staves.append(grp)
            sps.append(sp_est)
            i += 5
        else:
            i += 1

    if not staves:
        return [], 0.0
    return staves, float(np.median(sps))


def _measure_rastrum_response(
    img: Union[Image.Image, np.ndarray],
    profile: Optional[bytes] = None,
) -> Dict[str, Any]:
    """Measure coherent 5-line rastrum harmonic response."""
    if isinstance(img, Image.Image):
        gray = np.asarray(img.convert("L"), dtype=np.float32) / np.float32(255.0)
    else:
        arr = np.asarray(img, dtype=np.float32)
        if arr.ndim == 3:
            gray = np.float32(0.299) * arr[:, :, 0] + np.float32(0.587) * arr[:, :, 1] + np.float32(0.114) * arr[:, :, 2]
        else:
            gray = arr
        if float(gray.max()) > 1.5:
            gray = gray / np.float32(255.0)

    ink = np.clip(np.float32(1.0) - gray, np.float32(0.0), np.float32(1.0))
    skew_deg = _estimate_skew_deg(ink)
    if abs(skew_deg) > 0.25:
        pil_ink = Image.fromarray((ink * 255.0).astype(np.uint8), mode="L")
        ink = np.asarray(
            pil_ink.rotate(skew_deg, resample=Image.Resampling.BILINEAR), dtype=np.float32
        ) / np.float32(255.0)

    staves, sp_px = _locate_5line_staves(ink)
    if not staves or sp_px <= 0.0:
        return {"z_rastrum": 0.0, "staves_detected": 0, "sp_px": 0.0}

    h, _ = ink.shape
    nus, _ = _rastrum_harmonics(_resolve_profile(profile))
    win_half = max(2, int(round(0.32 * sp_px)))
    nu_grid = np.linspace(0.035, 0.240, 512, dtype=np.float64)
    psd_acc = np.zeros_like(nu_grid)
    used_staves = 0

    for grp in staves:
        y_top = max(0, int(math.floor(grp[0] - win_half)))
        y_bot = min(h, int(math.ceil(grp[-1] + win_half)) + 1)
        if y_bot <= y_top:
            continue
        col_energy = ink[y_top:y_bot, :].mean(axis=0)
        active = np.where(col_energy > max(0.03, float(np.median(col_energy)) * 1.5))[0]
        if active.size < int(max(80, 12 * sp_px)):
            continue
        x0 = int(active[0])
        x1 = int(active[-1]) + 1
        n_cols = x1 - x0
        if n_cols < 80:
            continue

        line_res = np.full((5, n_cols), np.nan, dtype=np.float64)
        xs_local = np.arange(n_cols, dtype=np.float64)
        for li, yc in enumerate(grp):
            ry0 = max(0, int(round(yc)) - win_half)
            ry1 = min(h, int(round(yc)) + win_half + 1)
            strip = ink[ry0:ry1, x0:x1].astype(np.float64)
            ys_idx = np.arange(ry0, ry1, dtype=np.float64)[:, None]
            w_sum = strip.sum(axis=0)
            pos = w_sum > 0.1
            if not np.any(pos):
                continue
            med_w = float(np.median(w_sum[pos]))
            if med_w <= 0.05:
                continue
            valid = (w_sum > 0.25 * med_w) & (w_sum < 2.4 * med_w)
            if int(valid.sum()) < max(40, n_cols // 5):
                continue
            cy = (ys_idx * strip).sum(axis=0) / np.maximum(w_sum, 1e-8)
            coef = np.polyfit(xs_local[valid], cy[valid], deg=3)
            res = cy - np.polyval(coef, xs_local)
            res[~valid] = np.nan
            line_res[li] = res

        has_any = np.any(np.isfinite(line_res), axis=0)
        med_res = np.full(n_cols, np.nan, dtype=np.float64)
        if np.any(has_any):
            med_res[has_any] = np.nanmedian(line_res[:, has_any], axis=0)
        finite = np.isfinite(med_res)
        if int(finite.sum()) < max(50, n_cols // 4):
            continue
        interp_res = np.interp(xs_local, xs_local[finite], med_res[finite])
        interp_res -= float(interp_res.mean())
        win = np.hanning(n_cols)
        n_fft_1d = 1 << int(math.ceil(math.log2(max(1024, n_cols * 2))))
        spec = np.abs(np.fft.rfft(interp_res * win, n=n_fft_1d)) ** 2
        freqs_sp = np.fft.rfftfreq(n_fft_1d, d=1.0) * sp_px
        psd_interp = np.interp(nu_grid, freqs_sp, spec)
        med_p = float(np.median(psd_interp))
        if med_p > 1e-12:
            psd_acc += np.log(psd_interp / med_p + 1e-3)
            used_staves += 1

    if used_staves == 0:
        return {"z_rastrum": 0.0, "staves_detected": len(staves), "sp_px": round(sp_px, 3)}

    psd_mean = psd_acc / float(used_staves)
    bg_mask = np.ones_like(nu_grid, dtype=bool)
    for nu in nus:
        bg_mask &= np.abs(nu_grid - nu) > 0.006

    bg_vals = psd_mean[bg_mask]
    bg_med = float(np.median(bg_vals))
    bg_mad = max(1e-6, 1.4826 * float(np.median(np.abs(bg_vals - bg_med))))

    best_z = -1e9
    for sc in np.linspace(0.97, 1.03, 13):
        hits = [float(np.interp(float(nu) * float(sc), nu_grid, psd_mean)) for nu in nus]
        z_val = (float(np.mean(hits)) - bg_med) / (bg_mad / math.sqrt(len(nus)))
        if z_val > best_z:
            best_z = z_val

    return {
        "z_rastrum": round(float(best_z), 4),
        "staves_detected": used_staves,
        "sp_px": round(sp_px, 3),
    }


def _check_cal(
    path_or_pil: Union[str, Path, Image.Image],
    profile: Optional[bytes] = None,
) -> Dict[str, Any]:
    """Run internal optical + rastrum calibration diagnostics."""
    if isinstance(path_or_pil, Image.Image):
        img = path_or_pil
        close_after = False
    else:
        img = Image.open(path_or_pil)
        close_after = True

    try:
        spec = _measure_spectral_response(img, profile=profile)
        rast = _measure_rastrum_response(img, profile=profile)
    finally:
        if close_after:
            img.close()

    z1 = float(spec["z_score"])
    z2 = float(rast["z_rastrum"])
    ok = bool(spec["calibrated"]) or (z2 >= 6.0 and int(rast["staves_detected"]) >= 2)
    score = 1.0 / (1.0 + math.exp(-0.85 * (z1 - 5.5)))
    if z2 >= 6.0 and int(rast["staves_detected"]) >= 2:
        score = max(score, 0.85)

    return {
        "calibrated": ok,
        "score": round(float(min(1.0, max(0.0, score))), 4),
        "spectral": spec,
        "rastrum": rast,
    }
