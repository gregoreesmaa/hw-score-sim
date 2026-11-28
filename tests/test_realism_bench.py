"""CPU-only unit and integration tests for `hw_score_sim.realism_bench`."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageOps

from hw_score_sim.realism_bench import (
    DEFAULT_EMBEDDING_FEATURE_SPEC,
    PATCH_VIEW_MODES,
    FullPageForensicCNN,
    PatchForensicCNN,
    _build_arg_parser,
    _fit_logistic_irls,
    _to_rgb_safe,
    _train_and_eval_cnn_kfold,
    bootstrap_group_auc_ci,
    build_subsystem_scorecard,
    canonicalize_for_probe,
    compute_frechet_distance,
    compute_gradcam,
    compute_knn_prdc,
    compute_roc_auc,
    diagnose_realism_gaps,
    estimate_staff_spacing_px,
    extract_default_embedding,
    group_kfold_indices,
    infer_manuscript_group,
    infer_manuscript_groups,
    letterbox_page,
    predegrade_sim_images,
    run_embedding_probe,
    run_null_control,
    run_realism_benchmark,
    sample_ink_patches,
    save_gradcam_overlay,
    transform_patch_view,
)


def _make_synthetic_page(
    size: tuple[int, int],
    bg_rgb: tuple[int, int, int],
    ink_rgb: tuple[int, int, int],
    seed: int,
    line_thickness: int = 2,
    staff_spacing: int = 8,
) -> Image.Image:
    """Create a small synthetic staff-and-noteheads page for fast unit testing."""
    rng = np.random.default_rng(seed)
    w, h = size
    noise = rng.integers(-6, 7, size=(h, w, 3), dtype=np.int16)
    base = np.clip(np.array(bg_rgb, dtype=np.int16)[None, None, :] + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(base, mode="RGB")
    draw = ImageDraw.Draw(img)

    for staff_idx in range(3):
        y_top = int(h * (0.15 + staff_idx * 0.28))
        for line in range(5):
            y = y_top + line * staff_spacing
            draw.line([(int(w * 0.08), y), (int(w * 0.92), y)], fill=ink_rgb, width=line_thickness)
        for _ in range(18):
            nx = int(rng.integers(int(w * 0.12), int(w * 0.88)))
            ny = y_top + int(rng.integers(-6, staff_spacing * 4 + 6))
            draw.ellipse([(nx - 4, ny - 3), (nx + 4, ny + 3)], fill=ink_rgb)
    return img


def test_to_rgb_safe_handles_alpha_and_high_bitdepth() -> None:
    """_to_rgb_safe composites transparent images over white and rescales 16-bit/float images."""
    # Transparent RGBA pixel (0, 0, 0, 0) must become white (255, 255, 255), NOT black.
    rgba = Image.new("RGBA", (16, 16), (0, 0, 0, 0))
    rgba.putpixel((0, 0), (10, 20, 30, 255))
    rgb = _to_rgb_safe(rgba)
    assert rgb.mode == "RGB"
    assert rgb.getpixel((8, 8)) == (255, 255, 255)
    assert rgb.getpixel((0, 0)) == (10, 20, 30)

    # LA with zero alpha -> white
    la = Image.new("LA", (8, 8), (0, 0))
    assert _to_rgb_safe(la).getpixel((0, 0)) == (255, 255, 255)

    # Palette image with transparency -> white
    pal = Image.new("P", (8, 8), 0)
    pal.putpalette([0, 0, 0, 200, 100, 50] + [0] * (256 * 3 - 6))
    pal.info["transparency"] = 0
    assert _to_rgb_safe(pal).getpixel((0, 0)) == (255, 255, 255)

    # 16-bit I;16 image rescaled cleanly to [0, 255]
    arr16 = np.array([[0, 32768], [65535, 16384]], dtype=np.uint16)
    im16 = Image.fromarray(arr16)
    assert im16.mode.startswith("I;16")
    rgb16 = np.asarray(_to_rgb_safe(im16))
    assert rgb16[0, 0, 0] == 0
    assert 125 <= rgb16[0, 1, 0] <= 130
    assert rgb16[1, 0, 0] == 255

    # Float F image in [0, 1] rescaled cleanly
    arr_f = np.array([[0.0, 0.5], [1.0, 0.25]], dtype=np.float32)
    im_f = Image.fromarray(arr_f)
    assert im_f.mode == "F"
    rgb_f = np.asarray(_to_rgb_safe(im_f))
    assert rgb_f[0, 0, 0] == 0
    assert 126 <= rgb_f[0, 1, 0] <= 129
    assert rgb_f[1, 0, 0] == 255


def test_compute_roc_auc_exactness_and_validation() -> None:
    """compute_roc_auc exactness for perfect/tied/inverted plus strict label validation."""
    y_true = [0, 0, 0, 1, 1, 1]
    assert compute_roc_auc(y_true, [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]) == 1.0
    assert compute_roc_auc(y_true, [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]) == 0.5
    assert compute_roc_auc([0, 1, 0, 1], [0.2, 0.2, 0.8, 0.8]) == 0.5
    assert compute_roc_auc(y_true, [0.9, 0.8, 0.7, 0.3, 0.2, 0.1]) == 0.0

    with pytest.raises(ValueError, match="binary"):
        compute_roc_auc([0, 1, 2], [0.1, 0.5, 0.9])
    with pytest.raises(ValueError, match="non-finite"):
        compute_roc_auc([0, np.nan], [0.1, 0.9])
    with pytest.raises(ValueError, match="non-finite"):
        compute_roc_auc([0, 1], [0.1, np.inf])


def test_group_kfold_indices_zero_leakage_balanced_and_validated() -> None:
    """group_kfold_indices guarantees zero leakage, balanced fold sizes, and min_class_count >= 2."""
    groups: list[str] = []
    labels: list[int] = []
    # 7 real groups + 7 sim groups across 5 folds -> running offset keeps fold sizes within +-1
    for page_i in range(7):
        groups.append(f"real_page_{page_i}")
        labels.append(0)
    for page_i in range(7):
        groups.append(f"sim_page_{page_i}")
        labels.append(1)

    splits = group_kfold_indices(groups=groups, labels=labels, n_splits=5, seed=123)
    assert len(splits) == 5
    fold_sizes = [len(v) for _, v in splits]
    assert max(fold_sizes) - min(fold_sizes) <= 1

    seen_val_indices: list[int] = []
    for train_idx, val_idx in splits:
        train_groups = {groups[int(i)] for i in train_idx}
        val_groups = {groups[int(i)] for i in val_idx}
        assert train_groups.isdisjoint(val_groups)
        assert len(np.intersect1d(train_idx, val_idx)) == 0
        assert {labels[int(i)] for i in val_idx} == {0, 1}
        seen_val_indices.extend(int(i) for i in val_idx)
    assert sorted(seen_val_indices) == list(range(len(groups)))

    # min_class_count < 2 must raise ValueError
    with pytest.raises(ValueError, match="min_class_count"):
        group_kfold_indices(groups=["r0", "s0", "s1"], labels=[0, 1, 1], n_splits=2)


def test_infer_manuscript_group() -> None:
    """Sibling pages of the same work collapse to one CV group; single-wrapper subdirs don't collapse domain."""
    root = Path("/data/real")
    assert infer_manuscript_group(root / "mozart_k458_p01.png", root, "real") == "real::mozart_k458"
    assert infer_manuscript_group(root / "mozart_k458_page12.jpg", root, "real") == "real::mozart_k458"
    assert infer_manuscript_group(root / "mozart_k458_pg03.jpg", root, "real") == "real::mozart_k458"
    assert infer_manuscript_group(root / "mozart_k458_f01r.jpg", root, "real") == "real::mozart_k458"
    assert infer_manuscript_group(root / "mozart_k458_fol02v.jpg", root, "real") == "real::mozart_k458"
    assert infer_manuscript_group(root / "mozart_k458_001.jpg", root, "real") == "real::mozart_k458"
    assert infer_manuscript_group(root / "mozart_k458-03.jpg", root, "real") == "real::mozart_k458"
    assert infer_manuscript_group(root / "debussy_l98" / "scan_01.png", root, "real") == "real::debussy_l98"
    assert infer_manuscript_group(root / "workA.jpg", root, "real") == "real::workA"
    assert infer_manuscript_group(root / "workB.jpg", root, "real") == "real::workB"

    # Single wrapper subfolder (root/rism_hw/*.jpg) must NOT collapse all manuscripts into 1 group.
    wrapped_paths = [
        root / "rism_hw" / "bach_bwv1_001.jpg",
        root / "rism_hw" / "bach_bwv1_f01r.jpg",
        root / "rism_hw" / "haydn_op76_001.jpg",
    ]
    wrapped_groups = infer_manuscript_groups(wrapped_paths, root=root, domain_prefix="real")
    assert wrapped_groups == ["real::bach_bwv1", "real::bach_bwv1", "real::haydn_op76"]

    # Multiple distinct subdirectories DO group by parent subdirectory.
    multi_sub_paths = [
        root / "ms_alpha" / "scan_001.jpg",
        root / "ms_alpha" / "scan_002.jpg",
        root / "ms_beta" / "scan_001.jpg",
    ]
    multi_groups = infer_manuscript_groups(multi_sub_paths, root=root, domain_prefix="real")
    assert multi_groups == ["real::ms_alpha", "real::ms_alpha", "real::ms_beta"]


def test_canonicalize_for_probe_exif_fallback_on_rgb_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RGB file path canonicalization succeeds even when EXIF transpose raises and falls back."""
    img = _make_synthetic_page((240, 320), bg_rgb=(235, 225, 200), ink_rgb=(40, 35, 30), seed=11)
    rgb_path = tmp_path / "rgb_page.jpg"
    img.save(rgb_path, format="JPEG", quality=92)

    def _boom(_im):
        raise ZeroDivisionError("malformed EXIF rational")

    monkeypatch.setattr(ImageOps, "exif_transpose", _boom)
    out = canonicalize_for_probe(rgb_path, max_dim=256)
    assert out.mode == "RGB"
    assert max(out.size) == 256
    assert np.asarray(out).shape[2] == 3


def test_canonicalize_for_probe_erases_png_vs_jpeg_shortcut(tmp_path: Path) -> None:
    """Identical image saved as PNG vs JPEG Q95 produces near-identical canonical tensor."""
    img = _make_synthetic_page((480, 640), bg_rgb=(238, 228, 206), ink_rgb=(42, 34, 28), seed=7)
    png_path = tmp_path / "page.png"
    jpg_path = tmp_path / "page.jpg"
    img.save(png_path, format="PNG")
    img.save(jpg_path, format="JPEG", quality=95)

    canon_png = canonicalize_for_probe(png_path, max_dim=512)
    canon_jpg = canonicalize_for_probe(jpg_path, max_dim=512)

    assert canon_png.size == canon_jpg.size
    assert max(canon_png.size) == 512

    arr_png = np.asarray(canon_png, dtype=np.float32) / 255.0
    arr_jpg = np.asarray(canon_jpg, dtype=np.float32) / 255.0
    mae = float(np.mean(np.abs(arr_png - arr_jpg)))
    assert mae < 0.015, f"PNG vs JPEG Q95 canonical MAE too high: {mae:.5f}"

    canon_gray = canonicalize_for_probe(png_path, max_dim=512, grayscale=True)
    assert canon_gray.mode == "L"


def test_staff_spacing_and_edge_gated_patch_sampling() -> None:
    """estimate_staff_spacing_px detects staff spacing and sample_ink_patches rejects flat dark borders."""
    page = _make_synthetic_page(
        (400, 520), bg_rgb=(235, 225, 205), ink_rgb=(35, 30, 25), seed=5, staff_spacing=10
    )
    gray = np.asarray(page.convert("L"), dtype=np.float32) / 255.0
    sp_px, prom = estimate_staff_spacing_px(gray)
    assert sp_px is not None
    assert 8.0 <= sp_px <= 12.0
    assert prom >= 0.08

    # Blank image -> no staff peak
    blank = np.full((200, 200), 0.9, dtype=np.float32)
    sp_blank, _ = estimate_staff_spacing_px(blank)
    assert sp_blank is None

    # Paint thick black scanner border around the outside; interior patches must stay bright/edged
    arr = np.asarray(page).copy()
    arr[:30, :] = 0
    arr[-30:, :] = 0
    arr[:, :30] = 0
    arr[:, -30:] = 0
    bordered = Image.fromarray(arr, mode="RGB")
    patches, stats = sample_ink_patches(bordered, num_patches=4, patch_size=128, seed=1, return_stats=True)
    assert len(patches) == 4
    assert all(p.size == (128, 128) for p in patches)
    assert stats["total"] == 4
    # Interior crop on letterbox_page
    lb = letterbox_page(bordered, size=128, interior_crop_frac=0.08)
    assert lb.size == (128, 128)


def test_orthogonal_six_view_patch_ladder() -> None:
    """All 6 orthogonal patch views + legacy aliases return finite (3, H, W) arrays in [-1, 1]."""
    page = _make_synthetic_page((256, 256), bg_rgb=(230, 215, 190), ink_rgb=(40, 35, 30), seed=3)
    for mode in (*PATCH_VIEW_MODES, "lowfreq", "highfreq"):
        t = transform_patch_view(page, mode=mode)
        assert t.shape == (3, 256, 256)
        assert t.dtype == np.float32
        assert np.all(np.isfinite(t))
        assert float(t.min()) >= -1.0 - 1e-5
        assert float(t.max()) <= 1.0 + 1e-5

    # Binary view must contain only {-1.0, +1.0}
    t_bin = transform_patch_view(page, mode="binary")
    unique_vals = set(np.unique(t_bin).tolist())
    assert unique_vals.issubset({-1.0, 1.0})


def test_cnn_bilateral_gradcam_and_atomic_overlay(tmp_path: Path) -> None:
    """GroupNorm+dilated PatchCNN & CoordConv+pyramid FullPageCNN Grad-CAM work even inside no_grad()."""
    import torch

    torch.manual_seed(0)
    patch_model = PatchForensicCNN(in_channels=3)
    patch_model.train()
    x_patch = torch.randn(2, 3, 128, 128)

    # Must succeed even when called inside torch.no_grad() and restore train() state
    with torch.no_grad():
        cam_pos = compute_gradcam(patch_model, x_patch, target_sign=1.0)
        cam_neg = compute_gradcam(patch_model, x_patch, target_sign=-1.0)
    assert patch_model.training is True
    assert len(patch_model.last_conv._forward_hooks) == 0
    assert all(p.grad is None for p in patch_model.parameters())
    assert not np.allclose(cam_pos, cam_neg)
    for cam in (cam_pos, cam_neg):
        assert cam.shape == (2, 128, 128)
        assert np.all(np.isfinite(cam))
        assert float(cam.min()) >= 0.0 and float(cam.max()) <= 1.0

    # Singleton batch (B=1) must work cleanly with GroupNorm in train() mode for BOTH CNNs
    out_p1 = patch_model(torch.randn(1, 3, 128, 128))
    assert out_p1.shape == (1,)

    fp_model = FullPageForensicCNN(in_channels=3)
    fp_model.train()
    x_fp = torch.randn(1, 3, 128, 128)
    out_fp = fp_model(x_fp)
    assert out_fp.shape == (1,)
    cam_fp = compute_gradcam(fp_model, x_fp, target_sign=1.0)
    assert cam_fp.shape == (1, 128, 128)
    assert np.all(np.isfinite(cam_fp))
    assert len(fp_model.last_conv._forward_hooks) == 0
    assert all(p.grad is None for p in fp_model.parameters())

    # Finite-loss guard raises FloatingPointError on NaN inputs
    nan_tensors = np.full((4, 3, 64, 64), np.nan, dtype=np.float32)
    nan_labels = np.array([0, 0, 1, 1], dtype=np.int64)
    nan_groups = ["r0", "r1", "s0", "s1"]
    with pytest.raises(FloatingPointError, match="Non-finite"):
        _train_and_eval_cnn_kfold(
            tensors=nan_tensors,
            labels=nan_labels,
            groups=nan_groups,
            model_factory=lambda: PatchForensicCNN(in_channels=3),
            folds=2,
            epochs=1,
            batch_size=2,
            device="cpu",
            seed=0,
        )

    # save_gradcam_overlay squeezes (1, H, W), sanitizes NaN, and writes atomically
    img = Image.new("RGB", (64, 64), (220, 210, 190))
    noisy_hm = np.full((1, 16, 16), np.nan, dtype=np.float32)
    out_png = save_gradcam_overlay(img, noisy_hm, tmp_path / "cam.png")
    assert out_png.is_file()
    assert not (tmp_path / "cam.png.tmp").exists()


def test_frechet_irls_woodbury_and_embedding_probe() -> None:
    """Test Frechet shrinkage/validation, IRLS Woodbury parity, PRDC, bootstrap CI & null control."""
    rng = np.random.default_rng(42)

    # Frechet validation
    with pytest.raises(ValueError, match="2D"):
        compute_frechet_distance(np.ones(5), np.ones(5))
    with pytest.raises(ValueError, match="non-finite"):
        compute_frechet_distance(np.array([[1.0, np.nan]]), np.array([[1.0, 2.0]]))
    with pytest.raises(ValueError, match="eps"):
        compute_frechet_distance(np.ones((3, 2)), np.ones((3, 2)), eps=-1e-5)

    base = rng.normal(0.0, 1.0, size=(30, 12))
    assert abs(compute_frechet_distance(base, base.copy())) < 1e-6
    assert abs(compute_frechet_distance(base, base.copy(), shrinkage=0.25)) < 1e-6

    # IRLS validation & true Woodbury (d > n) vs direct-Hessian solve numerical parity
    with pytest.raises(ValueError, match="l2_reg"):
        _fit_logistic_irls(np.ones((4, 2)), np.array([0.0, 0.0, 1.0, 1.0]), l2_reg=-1.0)

    x_hi = rng.normal(0.0, 1.0, size=(10, 32))
    y_hi = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=np.float64)
    l2 = 2.0
    w_wb, b_wb = _fit_logistic_irls(x_hi, y_hi, l2_reg=l2, max_iter=50, tol=1e-10)
    assert np.all(np.isfinite(w_wb)) and np.isfinite(b_wb)
    logits_wb = x_hi @ w_wb + b_wb
    assert compute_roc_auc(y_hi, logits_wb) > 0.95

    # Reference direct-Hessian Newton solve on the exact same d > n problem
    xb_ref = np.hstack([x_hi, np.ones((x_hi.shape[0], 1), dtype=np.float64)])
    reg_ref = np.full(x_hi.shape[1] + 1, l2, dtype=np.float64)
    reg_ref[-1] = 1e-6
    w_dir = np.zeros(x_hi.shape[1] + 1, dtype=np.float64)
    for _ in range(50):
        z_d = np.clip(xb_ref @ w_dir, -35.0, 35.0)
        p_d = 1.0 / (1.0 + np.exp(-z_d))
        wt_d = np.maximum(p_d * (1.0 - p_d), 1e-6)
        g_d = xb_ref.T @ (p_d - y_hi) + reg_ref * w_dir
        h_d = (xb_ref.T * wt_d) @ xb_ref + np.diag(reg_ref)
        step_d = np.linalg.solve(h_d, g_d)
        w_dir = w_dir - step_d
        if float(np.max(np.abs(step_d))) < 1e-12:
            break
    wb_full = np.append(w_wb, b_wb)
    assert float(np.max(np.abs(wb_full - w_dir))) < 1e-6

    # Default embedding ortho FFT produces 55 finite O(1) features
    page = _make_synthetic_page((256, 256), bg_rgb=(235, 225, 200), ink_rgb=(40, 35, 30), seed=9)
    emb = extract_default_embedding(page)
    assert emb.shape == (len(DEFAULT_EMBEDDING_FEATURE_SPEC),)
    assert np.all(np.isfinite(emb))
    assert float(np.max(np.abs(emb))) < 25.0

    # Separated vs identical distributions with PRDC + feature attribution + null control
    sep_r = rng.normal(-2.5, 0.6, size=(20, 8))
    sep_s = rng.normal(+2.5, 0.6, size=(20, 8))
    res_sep = run_embedding_probe(real_embeddings=sep_r, sim_embeddings=sep_s, folds=4, seed=42)
    assert res_sep["roc_auc"] > 0.95
    assert res_sep["standardized_fd"] > 10.0
    assert res_sep["coverage"] < 0.2
    assert len(res_sep["top_uncovered_real"]) == 5
    assert len(res_sep["feature_attribution"]["top_features"]) == 8
    assert res_sep["auc_ci_low"] <= res_sep["pooled_oof_auc"] <= res_sep["auc_ci_high"]

    # Null control on homogeneous Gaussian passes (repeated 5x with symmetric gate)
    nc = run_null_control(sep_r, [f"g_{i}" for i in range(len(sep_r))], folds=4, seed=42)
    assert nc["skipped"] is False
    assert nc["passed"] is True
    assert nc["n_repeats"] == 5
    assert len(nc["repeat_aucs"]) == 5

    # Bootstrap group CI sanity & upfront validation
    lo, hi = bootstrap_group_auc_ci([0, 0, 0, 1, 1, 1], [0.1, 0.2, 0.3, 0.7, 0.8, 0.9], [0, 1, 2, 3, 4, 5], n_boot=100)
    assert lo == 1.0 and hi == 1.0
    with pytest.raises(ValueError, match="binary"):
        bootstrap_group_auc_ci([0, 2], [0.1, 0.9], ["a", "b"])

    # PRDC mode collapse detection (tight collapse + missing second mode)
    wide_r = np.vstack([rng.normal(-5.0, 0.8, size=(20, 6)), rng.normal(+5.0, 0.8, size=(20, 6))])
    collapsed_s = rng.normal(-5.0, 0.1, size=(30, 6))
    prdc = compute_knn_prdc(wide_r, collapsed_s, k=5)
    assert prdc["variance_ratio"] < 0.05
    assert prdc["recall"] < 0.2
    assert prdc["coverage"] <= 0.55


def test_predegrade_sim_images_validation_and_strict_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """predegrade_sim_images handles empty list, negative seed, and raises RuntimeError on missing output."""
    assert predegrade_sim_images([], "fake-bin", tmp_path) == []
    img_path = tmp_path / "a.png"
    Image.new("RGB", (32, 32), (255, 255, 255)).save(img_path)

    with pytest.raises(ValueError, match="seed"):
        predegrade_sim_images([img_path], "fake-bin", tmp_path / "c", seed=-1)

    # Simulate photo-degrade returning 0 but producing no files -> must raise RuntimeError
    def _fake_run(cmd, **_kwargs):
        assert "--suffix" in cmd and "_degraded" in cmd
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", _fake_run)
    with pytest.raises(RuntimeError, match="did not produce outputs"):
        predegrade_sim_images([img_path], "fake-bin", tmp_path / "c2", seed=42)


def test_diagnostics_and_subsystem_scorecard_gating() -> None:
    """INDISTINGUISHABLE requires both low AUC AND healthy coverage/controls."""
    indist = diagnose_realism_gaps(0.52, 0.51, 0.50, 0.53, 0.54, 0.55, 0.12, coverage=0.85, variance_ratio=0.9)
    assert len(indist) == 1 and indist[0]["code"] == "INDISTINGUISHABLE"

    # Low coverage blocks INDISTINGUISHABLE and triggers MODE_COLLAPSE_RISK
    collapsed = diagnose_realism_gaps(0.52, 0.51, 0.50, 0.53, 0.54, 0.55, 0.12, coverage=0.40, variance_ratio=0.25)
    codes = {d["code"] for d in collapsed}
    assert "INDISTINGUISHABLE" not in codes
    assert "MODE_COLLAPSE_RISK" in codes

    # Underfit CNN probe blocks INDISTINGUISHABLE and triggers PROBE_UNDERFIT_WARNING
    underfit = diagnose_realism_gaps(
        0.52, 0.51, 0.50, 0.53, 0.54, 0.55, 0.12, coverage=0.85, variance_ratio=0.9, underfit_probes=["patch_rgb"]
    )
    uf_codes = {d["code"] for d in underfit}
    assert "INDISTINGUISHABLE" not in uf_codes
    assert "PROBE_UNDERFIT_WARNING" in uf_codes

    # Scale, aspect-ratio, SNR, null-control, and exposure gap diagnostics
    diag = diagnose_realism_gaps(
        0.92,
        0.80,
        0.60,
        0.62,
        0.65,
        0.68,
        2.0,
        auc_patch_gray_norm=0.60,
        auc_patch_binary=0.58,
        coverage=0.80,
        variance_ratio=0.85,
        domain_stats={
            "real": {"sp_px_p50": 10.0, "aspect_ratio_p50": 0.72, "landscape_frac": 0.0},
            "sim": {
                "sp_px_p50": 16.0,
                "aspect_ratio_p50": 1.35,
                "landscape_frac": 1.0,
                "michelson_contrast_mean": 0.15,
                "staff_peak_prominence_mean": 0.04,
            },
        },
        null_control={"passed": False, "null_auc": 0.78, "null_auc_ci_low": 0.65, "null_auc_ci_high": 0.90},
    )
    diag_codes = {d["code"] for d in diag}
    assert {
        "NULL_CONTROL_FAILED",
        "CONFOUNDER_SCALE_MISMATCH",
        "CONFOUNDER_ASPECT_RATIO",
        "LEGIBILITY_SNR_WARNING",
        "COLOR_TONE_GAP",
        "EXPOSURE_CONTRAST_GAP",
    }.issubset(diag_codes)

    card = build_subsystem_scorecard(
        auc_patch_rgb=0.88,
        auc_patch_gray=0.72,
        auc_patch_gray_norm=0.60,
        auc_patch_lowfreq_gray=0.65,
        auc_patch_highfreq_gray=0.78,
        auc_patch_binary=0.75,
        auc_fullpage=0.80,
        coverage=0.60,
        variance_ratio=0.55,
    )
    assert len(card) == 7
    assert all({"axis", "metric_name", "metric_value", "status", "target_modules", "action"} <= set(c) for c in card)

    # Ceiling-saturation masking on color_tint and exposure_contrast
    sat_card = build_subsystem_scorecard(
        auc_patch_rgb=0.96,
        auc_patch_gray=0.95,
        auc_patch_gray_norm=0.94,
        auc_patch_lowfreq_gray=0.90,
        auc_patch_highfreq_gray=0.92,
        auc_patch_binary=0.91,
        auc_fullpage=0.85,
        coverage=0.80,
        variance_ratio=0.85,
        embed_group_attribution={"color_moments": {"mean_abs_std_shift": 1.5}},
        domain_stats={"real": {"michelson_contrast_mean": 0.70}, "sim": {"michelson_contrast_mean": 0.65}},
    )
    sat_by_axis = {row["axis"]: row["status"] for row in sat_card}
    assert sat_by_axis["color_tint"] == "fail"
    assert sat_by_axis["exposure_contrast"] == "masked"


def test_end_to_end_realism_benchmark_and_arg_validation(tmp_path: Path) -> None:
    """End-to-end run_realism_benchmark skips corrupt files, writes report.json, bilateral Grad-CAMs & scorecard."""
    import torch

    real_dir = tmp_path / "real"
    sim_dir = tmp_path / "sim"
    out_dir = tmp_path / "out"
    real_dir.mkdir()
    sim_dir.mkdir()

    # Numeric argument validation
    with pytest.raises(ValueError, match="folds"):
        run_realism_benchmark(real_dir, sim_dir, out_dir, folds=1)
    with pytest.raises(ValueError, match="patch_size"):
        run_realism_benchmark(real_dir, sim_dir, out_dir, patch_size=8)
    with pytest.raises(ValueError, match="jpeg_quality"):
        run_realism_benchmark(real_dir, sim_dir, out_dir, jpeg_quality=0)
    with pytest.raises(ValueError, match="interior_crop_frac"):
        run_realism_benchmark(real_dir, sim_dir, out_dir, interior_crop_frac=0.9)
    with pytest.raises(ValueError, match="seed"):
        run_realism_benchmark(real_dir, sim_dir, out_dir, seed=-5)

    # CLI exposes --interior-crop-frac
    cli_args = _build_arg_parser().parse_args(
        ["--real-dir", str(real_dir), "--sim-dir", str(sim_dir), "--out-dir", str(out_dir), "--interior-crop-frac", "0.08"]
    )
    assert abs(cli_args.interior_crop_frac - 0.08) < 1e-9

    for i in range(4):
        real_im = _make_synthetic_page(
            (320, 420), bg_rgb=(220, 200, 165), ink_rgb=(65, 48, 32), seed=10 + i, line_thickness=3
        )
        real_im.save(real_dir / f"realwork{i}.jpg", format="JPEG", quality=88)

        sim_im = _make_synthetic_page(
            (320, 420), bg_rgb=(248, 248, 250), ink_rgb=(15, 15, 20), seed=100 + i, line_thickness=1
        )
        sim_im.save(sim_dir / f"simwork{i}.png", format="PNG")

    # Add a corrupt file to verify graceful skipping
    (real_dir / "corrupt.jpg").write_bytes(b"not-a-valid-jpeg")

    prev_cudnn = bool(torch.backends.cudnn.enabled)
    report = run_realism_benchmark(
        real_dir=real_dir,
        sim_dir=sim_dir,
        out_dir=out_dir,
        max_pages=10,
        patches_per_page=2,
        patch_size=128,
        fullpage_size=128,
        max_dim=320,
        epochs=1,
        folds=2,
        batch_size=4,
        device="cpu",
        top_k_gradcam=2,
        seed=42,
        no_cudnn=True,
    )
    assert bool(torch.backends.cudnn.enabled) == prev_cudnn

    report_file = out_dir / "report.json"
    assert report_file.is_file()
    assert not (out_dir / "report.json.tmp").exists()
    loaded = json.loads(report_file.read_text(encoding="utf-8"))

    assert loaded["config"]["skipped_real"] == 1
    assert loaded["config"]["skipped_sim"] == 0
    assert "domain_stats" in loaded and "real" in loaded["domain_stats"] and "sim" in loaded["domain_stats"]
    assert "null_control" in loaded
    assert "subsystem_scorecard" in loaded and len(loaded["subsystem_scorecard"]) == 7

    summary = loaded["summary"]
    for key in (
        "auc_patch_rgb",
        "auc_patch_gray",
        "auc_patch_gray_norm",
        "auc_patch_lowfreq_gray",
        "auc_patch_highfreq_gray",
        "auc_patch_binary",
        "auc_patch_lowfreq",
        "auc_patch_highfreq",
        "auc_fullpage",
        "auc_embed",
        "frechet_distance",
        "standardized_fd",
        "knn_precision",
        "knn_recall",
        "knn_density",
        "knn_coverage",
        "variance_ratio",
    ):
        assert key in summary
        assert np.isfinite(summary[key])

    # Verify bilateral Grad-CAMs on patch_rgb, patch_binary, patch_highfreq_gray, and fullpage
    for probe_key in ("patch_rgb", "patch_binary", "patch_highfreq_gray", "fullpage"):
        probe = loaded["probes"][probe_key]
        assert len(probe["gradcam_files"]) == 2
        assert len(probe["gradcam_real_files"]) == 2
        assert "fold_metrics" in probe and len(probe["fold_metrics"]) == 2
        for entry in probe["gradcam_files"] + probe["gradcam_real_files"]:
            cam_path = Path(entry["path"])
            assert cam_path.is_file()
            with Image.open(cam_path) as cam_img:
                assert cam_img.mode == "RGB"
                assert cam_img.size[0] > cam_img.size[1]

    # Verify Probe C feature attribution groups & top_uncovered_real fields
    attr_groups = loaded["probes"]["embedding"]["feature_attribution"]["groups"]
    assert set(attr_groups.keys()) == {
        "color_moments",
        "gradient_orientation",
        "radial_fft_8rings",
        "directional_gabor_2x4",
    }
    uncovered = loaded["probes"]["embedding"]["top_uncovered_real"]
    assert len(uncovered) > 0
    for item in uncovered:
        assert {"index", "group", "path", "nn_dist", "r_real", "covered"} <= set(item.keys())
        assert item["path"] is not None
