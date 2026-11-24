"""Generic ML dataset export & in-memory rendering utilities for `hw-score-sim`.

Provides:
- `RenderSample`: in-memory PIL image + schema-versioned GT metadata + per-system MusicXML slices.
- `render_sample(...)`: renders named styles or custom/sampled style dicts without mutating
  global `SCRIBE_STYLES`, releasing float render buffers before returning.
- `to_coco_annotations(...)` & `to_yolo_labels(...)`: pure-Python COCO & YOLO converters.
- `export_dataset_sample(...)`: atomic `*.tmp` + `os.replace` disk writer.
- `HandwrittenScoreDataset` & `to_hf_generator(...)`: lazy-`torch` PyTorch Dataset and
  HuggingFace `datasets.Dataset.from_generator` adapter.
"""

from __future__ import annotations

import copy
import gc
import hashlib
import json
import math
import os
import tempfile
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Union

from PIL import Image

from . import renderer, styles
from .musescore_compat import ParsedScore, export_score_to_musicxml, parse_score

SCHEMA_VERSION: str = "1.0"

COORD_SYSTEM: Dict[str, str] = {
    "origin": "top_left",
    "units": "px",
    "bbox_format": "xyxy",
    "bezier_eval": "uniform_x_cubic",
}

COCO_CATEGORIES: Tuple[Dict[str, Any], ...] = (
    {"id": 1, "name": "system", "supercategory": "music_layout"},
    {"id": 2, "name": "staff", "supercategory": "music_layout"},
)

_EPHEMERAL_STYLE_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Public Data Container
# ---------------------------------------------------------------------------


@dataclass
class RenderSample:
    """In-memory rendered handwritten score sample with ML-ready ground truth."""

    image: Image.Image
    meta: Dict[str, Any]
    musicxml_systems: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal Geometry, Style & Serialization Helpers
# ---------------------------------------------------------------------------


def _clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(val)))


def _eval_cubic_bezier_y(ctrl_pts: Sequence[Sequence[float]], t: float) -> float:
    if not ctrl_pts:
        return 0.0
    if len(ctrl_pts) == 4:
        mt = 1.0 - t
        return (
            (mt**3) * float(ctrl_pts[0][1])
            + 3.0 * (mt**2) * t * float(ctrl_pts[1][1])
            + 3.0 * mt * (t**2) * float(ctrl_pts[2][1])
            + (t**3) * float(ctrl_pts[3][1])
        )
    return (1.0 - t) * float(ctrl_pts[0][1]) + t * float(ctrl_pts[-1][1])


def _compute_staff_bbox_xyxy(
    staff: Dict[str, Any],
    page_w: int,
    page_h: int,
    n_samples: int = 33,
) -> List[float]:
    """Compute `[x0, y0, x1, y1]` enclosing the 5-line curved staff."""
    sp = float(staff.get("sp", 16.0))
    ctrl_pts = staff.get("ctrl_pts") or []
    x_left = float(staff.get("x_left", ctrl_pts[0][0] if ctrl_pts else 0.0))
    x_right = float(staff.get("x_right", ctrl_pts[-1][0] if ctrl_pts else float(page_w)))

    if ctrl_pts:
        ys = [
            _eval_cubic_bezier_y(ctrl_pts, i / float(max(1, n_samples - 1)))
            for i in range(n_samples)
        ]
        xs = [float(pt[0]) for pt in ctrl_pts]
        y_min_c = min(ys)
        y_max_c = max(ys)
        x0 = min(x_left, min(xs))
        x1 = max(x_right, max(xs))
    else:
        yc = float(staff.get("y_center", page_h * 0.5))
        y_min_c = yc
        y_max_c = yc
        x0 = x_left
        x1 = x_right

    x0_c = _clamp(x0, 0.0, float(page_w))
    x1_c = _clamp(x1, 0.0, float(page_w))
    y0_c = _clamp(y_min_c - 2.0 * sp, 0.0, float(page_h))
    y1_c = _clamp(y_max_c + 2.0 * sp, 0.0, float(page_h))
    if x1_c <= x0_c:
        x1_c = min(float(page_w), x0_c + 1.0)
    if y1_c <= y0_c:
        y1_c = min(float(page_h), y0_c + 1.0)
    return [round(x0_c, 2), round(y0_c, 2), round(x1_c, 2), round(y1_c, 2)]


def _compute_system_bbox_xyxy(
    staves: Sequence[Dict[str, Any]],
    page_w: int,
    page_h: int,
) -> List[float]:
    """Compute `[x0, y0, x1, y1]` enclosing all staves in a system plus notation margin."""
    if not staves:
        return [0.0, 0.0, 0.0, 0.0]

    staff_boxes = [
        st["bbox_xyxy"] if "bbox_xyxy" in st else _compute_staff_bbox_xyxy(st, page_w, page_h)
        for st in staves
    ]
    sp = float(staves[0].get("sp", 16.0))
    x0 = _clamp(min(b[0] for b in staff_boxes) - 2.5 * sp, 0.0, float(page_w))
    y0 = _clamp(min(b[1] for b in staff_boxes) - 3.2 * sp, 0.0, float(page_h))
    x1 = _clamp(max(b[2] for b in staff_boxes) + 1.5 * sp, 0.0, float(page_w))
    y1 = _clamp(max(b[3] for b in staff_boxes) + 3.2 * sp, 0.0, float(page_h))
    if x1 <= x0:
        x1 = min(float(page_w), x0 + 1.0)
    if y1 <= y0:
        y1 = min(float(page_h), y0 + 1.0)
    return [round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2)]


def _json_safe(obj: Any) -> Any:
    """Recursively convert tuples/numpy scalars/paths to plain JSON-serializable types."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "item") and callable(obj.item):
        try:
            return obj.item()
        except Exception:
            pass
    return obj


def _style_hash(style_dict: Dict[str, Any]) -> str:
    canonical = json.dumps(_json_safe(style_dict), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _extract_style_metadata(style_dict: Dict[str, Any], display_name: str) -> Dict[str, Any]:
    keys = (
        "era",
        "description",
        "notehead_archetype",
        "stem_archetype",
        "flag_archetype",
        "beam_archetype",
        "ledger_style",
        "clef_archetype",
        "keysig_archetype",
        "barline_archetype",
        "paper_type",
        "tags",
        "ink_rgb",
        "staff_ink_rgb",
        "paper_rgb",
        "stroke_scale",
        "wobble_px",
        "tremor_px",
        "slant_deg",
        "staff_curve_px",
        "staff_slope_deg",
        "ink_dropout",
        "paper_grain",
        "paper_mottle",
        "bleed_through",
    )
    meta: Dict[str, Any] = {"name": display_name}
    for k in keys:
        if k in style_dict and style_dict[k] is not None:
            meta[k] = _json_safe(style_dict[k])
    return meta


def _resolve_custom_style(scribe_dict: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """Validate and normalize a custom/sampled style dict without mutating global registry."""
    spec = copy.deepcopy(scribe_dict)
    display_name = str(spec.get("name") or "custom_scribe")

    parent_name = spec.get("extends")
    if parent_name is not None:
        base = styles.get_style(str(parent_name))
        spec.pop("extends", None)
        merged = copy.deepcopy(base)
        for k, v in spec.items():
            if isinstance(v, dict) and isinstance(merged.get(k), dict):
                merged[k].update(v)
            else:
                merged[k] = v
        spec = merged

    # Fill missing metadata defaults cleanly so partial custom dicts still validate if needed
    spec.setdefault("name", display_name)
    spec.setdefault("description", f"Custom scribe profile ({display_name}).")
    spec.setdefault("era", "custom")

    validated = styles.validate_style(spec, name=display_name)
    return display_name, validated


def _derive_item_seed(base_seed: int, index: int) -> int:
    payload = f"hw_score_sim:{int(base_seed)}:{int(index)}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "little") & 0x7FFFFFFF


def _is_inline_xml(value: Union[str, Path, ParsedScore]) -> bool:
    if not isinstance(value, str):
        return False
    stripped = value.lstrip()
    return stripped.startswith("<")


def _materialize_score_input(
    score_path_or_xml: Union[str, Path, ParsedScore],
    tmp_dir: Path,
) -> Tuple[Path, str, ParsedScore]:
    """Return `(fs_path, source_label, parsed_score)` for file path, inline XML, or ParsedScore."""
    if isinstance(score_path_or_xml, ParsedScore):
        tmp_xml = tmp_dir / "input_score.musicxml"
        export_score_to_musicxml(score_path_or_xml, tmp_xml)
        return tmp_xml, score_path_or_xml.source_path or "<ParsedScore>", score_path_or_xml

    if _is_inline_xml(score_path_or_xml):
        xml_text = str(score_path_or_xml)
        is_mscx = "<museScore" in xml_text or "<musescore" in xml_text.lower()
        ext = ".mscx" if is_mscx else ".musicxml"
        tmp_xml = tmp_dir / f"inline_score{ext}"
        tmp_xml.write_text(xml_text, encoding="utf-8")
        parsed = parse_score(tmp_xml)
        return tmp_xml, "<inline_xml>", parsed

    p = Path(score_path_or_xml)
    parsed = parse_score(p)
    return p, str(p), parsed


# ---------------------------------------------------------------------------
# In-Memory Rendering API
# ---------------------------------------------------------------------------


def render_sample(
    score_path_or_xml: Union[str, Path, ParsedScore],
    scribe: Union[str, Dict[str, Any]],
    *,
    seed: int = 42,
    page_w: int = 1650,
    page_h: int = 2200,
    include_musicxml: bool = True,
) -> RenderSample:
    """Render a score in memory and return a `RenderSample` with schema-1.0 GT metadata.

    Parameters
    ----------
    score_path_or_xml:
        Path to a `.musicxml`/`.xml`/`.mxl`/`.mscx`/`.mscz` file, raw XML string, or `ParsedScore`.
    scribe:
        Either a registered style name (`str`) or a custom/sampled style `dict`.
        Custom dicts are never persisted to the global `styles.SCRIBE_STYLES` registry.
    seed:
        Deterministic RNG seed.
    page_w, page_h:
        Output page dimensions in pixels.
    include_musicxml:
        When True, generates pure-Python per-system MusicXML 3.1 strings via
        `export_score_to_musicxml`.
    """
    if isinstance(scribe, str):
        display_name = scribe
        resolved_style = styles.get_style(scribe)
        ephemeral_key: Optional[str] = None
        render_style_name = scribe
    elif isinstance(scribe, dict):
        display_name, resolved_style = _resolve_custom_style(scribe)
        ephemeral_key = f"__hw_ephemeral_{uuid.uuid4().hex}__"
        render_style_name = ephemeral_key
    else:
        raise TypeError(f"scribe must be a style name (str) or style dict, got {type(scribe).__name__}")

    with tempfile.TemporaryDirectory(prefix="hw_score_sim_mem_") as td:
        tmp_dir = Path(td)
        fs_score_path, source_label, parsed_score = _materialize_score_input(score_path_or_xml, tmp_dir)
        out_png = tmp_dir / "render.png"

        if ephemeral_key is not None:
            with _EPHEMERAL_STYLE_LOCK:
                styles.SCRIBE_STYLES[ephemeral_key] = copy.deepcopy(resolved_style)
            try:
                raw_gt = renderer.render_score_handwritten(
                    score_path=fs_score_path,
                    style_name=render_style_name,
                    out_png=out_png,
                    seed=int(seed),
                    page_w=int(page_w),
                    page_h=int(page_h),
                    export_train_dir=None,
                )
            finally:
                with _EPHEMERAL_STYLE_LOCK:
                    styles.SCRIBE_STYLES.pop(ephemeral_key, None)
        else:
            raw_gt = renderer.render_score_handwritten(
                score_path=fs_score_path,
                style_name=render_style_name,
                out_png=out_png,
                seed=int(seed),
                page_w=int(page_w),
                page_h=int(page_h),
                export_train_dir=None,
            )

        # Load PIL image cleanly into memory before temp directory cleanup
        with Image.open(out_png) as opened:
            image = opened.convert("RGB").copy()

        # Enrich systems & staves with xyxy bounding boxes and per-system MusicXML slices
        enriched_systems: List[Dict[str, Any]] = []
        musicxml_systems: List[str] = []
        max_measure_end = 0

        for sys_dict in raw_gt.get("systems", []):
            is_blank = bool(sys_dict.get("is_blank_prerule", False))
            m_start = int(sys_dict.get("measure_start", 0))
            m_end = int(sys_dict.get("measure_end", 0))
            if not is_blank:
                max_measure_end = max(max_measure_end, m_end)

            enriched_staves: List[Dict[str, Any]] = []
            for st in sys_dict.get("staves", []):
                st_copy = dict(st)
                st_copy["bbox_xyxy"] = _compute_staff_bbox_xyxy(st_copy, int(page_w), int(page_h))
                enriched_staves.append(st_copy)

            sys_bbox = _compute_system_bbox_xyxy(enriched_staves, int(page_w), int(page_h))

            mxml_index: Optional[int] = None
            if include_musicxml and not is_blank and m_end > m_start:
                mxml_index = len(musicxml_systems)
                slice_path = tmp_dir / f"sys_{int(sys_dict.get('system_index', mxml_index)):03d}.musicxml"
                export_score_to_musicxml(
                    parsed_score,
                    slice_path,
                    measure_range=(m_start, m_end),
                )
                musicxml_systems.append(slice_path.read_text(encoding="utf-8"))

            enriched_systems.append(
                {
                    "system_index": int(sys_dict.get("system_index", len(enriched_systems))),
                    "measure_start": m_start,
                    "measure_end": m_end,
                    "is_blank_prerule": is_blank,
                    "bbox_xyxy": sys_bbox,
                    "musicxml_index": mxml_index,
                    "staves": enriched_staves,
                }
            )

    total_measures = int(parsed_score.num_measures)
    truncated = bool(max_measure_end < total_measures)

    try:
        from . import __version__ as pkg_version
    except Exception:
        pkg_version = "0.1.0"

    meta: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "coord_system": dict(COORD_SYSTEM),
        "reproducibility": {
            "seed": int(seed),
            "page_w": int(page_w),
            "page_h": int(page_h),
            "style_name": display_name,
            "style_hash": _style_hash(resolved_style),
            "score_source": source_label,
            "hw_score_sim_version": str(pkg_version),
        },
        "style_metadata": _extract_style_metadata(resolved_style, display_name),
        "score_title": str(raw_gt.get("score_title", parsed_score.title)),
        "source_path": source_label,
        "style": display_name,
        "seed": int(seed),
        "page_w": int(page_w),
        "page_h": int(page_h),
        "total_measures": total_measures,
        "rendered_measures": max_measure_end,
        "truncated": truncated,
        "systems": enriched_systems,
    }

    # Release transient float buffers from rendering promptly
    del raw_gt, parsed_score
    gc.collect()

    return RenderSample(image=image, meta=meta, musicxml_systems=musicxml_systems)


# ---------------------------------------------------------------------------
# Pure-Python COCO & YOLO Converters
# ---------------------------------------------------------------------------


def _ensure_system_and_staff_boxes(meta: Dict[str, Any]) -> Tuple[int, int, List[Dict[str, Any]]]:
    page_w = int(meta["page_w"])
    page_h = int(meta["page_h"])
    systems_out: List[Dict[str, Any]] = []
    for sys_item in meta.get("systems", []):
        staves_out: List[Dict[str, Any]] = []
        for st in sys_item.get("staves", []):
            st_copy = dict(st)
            if "bbox_xyxy" not in st_copy:
                st_copy["bbox_xyxy"] = _compute_staff_bbox_xyxy(st_copy, page_w, page_h)
            staves_out.append(st_copy)
        sys_copy = dict(sys_item)
        sys_copy["staves"] = staves_out
        if "bbox_xyxy" not in sys_copy:
            sys_copy["bbox_xyxy"] = _compute_system_bbox_xyxy(staves_out, page_w, page_h)
        systems_out.append(sys_copy)
    return page_w, page_h, systems_out


def _unwrap_meta(meta_or_sample: Union[Dict[str, Any], RenderSample]) -> Dict[str, Any]:
    if isinstance(meta_or_sample, RenderSample):
        return meta_or_sample.meta
    if isinstance(meta_or_sample, dict):
        return meta_or_sample
    raise TypeError(
        f"Expected RenderSample or metadata dict, got {type(meta_or_sample).__name__}"
    )


def to_coco_annotations(
    meta: Union[Dict[str, Any], RenderSample, Sequence[Union[Dict[str, Any], RenderSample]]],
    *,
    image_id: int = 1,
    file_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Convert a `RenderSample`, `RenderSample.meta` dict, or sequence of them into COCO format.

    Categories:
      - `1`: `system`
      - `2`: `staff`
    Bounding boxes follow COCO `[x, y, width, height]` pixel convention.
    """
    if isinstance(meta, (dict, RenderSample)):
        items: List[Dict[str, Any]] = [_unwrap_meta(meta)]
        single_mode = True
    elif isinstance(meta, Sequence) and not isinstance(meta, (str, bytes)):
        items = [_unwrap_meta(item) for item in meta]
        single_mode = False
    else:
        raise TypeError(
            f"Expected RenderSample, metadata dict, or sequence of them, got {type(meta).__name__}"
        )

    images: List[Dict[str, Any]] = []
    annotations: List[Dict[str, Any]] = []
    ann_id = 1
    schema_version = SCHEMA_VERSION

    for idx, item_meta in enumerate(items):
        cur_image_id = int(image_id) + idx
        schema_version = str(item_meta.get("schema_version", schema_version))
        page_w, page_h, systems_list = _ensure_system_and_staff_boxes(item_meta)
        resolved_name = (
            file_name
            if (single_mode and file_name)
            else str(item_meta.get("file_name") or f"image_{cur_image_id:06d}.png")
        )
        images.append(
            {
                "id": cur_image_id,
                "width": page_w,
                "height": page_h,
                "file_name": resolved_name,
            }
        )

        for sys_item in systems_list:
            sx0, sy0, sx1, sy1 = [float(v) for v in sys_item["bbox_xyxy"]]
            sw = round(max(0.0, sx1 - sx0), 2)
            sh = round(max(0.0, sy1 - sy0), 2)
            if sw > 0.0 and sh > 0.0:
                annotations.append(
                    {
                        "id": ann_id,
                        "image_id": cur_image_id,
                        "category_id": 1,
                        "bbox": [round(sx0, 2), round(sy0, 2), sw, sh],
                        "area": round(sw * sh, 2),
                        "iscrowd": 0,
                        "attributes": {
                            "system_index": int(sys_item.get("system_index", 0)),
                            "measure_start": int(sys_item.get("measure_start", 0)),
                            "measure_end": int(sys_item.get("measure_end", 0)),
                            "is_blank_prerule": bool(sys_item.get("is_blank_prerule", False)),
                        },
                    }
                )
                ann_id += 1

            for st in sys_item.get("staves", []):
                bx0, by0, bx1, by1 = [float(v) for v in st["bbox_xyxy"]]
                bw = round(max(0.0, bx1 - bx0), 2)
                bh = round(max(0.0, by1 - by0), 2)
                if bw > 0.0 and bh > 0.0:
                    annotations.append(
                        {
                            "id": ann_id,
                            "image_id": cur_image_id,
                            "category_id": 2,
                            "bbox": [round(bx0, 2), round(by0, 2), bw, bh],
                            "area": round(bw * bh, 2),
                            "iscrowd": 0,
                            "attributes": {
                                "system_index": int(sys_item.get("system_index", 0)),
                                "part_index": int(st.get("part_index", 0)),
                                "sp": float(st.get("sp", 16.0)),
                                "ctrl_pts": st.get("ctrl_pts", []),
                                "is_blank_prerule": bool(st.get("is_blank_prerule", False)),
                            },
                        }
                    )
                    ann_id += 1

    return {
        "info": {
            "description": "hw-score-sim synthetic handwritten sheet music annotations",
            "schema_version": schema_version,
        },
        "images": images,
        "annotations": annotations,
        "categories": [dict(c) for c in COCO_CATEGORIES],
    }


def to_yolo_labels(meta: Union[Dict[str, Any], RenderSample]) -> List[str]:
    """Convert a `RenderSample` or `RenderSample.meta` dict into normalized YOLO label lines.

    Classes:
      - `0`: `system`
      - `1`: `staff`
    Each line has the format: `class_id cx cy w h` with coordinates in `[0.0, 1.0]`.
    """
    resolved_meta = _unwrap_meta(meta)
    page_w, page_h, systems_list = _ensure_system_and_staff_boxes(resolved_meta)
    if page_w <= 0 or page_h <= 0:
        return []

    inv_w = 1.0 / float(page_w)
    inv_h = 1.0 / float(page_h)
    lines: List[str] = []

    def _append_box(class_id: int, box_xyxy: Sequence[float]) -> None:
        x0, y0, x1, y1 = [float(v) for v in box_xyxy]
        bw = (x1 - x0) * inv_w
        bh = (y1 - y0) * inv_h
        if bw <= 0.0 or bh <= 0.0:
            return
        cx = _clamp((x0 + x1) * 0.5 * inv_w, 0.0, 1.0)
        cy = _clamp((y0 + y1) * 0.5 * inv_h, 0.0, 1.0)
        bw = _clamp(bw, 0.0, 1.0)
        bh = _clamp(bh, 0.0, 1.0)
        lines.append(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    for sys_item in systems_list:
        _append_box(0, sys_item["bbox_xyxy"])
        for st in sys_item.get("staves", []):
            _append_box(1, st["bbox_xyxy"])

    return lines


# ---------------------------------------------------------------------------
# Atomic Disk Export
# ---------------------------------------------------------------------------


def _atomic_tmp_path(target: Path) -> Path:
    return target.with_name(f".{target.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")


def _atomic_write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _atomic_tmp_path(path)
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    except Exception:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        raise
    return path


def _atomic_save_png(path: Path, image: Image.Image) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _atomic_tmp_path(path)
    try:
        image.save(tmp, format="PNG")
        os.replace(tmp, path)
    except Exception:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        raise
    return path


def export_dataset_sample(
    sample: RenderSample,
    out_dir: Union[str, Path],
    stem: str,
    *,
    formats: Sequence[str] = ("json", "coco", "yolo", "musicxml"),
    image_id: int = 1,
) -> Dict[str, Any]:
    """Atomically write a `RenderSample` to disk (`*.tmp` + `os.replace`).

    Always writes `{stem}.png`, plus any requested formats from:
      - `"json"`     -> `{stem}.json`
      - `"coco"`     -> `{stem}.coco.json`
      - `"yolo"`     -> `{stem}.txt`
      - `"musicxml"` -> `{stem}_sys{idx:02d}.musicxml` (one per system slice)
    """
    valid_formats = {"json", "coco", "yolo", "musicxml"}
    req_formats = tuple(str(f).lower() for f in formats)
    unknown = set(req_formats) - valid_formats
    if unknown:
        raise ValueError(f"Unsupported export format(s): {sorted(unknown)}. Valid: {sorted(valid_formats)}")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    written: Dict[str, Any] = {}
    img_path = out_path / f"{stem}.png"
    _atomic_save_png(img_path, sample.image)
    written["image"] = str(img_path)

    if "json" in req_formats:
        json_path = out_path / f"{stem}.json"
        _atomic_write_text(json_path, json.dumps(_json_safe(sample.meta), indent=2))
        written["json"] = str(json_path)

    if "coco" in req_formats:
        coco_path = out_path / f"{stem}.coco.json"
        coco_payload = to_coco_annotations(sample.meta, image_id=image_id, file_name=img_path.name)
        _atomic_write_text(coco_path, json.dumps(coco_payload, indent=2))
        written["coco"] = str(coco_path)

    if "yolo" in req_formats:
        yolo_path = out_path / f"{stem}.txt"
        yolo_lines = to_yolo_labels(sample.meta)
        body = ("\n".join(yolo_lines) + "\n") if yolo_lines else ""
        _atomic_write_text(yolo_path, body)
        written["yolo"] = str(yolo_path)

    if "musicxml" in req_formats:
        mxml_paths: List[str] = []
        for idx, xml_str in enumerate(sample.musicxml_systems):
            mxml_path = out_path / f"{stem}_sys{idx:02d}.musicxml"
            _atomic_write_text(mxml_path, xml_str)
            mxml_paths.append(str(mxml_path))
        written["musicxml"] = mxml_paths

    return written


# ---------------------------------------------------------------------------
# PyTorch Dataset (Lazy-Torch) & HuggingFace Generator
# ---------------------------------------------------------------------------


def _select_scribe(
    scribes: Optional[
        Union[
            str,
            Dict[str, Any],
            Sequence[Union[str, Dict[str, Any]]],
            Callable[[int, int], Union[str, Dict[str, Any]]],
        ]
    ],
    index: int,
    item_seed: int,
) -> Union[str, Dict[str, Any]]:
    if scribes is None:
        all_names = styles.list_styles()
        return all_names[index % len(all_names)]
    if isinstance(scribes, (str, dict)):
        return scribes
    if callable(scribes):
        return scribes(index, item_seed)
    seq = list(scribes)
    if not seq:
        raise ValueError("scribes sequence must not be empty")
    return seq[index % len(seq)]


class HandwrittenScoreDataset:
    """Lightweight, lazy-`torch` PyTorch-compatible Dataset for on-the-fly rendering.

    Does not import `torch` unless `return_tensors=True` is requested (or when wrapped in a
    PyTorch `DataLoader`, which only requires `__len__` and `__getitem__`).
    """

    def __init__(
        self,
        score_paths: Sequence[Union[str, Path, ParsedScore]],
        *,
        scribes: Optional[
            Union[
                str,
                Dict[str, Any],
                Sequence[Union[str, Dict[str, Any]]],
                Callable[[int, int], Union[str, Dict[str, Any]]],
            ]
        ] = None,
        seed: int = 42,
        page_w: int = 1650,
        page_h: int = 2200,
        include_musicxml: bool = True,
        return_tensors: bool = False,
        transform: Optional[Callable[[RenderSample], Any]] = None,
    ) -> None:
        self.score_paths = list(score_paths)
        self.scribes = scribes
        self.seed = int(seed)
        self.page_w = int(page_w)
        self.page_h = int(page_h)
        self.include_musicxml = bool(include_musicxml)
        self.return_tensors = bool(return_tensors)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.score_paths)

    def __getitem__(self, index: int) -> Any:
        if index < 0:
            index += len(self.score_paths)
        if index < 0 or index >= len(self.score_paths):
            raise IndexError(f"Dataset index {index} out of range for length {len(self.score_paths)}")

        item_seed = _derive_item_seed(self.seed, index)
        scribe = _select_scribe(self.scribes, index, item_seed)
        sample = render_sample(
            self.score_paths[index],
            scribe,
            seed=item_seed,
            page_w=self.page_w,
            page_h=self.page_h,
            include_musicxml=self.include_musicxml,
        )

        if self.transform is not None:
            return self.transform(sample)

        if self.return_tensors:
            import numpy as np
            import torch  # Lazy import: only required when tensors are explicitly requested

            arr = np.asarray(sample.image, dtype=np.float32) / 255.0
            img_tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
            return {
                "image": img_tensor,
                "meta": sample.meta,
                "musicxml_systems": sample.musicxml_systems,
            }

        return sample


def to_hf_generator(
    score_paths: Sequence[Union[str, Path, ParsedScore]],
    *,
    scribes: Optional[
        Union[
            str,
            Dict[str, Any],
            Sequence[Union[str, Dict[str, Any]]],
            Callable[[int, int], Union[str, Dict[str, Any]]],
        ]
    ] = None,
    seed: int = 42,
    page_w: int = 1650,
    page_h: int = 2200,
    include_musicxml: bool = True,
    include_coco: bool = False,
    include_yolo: bool = False,
    max_samples: Optional[int] = None,
) -> Iterator[Dict[str, Any]]:
    """Generator yielding HuggingFace `datasets.Dataset.from_generator`-ready records.

    Usage with HuggingFace `datasets`::

        ds = datasets.Dataset.from_generator(
            to_hf_generator,
            gen_kwargs={"score_paths": paths, "seed": 42},
        )
    """
    limit = len(score_paths) if max_samples is None else min(len(score_paths), int(max_samples))
    for idx in range(limit):
        item_seed = _derive_item_seed(seed, idx)
        scribe = _select_scribe(scribes, idx, item_seed)
        sample = render_sample(
            score_paths[idx],
            scribe,
            seed=item_seed,
            page_w=page_w,
            page_h=page_h,
            include_musicxml=include_musicxml,
        )
        record: Dict[str, Any] = {
            "image": sample.image,
            "score_source": str(sample.meta["source_path"]),
            "style": str(sample.meta["style"]),
            "seed": int(sample.meta["seed"]),
            "page_w": int(sample.meta["page_w"]),
            "page_h": int(sample.meta["page_h"]),
            "truncated": bool(sample.meta["truncated"]),
            "meta_json": json.dumps(_json_safe(sample.meta)),
            "musicxml_systems": list(sample.musicxml_systems),
        }
        if include_coco:
            record["coco_json"] = json.dumps(to_coco_annotations(sample.meta, image_id=idx + 1))
        if include_yolo:
            record["yolo_labels"] = to_yolo_labels(sample.meta)
        yield record
