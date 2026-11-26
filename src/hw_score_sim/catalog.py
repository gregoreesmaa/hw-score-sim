"""Faceted 4-axis paleographic catalog and deterministic legibility scoring.

Maps every registered scribe style (all 234 built-in + pack styles, as well as
any user-registered style) onto four orthogonal canonical axes without mutating
any JSON style packs:

- ``era``: historical period (8 canonical values)
- ``medium``: musical context / manuscript function (10 canonical values)
- ``implement``: writing instrument (10 canonical values)
- ``legibility``: deterministic physics-scored readability tier ``1..5``
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Optional, Union

from . import styles as _styles_mod
from .styles import (
    NOTEHEAD_ARCHETYPES,
    PAPER_TYPES,
    SCRIBE_STYLES,
    get_style,
    list_styles,
)

_DESCRIBE_CACHE_VERSION: int = -1
_DESCRIBE_CACHE: dict[str, tuple[int, dict[str, Any]]] = {}

CANONICAL_ERAS: tuple[str, ...] = (
    "renaissance",
    "baroque",
    "classical",
    "romantic",
    "impressionist",
    "modernist",
    "mid_century",
    "contemporary",
)

HISTORICAL_ERAS: tuple[str, ...] = (
    "renaissance",
    "baroque",
    "classical",
    "romantic",
    "impressionist",
)

HISTORICAL_IMPLEMENTS: tuple[str, ...] = (
    "quill",
    "steel_nib",
    "fountain_pen",
    "graphite_pencil",
)

HISTORICAL_FORBIDDEN_MEDIA: tuple[str, ...] = (
    "jazz_lead_sheet",
    "film_studio",
)

ERA_PRESETS: dict[str, tuple[str, ...]] = {
    "historical": HISTORICAL_ERAS,
}

CANONICAL_MEDIA: tuple[str, ...] = (
    "autograph_sketch",
    "fair_copy",
    "orchestral_conductor",
    "chamber_solo",
    "choral_liturgical",
    "jazz_lead_sheet",
    "film_studio",
    "folk_vernacular",
    "student_everyday",
    "repro_archival",
)

CANONICAL_IMPLEMENTS: tuple[str, ...] = (
    "quill",
    "steel_nib",
    "fountain_pen",
    "technical_pen",
    "graphite_pencil",
    "mech_pencil",
    "grease_pencil",
    "felt_marker",
    "ballpoint",
    "gel_rollerball",
)

LEGIBILITY_LABELS: dict[int, str] = {
    1: "engraver",
    2: "neat",
    3: "working",
    4: "hurried",
    5: "wild",
}

_LABEL_TO_LEGIBILITY: dict[str, int] = {v: k for k, v in LEGIBILITY_LABELS.items()}

# Raw style["era"] -> canonical era mapping covering all 45 raw era labels.
_RAW_ERA_TO_CANONICAL: dict[str, str] = {
    "renaissance": "renaissance",
    "early_baroque": "baroque",
    "baroque": "baroque",
    "classical_baroque": "baroque",
    "galant_classical": "classical",
    "classical": "classical",
    "classical_romantic": "romantic",
    "late_classical_romantic": "romantic",
    "early_romantic": "romantic",
    "romantic": "romantic",
    "late_romantic": "romantic",
    "19c_american_folk": "romantic",
    "19c_celtic_folk": "romantic",
    "archival": "romantic",
    "impressionist": "impressionist",
    "late_romantic_impressionist": "impressionist",
    "early_20c_modernist": "modernist",
    "early_20c_folk": "modernist",
    "1920s_tin_pan_alley": "modernist",
    "modern": "modernist",
    "modern_classic": "modernist",
    "avant_garde": "modernist",
    "mid_century": "mid_century",
    "mid_century_brazilian": "mid_century",
    "mid_century_broadway": "mid_century",
    "mid_century_jazz": "mid_century",
    "mid_century_studio": "mid_century",
    "1940s_bebop": "mid_century",
    "1940s_swing": "mid_century",
    "1940s_tango": "mid_century",
    "1950s_orchestral_jazz": "mid_century",
    "1960s_blues_soul": "mid_century",
    "1960s_hardbop": "mid_century",
    "1970s_afrobeat_funk": "mid_century",
    "1970s_jazz": "mid_century",
    "20c_nordic_folk": "mid_century",
    "contemporary": "contemporary",
    "contemporary_broadway": "contemporary",
    "contemporary_folk": "contemporary",
    "contemporary_gospel": "contemporary",
    "contemporary_indie": "contemporary",
    "contemporary_marching": "contemporary",
    "contemporary_street": "contemporary",
    "contemporary_studio": "contemporary",
    "folk_traditional": "contemporary",
}


def _blob(style: dict[str, Any]) -> str:
    """Build a lowercase searchable text blob from name, description, era, and tags."""
    parts = [
        str(style.get("name", "")),
        str(style.get("description", "")),
        str(style.get("era", "")),
        " ".join(str(t) for t in style.get("tags", ())),
    ]
    return " ".join(parts).lower()


def classify_era(style: dict[str, Any]) -> str:
    """Map a style dict to one of ``CANONICAL_ERAS``."""
    raw = str(style.get("era", "")).strip().lower()
    if raw in _RAW_ERA_TO_CANONICAL:
        # Special case: modern_jazz_lead_sheet is mid-20th-century Real Book era.
        if style.get("name") == "modern_jazz_lead_sheet":
            return "mid_century"
        return _RAW_ERA_TO_CANONICAL[raw]

    text = _blob(style)
    if "renaissance" in text or "tudor" in text:
        return "renaissance"
    if "baroque" in text:
        return "baroque"
    if "classical" in text or "galant" in text:
        return "classical"
    if "impressionist" in text:
        return "impressionist"
    if "romantic" in text:
        return "romantic"
    if "modern" in text or "avant_garde" in text or "serialist" in text:
        return "modernist"
    if "mid_century" in text or "1940s" in text or "1950s" in text or "1960s" in text or "1970s" in text:
        return "mid_century"
    return "contemporary"


def classify_medium(style: dict[str, Any]) -> str:
    """Map a style dict to one of ``CANONICAL_MEDIA``."""
    text = _blob(style)
    tags = {str(t).strip().lower() for t in style.get("tags", ())}
    paper_type = str(style.get("paper_type") or "").strip().lower()

    # 1. Reproduction / archival duplicates
    if any(
        k in text
        for k in (
            "hectograph",
            "mimeo",
            "xerox",
            "photocopy",
            "carbon_copy",
            "carbon-paper",
            "faded_sepia_archival",
            "blueprint",
            "coffee_stained",
        )
    ):
        return "repro_archival"

    # 2. Film / studio / Broadway pit / session charts
    if (
        any(
            k in tags
            for k in (
                "film_scoring",
                "hollywood",
                "studio_copyist",
                "tokyo_studio",
                "broadway",
                "musical_theater",
                "pit_orchestra",
                "nashville",
            )
        )
        or any(
            k in text
            for k in (
                "film_scor",
                "film-scor",
                "film-composer",
                "hollywood",
                "cinecitta",
                "bbc radio",
                "broadway",
                "nashville",
                "tokyo studio",
                "session musician",
                "ozalid_vellum",
            )
        )
    ):
        return "film_studio"

    # 3. Jazz lead sheets / big-band / bebop / blues & funk charts
    if (
        any(k in tags for k in ("jazz", "real_book", "bebop", "hardbop", "big_band", "blues", "afrobeat", "tin_pan_alley"))
        or any(k in text for k in ("jazz", "real book", "real_book", "bebop", "big-band", "big_band", "lead-sheet", "lead_sheet", "tin pan alley"))
    ):
        return "jazz_lead_sheet"

    # 4. Choral / liturgical / church / cathedral / gospel
    if any(
        k in text
        for k in (
            "choir",
            "choral",
            "cathedral",
            "choirbook",
            "hymnal",
            "church_organist",
            "gospel",
            "villancico",
            "cantata",
            "liturg",
        )
    ):
        return "choral_liturgical"

    # 5. Orchestral / conductor / opera / marching / band scores
    if any(
        k in text
        for k in (
            "conductor",
            "orchestral",
            "opera",
            "theater",
            "harmoniemusik",
            "marching",
            "brass_band",
            "rehearsal_black_ink",
        )
    ):
        return "orchestral_conductor"

    # 6. Folk / vernacular / world / street / indie songwriter
    if (
        any(
            k in tags
            for k in (
                "folk",
                "world",
                "vernacular",
                "fiddle",
                "tango",
                "choro",
                "flamenco",
                "mariachi",
                "klezmer",
                "appalachian",
                "busker",
                "indie",
                "ethnomusicology",
            )
        )
        or any(
            k in text
            for k in (
                "fiddler",
                "fiddle",
                "folk",
                "tango",
                "choro",
                "samba",
                "flamenco",
                "mariachi",
                "klezmer",
                "busker",
                "indie/rock",
                "field transcription",
                "freehand_curved_staves",
            )
        )
    ):
        return "folk_vernacular"

    # 7. Student / everyday / exam / beginner / pocket / lefty homework
    if (
        any(
            k in tags
            for k in (
                "beginner",
                "child",
                "dictation",
                "pocket_notebook",
                "spiral_notebook",
                "left_handed",
            )
        )
        or any(
            k in text
            for k in (
                "student",
                "conservatory homework",
                "theory-class",
                "theory exam",
                "dictation",
                "amateur",
                "pocket_notebook",
                "pocket-notebook",
                "beginner",
                "left_handed",
                "lefthand",
                "dry_bic",
                "025mm_mech",
                "6b_carpenter",
                "eraser_ghost",
                "sharpie_chisel",
            )
        )
        or paper_type == "spiral_notebook"
    ):
        return "student_everyday"

    # 8. Fair copy / publisher copyist / engraver Stichvorlage
    if (
        any(k in tags for k in ("fair_copy", "copyist", "court_copyist", "calligraphic"))
        or any(
            k in text
            for k in (
                "fair copy",
                "fair_copy",
                "copyist",
                "stichvorlage",
                "engraver",
                "presentation score",
                "fountain_pen_calligraphy",
            )
        )
    ):
        return "fair_copy"

    # 9. Chamber / solo / lieder / keyboard / violin / organ / partbook
    if any(
        k in text
        for k in (
            "harpsichord",
            "clavecin",
            "chamber",
            "quartet",
            "partbook",
            "organ",
            "violin",
            "lieder",
            "salon",
            "piano",
            "solo",
            "consort",
            "sonata",
        )
    ):
        return "chamber_solo"

    # 10. Default: composer autograph / sketch / draft
    return "autograph_sketch"


def classify_implement(style: dict[str, Any]) -> str:
    """Map a style dict to one of ``CANONICAL_IMPLEMENTS``."""
    text = _blob(style)
    tags = {str(t).strip().lower() for t in style.get("tags", ())}
    era = classify_era(style)
    nib_min = float(style.get("nib_min_ratio", 0.5))
    grain = float(style.get("grain_strength", 0.1))
    pooling = float(style.get("ink_pooling", 0.1))

    # 1. Grease pencil / wax / crayon
    if any(k in text for k in ("grease-pencil", "grease_pencil", "blue_pencil", "carbon_copy", "carbon-paper")):
        return "grease_pencil"

    # 2. Mechanical pencil
    if any(k in text for k in ("mechanical pencil", "mechanical-pencil", "mechanical_pencil", "0.25mm", "0.3mm")):
        return "mech_pencil"

    # 3. Graphite / colored pencil
    if "pencil" in text or "graphite" in text or "blackwing" in text or "6b" in text:
        return "graphite_pencil"

    # 4. Technical pen / rapidograph / fineliner / drafting pen
    if any(
        k in text
        for k in (
            "rapidograph",
            "technical_pen",
            "technical pen",
            "technical-pen",
            "fineliner",
            "drafting-pen",
            "drafting pen",
            "0.18mm",
            "0.2mm",
        )
    ):
        return "technical_pen"

    # 5. Felt marker / sharpie / chisel marker
    if any(
        k in text
        for k in (
            "felt-tip",
            "felt_tip",
            "felt-marker",
            "felt marker",
            "felt_pen",
            "felt pen",
            "sharpie",
            "marker",
        )
    ):
        return "felt_marker"

    # 6. Ballpoint pen
    if "ballpoint" in text or "bic " in text or "bic_" in text:
        return "ballpoint"

    # 7. Gel / rollerball pen
    if "gel" in text or "rollerball" in text:
        return "gel_rollerball"

    # 8. Fountain pen
    if "fountain" in text or "fountain_pen" in tags:
        return "fountain_pen"

    # 9. Quill (goose quill, crow quill, Renaissance/Baroque/Classical ink)
    if "quill" in text or "crowquill" in text or "crow_quill" in text:
        return "quill"

    # 10. Explicit steel / dip / broad / chisel nib
    if any(k in text for k in ("steel-nib", "steel nib", "dip-pen", "dip pen", "dip ink", "pointed nib", "broad-nib", "broad nib", "chisel-nib", "india ink", "stub ink", "stravigor")):
        return "steel_nib"

    # Physics & era fallbacks when description doesn't name an instrument explicitly
    if grain >= 0.30 and pooling <= 0.08:
        return "graphite_pencil"
    if era in ("renaissance", "baroque", "classical"):
        return "quill"
    if era in ("romantic", "impressionist"):
        return "steel_nib" if nib_min <= 0.45 else "fountain_pen"
    if nib_min >= 0.85:
        return "gel_rollerball"
    if nib_min <= 0.35:
        return "fountain_pen"
    return "steel_nib"


def infer_paper_type(style: dict[str, Any]) -> str:
    """Return explicit ``paper_type`` or deterministically infer one from paper physics."""
    pt = style.get("paper_type")
    if isinstance(pt, str) and pt in PAPER_TYPES:
        return pt

    r, g, b = (float(c) for c in style["paper_rgb"])
    grain = float(style.get("paper_grain", 0.06))
    mottle = float(style.get("paper_mottle", 0.06))
    text = _blob(style)

    if "ozalid" in text or "blueprint" in text or (b > r + 0.02 and b > 0.85):
        return "ozalid_blue"
    if "soviet" in text or (g > r + 0.015 and g > b + 0.015):
        return "soviet_green_tint"
    if "spiral" in text or "pocket_notebook" in text:
        return "spiral_notebook"
    if r >= 0.95 and g >= 0.95 and b >= 0.93 and mottle <= 0.05:
        return "bright_white"
    if b <= 0.80 or mottle >= 0.14 or grain >= 0.10:
        return "aged_rag"
    return "cream_manuscript"


def compute_legibility_score(style: dict[str, Any]) -> float:
    """Compute a deterministic continuous illegibility/chaos score in ``[0.0, 1.0]``.

    ``0.0`` corresponds to engraver-clean fair copy; ``1.0`` corresponds to
    maximally degraded, tremulous, cramped, low-contrast manuscripts.
    """
    wobble = min(1.0, float(style.get("wobble_px", 0.0)) / 3.0)
    tremor = min(1.0, float(style.get("tremor_px", 0.0)) / 2.2)
    dropout = min(1.0, float(style.get("ink_dropout", 0.0)) / 0.40)
    ink_var = min(1.0, float(style.get("ink_var", 0.0)) / 0.45)
    feather = min(1.0, float(style.get("ink_feather", 0.0)) / 0.85)
    split_nib = min(1.0, float(style.get("split_nib_prob", 0.0)) / 0.80)
    spacing = min(1.0, max(0.0, (float(style.get("spacing_jitter", 0.08)) - 0.08) / 0.37))
    bar_var = min(1.0, max(0.0, (float(style.get("bar_width_var", 0.10)) - 0.10) / 0.25))
    cram = min(1.0, float(style.get("end_bar_cram", 0.0)) / 0.40)
    staff_curve = min(1.0, float(style.get("staff_curve_px", 0.0)) / 12.0)
    staff_slope = min(1.0, abs(float(style.get("staff_slope_deg", 0.0))) / 2.2)
    sp_jitter = min(1.0, float(style.get("sp_jitter_frac", 0.0)) / 0.08)
    rastrum = min(1.0, float(style.get("rastrum_wobble", 0.0)) / 4.5)
    overshoot = min(1.0, float(style.get("barline_overshoot_sp", 0.0)) / 0.85)
    hollow_open = min(1.0, float(style.get("hollow_open_prob", 0.0)))
    slant = min(1.0, abs(float(style.get("slant_deg", 0.0))) / 18.0)
    mottle = min(1.0, float(style.get("paper_mottle", 0.0)) / 0.30)
    bleed = min(1.0, float(style.get("bleed_through", 0.0)) / 0.25)
    eraser = min(1.0, float(style.get("eraser_smudge_prob", 0.0)))
    stain = min(1.0, float(style.get("stain_prob", 0.0)))

    # Morphological extremity (tiny/huge noteheads or extreme aspect ratio)
    nh_scale_dev = min(1.0, abs(float(style.get("notehead_scale", 1.0)) - 1.0) / 0.45)
    nh_aspect_dev = min(1.0, abs(float(style.get("notehead_aspect", 1.0)) - 1.0) / 0.85)

    # Ink-vs-paper luminance contrast penalty (lower contrast -> harder to read)
    ir, ig, ib = (float(c) for c in style["ink_rgb"])
    pr, pg, pb = (float(c) for c in style["paper_rgb"])
    ink_lum = 0.299 * ir + 0.587 * ig + 0.114 * ib
    paper_lum = 0.299 * pr + 0.587 * pg + 0.114 * pb
    contrast = max(0.0, paper_lum - ink_lum)
    low_contrast = min(1.0, max(0.0, (0.65 - contrast) / 0.50))

    raw_score = (
        0.13 * wobble
        + 0.11 * tremor
        + 0.09 * spacing
        + 0.07 * cram
        + 0.07 * staff_curve
        + 0.06 * dropout
        + 0.05 * sp_jitter
        + 0.05 * bar_var
        + 0.05 * low_contrast
        + 0.04 * overshoot
        + 0.04 * staff_slope
        + 0.04 * hollow_open
        + 0.04 * nh_aspect_dev
        + 0.03 * nh_scale_dev
        + 0.03 * ink_var
        + 0.02 * feather
        + 0.02 * split_nib
        + 0.02 * rastrum
        + 0.02 * mottle
        + 0.02 * bleed
        + 0.02 * eraser
        + 0.02 * stain
        + 0.02 * slant
    )
    # Normalize against sum of weights (1.06) and clamp cleanly to [0, 1].
    return round(min(1.0, max(0.0, raw_score / 1.06)), 4)


def score_legibility(style: dict[str, Any]) -> int:
    """Map continuous physics legibility score to discrete tier ``1..5``.

    1: ``engraver``, 2: ``neat``, 3: ``working``, 4: ``hurried``, 5: ``wild``.
    """
    s = compute_legibility_score(style)
    if s < 0.18:
        return 1
    if s < 0.27:
        return 2
    if s < 0.37:
        return 3
    if s < 0.49:
        return 4
    return 5


def _describe_style_cached(name: str) -> dict[str, Any]:
    """Return cached 4-axis catalog description without copying (internal use)."""
    global _DESCRIBE_CACHE_VERSION
    ver = _styles_mod._REGISTRY_VERSION
    if ver != _DESCRIBE_CACHE_VERSION:
        _DESCRIBE_CACHE.clear()
        _DESCRIBE_CACHE_VERSION = ver
    if name not in SCRIBE_STYLES:
        # Delegate to get_style for the standard KeyError message.
        get_style(name)
    style = SCRIBE_STYLES[name]
    obj_id = id(style)
    entry = _DESCRIBE_CACHE.get(name)
    if entry is not None and entry[0] == obj_id:
        return entry[1]
    score = compute_legibility_score(style)
    tier = 1 if score < 0.18 else (2 if score < 0.27 else (3 if score < 0.37 else (4 if score < 0.49 else 5)))
    desc = {
        "name": style["name"],
        "description": style["description"],
        "raw_era": style["era"],
        "era": classify_era(style),
        "medium": classify_medium(style),
        "implement": classify_implement(style),
        "legibility": tier,
        "legibility_label": LEGIBILITY_LABELS[tier],
        "legibility_score": score,
        "paper_type": infer_paper_type(style),
        "notehead_archetype": style["notehead_archetype"],
        "tags": list(style.get("tags", ())),
    }
    _DESCRIBE_CACHE[name] = (obj_id, desc)
    return desc


def describe_style(name: str) -> dict[str, Any]:
    """Return a structured 4-axis catalog description for a registered style name."""
    cached = _describe_style_cached(name)
    out = dict(cached)
    out["tags"] = list(cached["tags"])
    return out


def _normalize_enum_filter(
    value: Any,
    valid_values: tuple[str, ...],
    field_name: str,
) -> Optional[set[str]]:
    if value is None:
        return None
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, Iterable):
        items = list(value)
    else:
        raise ValueError(f"{field_name} must be a string or iterable of strings, got {type(value).__name__}")

    normalized: set[str] = set()
    valid_set = set(valid_values)
    for item in items:
        s = str(item).strip().lower()
        if field_name == "era" and s in ERA_PRESETS:
            normalized.update(ERA_PRESETS[s])
            continue
        if s not in valid_set:
            raise ValueError(f"Unknown {field_name} {item!r}; expected one of {valid_values}")
        normalized.add(s)
    return normalized


def _normalize_legibility_filter(value: Any) -> Optional[set[int]]:
    if value is None:
        return None
    if isinstance(value, (int, str)) and not isinstance(value, bool):
        items = [value]
    elif isinstance(value, Iterable):
        items = list(value)
    else:
        raise ValueError(f"legibility must be int (1..5), label, or iterable, got {type(value).__name__}")

    tiers: set[int] = set()
    for item in items:
        if isinstance(item, bool):
            raise ValueError(f"Invalid legibility value {item!r}")
        if isinstance(item, int):
            if item not in LEGIBILITY_LABELS:
                raise ValueError(f"legibility tier must be in 1..5, got {item}")
            tiers.add(item)
        elif isinstance(item, str):
            key = item.strip().lower()
            if key in _LABEL_TO_LEGIBILITY:
                tiers.add(_LABEL_TO_LEGIBILITY[key])
            elif key.isdigit() and int(key) in LEGIBILITY_LABELS:
                tiers.add(int(key))
            else:
                raise ValueError(
                    f"Unknown legibility label {item!r}; expected 1..5 or one of {tuple(LEGIBILITY_LABELS.values())}"
                )
        else:
            raise ValueError(f"Invalid legibility item {item!r}")
    return tiers


def filter_styles(
    *,
    era: Optional[Union[str, Iterable[str]]] = None,
    medium: Optional[Union[str, Iterable[str]]] = None,
    implement: Optional[Union[str, Iterable[str]]] = None,
    legibility: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
    paper_type: Optional[Union[str, Iterable[str]]] = None,
    notehead: Optional[Union[str, Iterable[str]]] = None,
    tag: Optional[Union[str, Iterable[str]]] = None,
) -> list[str]:
    """Filter all registered scribe styles by any combination of facets.

    All specified criteria are AND-combined; passing an iterable for a single
    facet acts as an OR across those values. Preserves registry ordering.
    """
    era_set = _normalize_enum_filter(era, CANONICAL_ERAS, "era")
    medium_set = _normalize_enum_filter(medium, CANONICAL_MEDIA, "medium")
    implement_set = _normalize_enum_filter(implement, CANONICAL_IMPLEMENTS, "implement")
    legibility_set = _normalize_legibility_filter(legibility)
    paper_set = _normalize_enum_filter(paper_type, PAPER_TYPES, "paper_type")
    notehead_set = _normalize_enum_filter(notehead, NOTEHEAD_ARCHETYPES, "notehead")

    tag_set: Optional[set[str]] = None
    if tag is not None:
        if isinstance(tag, str):
            tag_set = {tag.strip().lower()}
        elif isinstance(tag, Iterable):
            tag_set = {str(t).strip().lower() for t in tag}
        else:
            raise ValueError(f"tag must be a string or iterable of strings, got {type(tag).__name__}")

    matched: list[str] = []
    for name in list_styles():
        style = SCRIBE_STYLES[name]
        desc = _describe_style_cached(name)
        if era_set is not None and desc["era"] not in era_set:
            continue
        if medium_set is not None and desc["medium"] not in medium_set:
            continue
        if implement_set is not None and desc["implement"] not in implement_set:
            continue
        if legibility_set is not None and desc["legibility"] not in legibility_set:
            continue
        if paper_set is not None and desc["paper_type"] not in paper_set:
            continue
        if notehead_set is not None and desc["notehead_archetype"] not in notehead_set:
            continue
        if tag_set is not None:
            style_tokens = {
                str(t).strip().lower() for t in style.get("tags", ())
            } | {
                str(style.get("era", "")).strip().lower(),
                desc["era"],
                desc["medium"],
                desc["implement"],
                desc["paper_type"],
                desc["notehead_archetype"],
                desc["legibility_label"],
            }
            if not (tag_set & style_tokens):
                continue
        matched.append(name)
    return matched


def list_historical_styles(
    *,
    legibility: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> list[str]:
    """Return all registered 18th/19th/early-20th-century historical manuscript styles.

    Filters strictly to ``HISTORICAL_ERAS`` and ``HISTORICAL_IMPLEMENTS``, and
    excludes modern ``jazz_lead_sheet`` and ``film_studio`` media.
    """
    allowed_media = tuple(m for m in CANONICAL_MEDIA if m not in HISTORICAL_FORBIDDEN_MEDIA)
    return filter_styles(
        era=HISTORICAL_ERAS,
        medium=allowed_media,
        implement=HISTORICAL_IMPLEMENTS,
        legibility=legibility,
    )


__all__ = [
    "CANONICAL_ERAS",
    "HISTORICAL_ERAS",
    "HISTORICAL_IMPLEMENTS",
    "HISTORICAL_FORBIDDEN_MEDIA",
    "ERA_PRESETS",
    "CANONICAL_MEDIA",
    "CANONICAL_IMPLEMENTS",
    "LEGIBILITY_LABELS",
    "classify_era",
    "classify_medium",
    "classify_implement",
    "infer_paper_type",
    "compute_legibility_score",
    "score_legibility",
    "describe_style",
    "filter_styles",
    "list_historical_styles",
]
