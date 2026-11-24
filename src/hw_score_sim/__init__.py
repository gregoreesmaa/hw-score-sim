"""Procedural handwritten sheet-music rendering engine (`hw-score-sim`)."""

from __future__ import annotations

from .ml_export import (
    HandwrittenScoreDataset,
    RenderSample,
    export_dataset_sample,
    render_sample,
    to_coco_annotations,
    to_hf_generator,
    to_yolo_labels,
)
from .musescore_compat import (
    MeasureData,
    NoteEvent,
    ParsedScore,
    PartData,
    RestEvent,
    export_score_to_musicxml,
    parse_score,
)
from .renderer import (
    StaffLayout,
    SystemLayout,
    layout_score_pages,
    render_score,
    render_score_handwritten,
)
from .rest_variants import (
    ALL_NEW_PAUSE_ARCHETYPES,
    BAR_REST_ARCHETYPES,
    FLAGGED_REST_ARCHETYPES,
    NEW_QUARTER_REST_ARCHETYPES,
    draw_pause_variant,
)
from .structural_glyphs import (
    BARLINE_ARCHETYPES,
    CLEF_ARCHETYPES,
    KEYSIG_ARCHETYPES,
)
from .styles import (
    BEAM_ARCHETYPES,
    BEAM_GROUPING_MODES,
    BEAM_SLOPE_STYLES,
    FLAG_ARCHETYPES,
    LEDGER_STYLES,
    NOTEHEAD_ARCHETYPES,
    PAPER_TYPES,
    REST_ARCHETYPES,
    SCRIBE_STYLES,
    STEM_ARCHETYPES,
    get_style,
    interpolate_styles,
    list_styles,
    register_style,
    reload_style_packs,
    sample_random_scribe,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # Primary rendering & parsing facade
    "render_score",
    "render_score_handwritten",
    "parse_score",
    "layout_score_pages",
    "export_score_to_musicxml",
    # Generic ML export & in-memory rendering
    "RenderSample",
    "render_sample",
    "to_coco_annotations",
    "to_yolo_labels",
    "export_dataset_sample",
    "HandwrittenScoreDataset",
    "to_hf_generator",
    # Style registry & procedural scribe sampling
    "SCRIBE_STYLES",
    "list_styles",
    "get_style",
    "register_style",
    "reload_style_packs",
    "sample_random_scribe",
    "interpolate_styles",
    # Glyph & layout archetype registries
    "NOTEHEAD_ARCHETYPES",
    "STEM_ARCHETYPES",
    "FLAG_ARCHETYPES",
    "BEAM_ARCHETYPES",
    "BEAM_SLOPE_STYLES",
    "BEAM_GROUPING_MODES",
    "CLEF_ARCHETYPES",
    "KEYSIG_ARCHETYPES",
    "BARLINE_ARCHETYPES",
    "REST_ARCHETYPES",
    "NEW_QUARTER_REST_ARCHETYPES",
    "FLAGGED_REST_ARCHETYPES",
    "BAR_REST_ARCHETYPES",
    "ALL_NEW_PAUSE_ARCHETYPES",
    "draw_pause_variant",
    "LEDGER_STYLES",
    "PAPER_TYPES",
    # Data structures
    "ParsedScore",
    "PartData",
    "MeasureData",
    "NoteEvent",
    "RestEvent",
    "StaffLayout",
    "SystemLayout",
]

# Optional re-exports from sibling modules (sampler & catalog)
try:
    from .sampler import (
        DIFFICULTY_PRESETS,
        INK_PRESETS,
        PAPER_PRESETS,
        ScribeSampler,
        apply_ink_and_paper,
        derive_seed,
    )
except ImportError:
    pass
else:
    __all__.extend(
        [
            "ScribeSampler",
            "derive_seed",
            "DIFFICULTY_PRESETS",
            "INK_PRESETS",
            "PAPER_PRESETS",
            "apply_ink_and_paper",
        ]
    )

try:
    from .catalog import (
        describe_style,
        filter_styles,
    )
except ImportError:
    pass
else:
    __all__.extend(
        [
            "filter_styles",
            "describe_style",
        ]
    )
