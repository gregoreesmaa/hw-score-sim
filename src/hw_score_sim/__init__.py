"""Procedural handwritten sheet-music rendering engine (`hw-score-sim`)."""

from __future__ import annotations

from .expressive_glyphs import (
    ARPEGGIO_ARCHETYPES,
    ARTICULATION_ARCHETYPES,
    BRACE_ARCHETYPES,
    COURTESY_ARCHETYPES,
    DOT_ARCHETYPES,
    DYNAMIC_ARCHETYPES,
    EXPRESSIVE_REGISTRIES,
    GRACE_SLASH_ARCHETYPES,
    HAIRPIN_ARCHETYPES,
    MULTIBAR_ARCHETYPES,
    SLUR_ARCHETYPES,
    TIMESIG_ARCHETYPES,
    TUPLET_ARCHETYPES,
    VOLTA_ARCHETYPES,
    resolve_expressive_archetypes,
)
from .manuscript_marks import (
    ANNOTATION_PERSONAS,
    CANCELLATION_STYLES,
    CHORD_SYMBOL_STYLES,
    MEASURE_NUMBER_STYLES,
    PALM_SMEAR_HANDS,
    SafeZoneGuard,
)
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
from .catalog import (
    ERA_PRESETS,
    HISTORICAL_ERAS,
    HISTORICAL_FORBIDDEN_MEDIA,
    HISTORICAL_IMPLEMENTS,
    list_historical_styles,
)
from .profile_resolver import (
    SEPIA_MODES,
    apply_dense_landscape_profile,
    apply_historical_era_profile,
    apply_sepia_archival_profile,
    resolve_page_and_expressive_profile,
    sanitize_historical_clean_style,
)
from .styles import (
    BEAM_ARCHETYPES,
    BEAM_GROUPING_MODES,
    BEAM_SLOPE_STYLES,
    FLAG_ARCHETYPES,
    LEDGER_STYLES,
    LIGATURE_PATTERNS,
    NOTEHEAD_ARCHETYPES,
    PAPER_TYPES,
    REST_ARCHETYPES,
    SCRIBE_STYLES,
    SPACING_DUCTUS_PROFILES,
    STEM_ARCHETYPES,
    get_style,
    interpolate_styles,
    list_styles,
    register_style,
    reload_style_packs,
    resolve_ductus_param,
    sample_random_scribe,
)

__version__ = "0.3.0"

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
    # v0.3.0 Expressive archetypes, living-manuscript marks, ductus & auto-resolver
    "EXPRESSIVE_REGISTRIES",
    "SLUR_ARCHETYPES",
    "TIMESIG_ARCHETYPES",
    "DYNAMIC_ARCHETYPES",
    "HAIRPIN_ARCHETYPES",
    "ARTICULATION_ARCHETYPES",
    "DOT_ARCHETYPES",
    "TUPLET_ARCHETYPES",
    "GRACE_SLASH_ARCHETYPES",
    "ARPEGGIO_ARCHETYPES",
    "COURTESY_ARCHETYPES",
    "BRACE_ARCHETYPES",
    "VOLTA_ARCHETYPES",
    "MULTIBAR_ARCHETYPES",
    "resolve_expressive_archetypes",
    "ANNOTATION_PERSONAS",
    "CANCELLATION_STYLES",
    "MEASURE_NUMBER_STYLES",
    "CHORD_SYMBOL_STYLES",
    "PALM_SMEAR_HANDS",
    "SafeZoneGuard",
    "SPACING_DUCTUS_PROFILES",
    "LIGATURE_PATTERNS",
    "resolve_ductus_param",
    "SEPIA_MODES",
    "resolve_page_and_expressive_profile",
    "sanitize_historical_clean_style",
    "apply_historical_era_profile",
    "apply_sepia_archival_profile",
    "apply_dense_landscape_profile",
    "HISTORICAL_ERAS",
    "HISTORICAL_IMPLEMENTS",
    "HISTORICAL_FORBIDDEN_MEDIA",
    "ERA_PRESETS",
    "list_historical_styles",
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
