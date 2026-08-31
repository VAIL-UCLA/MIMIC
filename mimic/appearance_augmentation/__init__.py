"""Appearance augmentation: re-render a video under new lighting, weather and
time-of-day conditions while keeping geometry and the ego-trajectory unchanged.

Backed by Light-A-Video (ICCV 2025), linked into this folder as ``Light-A-Video``.
"""

from .prompts import (
    CATEGORIES,
    NEGATIVE_PROMPT,
    SIMULATOR_STYLE_PROMPTS,
    URBAN_RELIGHTING_PROMPTS,
    build_prompt_pool,
    sample_prompts,
)

__all__ = [
    "CATEGORIES",
    "NEGATIVE_PROMPT",
    "SIMULATOR_STYLE_PROMPTS",
    "URBAN_RELIGHTING_PROMPTS",
    "build_prompt_pool",
    "sample_prompts",
]
