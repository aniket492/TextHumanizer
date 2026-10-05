"""Shared constants for styles, intensities and quality thresholds."""
from __future__ import annotations

from enum import Enum


class Style(str, Enum):
    natural = "natural"
    professional = "professional"
    academic = "academic"
    casual = "casual"
    simple = "simple"
    concise = "concise"
    detailed = "detailed"


class Intensity(str, Enum):
    light = "light"
    medium = "medium"
    strong = "strong"


# Minimum semantic similarity a rewrite must keep with its source chunk.
MIN_SIMILARITY = {Intensity.light: 0.88, Intensity.medium: 0.82, Intensity.strong: 0.76}

# Allowed output/input word-count ratio per style.
LENGTH_BOUNDS = {
    Style.concise: (0.45, 1.05),
    Style.detailed: (0.9, 1.9),
    Style.simple: (0.6, 1.3),
}
DEFAULT_LENGTH_BOUNDS = (0.65, 1.4)

HISTORY_PAGE_SIZE = 20
