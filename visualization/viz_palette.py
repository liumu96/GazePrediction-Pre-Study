"""Shared visualization colors for ADT gaze analysis figures."""

from __future__ import annotations

# Okabe-Ito colors. Keep these stable across notebook and paper-style figures so
# the same semantic item keeps the same color everywhere.
BLUE = "#0072B2"
VERMILLION = "#D55E00"
GREEN = "#009E73"
PURPLE = "#CC79A7"
SKY_BLUE = "#56B4E9"
ORANGE = "#E69F00"
YELLOW = "#F0E442"
BLACK = "#222222"
GRAY = "#7A7A7A"
LIGHT_GRAY = "#CFCFCF"

GAZE_TRACK_COLORS = (
    BLUE,
    VERMILLION,
    GREEN,
    PURPLE,
    SKY_BLUE,
    ORANGE,
)

EVENT_FILL_COLORS = {
    "fixation": "rgba(0, 158, 115, 0.14)",
    "transition": "rgba(230, 159, 0, 0.14)",
    "invalid": "rgba(120, 120, 120, 0.10)",
}

EVENT_LINE_COLORS = {
    "fixation": GREEN,
    "transition": ORANGE,
    "invalid": GRAY,
}

NEUTRAL = {
    "paper": "#FFFFFF",
    "plot": "#FFFFFF",
    "grid": "#E6E6E6",
    "axis": LIGHT_GRAY,
    "text": BLACK,
    "legend_border": "#DDDDDD",
    "miss": "#CAD2DA",
}

SEMANTIC_COLORS = {
    "gt_gaze": BLUE,
    "gaze_ray": VERMILLION,
    "gaze_point": VERMILLION,
    "head": BLACK,
    "skeleton": GREEN,
    "skeleton_joint": "#007A59",
    "hit": BLUE,
    "hit_gap": PURPLE,
    "static_object": "rgba(150, 150, 150, 0.46)",
    "dynamic_object": "rgba(230, 159, 0, 0.78)",
    "hit_object": "rgba(0, 114, 178, 0.96)",
    "object_center": ORANGE,
    "image_boundary": "rgba(60, 60, 60, 0.50)",
    "image_fill": "rgba(250, 250, 250, 0.78)",
}
