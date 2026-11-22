"""Shared styling helpers for static geographic plots."""

from __future__ import annotations

import numpy as np


def _padded_map_extent(
    x, y, padding: float, aspect: float,
) -> tuple[float, float, float, float]:
    """Pad bounds and expand them to fill a panel without stretching geography."""
    if not np.isfinite(padding) or padding < 0:
        raise ValueError("map_padding must be a finite, non-negative fraction.")
    xmin, xmax = float(np.min(x)), float(np.max(x))
    ymin, ymax = float(np.min(y)), float(np.max(y))
    span_x, span_y = xmax - xmin, ymax - ymin
    fallback = max(span_x, span_y, 1.0)
    pad_x = padding * (span_x or fallback)
    pad_y = padding * (span_y or fallback)
    xmin, xmax = xmin - pad_x, xmax + pad_x
    ymin, ymax = ymin - pad_y, ymax + pad_y
    # Narrow study regions need surrounding map context, rather than axes that
    # shrink to a sliver inside a wide subplot cell.
    width, height = xmax - xmin, ymax - ymin
    if width < height * aspect:
        extra = (height * aspect - width) / 2
        xmin, xmax = xmin - extra, xmax + extra
    else:
        extra = (width / aspect - height) / 2
        ymin, ymax = ymin - extra, ymax + extra
    return xmin, xmax, ymin, ymax


def _hide_coordinate_ticks(ax) -> None:
    """Remove major/minor map ticks and their coordinate labels."""
    for minor in (False, True):
        ax.set_xticks([], minor=minor)
        ax.set_yticks([], minor=minor)
