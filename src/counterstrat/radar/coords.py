"""World <-> radar-image projection.

awpy ships equivalent helpers (``awpy.plot.utils.game_to_pixel_axis``) but they
read ``awpy.data.map_data.MAP_DATA``, which is populated only by a network
download that is unavailable here (verified empty in this venv). These functions
take the calibration this project extracts from the game files instead.
"""

import polars as pl

from counterstrat.radar.extract import RadarCalibration

ROUND_DP = 4

_SIDES = {"t": "T", "terrorist": "T", "ct": "CT", "counter-terrorist": "CT"}


def game_to_pixel(cal: RadarCalibration, x: float, y: float) -> tuple[float, float]:
    """World (X, Y) -> radar image pixel (u, v), origin top-left."""
    return ((x - cal.pos_x) / cal.scale, (cal.pos_y - y) / cal.scale)


def game_to_norm(cal: RadarCalibration, x: float, y: float) -> tuple[float, float]:
    """World (X, Y) -> image-relative (u, v) in [0, 1] for a 1:1 square radar."""
    u, v = game_to_pixel(cal, x, y)
    return (u / cal.image_px, v / cal.image_px)


def pixel_to_game(cal: RadarCalibration, u: float, v: float) -> tuple[float, float]:
    """Radar image pixel (u, v) -> world (X, Y). Inverse of :func:`game_to_pixel`."""
    return (u * cal.scale + cal.pos_x, cal.pos_y - v * cal.scale)


def norm_x_expr(cal: RadarCalibration, col: str = "X") -> pl.Expr:
    """Vectorised ``game_to_norm`` u-component over ``col``; keeps the column name."""
    return (pl.col(col) - cal.pos_x) / (cal.scale * cal.image_px)


def norm_y_expr(cal: RadarCalibration, col: str = "Y") -> pl.Expr:
    """Vectorised ``game_to_norm`` v-component over ``col``; keeps the column name."""
    return (cal.pos_y - pl.col(col)) / (cal.scale * cal.image_px)


def is_lower_level(cal: RadarCalibration, z: float) -> bool:
    """True when ``z`` belongs to the map's lower radar image (inclusive bound)."""
    if cal.lower_altitude_max is None:
        return False
    return z <= cal.lower_altitude_max


def level_expr(cal: RadarCalibration, col: str = "Z") -> pl.Expr:
    """Vectorised level label: ``"lower"`` or ``"default"``."""
    if cal.lower_altitude_max is None:
        return (
            pl.when(pl.col(col).is_not_null())
            .then(pl.lit("default"))
            .otherwise(pl.lit("default"))
            .alias(col)
        )
    return (
        pl.when(pl.col(col) <= cal.lower_altitude_max)
        .then(pl.lit("lower"))
        .otherwise(pl.lit("default"))
        .alias(col)
    )


def normalize_side(value: str | None) -> str | None:
    """Collapses the lake's side spellings (``TERRORIST``/``t``/``ct``) to T/CT."""
    if value is None:
        return None
    return _SIDES.get(str(value).strip().lower())


def side_expr(col: str) -> pl.Expr:
    """Vectorised :func:`normalize_side` over ``col``; keeps the column name."""
    return (
        pl.col(col)
        .cast(pl.String)
        .str.strip_chars()
        .str.to_lowercase()
        .replace_strict(_SIDES, default=None)
        .alias(col)
    )
