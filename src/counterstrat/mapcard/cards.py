"""Shipped map cards: calibration-grade cards for the supported map pool.

Compiled once by the operator from the curated calibration corpus
(``uv run python -m counterstrat.mapcard.calibrate --cards``) and committed
with the package, so a fresh clone gets a real card - zones, measured
topology, timings, rotates - for every calibrated map without a CS2 install
or the VRF CLI. Shipped cards speak engine vocabulary only; the user's own
compiled card under ``data/mapcards`` always wins at runtime, so custom
zones are never shadowed (see ``counterstrat.web.cards.resolve_card``).
"""

import logging
from pathlib import Path

import yaml

from counterstrat.mapcard.compile import MapCard

logger = logging.getLogger(__name__)

SHIPPED_CARDS_DIR = Path(__file__).resolve().parent / "cards"


def load_shipped_card(map_name: str, root: Path | None = None) -> MapCard | None:
    """The committed calibration card for ``map_name``, or None off the pool."""
    path = (root or SHIPPED_CARDS_DIR) / f"{map_name}.yaml"
    if not path.exists():
        return None
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return MapCard(**data) if isinstance(data, dict) else None
    except Exception as exc:  # noqa: BLE001 - a torn shipped card must not kill serving
        logger.warning("Unreadable shipped card for %s: %s", map_name, exc)
        return None


def save_shipped_card(card: MapCard, root: Path | None = None) -> Path:
    """Writes ``card`` into the shipped pool (calibration export only)."""
    out_dir = root or SHIPPED_CARDS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{card.map}.yaml"
    path.write_text(card.to_yaml(), encoding="utf-8")
    return path
