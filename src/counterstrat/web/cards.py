"""Card and lexicon resolution shared by chat sessions and the First Read.

Resolution order for a map card:

1. the user's own compiled card (``data/mapcards`` - carries their custom
   zones and their corpus's measured topology),
2. the shipped calibration card (``counterstrat.mapcard.cards`` - real zones,
   topology and timings for the calibrated pool, engine vocabulary),
3. a fresh VPK+VRF compile from the user's own lake (persisted for next time),
4. a degraded empty card.

Rung 4 means a fresh clone gets analysis on ANY map; rung 2 makes the
calibrated pool first-class. Before this module the First Read hard-404'd
without rung 1 while chat quietly degraded - the 'Map card for de_ancient
not found' field bug (2026-09-06).
"""

import logging
from collections.abc import Iterable
from pathlib import Path

import polars as pl
import yaml

from counterstrat.config import AppConfig
from counterstrat.mapcard.cards import load_shipped_card
from counterstrat.mapcard.compile import MapCard, compile_card
from counterstrat.mapcard.lexicon import Lexicon, build_lexicon, get_default_overlay_path
from counterstrat.mapcard.transitions import zone_graph
from counterstrat.mapcard.vents import parse_places, unique_places
from counterstrat.mapcard.vrf import extract_map_assets
from counterstrat.mining.tendencies import TeamBook
from counterstrat.roundscript.models import RoundScript
from counterstrat.web.ingest import _ensure_vrf_cli, _find_vpk_path

logger = logging.getLogger(__name__)


def _load_user_card(cfg: AppConfig, map_name: str) -> MapCard | None:
    card_path = cfg.data_root / "mapcards" / map_name / "card.yaml"
    if not card_path.exists():
        return None
    try:
        card_data = yaml.safe_load(card_path.read_text(encoding="utf-8"))
        return MapCard(**card_data) if isinstance(card_data, dict) else None
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to load map card from %s: %s", card_path, exc)
        return None


def _compile_user_card(cfg: AppConfig, map_name: str, teambook: TeamBook) -> MapCard | None:
    """Compile from the user's lake when the VPK and VRF CLI are available,
    persisting the card (and its sightlines) for every later session."""
    vpk_path = _find_vpk_path(map_name, cfg)
    vrf_cli = _ensure_vrf_cli()
    if vpk_path is None or vrf_cli is None:
        return None
    ticks_df: pl.DataFrame | None = None
    rounds_df: pl.DataFrame | None = None
    for mid in teambook.generated_from:
        tp = cfg.data_root / "lake" / mid / "ticks.parquet"
        rp = cfg.data_root / "lake" / mid / "rounds.parquet"
        if tp.exists() and rp.exists():
            try:
                ticks_df = pl.read_parquet(tp)
                rounds_df = pl.read_parquet(rp)
                break
            except Exception:  # noqa: BLE001, S112
                continue
    if ticks_df is None or rounds_df is None:
        lake_root = cfg.data_root / "lake"
        for rp in lake_root.glob("*/rounds.parquet"):
            tp = rp.parent / "ticks.parquet"
            if tp.exists():
                try:
                    tdf = pl.read_parquet(tp)
                    rdf = pl.read_parquet(rp)
                    if "map_name" in rdf.columns and map_name in rdf["map_name"].to_list():
                        ticks_df = tdf
                        rounds_df = rdf
                        break
                except Exception:  # noqa: BLE001, S112
                    continue
    if ticks_df is None or rounds_df is None:
        return None
    card_path = cfg.data_root / "mapcards" / map_name / "card.yaml"
    try:
        assets_dir = cfg.data_root / "tmp_assets" / map_name
        assets = extract_map_assets(vpk_path, vrf_cli, assets_dir)
        places = unique_places(parse_places(assets.vents))
        overlay_path = get_default_overlay_path(map_name)
        lexicon_card = build_lexicon(
            map_name, places, overlay_path if overlay_path.exists() else None
        )
        graph = zone_graph(ticks_df)
        card = compile_card(
            lexicon=lexicon_card,
            graph=graph,
            ticks=ticks_df,
            rounds=rounds_df,
            map_name=map_name,
            patch_version="unknown",
        )
        card_path.parent.mkdir(parents=True, exist_ok=True)
        card_path.write_text(card.to_yaml(), encoding="utf-8")
        try:
            from counterstrat.mapcard.visibility import refresh_card_sightlines

            if refresh_card_sightlines(cfg.data_root, map_name):
                card_data = yaml.safe_load(card_path.read_text(encoding="utf-8"))
                if isinstance(card_data, dict):
                    card = MapCard(**card_data)
        except Exception as exc:  # noqa: BLE001 - sightlines are optional
            logger.warning("Sightline refresh failed for %s: %s", map_name, exc)
    except Exception as exc:
        logger.exception("Mapcard compilation failed: %s", exc)  # noqa: TRY401
        return None
    return card


def _degraded_card(map_name: str) -> MapCard:
    return MapCard(
        map=map_name,
        game_version="unknown",
        nav_source="none",
        frame={},
        zones={},
        topology={},
        rotates=[],
        timings={},
        objectives={},
        sightlines=[],
        checksum="degraded",
    )


def resolve_card(cfg: AppConfig, map_name: str, teambook: TeamBook) -> MapCard:
    """The best available card for ``map_name`` - never raises, never None."""
    card = _load_user_card(cfg, map_name)
    if card is not None:
        return card
    card = load_shipped_card(map_name)
    if card is not None:
        return card
    card = _compile_user_card(cfg, map_name, teambook)
    if card is not None:
        return card
    return _degraded_card(map_name)


def resolve_lexicon(
    cfg: AppConfig,
    map_name: str,
    card: MapCard,
    scripts: Iterable[RoundScript],
    teambook: TeamBook,
) -> Lexicon:
    """Card zones when present; otherwise the vocabulary observed in the
    corpus itself (script places plus lake tick places plus overlay zones).
    The user's custom zones always join in: a shipped card cannot know them,
    yet re-zoned lakes and scripts already speak them."""
    from counterstrat.customzones import load_custom_zones

    custom = [z.name for z in load_custom_zones(cfg.data_root, map_name)]
    overlay_path = get_default_overlay_path(map_name)
    if card.zones:
        places = sorted(set(card.zones.keys()) | set(custom))
        try:
            return build_lexicon(map_name, places, overlay_path if overlay_path.exists() else None)
        except Exception:  # noqa: BLE001 - overlay/vocab clash must not fail closed
            return build_lexicon(map_name, places, None)

    places_set: set[str] = set()
    for s in scripts:
        for b in s.beats:
            for _, z in b.t_form.zones:
                if z:
                    places_set.add(z)
            for _, z in b.ct_form.zones:
                if z:
                    places_set.add(z)
        for k in s.kills:
            if k.zone:
                places_set.add(k.zone)
        if s.first_contact and s.first_contact.zone:
            places_set.add(s.first_contact.zone)
        for u in s.utility:
            if u.from_zone:
                places_set.add(u.from_zone)
            if u.to_zone:
                places_set.add(u.to_zone)
        if s.plant and s.plant.site:
            places_set.add(s.plant.site)
    for mid in teambook.generated_from:
        tp = cfg.data_root / "lake" / mid / "ticks.parquet"
        if tp.exists():
            try:
                tdf = pl.read_parquet(tp, columns=["last_place_name"])
                places_set.update(
                    str(p)
                    for p in tdf["last_place_name"].drop_nulls().unique().to_list()
                    if str(p).strip()
                )
            except Exception:  # noqa: BLE001, S110
                pass

    overlay_zones: list[str] = []
    valid_overlay: Path | None = None
    if overlay_path.exists():
        try:
            overlay_data = yaml.safe_load(overlay_path.read_text(encoding="utf-8")) or {}
            overlay_zones = list((overlay_data.get("zones") or {}).keys())
            valid_overlay = overlay_path
        except Exception:  # noqa: BLE001
            valid_overlay = None

    all_places = sorted(places_set | set(overlay_zones) | set(custom))
    if not all_places:
        all_places = ["Default"]

    try:
        return build_lexicon(map_name, all_places, valid_overlay)
    except Exception:  # noqa: BLE001
        return build_lexicon(map_name, all_places, None)
