"""Map Card compiler: compiles versioned textual atlas (spec §6.5 card format)."""

import hashlib
from typing import Any

import networkx as nx
import polars as pl
import yaml
from pydantic import BaseModel, Field

from counterstrat.constants import BOMB_SECONDS, DEFUSE_SECONDS, ROUND_SECONDS
from counterstrat.mapcard.lexicon import Lexicon
from counterstrat.mapcard.transitions import ZoneGraph


class MapCard(BaseModel):
    map: str
    card_version: str = "1.0"
    game_version: str
    nav_source: str = "nav"
    frame: dict[str, Any]
    zones: dict[str, dict[str, Any]]
    topology: dict[str, dict[str, float]]
    rotates: list[dict[str, Any]]
    timings: dict[str, dict[str, float]]
    objectives: dict[str, Any]
    sightlines: list[Any] = Field(default_factory=list)
    checksum: str

    def to_yaml(self) -> str:
        # Unit labels ride as comments so parsed content (and checksums, which
        # hash the comment-free dump in compile_card) never change.
        text = yaml.dump(self.model_dump(), sort_keys=False)
        text = text.replace(
            "\ntopology:\n", "\ntopology:  # seconds to move between adjacent zones\n", 1
        )
        text = text.replace(
            "\ntimings:\n", "\ntimings:  # earliest seconds each side reaches the zone\n", 1
        )
        text = text.replace(
            "\nrotates:\n", "\nrotates:  # site-to-site routes; run_s = seconds at run speed\n", 1
        )
        return text


def _compute_quadrant(
    cx: float | None,
    cy: float | None,
    min_x: float,
    max_x: float,
    min_y: float,
    max_y: float,
) -> str | None:
    if cx is None or cy is None:
        return None
    dx = (max_x - min_x) / 3.0
    dy = (max_y - min_y) / 3.0

    if dx > 0:
        if cx < min_x + dx:
            x_lbl = "west"
        elif cx < min_x + 2 * dx:
            x_lbl = "center"
        else:
            x_lbl = "east"
    else:
        x_lbl = "center"

    if dy > 0:
        if cy < min_y + dy:
            y_lbl = "south"
        elif cy < min_y + 2 * dy:
            y_lbl = "center"
        else:
            y_lbl = "north"
    else:
        y_lbl = "center"

    if x_lbl == "center" and y_lbl == "center":
        return "center"
    if x_lbl == "center":
        return f"center-{y_lbl}"
    if y_lbl == "center":
        return f"center-{x_lbl}"
    return f"{y_lbl}-{x_lbl}"


def _compute_elevation(
    q25: float | None,
    q75: float | None,
    z_median: float,
) -> str | None:
    if q25 is None or q75 is None:
        return None
    if q25 > z_median:
        return "upper"
    if q75 < z_median:
        return "lower"
    return "mixed"


def _site_zones(lexicon: Lexicon) -> list[str]:
    """Site zones: overlay tags first; engine Bombsite* names as the fallback.

    de_inferno ships no overlay, so every zone arrives untagged and the tag-only
    lookup left rotates/timings/objectives.sites empty (2026-09-04 plan, Task 1).
    """
    tagged = sorted(z for z, def_ in lexicon.zones.items() if "site" in def_.tags)
    if tagged:
        return tagged
    return sorted(z for z in lexicon.zones if z.lower().startswith("bombsite"))


def _build_rotates(
    lexicon: Lexicon,
    graph: ZoneGraph,
    sites: list[str],
) -> list[dict[str, Any]]:
    if len(sites) < 2:
        return []

    g_di = nx.DiGraph()
    g_fallback = nx.Graph()
    for (u, v), edge in graph.edges.items():
        if u in lexicon.zones and v in lexicon.zones:
            w = float(edge.median_transit_s)
            g_di.add_edge(u, v, weight=w)
            if g_fallback.has_edge(u, v):
                g_fallback[u][v]["weight"] = min(g_fallback[u][v]["weight"], w)
            else:
                g_fallback.add_edge(u, v, weight=w)

    rotates: list[dict[str, Any]] = []
    for s1 in sites:
        for s2 in sites:
            if s1 == s2:
                continue
            paths: list[list[str]] = []
            if g_di.has_node(s1) and g_di.has_node(s2):
                try:
                    gen = nx.shortest_simple_paths(g_di, s1, s2, weight="weight")
                    for path in gen:
                        paths.append(path)
                        if len(paths) == 2:
                            break
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    pass

            if len(paths) < 2 and g_fallback.has_node(s1) and g_fallback.has_node(s2):
                try:
                    gen = nx.shortest_simple_paths(g_fallback, s1, s2, weight="weight")
                    for path in gen:
                        if path not in paths:
                            paths.append(path)
                            if len(paths) == 2:
                                break
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    pass

            for path in paths:
                run_s = 0.0
                for i in range(len(path) - 1):
                    u, v = path[i], path[i + 1]
                    if g_di.has_edge(u, v):
                        run_s += g_di[u][v]["weight"]
                    elif g_fallback.has_edge(u, v):
                        run_s += g_fallback[u][v]["weight"]
                rotates.append(
                    {
                        "from": s1,
                        "to": s2,
                        "via": path[1:-1],
                        "run_s": round(run_s, 1),
                    }
                )

    return rotates


def _build_timings(
    lexicon: Lexicon,
    ticks: pl.DataFrame,
    sites: list[str],
) -> dict[str, dict[str, float]]:
    target_tags = {"site", "mid_control", "choke"}
    target_zones = {
        z for z, def_ in lexicon.zones.items() if any(t in target_tags for t in def_.tags)
    } | set(sites)
    timings: dict[str, dict[str, float]] = {"CT": {}, "T": {}}

    if ticks.is_empty() or not target_zones:
        return timings

    req_cols = {"last_place_name", "clock_s", "team_name"}
    if not req_cols.issubset(ticks.columns):
        return timings

    df = ticks.filter(
        pl.col("last_place_name").is_in(target_zones)
        & (pl.col("clock_s") >= 0.0)
        & pl.col("team_name").is_not_null()
    )
    if "is_alive" in df.columns:
        df = df.filter(pl.col("is_alive"))

    if df.is_empty():
        return timings

    df = df.with_columns(
        pl.when(pl.col("team_name").is_in(["TERRORIST", "T"]))
        .then(pl.lit("T"))
        .when(pl.col("team_name") == "CT")
        .then(pl.lit("CT"))
        .otherwise(pl.col("team_name"))
        .alias("_side")
    )

    for side in ["CT", "T"]:
        side_df = df.filter(pl.col("_side") == side)
        if not side_df.is_empty():
            agg = side_df.group_by("last_place_name").agg(
                pl.col("clock_s").min().alias("earliest_s")
            )
            for row in agg.iter_rows(named=True):
                zone = row["last_place_name"]
                timings[side][zone] = round(float(row["earliest_s"]), 1)
        timings[side] = dict(sorted(timings[side].items()))

    return timings


def compile_card(
    lexicon: Lexicon,
    graph: ZoneGraph,
    ticks: pl.DataFrame,
    rounds: pl.DataFrame,
    map_name: str,
    patch_version: str,
    nav_source: str = "nav",
) -> MapCard:
    # 1. Map bounding box and median Z
    has_coords = (
        not ticks.is_empty()
        and "X" in ticks.columns
        and "Y" in ticks.columns
        and "Z" in ticks.columns
        and ticks["X"].is_not_null().any()
    )
    if has_coords:
        valid_coords = ticks.filter(
            pl.col("X").is_not_null() & pl.col("Y").is_not_null() & pl.col("Z").is_not_null()
        )
        min_x = float(valid_coords["X"].min())
        max_x = float(valid_coords["X"].max())
        min_y = float(valid_coords["Y"].min())
        max_y = float(valid_coords["Y"].max())
        z_median = float(valid_coords["Z"].median())
        frame: dict[str, Any] = {
            "bbox": {
                "min_x": round(min_x, 1),
                "max_x": round(max_x, 1),
                "min_y": round(min_y, 1),
                "max_y": round(max_y, 1),
            },
            "z_median": round(z_median, 1),
        }
    else:
        min_x = max_x = min_y = max_y = z_median = 0.0
        frame = {
            "bbox": {"min_x": 0.0, "max_x": 0.0, "min_y": 0.0, "max_y": 0.0},
            "z_median": 0.0,
        }

    # 2. Zone stats (quadrant, elevation)
    zone_stats: dict[str, dict[str, float]] = {}
    if has_coords and "last_place_name" in ticks.columns:
        valid_zones = ticks.filter(
            pl.col("last_place_name").is_not_null() & (pl.col("last_place_name") != "")
        )
        if not valid_zones.is_empty():
            agg = valid_zones.group_by("last_place_name").agg(
                [
                    pl.col("X").mean().alias("mean_x"),
                    pl.col("Y").mean().alias("mean_y"),
                    pl.col("Z").quantile(0.25).alias("q25_z"),
                    pl.col("Z").quantile(0.75).alias("q75_z"),
                ]
            )
            for row in agg.iter_rows(named=True):
                zone_stats[row["last_place_name"]] = row

    sites = _site_zones(lexicon)

    zones: dict[str, dict[str, Any]] = {}
    for zone_id in sorted(lexicon.zones.keys()):
        zdef = lexicon.zones[zone_id]
        tags = list(zdef.tags)
        if zone_id in sites and "site" not in tags:
            tags.append("site")  # name-inferred: keep the card self-describing
        z_entry: dict[str, Any] = {
            "id": zone_id,
            "aliases": list(zdef.aliases),
            "tags": tags,
        }
        if zone_id in zone_stats:
            st = zone_stats[zone_id]
            quad = _compute_quadrant(st["mean_x"], st["mean_y"], min_x, max_x, min_y, max_y)
            if quad is not None:
                z_entry["quadrant"] = quad
            elev = _compute_elevation(st["q25_z"], st["q75_z"], z_median)
            if elev is not None:
                z_entry["elevation"] = elev
        zones[zone_id] = z_entry

    # 3. Topology (edges with n >= 5)
    topology: dict[str, dict[str, float]] = {}
    for from_zone in sorted(lexicon.zones.keys()):
        neighbors: dict[str, float] = {}
        for (src, dst), edge in graph.edges.items():
            if src == from_zone and edge.n >= 5 and dst in lexicon.zones:
                neighbors[dst] = round(float(edge.median_transit_s), 1)
        if neighbors:
            topology[from_zone] = dict(sorted(neighbors.items()))

    # 4. Rotates
    rotates = _build_rotates(lexicon, graph, sites)

    # 5. Timings
    timings = _build_timings(lexicon, ticks, sites)

    # 6. Objectives
    objectives: dict[str, Any] = {
        "sites": sites,
        "round_seconds": ROUND_SECONDS,
        "bomb_seconds": BOMB_SECONDS,
        "defuse_seconds": list(DEFUSE_SECONDS),
    }

    # 7. Check token budget and trim if necessary (<= 4k tokens ~ 16k chars)
    def _render_dict(
        timings_dict: dict[str, dict[str, float]], zones_dict: dict[str, dict[str, Any]]
    ) -> dict[str, Any]:
        return {
            "map": map_name,
            "card_version": "1.0",
            "game_version": patch_version,
            "nav_source": nav_source,
            "frame": frame,
            "zones": zones_dict,
            "topology": topology,
            "rotates": rotates,
            "timings": timings_dict,
            "objectives": objectives,
            "sightlines": [],
        }

    card_dict = _render_dict(timings, zones)
    yaml_text = yaml.dump(card_dict, sort_keys=False)

    if len(yaml_text) / 4.0 > 4000:
        # Trim timings: keep only site zones
        trimmed_timings = {
            side: {z: t for z, t in timings[side].items() if z in sites} for side in timings
        }
        card_dict = _render_dict(trimmed_timings, zones)
        yaml_text = yaml.dump(card_dict, sort_keys=False)

        if len(yaml_text) / 4.0 > 4000:
            # Trim untagged zone aliases
            trimmed_zones = {}
            for zid, zval in zones.items():
                zcopy = dict(zval)
                if not zcopy.get("tags"):
                    zcopy["aliases"] = []
                trimmed_zones[zid] = zcopy
            card_dict = _render_dict(trimmed_timings, trimmed_zones)
            yaml_text = yaml.dump(card_dict, sort_keys=False)

    checksum = hashlib.sha256(yaml_text.encode("utf-8")).hexdigest()[:12]
    card_dict["checksum"] = checksum
    return MapCard(**card_dict)
