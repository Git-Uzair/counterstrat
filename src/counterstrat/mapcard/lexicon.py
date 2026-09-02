import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

DEFAULT_OVERLAYS_DIR = Path(__file__).resolve().parent / "overlays"


class ZoneDef(BaseModel):
    id: str
    aliases: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    engine_place: str


class Lexicon(BaseModel):
    map_name: str
    zones: dict[str, ZoneDef]
    checksum: str

    def is_valid_zone(self, zone_id: str) -> bool:
        return zone_id in self.zones


def get_default_overlay_path(map_name: str) -> Path:
    return DEFAULT_OVERLAYS_DIR / f"{map_name}.yaml"


def _compute_checksum(map_name: str, zones: dict[str, ZoneDef]) -> str:
    canonical = {
        "map_name": map_name,
        "zones": {k: zones[k].model_dump() for k in sorted(zones.keys())},
    }
    canonical_json = json.dumps(canonical, sort_keys=True)
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()[:12]


def build_lexicon(
    map_name: str,
    places: Sequence[str] | list[str],
    overlay_path: Path | str | None = None,
) -> Lexicon:
    sorted_places = sorted(set(places))
    zones: dict[str, ZoneDef] = {
        p: ZoneDef(id=p, aliases=[], tags=[], engine_place=p) for p in sorted_places
    }

    if overlay_path is not None:
        path = Path(overlay_path)
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                overlay_data = yaml.safe_load(f) or {}

            overlay_map = overlay_data.get("map")
            if overlay_map and overlay_map != map_name:
                raise ValueError(
                    f"Overlay map mismatch: expected '{map_name}', got '{overlay_map}'"
                )

            overlay_zones = overlay_data.get("zones") or {}
            if isinstance(overlay_zones, dict):
                for zone_id, zone_spec in overlay_zones.items():
                    if zone_spec is None:
                        zone_spec = {}
                    if not isinstance(zone_spec, dict):
                        continue
                    if "parent" in zone_spec:
                        raise ValueError(
                            f"Subdivision zones with 'parent' are not supported yet: {zone_id}"
                        )
                    if zone_id not in zones:
                        raise ValueError(f"Unknown zone in overlay: {zone_id}")

                    aliases_raw = zone_spec.get("aliases") or []
                    aliases = (
                        [str(a) for a in aliases_raw]
                        if isinstance(aliases_raw, list)
                        else [str(aliases_raw)]
                    )

                    tags_raw = zone_spec.get("tags") or []
                    tags = (
                        [str(t) for t in tags_raw]
                        if isinstance(tags_raw, list)
                        else [str(tags_raw)]
                    )

                    zones[zone_id] = ZoneDef(
                        id=zone_id,
                        aliases=aliases,
                        tags=tags,
                        engine_place=zones[zone_id].engine_place,
                    )

    checksum = _compute_checksum(map_name, zones)
    return Lexicon(map_name=map_name, zones=zones, checksum=checksum)
