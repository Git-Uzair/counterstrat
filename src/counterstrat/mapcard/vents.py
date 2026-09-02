import re
from pathlib import Path

from pydantic import BaseModel

_BLOCK = re.compile(r"^====\d+====$", re.MULTILINE)
_KV = re.compile(r"^(\w+)\s+(.*)$")


class PlaceVolume(BaseModel):
    place_name: str
    origin: tuple[float, float, float]
    hammer_id: str
    model: str


def parse_places(vents_path: Path) -> list[PlaceVolume]:
    text = vents_path.read_text(encoding="utf-8", errors="replace")
    out: list[PlaceVolume] = []
    for block in _BLOCK.split(text):
        kv = {}
        for line in block.splitlines():
            m = _KV.match(line.strip())
            if m:
                kv[m.group(1)] = m.group(2).strip()
        if kv.get("classname") != '"env_cs_place"':
            continue
        parts = [float(x) for x in kv["origin"].strip("[] ").split(",")]
        origin = (parts[0], parts[1], parts[2])
        out.append(
            PlaceVolume(
                place_name=kv["place_name"].strip('"'),
                origin=origin,
                hammer_id=kv.get("hammeruniqueid", "").strip('"'),
                model=kv.get("model", ""),
            )
        )
    return out


def unique_places(vols: list[PlaceVolume]) -> list[str]:
    return sorted({v.place_name for v in vols})
