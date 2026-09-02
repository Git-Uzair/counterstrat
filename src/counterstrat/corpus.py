"""Corpus manifest: content-addressed demo registry."""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from demoparser2 import DemoParser
from pydantic import BaseModel


class DemoRecord(BaseModel):
    match_id: str
    path: str
    map_name: str
    patch_version: str
    demo_version_guid: str
    server_name: str
    registered_at: str


def _sha256_16(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def register_demo(dem: Path, manifest: Path) -> DemoRecord:
    match_id = _sha256_16(dem)
    existing = load_manifest(manifest)
    if match_id in existing:
        return existing[match_id]
    header = DemoParser(str(dem)).parse_header()
    rec = DemoRecord(
        match_id=match_id,
        path=str(dem),
        map_name=header["map_name"],
        patch_version=header["patch_version"],
        demo_version_guid=header["demo_version_guid"],
        server_name=header.get("server_name", ""),
        registered_at=datetime.now(UTC).isoformat(),
    )
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("a", encoding="utf-8") as f:
        f.write(rec.model_dump_json() + "\n")
    return rec


def load_manifest(manifest: Path) -> dict[str, DemoRecord]:
    if not manifest.exists():
        return {}
    out: dict[str, DemoRecord] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = DemoRecord(**json.loads(line))
            out[rec.match_id] = rec
    return out
