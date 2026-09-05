"""A fresh clone must work out of the box: list calibrated maps, render the
callout editor, and accept personal custom callouts - with ONLY what git
ships (shipped anchors + packaged radar assets; no data/, no maps/ VPKs,
no CS2 install, no calibration caches).

Personal callouts never ship: these tests run against the real packaged
data, so they also fail if anyone's zones/aliases leak into the package.
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from counterstrat.config import AppConfig
from counterstrat.mapcard.anchors import load_shipped_anchors, shipped_anchor_maps
from counterstrat.web.app import create_app

CALIBRATED = sorted(shipped_anchor_maps())


@pytest.fixture
def clone(tmp_path: Path, monkeypatch) -> TestClient:
    """An app over an empty data root and a repo without vendored VPKs."""
    fake_repo = tmp_path / "fresh_repo"
    (fake_repo / "maps").mkdir(parents=True)
    monkeypatch.setattr("counterstrat.web.ingest.REPO_ROOT", fake_repo)
    monkeypatch.chdir(tmp_path)
    cfg = AppConfig(data_root=tmp_path / "data")
    cfg.data_root.mkdir(parents=True)
    return TestClient(create_app(cfg))


def test_shipped_package_contains_no_personal_callouts():
    """Shipped anchors are game vocabulary only: every name comes from the
    engine's place list, never from someone's zones.json."""
    assert CALIBRATED, "no shipped anchor files - calibration data missing from the package"
    for map_name in CALIBRATED:
        for name in load_shipped_anchors(map_name):
            # Engine place names are single CamelCase words (BombsiteA, TSpawn,
            # Underpass...). Personal callouts in this repo's history were
            # lowercase/spaced ('headshot', 'b ramp') - none may ship.
            assert name[:1].isupper() and " " not in name, (
                f"{map_name}: suspicious non-engine zone name {name!r} in shipped anchors"
            )


def test_fresh_clone_lists_calibrated_maps(clone: TestClient):
    listed = clone.get("/api/maps").json()
    for map_name in CALIBRATED:
        assert map_name in listed


def test_fresh_clone_serves_callout_editor(clone: TestClient):
    body = clone.get("/api/maps/de_anubis/callouts").json()
    zones = {z["name"]: z for z in body["zones"]}
    assert len(zones) >= 20  # the full calibrated game vocabulary
    anchored = [z for z in zones.values() if z["u"] is not None]
    assert len(anchored) == len(zones)  # every zone label is positioned
    boxed = [z for z in zones.values() if z["bounds"]]
    assert len(boxed) == len(zones)  # every game zone ships its occupancy box
    # The radar image itself ships with the package.
    img = clone.get("/api/radar/de_anubis/image")
    assert img.status_code == 200
    assert img.content[:8] == b"\x89PNG\r\n\x1a\n"


def test_fresh_clone_accepts_personal_callouts(clone: TestClient):
    """Place a custom rect with zero local data: grounding comes from the
    shipped anchors' ground Z."""
    mid = load_shipped_anchors("de_anubis")["Middle"]
    r = clone.put(
        "/api/maps/de_anubis/zones",
        json={
            "zones": [
                {
                    "name": "myspot",
                    "u": mid[0] - 0.02,
                    "v": mid[1] - 0.02,
                    "u2": mid[0] + 0.02,
                    "v2": mid[1] + 0.02,
                }
            ]
        },
    )
    assert r.status_code == 200, r.text
    zones = {z["name"]: z for z in clone.get("/api/maps/de_anubis/callouts").json()["zones"]}
    assert zones["myspot"]["custom"] is True
    assert zones["myspot"]["half_u"] and zones["myspot"]["half_v"]
