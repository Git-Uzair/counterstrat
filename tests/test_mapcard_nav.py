from pathlib import Path

import pytest

from counterstrat.mapcard.nav36 import NavUnsupportedError, read_nav


@pytest.mark.demo
def test_read_nav_anubis(anubis_assets):
    mesh = read_nav(anubis_assets.nav)
    assert mesh.version == 36
    assert len(mesh.areas) == 2633  # verified via VRF CLI
    area = next(iter(mesh.areas.values()))
    assert len(area.corners) >= 3
    assert all(len(c) == 3 for c in area.corners)
    # adjacency must be symmetric-ish and reference real areas
    assert area.connections
    assert all(cid in mesh.areas for a in mesh.areas.values() for cid in a.connections)


def test_read_nav_rejects_garbage(tmp_path: Path):
    bad = tmp_path / "bad.nav"
    bad.write_bytes(b"\x00" * 64)
    with pytest.raises(NavUnsupportedError):
        read_nav(bad)


def test_read_nav_rejects_truncated(tmp_path: Path):
    bad = tmp_path / "short.nav"
    bad.write_bytes(bytes.fromhex("cefaedfe") + (36).to_bytes(4, "little"))
    with pytest.raises(NavUnsupportedError):
        read_nav(bad)


def test_read_nav_missing(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        read_nav(tmp_path / "nope.nav")
