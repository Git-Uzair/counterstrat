from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
DEMO = REPO / "demos" / "1-7065ab7c-bc8f-4995-adf1-ac774327c5db-1-1.dem"
ANUBIS_VPK = REPO / "maps" / "de_anubis" / "de_anubis.vpk"
VRF = REPO / "tools" / "vrf" / "Source2Viewer-CLI.exe"


@pytest.fixture(scope="session")
def demo_path() -> Path:
    if not DEMO.exists():
        pytest.skip("real demo fixture not present")
    return DEMO


@pytest.fixture(scope="session")
def anubis_vpk() -> Path:
    if not ANUBIS_VPK.exists():
        pytest.skip("anubis vpk not present")
    return ANUBIS_VPK


@pytest.fixture(scope="session")
def vrf_cli() -> Path:
    if not VRF.exists():
        pytest.skip("VRF CLI not vendored (see plan Context for URL+sha256)")
    return VRF
