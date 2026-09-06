"""The VRF CLI bootstrap: found -> reused; missing -> one pinned download; and
the kill switch plus failure paths never crash a caller."""

import contextlib
import io
import zipfile
from pathlib import Path

from counterstrat.web import ingest


def _fake_cli_zip() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("Source2Viewer-CLI.exe", b"MZ fake decompiler")
        zf.writestr("libSkiaSharp.dll", b"fake dll")
    return buf.getvalue()


def test_ensure_returns_existing_cli_without_download(monkeypatch, tmp_path: Path):
    exe = tmp_path / "tools" / "vrf" / "Source2Viewer-CLI.exe"
    exe.parent.mkdir(parents=True)
    exe.write_bytes(b"MZ")
    monkeypatch.setattr(ingest, "REPO_ROOT", tmp_path)

    def boom(*_a, **_k):  # any network touch is a failure
        raise AssertionError("download attempted despite existing CLI")

    monkeypatch.setattr("urllib.request.urlopen", boom)
    assert ingest._ensure_vrf_cli() == exe


def test_ensure_respects_kill_switch(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(ingest, "REPO_ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)  # the finder also probes cwd-relative tools/vrf
    monkeypatch.setenv("COUNTERSTRAT_NO_VRF_DOWNLOAD", "1")

    def boom(*_a, **_k):
        raise AssertionError("download attempted despite kill switch")

    monkeypatch.setattr("urllib.request.urlopen", boom)
    assert ingest._ensure_vrf_cli() is None


def test_ensure_downloads_and_extracts_once(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(ingest, "REPO_ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("COUNTERSTRAT_NO_VRF_DOWNLOAD", raising=False)
    payload = _fake_cli_zip()
    calls: list[str] = []

    def fake_urlopen(url, timeout=0):
        calls.append(url)
        return contextlib.closing(io.BytesIO(payload))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    cli = ingest._ensure_vrf_cli()
    assert cli is not None and cli.exists()
    assert cli.name.startswith("Source2Viewer-CLI")
    assert len(calls) == 1
    assert ingest.VRF_VERSION in calls[0] and calls[0].startswith("https://github.com/")
    assert not list(cli.parent.glob(".download_*.zip")), "temp archive must be cleaned up"

    # Second call finds the extracted CLI - no second download.
    assert ingest._ensure_vrf_cli() == cli
    assert len(calls) == 1


def test_ensure_degrades_when_offline(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(ingest, "REPO_ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("COUNTERSTRAT_NO_VRF_DOWNLOAD", raising=False)

    def offline(*_a, **_k):
        raise OSError("no route to host")

    monkeypatch.setattr("urllib.request.urlopen", offline)
    assert ingest._ensure_vrf_cli() is None  # degrade, never raise
