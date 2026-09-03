"""Radar asset extraction and overview calibration parsing (spec item N2).

The CS2 overhead radar art and its world->image calibration live in the game's
``pak01_dir.vpk``, not in the per-map VPKs this repo vendors. Both are pulled
out with the vendored Source2Viewer CLI and cached per map under
``<data_root>/radar/<map_name>/``.
"""

import logging
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Every shipped CS2 overhead radar decompiles to a 1024x1024 RGBA PNG.
RADAR_IMAGE_PX = 1024

_VPK_REL = Path("game") / "csgo" / "pak01_dir.vpk"

# KeyValues values are quoted and tab separated, with trailing "// comment" text.
_FULL_LINE_COMMENT = re.compile(r"(?m)^[ \t]*//.*$")
_TRAILING_COMMENT = re.compile(r"[ \t]//[^\n]*")
_PAIR = re.compile(r'"([^"]+)"[ \t\r\n]*"([^"]*)"')
_LOWER_BLOCK = re.compile(r'"lower"[^{]*\{(.*?)\}', re.DOTALL | re.IGNORECASE)


class RadarCalibration(BaseModel):
    """World->radar-image transform for one map, read from resource/overviews."""

    map_name: str
    pos_x: float  # world X at image column 0
    pos_y: float  # world Y at image row 0
    scale: float  # world units per image pixel
    image_px: int = RADAR_IMAGE_PX
    lower_altitude_max: float | None = None  # None => single-level map


class RadarAssets(BaseModel):
    """Cached on-disk radar bundle for one map."""

    map_name: str
    image: Path
    lower_image: Path | None = None
    overview: Path
    calibration: RadarCalibration

    @property
    def levels(self) -> list[str]:
        return ["default", "lower"] if self.lower_image else ["default"]


def radar_cache_dir(data_root: Path, map_name: str) -> Path:
    return Path(data_root) / "radar" / map_name


def _strip_comments(text: str) -> str:
    return _TRAILING_COMMENT.sub("", _FULL_LINE_COMMENT.sub("", text))


def parse_calibration(text: str, map_name: str, image_px: int = RADAR_IMAGE_PX) -> RadarCalibration:
    """Parses an overview KeyValues file body into a :class:`RadarCalibration`.

    Only the keys this project needs are read: ``pos_x``, ``pos_y``, ``scale``
    and ``verticalsections."lower".AltitudeMax``. ``rotate``/``zoom`` are
    recognised and ignored (every shipped competitive map has them at 0).
    """
    body = _strip_comments(text)
    flat = {k.lower(): v.strip() for k, v in _PAIR.findall(body)}

    missing = [k for k in ("pos_x", "pos_y", "scale") if k not in flat]
    if missing:
        raise ValueError(f"overview for '{map_name}' is missing {', '.join(missing)}")

    for key in ("rotate", "zoom"):
        raw = flat.get(key, "0")
        try:
            if float(raw) != 0.0:
                logger.warning(
                    "overview for %s sets %s=%s; radar projection ignores it", map_name, key, raw
                )
        except ValueError:
            logger.warning("overview for %s has non-numeric %s=%r", map_name, key, raw)

    lower_max: float | None = None
    lower = _LOWER_BLOCK.search(body)
    if lower:
        inner = {k.lower(): v.strip() for k, v in _PAIR.findall(lower.group(1))}
        if "altitudemax" in inner:
            lower_max = float(inner["altitudemax"])

    scale = float(flat["scale"])
    if scale <= 0:
        raise ValueError(f"overview for '{map_name}' has non-positive scale {scale}")

    return RadarCalibration(
        map_name=map_name,
        pos_x=float(flat["pos_x"]),
        pos_y=float(flat["pos_y"]),
        scale=scale,
        image_px=image_px,
        lower_altitude_max=lower_max,
    )


def _run_vrf(
    vrf_cli: Path, vpk: Path, out_dir: Path, filters: list[str], *, decompile: bool, timeout: float
) -> None:
    """Runs one VRF extraction. Filter entries absent from the VPK are skipped."""
    args = [str(vrf_cli), "-i", str(vpk), "-o", str(out_dir), "-f", ",".join(filters)]
    if decompile:
        args.append("-d")
    subprocess.run(args, check=True, capture_output=True, timeout=timeout)


def load_cached_assets(data_root: Path, map_name: str) -> RadarAssets | None:
    """Returns the cached bundle for ``map_name``, or ``None`` when not extracted."""
    cache = radar_cache_dir(data_root, map_name)
    image, overview = cache / "radar.png", cache / "overview.txt"
    if not (image.exists() and overview.exists()):
        return None
    cal_json = cache / "calibration.json"
    if cal_json.exists():
        cal = RadarCalibration.model_validate_json(cal_json.read_text(encoding="utf-8"))
    else:
        cal = parse_calibration(overview.read_text(encoding="utf-8", errors="replace"), map_name)
    lower = cache / "radar_lower.png"
    return RadarAssets(
        map_name=map_name,
        image=image,
        lower_image=lower if lower.exists() else None,
        overview=overview,
        calibration=cal,
    )


def extract_radar_assets(
    cs2_install: Path, vrf_cli: Path, map_name: str, data_root: Path, *, timeout: float = 300.0
) -> RadarAssets:
    """Extracts radar PNG(s) + overview for ``map_name`` into the per-map cache."""
    vpk = Path(cs2_install) / _VPK_REL
    if not vpk.exists():
        raise FileNotFoundError(f"CS2 VPK index not found at {vpk}")

    cache = radar_cache_dir(data_root, map_name)
    cache.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="csradar_") as tmp:
        staged = Path(tmp)
        _run_vrf(
            vrf_cli,
            vpk,
            staged,
            [
                f"panorama/images/overheadmaps/{map_name}_radar_psd.vtex_c",
                f"panorama/images/overheadmaps/{map_name}_lower_radar_psd.vtex_c",
            ],
            decompile=True,
            timeout=timeout,
        )
        _run_vrf(
            vrf_cli,
            vpk,
            staged,
            [f"resource/overviews/{map_name}.txt"],
            decompile=False,
            timeout=timeout,
        )

        overheads = staged / "panorama" / "images" / "overheadmaps"
        src_image = overheads / f"{map_name}_radar_psd.png"
        src_lower = overheads / f"{map_name}_lower_radar_psd.png"
        src_overview = staged / "resource" / "overviews" / f"{map_name}.txt"
        if not src_image.exists() or not src_overview.exists():
            raise FileNotFoundError(
                f"pak01 has no radar assets for '{map_name}' "
                f"(image={src_image.exists()}, overview={src_overview.exists()})"
            )

        image = cache / "radar.png"
        overview = cache / "overview.txt"
        shutil.copyfile(src_image, image)
        shutil.copyfile(src_overview, overview)
        lower_image: Path | None = None
        if src_lower.exists():
            lower_image = cache / "radar_lower.png"
            shutil.copyfile(src_lower, lower_image)

    cal = parse_calibration(overview.read_text(encoding="utf-8", errors="replace"), map_name)
    (cache / "calibration.json").write_text(cal.model_dump_json(indent=2), encoding="utf-8")
    return RadarAssets(
        map_name=map_name,
        image=image,
        lower_image=lower_image,
        overview=overview,
        calibration=cal,
    )


def get_radar_assets(
    data_root: Path,
    map_name: str,
    cs2_install: Path | None,
    vrf_cli: Path | None,
    *,
    force: bool = False,
) -> RadarAssets:
    """Cache-first accessor: extracts on miss, raises when N2 is unsatisfied."""
    if not force:
        cached = load_cached_assets(data_root, map_name)
        if cached is not None:
            return cached
    if cs2_install is None:
        raise FileNotFoundError(
            f"no cached radar for '{map_name}' and cs2_install_path is not configured (see item N2)"
        )
    if vrf_cli is None:
        raise FileNotFoundError("Source2Viewer-CLI not found under tools/vrf/")
    return extract_radar_assets(cs2_install, vrf_cli, map_name, data_root)
