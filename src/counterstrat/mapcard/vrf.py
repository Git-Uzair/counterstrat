import subprocess
from pathlib import Path

from pydantic import BaseModel


class MapAssets(BaseModel):
    vents: Path
    nav: Path


def extract_map_assets(vpk: Path, vrf_cli: Path, out_dir: Path) -> MapAssets:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("vents_c", "nav"):
        args = [str(vrf_cli), "-i", str(vpk), "-e", ext, "-o", str(out_dir)]
        if ext == "vents_c":
            args.insert(5, "-d")  # decompile entities to text
        subprocess.run(args, check=True, capture_output=True, timeout=300)
    map_name = vpk.stem  # de_anubis
    vents = out_dir / "maps" / map_name / "entities" / "default_ents.vents"
    nav = out_dir / "maps" / f"{map_name}.nav"
    if not vents.exists() or not nav.exists():
        raise FileNotFoundError(f"VRF extraction incomplete under {out_dir}")
    return MapAssets(vents=vents, nav=nav)
