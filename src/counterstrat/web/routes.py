"""Catalog, upload, job, and settings API endpoints."""

import bz2
import gzip
import json
import logging
import shutil
import time
import uuid
from collections import defaultdict
from typing import Annotated, Any, Literal

import polars as pl
import yaml
import zstandard
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    HTTPException,
    Request,
    Response,
    UploadFile,
)
from pydantic import BaseModel

from counterstrat.config import AppConfig
from counterstrat.corpus import load_manifest
from counterstrat.mapcard.compile import MapCard
from counterstrat.mapcard.lexicon import build_lexicon, get_default_overlay_path
from counterstrat.mining.tendencies import TeamBook
from counterstrat.roundscript.models import RoundScript
from counterstrat.web.ingest import JobState, load_job_state, run_ingest, save_job_state

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")

_models_cache: dict[str, tuple[float, list[str]]] = {}
CACHE_TTL = 300.0  # 5 minutes

DEFAULT_MODELS = {
    "anthropic": [
        "claude-sonnet-5",
        "claude-3-7-sonnet-20250219",
        "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku-20241022",
    ],
    "gemini": [
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-2.0-flash",
    ],
}


def get_cfg(request: Request) -> AppConfig:
    cfg = getattr(request.app.state, "cfg", None)
    if cfg is None:
        cfg = AppConfig.load()
        request.app.state.cfg = cfg
    return cfg


ConfigDep = Annotated[AppConfig, Depends(get_cfg)]


class SettingsResponse(BaseModel):
    provider: str
    model: str
    keys_present: dict[str, bool]


class SettingsUpdateRequest(BaseModel):
    provider: Literal["anthropic", "gemini"]
    model: str
    api_key: str | None = None


# --- 1. Demos Upload & Jobs ---


@router.post("/demos")
def upload_demo(
    background_tasks: BackgroundTasks,
    demo: Annotated[UploadFile, File()],
    cfg: ConfigDep,
) -> dict[str, str]:
    filename = demo.filename or "uploaded.dem"
    lower = filename.lower()
    if lower.endswith(".dem.zst"):
        stem = filename[:-8]
        fmt = "zst"
    elif lower.endswith(".zst"):
        stem = filename[:-4]
        fmt = "zst"
    elif lower.endswith(".dem.gz"):
        stem = filename[:-7]
        fmt = "gz"
    elif lower.endswith(".gz"):
        stem = filename[:-3]
        fmt = "gz"
    elif lower.endswith(".dem.bz2"):
        stem = filename[:-8]
        fmt = "bz2"
    elif lower.endswith(".bz2"):
        stem = filename[:-4]
        fmt = "bz2"
    elif lower.endswith(".dem"):
        stem = filename[:-4]
        fmt = "dem"
    else:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file format: expected .dem, .dem.zst, .dem.gz, or .dem.bz2",
        )

    if not stem:
        stem = "demo"
    stem = stem.removesuffix(".dem")

    upload_dir = cfg.data_root / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest_path = upload_dir / f"{stem}.dem"

    try:
        if fmt == "zst":
            dctx = zstandard.ZstdDecompressor()
            with open(dest_path, "wb") as out_f:
                dctx.copy_stream(demo.file, out_f)
        elif fmt == "gz":
            with (
                gzip.GzipFile(fileobj=demo.file, mode="rb") as gz_f,
                open(dest_path, "wb") as out_f,
            ):
                shutil.copyfileobj(gz_f, out_f)
        elif fmt == "bz2":
            with bz2.BZ2File(demo.file, mode="rb") as bz2_f, open(dest_path, "wb") as out_f:
                shutil.copyfileobj(bz2_f, out_f)
        else:
            with open(dest_path, "wb") as out_f:
                shutil.copyfileobj(demo.file, out_f)
    except Exception as exc:
        if dest_path.exists():
            dest_path.unlink()
        raise HTTPException(
            status_code=400, detail=f"Failed to decompress demo upload: {exc}"
        ) from exc

    # Magic byte validation (first 8 bytes must start with PBDEMS2)
    with open(dest_path, "rb") as f:
        magic = f.read(8)

    if not magic.startswith(b"PBDEMS2"):
        if dest_path.exists():
            dest_path.unlink()
        raise HTTPException(
            status_code=400, detail="Invalid demo file: missing PBDEMS2 header magic"
        )

    job_id = uuid.uuid4().hex[:12]
    job_state = JobState(job_id=job_id, stage="queued")
    save_job_state(cfg.data_root, job_state)

    background_tasks.add_task(run_ingest, job_id, dest_path, cfg)
    return {"job_id": job_id, "filename": filename}


@router.get("/jobs/{job_id}", response_model=JobState)
def get_job_status(job_id: str, cfg: ConfigDep) -> JobState:
    state = load_job_state(cfg.data_root, job_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return state


# --- 2. Catalog (Demos, Teams, TeamBooks, Reports) ---


@router.get("/demos")
def list_demos(cfg: ConfigDep) -> list[dict[str, Any]]:
    manifest_path = cfg.data_root / "corpus.jsonl"
    manifest = load_manifest(manifest_path)
    lake_dir = cfg.data_root / "lake"

    out = []
    for match_id, rec in manifest.items():
        rec_dict = rec.model_dump()
        team_keys: list[str] = []
        rosters_p = lake_dir / match_id / "rosters.parquet"
        if rosters_p.exists():
            try:
                df = pl.read_parquet(rosters_p)
                team_keys = sorted(
                    [
                        str(k)
                        for k in df["team_key"].unique().to_list()
                        if k and str(k) not in ("T", "CT", "None")
                    ]
                )
            except Exception:  # noqa: BLE001, S110
                pass
        rec_dict["team_keys"] = team_keys
        out.append(rec_dict)
    return out


@router.get("/teams")
def list_teams(cfg: ConfigDep) -> list[dict[str, Any]]:
    manifest = load_manifest(cfg.data_root / "corpus.jsonl")
    lake_dir = cfg.data_root / "lake"
    if not lake_dir.exists():
        return []

    teams_data: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "names": set(),
            "maps": set(),
            "demos": set(),
            "rounds": set(),
        }
    )

    for match_id, rec in manifest.items():
        rosters_path = lake_dir / match_id / "rosters.parquet"
        if not rosters_path.exists():
            continue
        try:
            df = pl.read_parquet(rosters_path)
            for row in df.iter_rows(named=True):
                tk = row.get("team_key")
                if not tk or str(tk) in ("T", "CT", "None"):
                    continue
                tk_str = str(tk)
                entry = teams_data[tk_str]
                clan = row.get("clan_name")
                if clan and str(clan).strip():
                    entry["names"].add(str(clan).strip())
                entry["maps"].add(rec.map_name)
                entry["demos"].add(match_id)
                r_num = row.get("round_num")
                if r_num is not None:
                    entry["rounds"].add((match_id, r_num))
        except Exception:  # noqa: BLE001, S112
            continue

    result = []
    for tk in sorted(teams_data.keys()):
        data = teams_data[tk]
        result.append(
            {
                "team_key": tk,
                "names": sorted(data["names"]),
                "maps": sorted(data["maps"]),
                "demos": len(data["demos"]),
                "rounds": len(data["rounds"]),
            }
        )
    return result


@router.get("/teams/{team_key}/{map_name}/teambook")
def get_teambook(team_key: str, map_name: str, cfg: ConfigDep) -> dict[str, Any]:
    tb_path = cfg.data_root / "teambooks" / team_key / map_name / "teambook.json"
    if not tb_path.exists():
        raise HTTPException(
            status_code=404, detail=f"TeamBook for {team_key} on {map_name} not found"
        )
    try:
        return json.loads(tb_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read TeamBook: {exc}") from exc


@router.get("/reports/{team_key}/{map_name}")
def get_report(team_key: str, map_name: str, cfg: ConfigDep, mock: str | None = None) -> Response:
    dossier_path = cfg.data_root / "teambooks" / team_key / map_name / "dossier.md"
    if dossier_path.exists():
        return Response(
            content=dossier_path.read_text(encoding="utf-8"), media_type="text/markdown"
        )

    key = cfg.anthropic_api_key if cfg.provider == "anthropic" else cfg.gemini_api_key
    if not key:
        if mock == "1":
            mock_content = f"# Anti-Strat Dossier: {team_key} on {map_name}\n\n## 1. Executive Summary\nOffline UI test dossier."
            return Response(content=mock_content, media_type="text/markdown")
        raise HTTPException(
            status_code=503,
            detail=f"No API key configured for provider '{cfg.provider}'. Please set it in Settings.",
        )

    card_path = cfg.data_root / "mapcards" / map_name / "card.yaml"
    if not card_path.exists():
        raise HTTPException(status_code=404, detail=f"Map card for {map_name} not found")

    tb_path = cfg.data_root / "teambooks" / team_key / map_name / "teambook.json"
    if not tb_path.exists():
        raise HTTPException(
            status_code=404, detail=f"TeamBook for {team_key} on {map_name} not found"
        )

    try:
        card = MapCard(**yaml.safe_load(card_path.read_text(encoding="utf-8")))
        teambook = TeamBook.model_validate_json(tb_path.read_text(encoding="utf-8"))
        overlay_path = get_default_overlay_path(map_name)
        lex = build_lexicon(
            map_name, list(card.zones.keys()), overlay_path if overlay_path.exists() else None
        )

        scripts: list[RoundScript] = []
        for mid in teambook.generated_from:
            scripts_dir = cfg.data_root / "scripts" / mid
            if scripts_dir.exists():
                for sp in sorted(scripts_dir.glob("round_*.json")):
                    scripts.append(RoundScript.model_validate_json(sp.read_text(encoding="utf-8")))

        from counterstrat.llm.base import make_client
        from counterstrat.llm.dossier import generate as generate_dossier

        client = make_client(cfg)
        dossier = generate_dossier(client, card, teambook, scripts, lex)
        dossier_path.parent.mkdir(parents=True, exist_ok=True)
        dossier_path.write_text(dossier.text, encoding="utf-8")
        return Response(content=dossier.text, media_type="text/markdown")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to generate dossier: {exc}") from exc


# --- 3. Settings & Models ---


@router.get("/settings", response_model=SettingsResponse)
def get_settings(cfg: ConfigDep) -> SettingsResponse:
    model_name = cfg.anthropic_model if cfg.provider == "anthropic" else cfg.gemini_model
    return SettingsResponse(
        provider=cfg.provider,
        model=model_name,
        keys_present={
            "anthropic": bool(cfg.anthropic_api_key),
            "gemini": bool(cfg.gemini_api_key),
        },
    )


@router.post("/settings", response_model=SettingsResponse)
def update_settings(
    req: SettingsUpdateRequest,
    cfg: ConfigDep,
) -> SettingsResponse:
    settings_path = cfg.data_root / "settings.json"
    data: dict[str, Any] = {}
    if settings_path.exists():
        try:
            loaded = json.loads(settings_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                data = loaded
        except Exception:  # noqa: BLE001
            data = {}

    data["provider"] = req.provider
    if req.provider == "anthropic":
        data["anthropic_model"] = req.model
        if req.api_key:
            data["anthropic_api_key"] = req.api_key
    elif req.provider == "gemini":
        data["gemini_model"] = req.model
        if req.api_key:
            data["gemini_api_key"] = req.api_key

    data["data_root"] = str(cfg.data_root)

    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    # Update in-memory configuration
    cfg.provider = req.provider
    if req.provider == "anthropic":
        cfg.anthropic_model = req.model
        if req.api_key:
            cfg.anthropic_api_key = req.api_key
    elif req.provider == "gemini":
        cfg.gemini_model = req.model
        if req.api_key:
            cfg.gemini_api_key = req.api_key

    return SettingsResponse(
        provider=cfg.provider,
        model=cfg.anthropic_model if cfg.provider == "anthropic" else cfg.gemini_model,
        keys_present={
            "anthropic": bool(cfg.anthropic_api_key),
            "gemini": bool(cfg.gemini_api_key),
        },
    )


@router.get("/models")
def list_models(cfg: ConfigDep, provider: str | None = None) -> list[str]:
    selected_provider = provider or cfg.provider
    now = time.time()
    if selected_provider in _models_cache:
        cached_time, models = _models_cache[selected_provider]
        if now - cached_time < CACHE_TTL:
            return models

    key = cfg.anthropic_api_key if selected_provider == "anthropic" else cfg.gemini_api_key
    if not key:
        return DEFAULT_MODELS.get(selected_provider, [])

    try:
        if selected_provider == "anthropic":
            import anthropic

            client = anthropic.Anthropic(api_key=key)
            page = client.models.list(limit=100)
            models = [m.id for m in page.data]
        elif selected_provider == "gemini":
            from google import genai

            client = genai.Client(api_key=key)
            models = [m.name.removeprefix("models/") for m in client.models.list()]
        else:
            models = DEFAULT_MODELS.get(selected_provider, [])

        _models_cache[selected_provider] = (now, models)
        return models
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Upstream provider failure: {exc}") from exc
