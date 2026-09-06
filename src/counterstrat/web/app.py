"""FastAPI application factory and server entrypoint."""

from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from counterstrat.config import AppConfig
from counterstrat.web.chat import router as chat_router
from counterstrat.web.radar_api import router as radar_router
from counterstrat.web.routes import router


def create_app(cfg: AppConfig | None = None, client_factory: Any = None) -> FastAPI:
    """Create and configure the Counter-Strat FastAPI application.

    ``client_factory`` (``AppConfig -> LLMClient``) overrides provider client
    construction; tests inject scripted clients through it.
    """
    app = FastAPI(title="Counter-Strat", version="0.1.0")
    app.state.cfg = cfg or AppConfig.load()
    app.state.client_factory = client_factory
    app.state.chat_sessions = {}

    # Setup work (decompiler download, map-asset warmup) starts at boot, not
    # on a user's first click; no-op under COUNTERSTRAT_NO_VRF_DOWNLOAD.
    from counterstrat.web import readiness

    readiness.start_bootstrap(app.state.cfg)

    app.include_router(router)
    app.include_router(chat_router)
    app.include_router(radar_router)

    static_dir = Path(__file__).resolve().parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

        @app.get("/")
        def index() -> FileResponse:
            return FileResponse(static_dir / "index.html", media_type="text/html")

    return app


def main() -> None:
    """Entry point for the counterstrat command."""
    import uvicorn

    uvicorn.run(create_app(AppConfig.load()), host="127.0.0.1", port=8710)


if __name__ == "__main__":
    main()
