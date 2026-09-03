"""FastAPI application factory and server entrypoint."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from counterstrat.config import AppConfig
from counterstrat.web.routes import router


def create_app(cfg: AppConfig | None = None) -> FastAPI:
    """Create and configure the Counter-Strat FastAPI application."""
    app = FastAPI(title="Counter-Strat", version="0.1.0")
    app.state.cfg = cfg or AppConfig.load()
    app.include_router(router)

    static_dir = Path(__file__).resolve().parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    return app


def main() -> None:
    """Entry point for the counterstrat command."""
    import uvicorn

    uvicorn.run(create_app(AppConfig.load()), host="127.0.0.1", port=8710)


if __name__ == "__main__":
    main()
