from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI

from src.api.process import router as process_router
from src.api.v1.health import router as health_v1_router
from src.api.v1.tasks import router as tasks_v1_router
from src.config import get_app_version
from src.db import close_db_pool, init_db
from src.inference import load_model_with_retries

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model_service = await load_model_with_retries()
    await init_db()
    try:
        yield
    finally:
        broker = getattr(app.state, "rabbit_broker", None)
        if broker is not None:
            try:
                await broker.stop()
            except (OSError, RuntimeError) as exc:
                logger.warning("broker_close_warning", error=str(exc))
        await close_db_pool()


def create_app() -> FastAPI:
    app = FastAPI(
        title="MLOps Project API",
        version=get_app_version(),
        description="FastAPI сервис с мониторингом состояния и компонентов",
        lifespan=lifespan,
    )

    @app.get("/healthz", tags=["Probes"])
    async def healthz() -> dict[str, str]:
        logger.info("liveness_probe_called", endpoint="/healthz", status="ok")
        return {"status": "ok"}

    app.include_router(health_v1_router, prefix="/api/v1")
    app.include_router(tasks_v1_router, prefix="/api/v1")
    app.include_router(process_router)

    return app


app = create_app()
