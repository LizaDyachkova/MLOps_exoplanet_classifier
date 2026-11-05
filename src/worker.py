import asyncio
import os
import time
from contextlib import asynccontextmanager

import asyncpg
import structlog
import uvicorn
from fastapi import FastAPI
from faststream import ContextRepo, FastStream
from faststream.middlewares.acknowledgement.config import AckPolicy
from faststream.rabbit import Channel, RabbitBroker, RabbitMessage

from src.astrobiology import AstrobiologyEngine
from src.config import (
    GRACEFUL_TIMEOUT_SECONDS,
    INFERENCE_DELAY_SECONDS,
    MAX_DELIVERY_ATTEMPTS,
    MODEL_ALIAS,
    MODEL_NAME,
    PREFETCH_COUNT,
    RABBITMQ_URL,
    RETRY_DELAY_SECONDS,
    WORKER_HEALTH_PORT,
)
from src.db import (
    close_db_pool,
    get_db_pool,
    init_db,
    save_task_failure,
    save_task_result,
)
from src.inference import ModelService, load_model_with_retries
from src.messaging import dlq_queue, tasks_queue

logger = structlog.get_logger()

_model_service: ModelService | None = None
_health_server: uvicorn.Server | None = None

health_app = FastAPI(title="Worker Health Check API")


@health_app.get("/healthz")
async def healthz() -> dict:
    return {
        "status": "ok",
        "model_loaded": _model_service is not None,
        "model_name": MODEL_NAME,
        "model_alias": MODEL_ALIAS,
    }


def compute_sync(
    service: ModelService,
    features: dict[str, float | None],
    delay_sec: float = 0.0,
) -> dict:
    if delay_sec > 0:
        time.sleep(delay_sec)

    if not service.knows_any_feature(features):
        raise ValueError("None of the provided features are known by the model")

    prediction_result = service.predict(features)
    radius = features.get("koi_prad")
    teq = features.get("koi_teq")
    esi_metrics = AstrobiologyEngine.compute_all_metrics(radius, teq)

    return {
        "prediction": prediction_result["prediction"],
        "probabilities": prediction_result["probabilities"],
        "missing_features": prediction_result["missing_features"],
        **esi_metrics,
    }


worker_channel = Channel(prefetch_count=PREFETCH_COUNT)
broker = RabbitBroker(
    RABBITMQ_URL,
    default_channel=worker_channel,
    graceful_timeout=GRACEFUL_TIMEOUT_SECONDS,
)


async def _handle_failure(task_id: str, attempt: int, exc: Exception) -> None:
    if attempt < MAX_DELIVERY_ATTEMPTS:
        logger.warning(
            "worker_task_retry_scheduled",
            task_id=task_id,
            attempt=attempt,
            max_attempts=MAX_DELIVERY_ATTEMPTS,
            error=str(exc),
        )
        await asyncio.sleep(RETRY_DELAY_SECONDS)
        return

    logger.error(
        "worker_task_moved_to_dlq", task_id=task_id, attempts=attempt, error=str(exc)
    )
    try:
        pool = await get_db_pool()
        await save_task_failure(pool, task_id, str(exc))
    except (OSError, RuntimeError, ValueError, asyncpg.PostgresError) as db_exc:
        logger.error(
            "worker_db_failure_save_failed", task_id=task_id, error=str(db_exc)
        )


@broker.subscriber(
    tasks_queue, channel=worker_channel, ack_policy=AckPolicy.NACK_ON_ERROR
)
async def process_task(payload: dict, message: RabbitMessage) -> None:
    task_id = str(payload.get("task_id", "anonymous"))
    attempt = int(message.headers.get("x-delivery-count") or 0) + 1
    started_at = time.time()

    logger.info("worker_task_received", task_id=task_id, attempt=attempt)

    try:
        features = payload.get("features")
        if not isinstance(features, dict):
            raise ValueError("Message has no features object")

        result = await asyncio.to_thread(
            compute_sync,
            _model_service,
            features,
            INFERENCE_DELAY_SECONDS,
        )
        pool = await get_db_pool()
        await save_task_result(
            pool,
            task_id,
            result["prediction"],
            result["probabilities"],
            esi=result.get("esi"),
            esi_percent=result.get("esi_percent"),
            is_habitable_zone=result.get("is_habitable_zone"),
            is_rocky=result.get("is_rocky"),
            planet_category=result.get("planet_category"),
        )
    except Exception as exc:
        await _handle_failure(task_id, attempt, exc)
        raise

    finished_at = time.time()
    enqueued_at = payload.get("enqueued_at")
    logger.info(
        "worker_task_completed",
        task_id=task_id,
        prediction=result.get("prediction"),
        esi_percent=result.get("esi_percent"),
        process_sec=round(finished_at - started_at, 3),
        total_sec=round(finished_at - enqueued_at, 3) if enqueued_at else None,
    )


@asynccontextmanager
async def lifespan(context: ContextRepo):
    global _model_service, _health_server
    logger.info(
        "worker_starting",
        model_name=MODEL_NAME,
        model_alias=MODEL_ALIAS,
        prefetch_count=PREFETCH_COUNT,
        inference_delay_seconds=INFERENCE_DELAY_SECONDS,
    )

    _model_service = await load_model_with_retries()
    if _model_service is None:
        raise RuntimeError(f"Model {MODEL_NAME}@{MODEL_ALIAS} is not loaded")

    logger.info(
        "worker_model_loaded",
        model_name=_model_service.model_name,
        version=_model_service.model_version,
        alias=MODEL_ALIAS,
    )
    context.set_global("model_service", _model_service)
    await init_db()

    server_task = None
    if os.getenv("ENABLE_WORKER_HEALTH_SERVER", "true").lower() in ("true", "1"):
        try:
            cfg = uvicorn.Config(
                health_app,
                host="0.0.0.0",
                port=WORKER_HEALTH_PORT,
                log_level="warning",
            )
            _health_server = uvicorn.Server(cfg)
            server_task = asyncio.create_task(_health_server.serve())
            logger.info("worker_health_server_started", port=WORKER_HEALTH_PORT)
        except (OSError, RuntimeError) as exc:
            logger.warning("worker_health_server_start_failed", error=str(exc))

    try:
        yield
    finally:
        logger.info("worker_shutting_down")
        if _health_server:
            _health_server.should_exit = True
        if server_task:
            try:
                await server_task
            except (asyncio.CancelledError, OSError, RuntimeError):
                pass
        await close_db_pool()
        logger.info("worker_shutdown_complete")


app = FastStream(broker, lifespan=lifespan)


@app.after_startup
async def declare_dlq() -> None:
    await broker.declare_queue(dlq_queue)


if __name__ == "__main__":
    asyncio.run(app.run())
