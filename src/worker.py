import asyncio
import os
import time
from contextlib import asynccontextmanager

import structlog
import uvicorn
from fastapi import FastAPI
from faststream import ContextRepo, FastStream
from faststream.middlewares.acknowledgement.config import AckPolicy
from faststream.rabbit import Channel, RabbitBroker, RabbitMessage, RabbitQueue

from src.astrobiology import AstrobiologyEngine
from src.config import (
    DLQ_NAME,
    INFERENCE_DELAY_SECONDS,
    MAX_DELIVERY_ATTEMPTS,
    MODEL_ALIAS,
    MODEL_NAME,
    PREFETCH_COUNT,
    QUEUE_MAX_LENGTH,
    QUEUE_OVERFLOW,
    RABBITMQ_URL,
    TASKS_QUEUE_NAME,
    WORKER_HEALTH_PORT,
)
from src.db import get_db_pool, save_task_failure, save_task_result
from src.inference import ModelService, load_model_with_retries

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


# prefetch=1: inference идёт в одном потоке (to_thread), больше одного
# неподтверждённого сообщения воркер всё равно не обработает параллельно.
worker_channel = Channel(prefetch_count=PREFETCH_COUNT)
broker = RabbitBroker(RABBITMQ_URL, default_channel=worker_channel)
tasks_queue = RabbitQueue(
    name=TASKS_QUEUE_NAME,
    durable=True,
    arguments={
        "x-max-length": QUEUE_MAX_LENGTH,
        "x-overflow": QUEUE_OVERFLOW,
    },
)
# Терминальная очередь для сообщений, исчерпавших лимит попыток.
# Без спец-аргументов: никто её не перечитывает автоматически.
dlq_queue = RabbitQueue(name=DLQ_NAME, durable=True)


async def _send_to_retry_or_dlq(
    message: RabbitMessage, payload: dict, task_id: str, retry_count: int, error: str
) -> None:
    """Переслать сообщение на повтор или, если попытки исчерпаны, в DLQ.

    Счётчик попыток храним в заголовке x-retry-count самого сообщения:
    republish с увеличенным счётчиком + ack оригинала. Так сообщение
    никогда не "зависает" неподтверждённым и не уходит в бесконечный requeue.
    """
    next_attempt = retry_count + 1

    if next_attempt < MAX_DELIVERY_ATTEMPTS:
        logger.warning(
            "worker_task_retry_scheduled",
            task_id=task_id,
            attempt=next_attempt + 1,
            max_attempts=MAX_DELIVERY_ATTEMPTS,
            error=error,
        )
        await broker.publish(
            payload,
            queue=tasks_queue,
            headers={"x-retry-count": str(next_attempt)},
        )
        await message.ack()
        return

    logger.error(
        "worker_task_moved_to_dlq",
        task_id=task_id,
        attempts=next_attempt,
        error=error,
    )
    try:
        pool = await get_db_pool()
        await save_task_failure(pool, task_id, error)
    except (OSError, RuntimeError, ValueError) as exc:
        logger.error("worker_db_failure_save_failed", task_id=task_id, error=str(exc))

    await broker.publish(payload, queue=dlq_queue)
    await message.ack()


@broker.subscriber(tasks_queue, channel=worker_channel, ack_policy=AckPolicy.MANUAL)
async def process_task(payload: dict, message: RabbitMessage) -> None:
    """Обработчик задач из очереди.

    ack_policy=MANUAL выбран осознанно: сообщение подтверждается только
    после того, как результат успешно записан в БД (или после того, как
    оно переотправлено на повтор / в DLQ). Если воркер упадёт во время
    inference, сообщение останется неподтверждённым и будет повторно
    доставлено другому воркеру — данные не теряются. Повторная доставка
    безопасна благодаря идемпотентному UPSERT в БД по task_id.
    """
    global _model_service
    task_id = payload.get("task_id", "anonymous")
    features = (
        payload.get("features")
        if isinstance(payload.get("features"), dict)
        else payload
    )
    retry_count = int(message.headers.get("x-retry-count", 0))

    logger.info("worker_task_received", task_id=task_id, attempt=retry_count + 1)

    try:
        if _model_service is None:
            raise RuntimeError("Model service is not loaded from MLflow")

        result = await asyncio.to_thread(
            compute_sync,
            _model_service,
            features,
            INFERENCE_DELAY_SECONDS,
        )
    except (ValueError, RuntimeError) as exc:
        logger.warning(
            "worker_task_failed",
            task_id=task_id,
            attempt=retry_count + 1,
            error=str(exc),
        )
        await _send_to_retry_or_dlq(message, payload, task_id, retry_count, str(exc))
        return

    try:
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
    except (OSError, RuntimeError, ValueError) as exc:
        logger.error("worker_db_save_failed", task_id=task_id, error=str(exc))
        await _send_to_retry_or_dlq(message, payload, task_id, retry_count, str(exc))
        return

    logger.info(
        "worker_task_completed",
        task_id=task_id,
        prediction=result.get("prediction"),
        esi_percent=result.get("esi_percent"),
    )
    await message.ack()


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
    if _model_service:
        logger.info(
            "worker_model_loaded",
            model_name=_model_service.model_name,
            version=_model_service.model_version,
            alias=MODEL_ALIAS,
        )
        context.set_global("model_service", _model_service)
    else:
        logger.warning(
            "worker_model_load_failed",
            detail="Proceeding without model; will fail on inference",
        )

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
        logger.info("worker_shutdown_complete")


app = FastStream(broker, lifespan=lifespan)


@app.after_startup
async def declare_dlq() -> None:
    await broker.declare_queue(dlq_queue)


if __name__ == "__main__":
    asyncio.run(app.run())
