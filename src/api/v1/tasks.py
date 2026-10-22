import uuid
from typing import Any

import aiormq.exceptions
import structlog
from aio_pika.exceptions import DeliveryError, PublishError
from fastapi import APIRouter, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse
from faststream.rabbit import Channel, RabbitBroker, RabbitQueue
from pydantic import BaseModel, Field

from src.config import (
    QUEUE_MAX_LENGTH,
    QUEUE_OVERFLOW,
    RABBITMQ_URL,
    TASKS_QUEUE_NAME,
)
from src.db import create_pending_task, get_db_pool, get_task

logger = structlog.get_logger()
router = APIRouter(prefix="/tasks", tags=["Tasks"])

RETRY_AFTER_SECONDS = 5

tasks_queue = RabbitQueue(
    name=TASKS_QUEUE_NAME,
    durable=True,
    arguments={
        "x-max-length": QUEUE_MAX_LENGTH,
        "x-overflow": QUEUE_OVERFLOW,
    },
)


class TaskCreateRequest(BaseModel):
    features: dict[str, float | None] = Field(
        ...,
        min_length=1,
        description="Exoplanet candidate features (KOI parameters)",
        examples=[
            {
                "koi_period": 9.488,
                "koi_duration": 2.96,
                "koi_depth": 615.8,
                "koi_prad": 2.26,
                "koi_teq": 793.0,
                "koi_steff": 5455.0,
                "koi_srad": 0.927,
            }
        ],
    )


class TaskCreatedResponse(BaseModel):
    task_id: str
    status: str
    message: str


class TaskDetailResponse(BaseModel):
    task_id: str
    status: str
    features: dict[str, Any] | None = None
    prediction: str | None = None
    probabilities: dict[str, float] | None = None
    esi: float | None = None
    esi_percent: float | None = None
    is_habitable_zone: bool | None = None
    is_rocky: bool | None = None
    planet_category: str | None = None
    error: str | None = None


async def get_or_create_broker(request: Request) -> RabbitBroker:
    broker: RabbitBroker | None = getattr(request.app.state, "rabbit_broker", None)
    if broker is None:
        channel = Channel(publisher_confirms=True)
        broker = RabbitBroker(RABBITMQ_URL, default_channel=channel)
        await broker.connect()
        await broker.declare_queue(tasks_queue)
        request.app.state.rabbit_broker = broker
    return broker


@router.post(
    "",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=TaskCreatedResponse,
    responses={
        503: {
            "description": "System overloaded, queue capacity exceeded",
            "headers": {
                "Retry-After": {
                    "description": "Suggested wait time before retry",
                    "schema": {"type": "integer", "example": 5},
                }
            },
        }
    },
)
async def create_task(payload: TaskCreateRequest, request: Request) -> Response:
    task_id = str(uuid.uuid4())
    message = {"task_id": task_id, "features": payload.features}

    try:
        broker = await get_or_create_broker(request)
        await broker.publish(message, queue=tasks_queue)
    except (DeliveryError, PublishError, aiormq.exceptions.DeliveryError) as exc:
        logger.warning(
            "load_shedding_503_triggered",
            task_id=task_id,
            retry_after=RETRY_AFTER_SECONDS,
            error=str(exc),
        )
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            headers={"Retry-After": str(RETRY_AFTER_SECONDS)},
            content={
                "detail": "Queue is full, system overloaded (Load Shedding)",
                "retry_after": RETRY_AFTER_SECONDS,
            },
        )
    except (OSError, RuntimeError) as exc:
        logger.error("rabbit_publish_failed", error=str(exc))
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            headers={"Retry-After": str(RETRY_AFTER_SECONDS)},
            content={
                "detail": "Message broker unavailable",
                "retry_after": RETRY_AFTER_SECONDS,
            },
        )

    try:
        pool = await get_db_pool()
        await create_pending_task(pool, task_id, payload.features)
    except (OSError, RuntimeError) as exc:
        logger.warning(
            "db_task_pending_creation_warning", task_id=task_id, error=str(exc)
        )

    logger.info("task_enqueued_successfully", task_id=task_id)
    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content={
            "task_id": task_id,
            "status": "PENDING",
            "message": "Task accepted for processing",
        },
    )


@router.get("/{task_id}", response_model=TaskDetailResponse)
async def get_task_status(task_id: str) -> dict[str, Any]:
    try:
        pool = await get_db_pool()
        task = await get_task(pool, task_id)
    except (OSError, RuntimeError) as exc:
        logger.error("db_get_task_failed", task_id=task_id, error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error while fetching task status",
        ) from exc

    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task {task_id} not found",
        )

    return task
