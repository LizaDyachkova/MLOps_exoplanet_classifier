import asyncio
import csv
import os
import uuid
from collections.abc import Generator
from pathlib import Path
from typing import Any

import aiormq.exceptions
import structlog
from aio_pika.exceptions import DeliveryError, PublishError
from faststream.rabbit import Channel, RabbitBroker, RabbitQueue

from src.config import (
    PRODUCER_BACKOFF_FACTOR,
    PRODUCER_BACKOFF_STEP,
    PRODUCER_INITIAL_DELAY,
    PRODUCER_MAX_DELAY,
    PRODUCER_MIN_DELAY,
    PRODUCER_RECOVERY_FACTOR,
    QUEUE_MAX_LENGTH,
    QUEUE_OVERFLOW,
    RABBITMQ_URL,
    RAW_DATA_PATH,
    TASKS_QUEUE_NAME,
)

logger = structlog.get_logger()

NON_FEATURE_COLS = {
    "rowid",
    "kepid",
    "kepoi_name",
    "kepler_name",
    "koi_disposition",
    "koi_pdisposition",
    "koi_tce_delivname",
}


def stream_csv_tasks(
    file_path: Path | str,
) -> Generator[dict[str, Any], None, None]:
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found at {path}")

    with open(path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=1):
            name = row.get("kepoi_name") or f"task_{idx}"
            task_id = str(uuid.uuid5(uuid.NAMESPACE_OID, name))
            features: dict[str, float | None] = {}
            for col, val in row.items():
                if col in NON_FEATURE_COLS:
                    continue
                if val is None or val == "":
                    features[col] = None
                else:
                    try:
                        features[col] = float(val)
                    except ValueError:
                        continue
            yield {"task_id": task_id, "features": features}


class AdaptiveBackoff:
    def __init__(
        self,
        initial_delay: float = PRODUCER_INITIAL_DELAY,
        min_delay: float = PRODUCER_MIN_DELAY,
        max_delay: float = PRODUCER_MAX_DELAY,
        backoff_factor: float = PRODUCER_BACKOFF_FACTOR,
        backoff_step: float = PRODUCER_BACKOFF_STEP,
        recovery_factor: float = PRODUCER_RECOVERY_FACTOR,
    ) -> None:
        self.delay = initial_delay
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.backoff_step = backoff_step
        self.recovery_factor = recovery_factor
        self.consecutive_rejections = 0
        self.consecutive_successes = 0

    def on_rejected(self) -> float:
        self.consecutive_successes = 0
        self.consecutive_rejections += 1
        self.delay = min(
            self.max_delay,
            max(self.delay * self.backoff_factor, self.delay + self.backoff_step, 0.05),
        )
        return self.delay

    def on_success(self) -> float:
        self.consecutive_rejections = 0
        self.consecutive_successes += 1
        self.delay = max(self.min_delay, self.delay * self.recovery_factor)
        return self.delay


async def publish_batch(
    file_path: Path | str = RAW_DATA_PATH,
    broker_url: str = RABBITMQ_URL,
    max_tasks: int | None = None,
) -> tuple[int, int]:
    logger.info(
        "producer_starting",
        file_path=str(file_path),
        queue_max_length=QUEUE_MAX_LENGTH,
        queue_overflow=QUEUE_OVERFLOW,
        broker_url=broker_url,
    )

    channel = Channel(publisher_confirms=True)
    broker = RabbitBroker(broker_url, default_channel=channel)
    await broker.connect()

    tasks_queue = RabbitQueue(
        name=TASKS_QUEUE_NAME,
        durable=True,
        arguments={
            "x-max-length": QUEUE_MAX_LENGTH,
            "x-overflow": QUEUE_OVERFLOW,
        },
    )
    await broker.declare_queue(tasks_queue)

    backoff = AdaptiveBackoff()
    total_published = 0
    total_rejections = 0

    try:
        generator = stream_csv_tasks(file_path)
        for task in generator:
            if max_tasks is not None and total_published >= max_tasks:
                break

            published = False
            while not published:
                try:
                    await broker.publish(task, queue=tasks_queue)
                    published = True
                    total_published += 1
                    current_delay = backoff.on_success()

                    if total_published % 20 == 0 or total_rejections > 0:
                        logger.info(
                            "task_published",
                            task_id=task["task_id"],
                            total_published=total_published,
                            total_rejections=total_rejections,
                            adaptive_delay_sec=round(current_delay, 4),
                        )

                    if current_delay > 0:
                        await asyncio.sleep(current_delay)

                except (
                    DeliveryError,
                    PublishError,
                    aiormq.exceptions.DeliveryError,
                ):
                    total_rejections += 1
                    current_delay = backoff.on_rejected()
                    logger.warning(
                        "queue_overflow_backpressure_signal",
                        task_id=task["task_id"],
                        rejection_count=total_rejections,
                        adaptive_backoff_delay_sec=round(current_delay, 4),
                        action="Waiting for worker before retrying the same task",
                    )
                    await asyncio.sleep(current_delay)

    finally:
        await broker.stop()
        logger.info(
            "producer_finished",
            total_published=total_published,
            total_rejections=total_rejections,
        )

    return total_published, total_rejections


def main() -> None:
    max_tasks_env = os.getenv("MAX_TASKS")
    max_tasks = int(max_tasks_env) if max_tasks_env else None
    asyncio.run(publish_batch(max_tasks=max_tasks))


if __name__ == "__main__":
    main()
