from faststream.rabbit import RabbitBroker, RabbitQueue

from src.config import (
    DLQ_NAME,
    MAX_DELIVERY_ATTEMPTS,
    QUEUE_MAX_LENGTH,
    QUEUE_OVERFLOW,
    TASKS_QUEUE_NAME,
)


def build_tasks_queue_arguments(
    max_length: int = QUEUE_MAX_LENGTH,
    overflow: str = QUEUE_OVERFLOW,
    max_attempts: int = MAX_DELIVERY_ATTEMPTS,
    dlq_name: str = DLQ_NAME,
) -> dict[str, str | int]:
    arguments: dict[str, str | int] = {
        "x-queue-type": "quorum",
        "x-delivery-limit": max_attempts - 1,
        "x-dead-letter-exchange": "",
        "x-dead-letter-routing-key": dlq_name,
    }
    if max_length > 0:
        arguments["x-max-length"] = max_length - 1
        arguments["x-overflow"] = overflow
        arguments["x-dead-letter-strategy"] = "at-least-once"
    return arguments


tasks_queue = RabbitQueue(
    name=TASKS_QUEUE_NAME,
    durable=True,
    arguments=build_tasks_queue_arguments(),
)
dlq_queue = RabbitQueue(name=DLQ_NAME, durable=True)


async def declare_topology(broker: RabbitBroker) -> None:
    await broker.declare_queue(dlq_queue)
    await broker.declare_queue(tasks_queue)
