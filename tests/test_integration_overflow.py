import asyncio
import shutil
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import aio_pika
import pytest
from aio_pika.exceptions import DeliveryError, PublishError
from faststream.middlewares.acknowledgement.config import AckPolicy
from faststream.rabbit import Channel, RabbitBroker, RabbitMessage, RabbitQueue

from src import worker
from src.batch_producer import publish_batch
from src.config import QUEUE_MAX_LENGTH
from src.messaging import build_tasks_queue_arguments, tasks_queue
from src.worker import process_task

try:
    from testcontainers.core.container import DockerContainer
    from testcontainers.core.waiting_utils import wait_for_logs
except ImportError:  # pragma: no cover
    DockerContainer = None

pytestmark = pytest.mark.skipif(
    DockerContainer is None or shutil.which("docker") is None,
    reason="Docker or testcontainers is not available",
)

QUEUE_LIMIT = 5
TASK_ID = "22222222-2222-2222-2222-222222222222"


@pytest.fixture(scope="module")
def rabbit_url():
    container = DockerContainer("rabbitmq:3-management").with_exposed_ports(5672)
    with container:
        wait_for_logs(container, "Server startup complete", timeout=120)
        yield (
            f"amqp://guest:guest@{container.get_container_host_ip()}"
            f":{container.get_exposed_port(5672)}/"
        )


def project_queue(name: str, dlq_name: str, max_length: int = QUEUE_LIMIT):
    return RabbitQueue(
        name=name,
        durable=True,
        arguments=build_tasks_queue_arguments(
            max_length=max_length, max_attempts=3, dlq_name=dlq_name
        ),
    )


async def message_count(url: str, name: str) -> int:
    connection = await aio_pika.connect(url)
    async with connection:
        channel = await connection.channel()
        queue = await channel.declare_queue(name, passive=True)
        return queue.declaration_result.message_count


async def wait_for_count(url: str, name: str, expected: int, timeout: float = 15):
    for _ in range(int(timeout / 0.2)):
        if await message_count(url, name) == expected:
            return
        await asyncio.sleep(0.2)
    assert await message_count(url, name) == expected


@pytest.mark.asyncio
async def test_overflow_rejects_publish_at_the_limit(rabbit_url):
    broker = RabbitBroker(rabbit_url, default_channel=Channel(publisher_confirms=True))
    queue = project_queue("overflow_test", "overflow_test_dlq")
    await broker.connect()
    await broker.declare_queue(RabbitQueue("overflow_test_dlq", durable=True))
    await broker.declare_queue(queue)

    accepted = rejected = 0
    try:
        for i in range(QUEUE_LIMIT + 3):
            try:
                await broker.publish({"task_id": f"t{i}"}, queue=queue, persist=True)
                accepted += 1
            except (DeliveryError, PublishError):
                rejected += 1
    finally:
        await broker.stop()

    assert accepted == QUEUE_LIMIT
    assert rejected == 3


@pytest.mark.asyncio
async def test_poison_messages_reach_dlq_and_good_task_is_processed(rabbit_url):
    queue = project_queue("poison_test", "poison_test_dlq")
    dlq = RabbitQueue("poison_test_dlq", durable=True)
    broker = RabbitBroker(rabbit_url, default_channel=Channel(prefetch_count=1))

    @broker.subscriber(queue, ack_policy=AckPolicy.NACK_ON_ERROR)
    async def handler(payload: dict, message: RabbitMessage) -> None:
        await process_task(payload, message)

    service = MagicMock()
    service.knows_any_feature.side_effect = lambda features: "koi_period" in features
    service.predict.return_value = {
        "prediction": "CONFIRMED",
        "probabilities": {"CONFIRMED": 1.0},
        "missing_features": 0,
    }
    worker._model_service = service

    with (
        patch("src.worker.INFERENCE_DELAY_SECONDS", 0),
        patch("src.worker.RETRY_DELAY_SECONDS", 0),
        patch("src.worker.get_db_pool", new=AsyncMock()),
        patch("src.worker.save_task_failure", new=AsyncMock()) as mock_fail,
        patch("src.worker.save_task_result", new=AsyncMock()) as mock_save,
    ):
        await broker.start()
        await broker.declare_queue(dlq)
        connection = await aio_pika.connect(rabbit_url)
        async with connection:
            channel = await connection.channel()
            for body, content_type in (
                (b"this is not json", "text/plain"),
                (b"[1, 2]", "application/json"),
            ):
                await channel.default_exchange.publish(
                    aio_pika.Message(body, content_type=content_type),
                    routing_key=queue.name,
                )
        await broker.publish(
            {"task_id": TASK_ID, "features": {"unknown": 1.0}}, queue=queue
        )
        await broker.publish(
            {"task_id": TASK_ID, "features": {"koi_period": 1.0}}, queue=queue
        )

        await wait_for_count(rabbit_url, dlq.name, 3)
        await asyncio.sleep(0.5)
        await broker.stop()

    assert await message_count(rabbit_url, queue.name) == 0
    mock_fail.assert_awaited_once()
    mock_save.assert_awaited_once()


@pytest.mark.asyncio
async def test_producer_slows_down_without_losing_tasks(rabbit_url, tmp_path: Path):
    total = QUEUE_MAX_LENGTH + 10
    csv_file = tmp_path / "tasks.csv"
    csv_file.write_text(
        "kepoi_name,koi_period\n" + "".join(f"K{i},{i}.0\n" for i in range(total)),
        encoding="utf-8",
    )
    received = []

    async def slow_consumer() -> None:
        await asyncio.sleep(3)
        connection = await aio_pika.connect(rabbit_url)
        async with connection:
            channel = await connection.channel()
            await channel.set_qos(1)
            queue = await channel.declare_queue(tasks_queue.name, passive=True)
            async with queue.iterator() as messages:
                async for message in messages:
                    async with message.process():
                        received.append(message.body)
                    await asyncio.sleep(0.05)
                    if len(received) == total:
                        return

    consumer = asyncio.create_task(slow_consumer())
    published, rejections = await publish_batch(csv_file, broker_url=rabbit_url)
    await asyncio.wait_for(consumer, timeout=60)

    assert published == total
    assert rejections > 0
    assert len(set(received)) == total
