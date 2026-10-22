import shutil

import pytest
from aio_pika.exceptions import DeliveryError, PublishError
from faststream.rabbit import Channel, RabbitBroker, RabbitQueue

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


@pytest.mark.asyncio
async def test_queue_overflow_rejects_publish_on_real_rabbitmq():
    container = DockerContainer("rabbitmq:3-management").with_exposed_ports(5672)
    with container:
        wait_for_logs(container, "Server startup complete", timeout=90)
        url = (
            f"amqp://guest:guest@{container.get_container_host_ip()}"
            f":{container.get_exposed_port(5672)}/"
        )
        broker = RabbitBroker(url, default_channel=Channel(publisher_confirms=True))
        queue = RabbitQueue(
            name="overflow_test",
            durable=True,
            arguments={"x-max-length": QUEUE_LIMIT, "x-overflow": "reject-publish"},
        )
        await broker.connect()
        await broker.declare_queue(queue)

        accepted = 0
        rejected = 0
        try:
            for i in range(QUEUE_LIMIT + 3):
                try:
                    await broker.publish({"task_id": f"t{i}"}, queue=queue)
                    accepted += 1
                except (DeliveryError, PublishError):
                    rejected += 1
        finally:
            await broker.stop()

        assert accepted == QUEUE_LIMIT
        assert rejected == 3
