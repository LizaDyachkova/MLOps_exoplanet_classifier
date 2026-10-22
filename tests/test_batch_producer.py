import uuid
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import aiormq.exceptions
import pytest

from src.batch_producer import AdaptiveBackoff, main, publish_batch, stream_csv_tasks


def test_stream_csv_tasks_not_found():
    with pytest.raises(FileNotFoundError):
        list(stream_csv_tasks("non_existent_file.csv"))


def test_stream_csv_tasks_success(tmp_path: Path):
    csv_file = tmp_path / "test.csv"
    csv_file.write_text(
        "rowid,kepid,kepoi_name,kepler_name,koi_disposition,koi_period,koi_prad,koi_teq\n"
        "1,1001,K0001.01,Kepler-1,CONFIRMED,9.5,2.3,\n"
        "2,1002,K0002.01,,CANDIDATE,invalid,1.5,300.0\n",
        encoding="utf-8",
    )

    tasks = list(stream_csv_tasks(csv_file))
    assert len(tasks) == 2

    assert tasks[0]["task_id"] == str(uuid.uuid5(uuid.NAMESPACE_OID, "K0001.01"))
    assert tasks[0]["features"]["koi_period"] == 9.5
    assert tasks[0]["features"]["koi_prad"] == 2.3
    assert tasks[0]["features"]["koi_teq"] is None
    assert "koi_disposition" not in tasks[0]["features"]

    assert tasks[1]["task_id"] == str(uuid.uuid5(uuid.NAMESPACE_OID, "K0002.01"))
    assert "koi_period" not in tasks[1]["features"]
    assert tasks[1]["features"]["koi_prad"] == 1.5
    assert tasks[1]["features"]["koi_teq"] == 300.0


def test_adaptive_backoff_logic():
    backoff = AdaptiveBackoff(
        initial_delay=0.01,
        min_delay=0.001,
        max_delay=1.0,
        backoff_factor=2.0,
        backoff_step=0.05,
        recovery_factor=0.5,
    )

    assert backoff.delay == 0.01

    d1 = backoff.on_rejected()
    assert d1 >= 0.05
    assert backoff.consecutive_rejections == 1

    d2 = backoff.on_rejected()
    assert d2 > d1
    assert backoff.consecutive_rejections == 2

    d3 = backoff.on_success()
    assert d3 < d2
    assert backoff.consecutive_rejections == 0
    assert backoff.consecutive_successes == 1

    for _ in range(20):
        backoff.on_success()
    assert backoff.delay >= backoff.min_delay


@pytest.mark.asyncio
async def test_publish_batch_success(tmp_path: Path):
    csv_file = tmp_path / "test.csv"
    csv_file.write_text(
        "rowid,kepid,kepoi_name,kepler_name,koi_disposition,koi_period,koi_prad\n"
        "1,1001,K0001.01,Kepler-1,CONFIRMED,9.5,2.3\n"
        "2,1002,K0002.01,Kepler-2,CONFIRMED,14.2,1.8\n",
        encoding="utf-8",
    )

    mock_broker = MagicMock()
    mock_broker.connect = AsyncMock()
    mock_broker.declare_queue = AsyncMock()
    mock_broker.publish = AsyncMock()
    mock_broker.stop = AsyncMock()

    with patch("src.batch_producer.RabbitBroker", return_value=mock_broker):
        pub, rej = await publish_batch(file_path=csv_file, max_tasks=2)

    assert pub == 2
    assert rej == 0
    assert mock_broker.publish.call_count == 2
    mock_broker.stop.assert_called_once()


@pytest.mark.asyncio
async def test_publish_batch_with_rejections(tmp_path: Path):
    csv_file = tmp_path / "test.csv"
    csv_file.write_text(
        "rowid,kepid,kepoi_name,kepler_name,koi_disposition,koi_period,koi_prad\n"
        "1,1001,K0001.01,Kepler-1,CONFIRMED,9.5,2.3\n",
        encoding="utf-8",
    )

    mock_broker = MagicMock()
    mock_broker.connect = AsyncMock()
    mock_broker.declare_queue = AsyncMock()

    mock_frame = MagicMock()
    mock_broker.publish = AsyncMock(
        side_effect=[
            aiormq.exceptions.DeliveryError(message="Overflow", frame=mock_frame),
            None,
        ]
    )
    mock_broker.stop = AsyncMock()

    with (
        patch("src.batch_producer.RabbitBroker", return_value=mock_broker),
        patch("asyncio.sleep", new=AsyncMock()),
    ):
        pub, rej = await publish_batch(file_path=csv_file, max_tasks=1)

    assert pub == 1
    assert rej == 1
    assert mock_broker.publish.call_count == 2


def test_main():
    with (
        patch.dict("os.environ", {"MAX_TASKS": "5"}),
        patch("src.batch_producer.publish_batch", new=AsyncMock()) as mock_pub,
    ):
        main()
        mock_pub.assert_called_once_with(max_tasks=5)
