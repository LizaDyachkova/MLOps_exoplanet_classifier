from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from faststream.rabbit import TestRabbitBroker

from src import worker
from src.worker import (
    broker,
    compute_sync,
    dlq_queue,
    health_app,
    healthz,
    lifespan,
    process_task,
    tasks_queue,
)


def make_dummy_service():
    service = MagicMock()
    service.model_name = "exoplanet_classifier"
    service.model_version = "1"
    service.knows_any_feature.return_value = True
    service.predict.return_value = {
        "prediction": "CONFIRMED",
        "probabilities": {
            "CANDIDATE": 0.05,
            "CONFIRMED": 0.90,
            "FALSE POSITIVE": 0.05,
        },
        "missing_features": 0,
    }
    return service


def make_message(headers: dict | None = None) -> MagicMock:
    message = MagicMock()
    message.headers = headers or {}
    message.ack = AsyncMock()
    return message


def test_healthz():
    client = TestClient(health_app)
    worker._model_service = None
    res = client.get("/healthz")
    assert res.status_code == 200
    assert res.json()["model_loaded"] is False

    worker._model_service = make_dummy_service()
    res = client.get("/healthz")
    assert res.status_code == 200
    assert res.json()["model_loaded"] is True


@pytest.mark.asyncio
async def test_healthz_direct():
    worker._model_service = None
    data = await healthz()
    assert data["model_loaded"] is False


def test_compute_sync_success():
    service = make_dummy_service()
    features = {"koi_period": 10.0, "koi_prad": 1.0, "koi_teq": 255.0}
    res = compute_sync(service, features, delay_sec=0.01)

    assert res["prediction"] == "CONFIRMED"
    assert res["esi"] == 1.0
    assert res["esi_percent"] == 100.0
    assert res["is_habitable_zone"] is True
    assert res["planet_category"] == "Землеподобный супер-кандидат"


def test_compute_sync_unknown_features():
    service = make_dummy_service()
    service.knows_any_feature.return_value = False

    with pytest.raises(ValueError, match="None of the provided features"):
        compute_sync(service, {"unknown_feature": 123.0})


@pytest.mark.asyncio
async def test_process_task_success_acks_after_db_write():
    worker._model_service = make_dummy_service()
    pool = MagicMock()
    message = make_message()

    with patch("src.worker.get_db_pool", new=AsyncMock(return_value=pool)):
        with patch("src.worker.save_task_result", new=AsyncMock()) as mock_save:
            await process_task(
                {"task_id": "test-1", "features": {"koi_period": 10.0}}, message
            )

    mock_save.assert_awaited_once()
    assert mock_save.await_args.args[1] == "test-1"
    message.ack.assert_awaited_once()


@pytest.mark.asyncio
async def test_process_task_without_model_schedules_retry():
    worker._model_service = None
    message = make_message()

    with patch("src.worker.broker.publish", new=AsyncMock()) as mock_publish:
        await process_task(
            {"task_id": "test-1", "features": {"koi_period": 10.0}}, message
        )

    mock_publish.assert_awaited_once()
    assert mock_publish.call_args.kwargs["headers"] == {"x-retry-count": "1"}
    assert mock_publish.call_args.kwargs["queue"] is tasks_queue
    message.ack.assert_awaited_once()


@pytest.mark.asyncio
async def test_process_task_exhausted_retries_goes_to_dlq():
    worker._model_service = None
    message = make_message(headers={"x-retry-count": "2"})

    with (
        patch("src.worker.broker.publish", new=AsyncMock()) as mock_publish,
        patch("src.worker.get_db_pool", new=AsyncMock()),
        patch("src.worker.save_task_failure", new=AsyncMock()) as mock_fail,
    ):
        await process_task(
            {"task_id": "test-1", "features": {"koi_period": 10.0}}, message
        )

    mock_fail.assert_awaited_once()
    mock_publish.assert_awaited_once()
    assert mock_publish.call_args.kwargs["queue"] is dlq_queue
    message.ack.assert_awaited_once()


@pytest.mark.asyncio
async def test_process_task_unknown_features_raw_dict():
    worker._model_service = make_dummy_service()
    worker._model_service.knows_any_feature.return_value = False
    message = make_message()

    with patch("src.worker.broker.publish", new=AsyncMock()) as mock_publish:
        await process_task({"koi_period": 10.0}, message)

    mock_publish.assert_awaited_once()
    message.ack.assert_awaited_once()


@pytest.mark.asyncio
async def test_pipeline_via_test_rabbit_broker_success():
    worker._model_service = make_dummy_service()
    pool = MagicMock()

    async with TestRabbitBroker(broker) as br:
        with patch("src.worker.get_db_pool", new=AsyncMock(return_value=pool)):
            with patch("src.worker.save_task_result", new=AsyncMock()) as mock_save:
                await br.publish(
                    {"task_id": "demo-1", "features": {"koi_period": 1.0}},
                    queue=tasks_queue,
                )

    mock_save.assert_awaited_once()


@pytest.mark.asyncio
async def test_pipeline_via_test_rabbit_broker_moves_to_dlq():
    worker._model_service = None
    received_in_dlq = []

    @broker.subscriber(dlq_queue)
    async def _dlq_watcher(payload: dict) -> None:
        received_in_dlq.append(payload)

    async with TestRabbitBroker(broker) as br:
        with (
            patch("src.worker.get_db_pool", new=AsyncMock()),
            patch("src.worker.save_task_failure", new=AsyncMock()) as mock_fail,
        ):
            await br.publish(
                {"task_id": "demo-2", "features": {"koi_period": 1.0}},
                queue=tasks_queue,
                headers={"x-retry-count": "2"},
            )

    mock_fail.assert_awaited_once()
    assert received_in_dlq == [{"task_id": "demo-2", "features": {"koi_period": 1.0}}]


@pytest.mark.asyncio
async def test_lifespan_success():
    context = MagicMock()
    mock_service = make_dummy_service()

    with (
        patch(
            "src.worker.load_model_with_retries",
            new=AsyncMock(return_value=mock_service),
        ),
        patch("src.worker.broker.declare_queue", new=AsyncMock()),
        patch("src.worker.uvicorn.Server") as mock_server_cls,
    ):
        mock_server = MagicMock()
        mock_server.serve = AsyncMock()
        mock_server_cls.return_value = mock_server

        async with lifespan(context):
            assert worker._model_service == mock_service
            context.set_global.assert_called_once_with("model_service", mock_service)

        assert mock_server.should_exit is True


@pytest.mark.asyncio
async def test_lifespan_model_load_failure():
    context = MagicMock()

    with (
        patch("src.worker.load_model_with_retries", new=AsyncMock(return_value=None)),
        patch("src.worker.broker.declare_queue", new=AsyncMock()),
        patch("src.worker.uvicorn.Server") as mock_server_cls,
    ):
        mock_server_cls.side_effect = OSError("Port already bound")
        async with lifespan(context):
            assert worker._model_service is None
