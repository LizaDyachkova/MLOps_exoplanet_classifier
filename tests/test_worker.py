import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import asyncpg
import pytest
from fastapi.testclient import TestClient
from faststream.rabbit import TestRabbitBroker

from src import worker
from src.config import MAX_DELIVERY_ATTEMPTS
from src.messaging import build_tasks_queue_arguments
from src.worker import (
    broker,
    compute_sync,
    health_app,
    healthz,
    lifespan,
    process_task,
    tasks_queue,
)

TASK_ID = "11111111-1111-1111-1111-111111111111"
PAYLOAD = {"task_id": TASK_ID, "features": {"koi_period": 10.0}}


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


def make_message(delivery_count: int | None = None) -> MagicMock:
    message = MagicMock()
    message.headers = (
        {} if delivery_count is None else {"x-delivery-count": delivery_count}
    )
    return message


@pytest.fixture(autouse=True)
def no_real_delays():
    with (
        patch("src.worker.INFERENCE_DELAY_SECONDS", 0),
        patch("src.worker.RETRY_DELAY_SECONDS", 0),
    ):
        yield


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


def test_delivery_limit_matches_attempts():
    arguments = build_tasks_queue_arguments(max_attempts=3)
    assert arguments["x-delivery-limit"] == 2
    assert arguments["x-queue-type"] == "quorum"


def test_queue_arguments_without_limit():
    arguments = build_tasks_queue_arguments(max_length=0)
    assert "x-max-length" not in arguments
    assert "x-overflow" not in arguments


def test_queue_limit_compensates_quorum_extra_message():
    assert build_tasks_queue_arguments(max_length=50)["x-max-length"] == 49


@pytest.mark.asyncio
async def test_process_task_saves_result():
    worker._model_service = make_dummy_service()

    with (
        patch("src.worker.get_db_pool", new=AsyncMock()),
        patch("src.worker.save_task_result", new=AsyncMock()) as mock_save,
    ):
        await process_task({**PAYLOAD, "enqueued_at": 1.0}, make_message())

    mock_save.assert_awaited_once()
    assert mock_save.await_args.args[1] == TASK_ID


@pytest.mark.asyncio
async def test_process_task_error_before_last_attempt_is_reraised_without_failure():
    worker._model_service = make_dummy_service()
    worker._model_service.knows_any_feature.return_value = False

    with patch("src.worker.save_task_failure", new=AsyncMock()) as mock_fail:
        with pytest.raises(ValueError):
            await process_task(PAYLOAD, make_message())

    mock_fail.assert_not_awaited()


@pytest.mark.asyncio
async def test_process_task_last_attempt_marks_failed_and_reraises():
    worker._model_service = make_dummy_service()
    worker._model_service.knows_any_feature.return_value = False

    with (
        patch("src.worker.get_db_pool", new=AsyncMock()),
        patch("src.worker.save_task_failure", new=AsyncMock()) as mock_fail,
    ):
        with pytest.raises(ValueError):
            await process_task(PAYLOAD, make_message(MAX_DELIVERY_ATTEMPTS - 1))

    mock_fail.assert_awaited_once()


@pytest.mark.asyncio
async def test_process_task_failed_status_write_error_keeps_original_error():
    worker._model_service = make_dummy_service()
    worker._model_service.knows_any_feature.return_value = False

    with patch(
        "src.worker.get_db_pool", new=AsyncMock(side_effect=OSError("db is down"))
    ):
        with pytest.raises(ValueError):
            await process_task(PAYLOAD, make_message(MAX_DELIVERY_ATTEMPTS - 1))


@pytest.mark.asyncio
async def test_process_task_without_features_object_is_reraised():
    worker._model_service = make_dummy_service()

    with pytest.raises(ValueError, match="no features"):
        await process_task({"task_id": TASK_ID, "features": [1, 2]}, make_message())


@pytest.mark.asyncio
async def test_process_task_database_error_is_reraised():
    worker._model_service = make_dummy_service()
    db_error = asyncpg.UndefinedTableError("relation tasks does not exist")

    with (
        patch("src.worker.get_db_pool", new=AsyncMock()),
        patch("src.worker.save_task_result", new=AsyncMock(side_effect=db_error)),
    ):
        with pytest.raises(asyncpg.UndefinedTableError):
            await process_task(PAYLOAD, make_message())


@pytest.mark.asyncio
async def test_inference_runs_outside_event_loop():
    worker._model_service = make_dummy_service()
    ticks = 0

    async def ticker():
        nonlocal ticks
        while True:
            ticks += 1
            await asyncio.sleep(0.01)

    task = asyncio.create_task(ticker())
    with (
        patch("src.worker.INFERENCE_DELAY_SECONDS", 0.2),
        patch("src.worker.get_db_pool", new=AsyncMock()),
        patch("src.worker.save_task_result", new=AsyncMock()),
    ):
        await process_task(PAYLOAD, make_message())
    task.cancel()

    assert ticks >= 5


@pytest.mark.asyncio
async def test_pipeline_via_test_rabbit_broker_success():
    worker._model_service = make_dummy_service()

    async with TestRabbitBroker(broker) as br:
        with (
            patch("src.worker.get_db_pool", new=AsyncMock()),
            patch("src.worker.save_task_result", new=AsyncMock()) as mock_save,
        ):
            await br.publish(PAYLOAD, queue=tasks_queue)

    mock_save.assert_awaited_once()


@pytest.mark.asyncio
async def test_pipeline_via_test_rabbit_broker_failure_reaches_failed_status():
    worker._model_service = make_dummy_service()
    worker._model_service.knows_any_feature.return_value = False

    async with TestRabbitBroker(broker) as br:
        with (
            patch("src.worker.get_db_pool", new=AsyncMock()),
            patch("src.worker.save_task_failure", new=AsyncMock()) as mock_fail,
        ):
            with pytest.raises(ValueError):
                await br.publish(
                    PAYLOAD,
                    queue=tasks_queue,
                    headers={"x-delivery-count": MAX_DELIVERY_ATTEMPTS - 1},
                )

    mock_fail.assert_awaited_once()


@pytest.mark.asyncio
async def test_lifespan_success():
    context = MagicMock()
    mock_service = make_dummy_service()

    with (
        patch(
            "src.worker.load_model_with_retries",
            new=AsyncMock(return_value=mock_service),
        ) as mock_load,
        patch("src.worker.init_db", new=AsyncMock()) as mock_init,
        patch("src.worker.close_db_pool", new=AsyncMock()) as mock_close,
        patch("src.worker.uvicorn.Server") as mock_server_cls,
    ):
        mock_server = MagicMock()
        mock_server.serve = AsyncMock()
        mock_server_cls.return_value = mock_server

        async with lifespan(context):
            assert worker._model_service == mock_service
            context.set_global.assert_called_once_with("model_service", mock_service)
            mock_load.assert_awaited_once()
            mock_init.assert_awaited_once()

        assert mock_server.should_exit is True
        mock_close.assert_awaited_once()


@pytest.mark.asyncio
async def test_lifespan_stops_when_model_is_not_loaded():
    with patch("src.worker.load_model_with_retries", new=AsyncMock(return_value=None)):
        with pytest.raises(RuntimeError, match="is not loaded"):
            async with lifespan(MagicMock()):
                pass
