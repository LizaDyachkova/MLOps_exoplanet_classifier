import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import aiormq.exceptions
import pytest
from fastapi.testclient import TestClient

from src.app import app
from src.db import (
    close_db_pool,
    create_pending_task,
    get_task,
    init_db,
    save_task_failure,
    save_task_result,
)

client = TestClient(app)


def test_create_task_success():
    mock_broker = MagicMock()
    mock_broker.publish = AsyncMock(return_value=None)
    mock_pool = MagicMock()
    mock_pool.execute = AsyncMock()

    with (
        patch(
            "src.api.v1.tasks.get_or_create_broker",
            new=AsyncMock(return_value=mock_broker),
        ),
        patch("src.api.v1.tasks.get_db_pool", new=AsyncMock(return_value=mock_pool)),
        patch(
            "src.api.v1.tasks.create_pending_task", new=AsyncMock()
        ) as mock_create_db,
    ):
        response = client.post(
            "/api/v1/tasks",
            json={"features": {"koi_period": 9.488, "koi_prad": 2.26}},
        )

    assert response.status_code == 202
    data = response.json()
    assert "task_id" in data
    assert data["status"] == "PENDING"
    mock_broker.publish.assert_called_once()
    mock_create_db.assert_called_once()


def test_create_task_load_shedding_503():
    mock_broker = MagicMock()
    mock_frame = MagicMock()
    mock_broker.publish = AsyncMock(
        side_effect=aiormq.exceptions.DeliveryError(
            message="Queue full", frame=mock_frame
        )
    )

    with (
        patch(
            "src.api.v1.tasks.get_or_create_broker",
            new=AsyncMock(return_value=mock_broker),
        ),
        patch(
            "src.api.v1.tasks.create_pending_task", new=AsyncMock()
        ) as mock_create_db,
    ):
        response = client.post(
            "/api/v1/tasks",
            json={"features": {"koi_period": 9.488, "koi_prad": 2.26}},
        )

    assert response.status_code == 503
    assert response.headers.get("retry-after") == "5"
    data = response.json()
    assert data["retry_after"] == 5
    assert "Load Shedding" in data["detail"]

    mock_create_db.assert_not_called()


def test_create_task_broker_connection_error_503():
    mock_broker = MagicMock()
    mock_broker.publish = AsyncMock(side_effect=OSError("RabbitMQ unreachable"))

    with (
        patch(
            "src.api.v1.tasks.get_or_create_broker",
            new=AsyncMock(return_value=mock_broker),
        ),
        patch(
            "src.api.v1.tasks.create_pending_task", new=AsyncMock()
        ) as mock_create_db,
    ):
        response = client.post(
            "/api/v1/tasks",
            json={"features": {"koi_period": 9.488}},
        )

    assert response.status_code == 503
    assert response.headers.get("retry-after") == "5"
    mock_create_db.assert_not_called()


def test_get_task_status_found():
    sample_id = str(uuid.uuid4())
    mock_task = {
        "task_id": sample_id,
        "status": "COMPLETED",
        "prediction": "CONFIRMED",
        "probabilities": {"CONFIRMED": 0.95},
        "esi": 0.98,
        "esi_percent": 98.0,
        "is_habitable_zone": True,
        "is_rocky": True,
        "planet_category": "Землеподобный супер-кандидат",
        "error": None,
    }

    with (
        patch("src.api.v1.tasks.get_db_pool", new=AsyncMock()),
        patch("src.api.v1.tasks.get_task", new=AsyncMock(return_value=mock_task)),
    ):
        response = client.get(f"/api/v1/tasks/{sample_id}")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "COMPLETED"
    assert data["prediction"] == "CONFIRMED"


def test_get_task_status_not_found():
    sample_id = str(uuid.uuid4())
    with (
        patch("src.api.v1.tasks.get_db_pool", new=AsyncMock()),
        patch("src.api.v1.tasks.get_task", new=AsyncMock(return_value=None)),
    ):
        response = client.get(f"/api/v1/tasks/{sample_id}")

    assert response.status_code == 404


def test_get_task_status_db_error():
    sample_id = str(uuid.uuid4())
    with (
        patch("src.api.v1.tasks.get_db_pool", new=AsyncMock()),
        patch("src.api.v1.tasks.get_task", side_effect=RuntimeError("DB error")),
    ):
        response = client.get(f"/api/v1/tasks/{sample_id}")

    assert response.status_code == 500


@pytest.mark.asyncio
async def test_db_helpers():
    mock_pool = MagicMock()
    mock_pool.execute = AsyncMock()
    mock_pool.fetchrow = AsyncMock(
        return_value={
            "task_id": str(uuid.uuid4()),
            "status": "COMPLETED",
            "features": '{"koi_period": 1.0}',
            "prediction": "CONFIRMED",
            "probabilities": '{"CONFIRMED": 1.0}',
            "esi": 0.9,
            "esi_percent": 90.0,
            "is_habitable_zone": True,
            "is_rocky": True,
            "planet_category": "Тест",
            "error": None,
            "created_at": None,
            "updated_at": None,
        }
    )

    tid = uuid.uuid4()
    await create_pending_task(mock_pool, tid, {"koi_period": 1.0})
    assert mock_pool.execute.call_count == 1

    await save_task_result(
        mock_pool,
        tid,
        "CONFIRMED",
        {"CONFIRMED": 1.0},
        esi=0.9,
        esi_percent=90.0,
        is_habitable_zone=True,
        is_rocky=True,
        planet_category="Тест",
    )
    assert mock_pool.execute.call_count == 2

    await save_task_failure(mock_pool, tid, "Calculation error")
    assert mock_pool.execute.call_count == 3

    task_data = await get_task(mock_pool, tid)
    assert task_data is not None
    assert task_data["features"] == {"koi_period": 1.0}
    assert task_data["probabilities"] == {"CONFIRMED": 1.0}


@pytest.mark.asyncio
async def test_init_db_and_close():
    mock_pool = MagicMock()
    mock_conn = MagicMock()
    mock_conn.execute = AsyncMock()

    class AcquireCtx:
        async def __aenter__(self):
            return mock_conn

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    mock_pool.acquire.return_value = AcquireCtx()

    with patch("src.db.get_db_pool", new=AsyncMock(return_value=mock_pool)):
        await init_db()
        mock_conn.execute.assert_called_once()

    with patch("src.db._pool", mock_pool):
        mock_pool._closed = False
        mock_pool.close = AsyncMock()
        await close_db_pool()
        mock_pool.close.assert_called_once()
