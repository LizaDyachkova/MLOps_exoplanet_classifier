import json
import uuid
from typing import Any

import asyncpg
import structlog

from src.config import DATABASE_URL

logger = structlog.get_logger()

_pool: asyncpg.Pool | None = None

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS tasks (
    task_id UUID PRIMARY KEY,
    status VARCHAR(20) NOT NULL,
    features JSONB,
    prediction VARCHAR(50),
    probabilities JSONB,
    esi DOUBLE PRECISION,
    esi_percent DOUBLE PRECISION,
    is_habitable_zone BOOLEAN,
    is_rocky BOOLEAN,
    planet_category VARCHAR(100),
    error TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
"""


async def get_db_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None or _pool._closed:
        _pool = await asyncpg.create_pool(
            dsn=DATABASE_URL,
            min_size=2,
            max_size=10,
            command_timeout=10.0,
        )
    return _pool


async def close_db_pool() -> None:
    global _pool
    if _pool is not None and not _pool._closed:
        await _pool.close()
        _pool = None


async def init_db() -> None:
    try:
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            await conn.execute(CREATE_TABLE_SQL)
        logger.info("db_schema_initialized", table="tasks")
    except (OSError, asyncpg.PostgresError) as exc:
        logger.warning("db_init_skipped_or_failed", error=str(exc))


async def create_pending_task(
    pool: asyncpg.Pool, task_id: str | uuid.UUID, features: dict[str, Any]
) -> None:
    uid = uuid.UUID(str(task_id))
    query = """
    INSERT INTO tasks (task_id, status, features, created_at, updated_at)
    VALUES ($1, 'PENDING', $2::jsonb, NOW(), NOW())
    ON CONFLICT (task_id) DO NOTHING;
    """
    await pool.execute(query, uid, json.dumps(features))


async def save_task_result(
    pool: asyncpg.Pool,
    task_id: str | uuid.UUID,
    prediction: str,
    probabilities: dict[str, float],
    esi: float | None = None,
    esi_percent: float | None = None,
    is_habitable_zone: bool | None = None,
    is_rocky: bool | None = None,
    planet_category: str | None = None,
) -> None:
    uid = uuid.UUID(str(task_id))
    query = """
    INSERT INTO tasks (
        task_id, status, prediction, probabilities,
        esi, esi_percent, is_habitable_zone, is_rocky, planet_category,
        updated_at
    )
    VALUES ($1, 'COMPLETED', $2, $3::jsonb, $4, $5, $6, $7, $8, NOW())
    ON CONFLICT (task_id) DO UPDATE SET
        status = 'COMPLETED',
        prediction = EXCLUDED.prediction,
        probabilities = EXCLUDED.probabilities,
        esi = EXCLUDED.esi,
        esi_percent = EXCLUDED.esi_percent,
        is_habitable_zone = EXCLUDED.is_habitable_zone,
        is_rocky = EXCLUDED.is_rocky,
        planet_category = EXCLUDED.planet_category,
        error = NULL,
        updated_at = NOW();
    """
    await pool.execute(
        query,
        uid,
        prediction,
        json.dumps(probabilities),
        esi,
        esi_percent,
        is_habitable_zone,
        is_rocky,
        planet_category,
    )


async def save_task_failure(
    pool: asyncpg.Pool, task_id: str | uuid.UUID, error: str
) -> None:
    uid = uuid.UUID(str(task_id))
    query = """
    INSERT INTO tasks (task_id, status, error, updated_at)
    VALUES ($1, 'FAILED', $2, NOW())
    ON CONFLICT (task_id) DO UPDATE SET
        status = 'FAILED',
        error = EXCLUDED.error,
        updated_at = NOW();
    """
    await pool.execute(query, uid, error)


async def get_task(
    pool: asyncpg.Pool, task_id: str | uuid.UUID
) -> dict[str, Any] | None:
    uid = uuid.UUID(str(task_id))
    query = """
    SELECT
        task_id::text,
        status,
        features::text as features,
        prediction,
        probabilities::text as probabilities,
        esi,
        esi_percent,
        is_habitable_zone,
        is_rocky,
        planet_category,
        error,
        created_at,
        updated_at
    FROM tasks
    WHERE task_id = $1;
    """
    row = await pool.fetchrow(query, uid)
    if not row:
        return None
    data = dict(row)
    if data.get("features"):
        data["features"] = json.loads(data["features"])
    if data.get("probabilities"):
        data["probabilities"] = json.loads(data["probabilities"])
    return data
