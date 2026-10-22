import os
import tomllib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
PYPROJECT_PATH = BASE_DIR / "pyproject.toml"


def get_app_version() -> str:
    try:
        with open(PYPROJECT_PATH, "rb") as f:
            data = tomllib.load(f)
            return data.get("project", {}).get("version", "0.1.0")
    except (OSError, tomllib.TOMLDecodeError):
        return "0.1.0"


DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/exoplanet_classifier"
)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "kepler_exoplanet_classifier")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "champion")
MODEL_LOAD_ATTEMPTS = int(os.getenv("MODEL_LOAD_ATTEMPTS", "5"))
MODEL_LOAD_DELAY_SECONDS = float(os.getenv("MODEL_LOAD_DELAY_SECONDS", "3"))

RABBITMQ_URL = os.getenv("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/")
TASKS_QUEUE_NAME = os.getenv("TASKS_QUEUE_NAME", "tasks")
PREFETCH_COUNT = int(os.getenv("PREFETCH_COUNT", "1"))
INFERENCE_DELAY_SECONDS = float(os.getenv("INFERENCE_DELAY_SECONDS", "0.5"))
WORKER_HEALTH_PORT = int(os.getenv("WORKER_HEALTH_PORT", "8001"))

QUEUE_MAX_LENGTH = int(os.getenv("QUEUE_MAX_LENGTH", "50"))
QUEUE_OVERFLOW = os.getenv("QUEUE_OVERFLOW", "reject-publish")
DLQ_NAME = os.getenv("DLQ_NAME", "tasks_dlq")
MAX_DELIVERY_ATTEMPTS = int(os.getenv("MAX_DELIVERY_ATTEMPTS", "3"))
RAW_DATA_PATH = Path(
    os.getenv("RAW_DATA_PATH", str(BASE_DIR / "data" / "raw" / "cumulative.csv"))
)
PRODUCER_INITIAL_DELAY = float(os.getenv("PRODUCER_INITIAL_DELAY", "0.001"))
PRODUCER_MIN_DELAY = float(os.getenv("PRODUCER_MIN_DELAY", "0.0001"))
PRODUCER_MAX_DELAY = float(os.getenv("PRODUCER_MAX_DELAY", "5.0"))
PRODUCER_BACKOFF_FACTOR = float(os.getenv("PRODUCER_BACKOFF_FACTOR", "1.5"))
PRODUCER_BACKOFF_STEP = float(os.getenv("PRODUCER_BACKOFF_STEP", "0.05"))
PRODUCER_RECOVERY_FACTOR = float(os.getenv("PRODUCER_RECOVERY_FACTOR", "0.9"))
