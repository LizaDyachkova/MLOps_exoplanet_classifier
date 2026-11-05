import asyncio
import csv
import os
import re
import signal
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

import aio_pika

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "experiment"
RABBIT, POSTGRES = "experiment_rabbit", "experiment_postgres"
AMQP_PORT, PG_PORT = 29672, 29432
URL = f"amqp://guest:guest@localhost:{AMQP_PORT}/"
PHASE_SECONDS = 90
SAMPLE_SECONDS = 3

ENV = {
    **os.environ,
    "PYTHONDONTWRITEBYTECODE": "1",
    "RABBITMQ_URL": URL,
    "DATABASE_URL": f"postgresql://postgres:postgres@localhost:{PG_PORT}/experiment",
    "MLFLOW_TRACKING_URI": os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"),
    "INFERENCE_DELAY_SECONDS": "0.5",
    "PREFETCH_COUNT": "1",
}
SCENARIOS = {"limited": "50", "unlimited": "0"}


def run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, text=True, check=False)


def start_rabbit() -> None:
    for _ in range(4):
        run("docker", "rm", "-f", "-v", RABBIT)
        run(
            "docker", "run", "-d", "--name", RABBIT,
            "-p", f"{AMQP_PORT}:5672", "rabbitmq:3-management",
        )  # fmt: skip
        time.sleep(10)
        for _ in range(45):
            ready = run(
                "docker", "exec", RABBIT, "rabbitmq-diagnostics", "-q",
                "check_port_connectivity",
            )  # fmt: skip
            if ready.returncode == 0:
                return
            state = run("docker", "inspect", "-f", "{{.State.Running}}", RABBIT)
            if "true" not in state.stdout:
                break
            time.sleep(2)
    raise RuntimeError("RabbitMQ did not start")


def start_infrastructure() -> None:
    run("docker", "rm", "-f", "-v", POSTGRES)
    start_rabbit()
    run(
        "docker", "run", "-d", "--name", POSTGRES, "-p", f"{PG_PORT}:5432",
        "-e", "POSTGRES_PASSWORD=postgres", "-e", "POSTGRES_DB=experiment",
        "postgres:16-alpine",
    )  # fmt: skip
    for _ in range(60):
        if (
            run("docker", "exec", POSTGRES, "pg_isready", "-U", "postgres").returncode
            == 0
        ):
            time.sleep(3)
            return
        time.sleep(1)
    raise RuntimeError("PostgreSQL did not start")


def stop_infrastructure() -> None:
    run("docker", "rm", "-f", "-v", RABBIT, POSTGRES)


def spawn(log_path: Path, module: str, **env: str) -> subprocess.Popen:
    with open(log_path, "w", encoding="utf-8") as log:
        return subprocess.Popen(
            [sys.executable, "-m", module],
            cwd=ROOT,
            env={**ENV, **env},
            stdout=log,
            stderr=subprocess.STDOUT,
        )


def wait_worker(port: int) -> None:
    for _ in range(120):
        try:
            urllib.request.urlopen(f"http://localhost:{port}/healthz", timeout=1)
            time.sleep(1)
            return
        except OSError:
            time.sleep(0.5)
    raise RuntimeError("Worker did not start")


def rss_mb(process: subprocess.Popen) -> float:
    if process.poll() is not None:
        return float("nan")
    output = run("ps", "-o", "rss=", "-p", str(process.pid)).stdout.strip()
    return int(output) / 1024 if output else float("nan")


def rabbit_memory_mb() -> float:
    output = run(
        "docker", "stats", "--no-stream", "--format", "{{.MemUsage}}", RABBIT
    ).stdout  # fmt: skip
    match = re.match(r"([\d.]+)(MiB|GiB)", output)
    if not match:
        return float("nan")
    return float(match[1]) * (1024 if match[2] == "GiB" else 1)


class DepthProbe:
    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.loop.run_forever, daemon=True).start()
        self.channel = self._wait(self._open())

    def _wait(self, coroutine):
        return asyncio.run_coroutine_threadsafe(coroutine, self.loop).result(30)

    async def _open(self):
        self.connection = await aio_pika.connect(URL)
        return await self.connection.channel()

    async def _depth(self) -> int:
        queue = await self.channel.declare_queue("tasks", passive=True)
        return queue.declaration_result.message_count

    def depth(self) -> int:
        return self._wait(self._depth())

    def close(self) -> None:
        self._wait(self.connection.close())
        self.loop.call_soon_threadsafe(self.loop.stop)


def run_scenario(name: str, max_length: str, logs: Path) -> list[dict]:
    env = {"QUEUE_MAX_LENGTH": max_length}
    run("docker", "exec", RABBIT, "rabbitmqctl", "-q", "delete_queue", "tasks")
    run("docker", "exec", RABBIT, "rabbitmqctl", "-q", "delete_queue", "tasks_dlq")

    worker_logs = [logs / f"{name}_worker1.log", logs / f"{name}_worker2.log"]
    producer_log = logs / f"{name}_producer.log"
    worker_1 = spawn(worker_logs[0], "src.worker", WORKER_HEALTH_PORT="29101", **env)
    wait_worker(29101)
    probe = DepthProbe()
    producer = spawn(producer_log, "src.batch_producer", **env)

    worker_2 = None
    rows: list[dict] = []
    started = time.time()
    published_before = 0
    seen_at = started
    while time.time() - started < PHASE_SECONDS * 2:
        time.sleep(SAMPLE_SECONDS)
        now = time.time()
        if worker_2 is None and now - started >= PHASE_SECONDS:
            worker_2 = spawn(
                worker_logs[1], "src.worker", WORKER_HEALTH_PORT="29102", **env
            )

        text = producer_log.read_text(encoding="utf-8")
        published = re.findall(r"total_published=(\d+)", text)
        published_now = int(published[-1]) if published else 0
        totals: list[float] = []
        for path in worker_logs:
            if path.exists():
                totals += [
                    float(value)
                    for value in re.findall(
                        r"total_sec=([\d.]+)", path.read_text(encoding="utf-8")
                    )
                ]
        rows.append(
            {
                "scenario": name,
                "t_sec": round(now - started, 1),
                "queue_depth": probe.depth(),
                "source_tasks_per_sec": round(
                    (published_now - published_before) / (now - seen_at), 2
                ),
                "published_total": published_now,
                "completed_total": len(totals),
                "task_time_sec": round(statistics.fmean(totals[-10:]), 2)
                if totals
                else "",
                "producer_rss_mb": round(rss_mb(producer), 1),
                "worker_rss_mb": round(rss_mb(worker_1), 1),
                "rabbit_memory_mb": round(rabbit_memory_mb(), 1),
                "workers": 1 if worker_2 is None else 2,
            }
        )
        print(rows[-1], flush=True)
        published_before, seen_at = published_now, now

    probe.close()
    for process in (producer, worker_1, worker_2):
        if process is not None and process.poll() is None:
            process.send_signal(signal.SIGTERM)
    for process in (producer, worker_1, worker_2):
        if process is not None:
            try:
                process.wait(25)
            except subprocess.TimeoutExpired:
                process.kill()
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    with tempfile.TemporaryDirectory() as tmp:
        try:
            start_infrastructure()
            for name, max_length in SCENARIOS.items():
                rows += run_scenario(name, max_length, Path(tmp))
        finally:
            stop_infrastructure()
    with open(OUT / "results.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {OUT / 'results.csv'}")


if __name__ == "__main__":
    main()
