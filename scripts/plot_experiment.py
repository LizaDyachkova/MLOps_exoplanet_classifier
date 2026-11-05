from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "docs" / "experiment"
SCENARIOS = {
    "limited": ("С лимитом очереди 50", "#2a78d6"),
    "unlimited": ("Без лимита", "#eb6834"),
}
SECOND_WORKER_SEC = 90
RATE_WINDOW = 5
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
PANELS = [
    ("queue_depth", "Глубина очереди, сообщений", True),
    ("source_rate", "Скорость источника, задач/с (окно 15 с)", True),
    ("task_time_sec", "Время выполнения задачи, с", True),
    ("rabbit_memory_mb", "Память RabbitMQ, МБ", False),
    ("producer_rss_mb", "Память источника, МБ", False),
    ("throughput", "Скорость воркеров, задач/с (окно 15 с)", False),
]


def prepare(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    window_sec = data["t_sec"].diff(RATE_WINDOW)
    data["source_rate"] = data["published_total"].diff(RATE_WINDOW) / window_sec
    data["throughput"] = data["completed_total"].diff(RATE_WINDOW) / window_sec
    return data


def main() -> None:
    results = pd.read_csv(OUT / "results.csv")
    figure, axes = plt.subplots(3, 2, figsize=(11, 10), facecolor=SURFACE)
    for axis, (column, title, log_scale) in zip(axes.flat, PANELS, strict=True):
        axis.set_facecolor(SURFACE)
        for name, (label, color) in SCENARIOS.items():
            data = prepare(results[results["scenario"] == name])
            values = data[column]
            if log_scale:
                values = values.where(values > 0)
            axis.plot(data["t_sec"], values, color=color, linewidth=2, label=label)
        axis.axvline(SECOND_WORKER_SEC, color=MUTED, linewidth=1, linestyle="--")
        if log_scale:
            axis.set_yscale("log")
        else:
            axis.set_ylim(bottom=0)
        axis.set_xlim(0, 190)
        axis.set_title(title, loc="left", fontsize=11, color=INK)
        axis.grid(axis="y", color=GRID, linewidth=0.8)
        axis.set_axisbelow(True)
        axis.tick_params(colors=MUTED, length=0, labelsize=9)
        for side in ("top", "right", "left"):
            axis.spines[side].set_visible(False)
        axis.spines["bottom"].set_color(GRID)
    for axis in axes[-1]:
        axis.set_xlabel("Время от старта, с", color=MUTED, fontsize=9)

    handles, labels = axes[0][0].get_legend_handles_labels()
    handles.append(plt.Line2D([], [], color=MUTED, linewidth=1, linestyle="--"))
    labels.append("Подключён второй воркер")
    figure.legend(
        handles,
        labels,
        loc="upper left",
        ncol=3,
        frameon=False,
        fontsize=10,
        bbox_to_anchor=(0.01, 1.0),
    )
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(OUT / "comparison.png", dpi=130, facecolor=SURFACE)


if __name__ == "__main__":
    main()
