"""
Построение графиков обучения из сохранённого metrics.json.

Запуск:
    python plot_metrics.py [--metrics ПУТЬ] [--out_dir ПАПКА]
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot(history: dict, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    epochs = history["epoch"]

    specs = [
        (
            "losses_generator.png",
            "Generator losses (epoch avg)",
            [("g_loss", "G total"), ("adv_loss", "Adversarial"), ("fm_loss", "Feature matching"), ("mel_loss", "Mel")],
        ),
        (
            "losses_discriminator.png",
            "Discriminator loss (epoch avg)",
            [("d_loss", "D total")],
        ),
        (
            "learning_rate.png",
            "Learning rate",
            [("lr", "LR")],
        ),
        (
            "losses_all.png",
            "All losses (epoch avg)",
            [
                ("d_loss", "D total"),
                ("g_loss", "G total"),
                ("adv_loss", "Adversarial"),
                ("fm_loss", "Feature matching"),
                ("mel_loss", "Mel"),
            ],
        ),
    ]

    for fname, title, series in specs:
        fig, ax = plt.subplots(figsize=(10, 4))
        for key, label in series:
            if key in history and history[key]:
                ax.plot(epochs, history[key], label=label, linewidth=1.5)
        ax.set_xlabel("Epoch")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = os.path.join(out_dir, fname)
        fig.savefig(path, dpi=130)
        plt.close(fig)
        print(f"Сохранено: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics",  default="./checkpoints/metrics.json", help="Путь к metrics.json")
    parser.add_argument("--out_dir",  default="./checkpoints",              help="Куда сохранять графики")
    args = parser.parse_args()

    if not os.path.exists(args.metrics):
        raise FileNotFoundError(f"Файл не найден: {args.metrics}")

    with open(args.metrics) as f:
        history = json.load(f)

    print(f"Загружено {len(history['epoch'])} эпох из {args.metrics}")
    plot(history, args.out_dir)


if __name__ == "__main__":
    main()
