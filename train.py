"""
Запуск:
    python train.py [--data_path ПУТЬ] [--checkpoint_dir ПАПКА] [--resume ЧЕКПОИНТ]
"""

import argparse
import os
import sys
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import TrainConfig
from dataset import LJSpeechDataset, collate_fn
from model import Generator, MultiPeriodDiscriminator, MultiScaleDiscriminator
from losses import discriminator_loss, generator_loss


def save_metrics(metrics_path: str, history: dict):
    with open(metrics_path, "w") as f:
        json.dump(history, f, indent=2)


def plot_metrics(history: dict, out_dir: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib не установлен — пропускаем построение графиков.")
        return

    epochs = history["epoch"]
    plots = {
        "losses_generator.png": {
            "title": "Generator losses",
            "series": [
                ("g_loss", "G total"),
                ("adv_loss", "Adversarial"),
                ("fm_loss", "Feature matching"),
                ("mel_loss", "Mel"),
            ],
        },
        "losses_discriminator.png": {
            "title": "Discriminator loss",
            "series": [("d_loss", "D total")],
        },
        "learning_rate.png": {
            "title": "Learning rate",
            "series": [("lr", "LR")],
        },
    }

    for fname, cfg in plots.items():
        fig, ax = plt.subplots(figsize=(9, 4))
        for key, label in cfg["series"]:
            if key in history:
                ax.plot(epochs, history[key], label=label)
        ax.set_xlabel("Epoch")
        ax.set_title(cfg["title"])
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out_path = os.path.join(out_dir, fname)
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        print(f"  График сохранён: {out_path}")


def save_checkpoint(path, generator, mpd, msd, opt_g, opt_d, epoch, step):
    torch.save({
        "generator": generator.state_dict(),
        "mpd": mpd.state_dict(),
        "msd": msd.state_dict(),
        "opt_g": opt_g.state_dict(),
        "opt_d": opt_d.state_dict(),
        "epoch": epoch,
        "step": step,
    }, path)
    print(f"  Чекпоинт сохранён: {path}")


def load_checkpoint(path, generator, mpd, msd, opt_g, opt_d, device):
    ckpt = torch.load(path, map_location=device)
    generator.load_state_dict(ckpt["generator"])
    mpd.load_state_dict(ckpt["mpd"])
    msd.load_state_dict(ckpt["msd"])
    opt_g.load_state_dict(ckpt["opt_g"])
    opt_d.load_state_dict(ckpt["opt_d"])
    return ckpt["epoch"], ckpt["step"]


def train(cfg: TrainConfig, resume: str = None):
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Устройство: {device}")

    Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    with open(os.path.join(cfg.checkpoint_dir, "config.json"), "w") as f:
        json.dump(cfg.__dict__, f, indent=2)

    train_ds = LJSpeechDataset(cfg, split="train")
    val_ds   = LJSpeechDataset(cfg, split="val")

    # spawn-контекст безопаснее fork на macOS/CUDA; pin_memory только для CUDA
    num_workers = cfg.num_workers
    mp_context  = "spawn" if num_workers > 0 else None
    pin_memory  = device == "cuda"

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=num_workers,
        multiprocessing_context=mp_context,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        collate_fn=collate_fn,
        drop_last=True,
    )
    print(f"Обучение: {len(train_ds)} сэмплов, валидация: {len(val_ds)}")

    generator = Generator(cfg).to(device)
    mpd = MultiPeriodDiscriminator().to(device)
    msd = MultiScaleDiscriminator().to(device)

    opt_g = torch.optim.AdamW(
        generator.parameters(), lr=cfg.learning_rate,
        betas=(cfg.adam_b1, cfg.adam_b2),
    )
    opt_d = torch.optim.AdamW(
        list(mpd.parameters()) + list(msd.parameters()),
        lr=cfg.learning_rate, betas=(cfg.adam_b1, cfg.adam_b2),
    )

    sched_g = torch.optim.lr_scheduler.ExponentialLR(opt_g, gamma=cfg.lr_decay)
    sched_d = torch.optim.lr_scheduler.ExponentialLR(opt_d, gamma=cfg.lr_decay)

    start_epoch, global_step = 0, 0
    if resume:
        start_epoch, global_step = load_checkpoint(
            resume, generator, mpd, msd, opt_g, opt_d, device
        )
        print(f"Продолжаем с эпохи {start_epoch}, шаг {global_step}")

    metrics_path = os.path.join(cfg.checkpoint_dir, "metrics.json")
    history: dict = {"epoch": [], "d_loss": [], "g_loss": [], "adv_loss": [], "fm_loss": [], "mel_loss": [], "lr": []}

    # подгружаем историю при возобновлении
    if resume and os.path.exists(metrics_path):
        with open(metrics_path) as f:
            history = json.load(f)

    epoch_bar = tqdm(range(start_epoch, cfg.num_epochs), desc="Эпохи", unit="ep")

    for epoch in epoch_bar:
        generator.train()
        mpd.train()
        msd.train()

        sum_d, sum_g, sum_adv, sum_fm, sum_mel, n_batches = 0.0, 0.0, 0.0, 0.0, 0.0, 0

        batch_bar = tqdm(train_loader, desc=f"Эпоха {epoch+1:3d}", unit="batch", leave=False)

        for mel, audio in batch_bar:
            mel   = mel.to(device)              # [B, 80, T_mel]
            audio = audio.to(device)
            audio = audio.unsqueeze(1)          # [B, 1, T_audio]

            audio_fake = generator(mel)

            # выравниваем длины
            min_len = min(audio.shape[-1], audio_fake.shape[-1])
            audio      = audio[..., :min_len]
            audio_fake = audio_fake[..., :min_len]

            # шаг дискриминатора
            opt_d.zero_grad()
            mpd_r_outs, mpd_f_outs, _, _ = mpd(audio, audio_fake.detach())
            msd_r_outs, msd_f_outs, _, _ = msd(audio, audio_fake.detach())
            d_loss = (
                discriminator_loss(mpd_r_outs, mpd_f_outs)
                + discriminator_loss(msd_r_outs, msd_f_outs)
            )
            d_loss.backward()
            opt_d.step()

            # шаг генератора
            opt_g.zero_grad()
            mpd_r_outs, mpd_f_outs, mpd_r_fmaps, mpd_f_fmaps = mpd(audio, audio_fake)
            msd_r_outs, msd_f_outs, msd_r_fmaps, msd_f_fmaps = msd(audio, audio_fake)
            g_loss, adv_val, fm_val, mel_val = generator_loss(
                audio.squeeze(1), audio_fake,
                mpd_r_outs, mpd_f_outs, mpd_r_fmaps, mpd_f_fmaps,
                msd_r_outs, msd_f_outs, msd_r_fmaps, msd_f_fmaps,
                cfg,
            )
            g_loss.backward()
            opt_g.step()

            global_step += 1
            n_batches   += 1
            sum_d   += d_loss.item()
            sum_g   += g_loss.item()
            sum_adv += adv_val
            sum_fm  += fm_val
            sum_mel += mel_val

            batch_bar.set_postfix(
                D=f"{d_loss.item():.3f}",
                G=f"{g_loss.item():.3f}",
                adv=f"{adv_val:.3f}",
                fm=f"{fm_val:.3f}",
                mel=f"{mel_val:.3f}",
            )

        sched_g.step()
        sched_d.step()

        cur_lr = opt_g.param_groups[0]["lr"]
        avg_d, avg_g = sum_d / n_batches, sum_g / n_batches
        avg_adv, avg_fm, avg_mel = sum_adv / n_batches, sum_fm / n_batches, sum_mel / n_batches

        history["epoch"].append(epoch + 1)
        history["d_loss"].append(round(avg_d, 5))
        history["g_loss"].append(round(avg_g, 5))
        history["adv_loss"].append(round(avg_adv, 5))
        history["fm_loss"].append(round(avg_fm, 5))
        history["mel_loss"].append(round(avg_mel, 5))
        history["lr"].append(cur_lr)
        save_metrics(metrics_path, history)

        epoch_bar.set_postfix(
            D=f"{avg_d:.3f}",
            G=f"{avg_g:.3f}",
            mel=f"{avg_mel:.3f}",
            lr=f"{cur_lr:.2e}",
        )

        if (epoch + 1) % cfg.checkpoint_interval == 0:
            ckpt_path = os.path.join(cfg.checkpoint_dir, f"vocoder_epoch{epoch+1:04d}.pt")
            save_checkpoint(ckpt_path, generator, mpd, msd, opt_g, opt_d, epoch + 1, global_step)

    save_checkpoint(
        os.path.join(cfg.checkpoint_dir, "vocoder_final.pt"),
        generator, mpd, msd, opt_g, opt_d, cfg.num_epochs, global_step,
    )
    print("Обучение завершено.")
    plot_metrics(history, cfg.checkpoint_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path",      default="./data/LJSpeech-1.1")
    parser.add_argument("--checkpoint_dir", default="./checkpoints")
    parser.add_argument("--batch_size",     type=int,   default=16)
    parser.add_argument("--num_epochs",     type=int,   default=10)
    parser.add_argument("--lr",             type=float, default=2e-4)
    parser.add_argument("--resume",         default=None)
    args = parser.parse_args()

    cfg = TrainConfig(
        data_path=args.data_path,
        checkpoint_dir=args.checkpoint_dir,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.lr,
    )
    train(cfg, resume=args.resume)
