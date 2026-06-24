import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset
from tqdm import tqdm

from config import TrainConfig


def compute_mel_spec(audio: torch.Tensor, cfg: TrainConfig) -> torch.Tensor:
    # всегда на CPU — torch.stft не поддерживается на MPS
    audio = audio.cpu()
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)

    window = torch.hann_window(cfg.win_length)
    stft = torch.stft(
        audio.squeeze(0),
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        win_length=cfg.win_length,
        window=window,
        return_complex=True,
    )
    magnitude = stft.abs()

    mel_fb = torchaudio.functional.melscale_fbanks(
        n_freqs=cfg.n_fft // 2 + 1,
        f_min=cfg.mel_fmin,
        f_max=cfg.mel_fmax,
        n_mels=cfg.num_mels,
        sample_rate=cfg.sample_rate,
        norm=None,
        mel_scale="htk",
    )

    mel = torch.matmul(magnitude.T, mel_fb).T        # [num_mels, T]
    mel = mel ** cfg.mel_power
    mel = torch.log(torch.clamp(mel, min=1e-5)) * cfg.spec_gain
    return mel


def _load_audio(path: Path, cfg: TrainConfig) -> torch.Tensor:
    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    audio = torch.from_numpy(data)
    if audio.dim() == 2:
        audio = audio.mean(dim=1)  # стерео -> моно
    if sr != cfg.sample_rate:
        audio = torchaudio.functional.resample(audio, sr, cfg.sample_rate)
    return audio


class LJSpeechDataset(Dataset):
    """
    Датасет LJSpeech для обучения вокодера.
    При первом запуске кеширует мел-спектрограммы на диск.
    Возвращает (мел [80, T_mel], аудио [segment_size]).
    """

    def __init__(self, cfg: TrainConfig, split: str = "train"):
        self.cfg = cfg
        self.segment_size = cfg.segment_size
        self.mel_segment  = cfg.segment_size // cfg.hop_length
        self.cache_dir    = Path(cfg.mel_cache_dir) if cfg.mel_cache_dir else None

        wav_dir = Path(cfg.data_path) / "wavs"
        all_files = sorted(wav_dir.glob("*.wav"))
        assert len(all_files) > 0, f"Нет wav-файлов в {wav_dir}"

        rng = random.Random(42)
        rng.shuffle(all_files)
        val_size = min(200, int(0.05 * len(all_files)))
        self.files = all_files[val_size:] if split == "train" else all_files[:val_size]

        if self.cache_dir:
            self._build_cache()

    def _build_cache(self):
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        missing = [p for p in self.files if not (self.cache_dir / (p.stem + ".npy")).exists()]
        if not missing:
            return
        print(f"Кеширование мел-спектрограмм: {len(missing)} файлов -> {self.cache_dir}")
        for path in tqdm(missing, desc="Мел-кеш", unit="файл"):
            audio = _load_audio(path, self.cfg)
            mel   = compute_mel_spec(audio, self.cfg)
            np.save(str(self.cache_dir / (path.stem + ".npy")), mel.numpy())

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]

        # загружаем мел из кеша или считаем на лету
        if self.cache_dir:
            mel_full = torch.from_numpy(
                np.load(str(self.cache_dir / (path.stem + ".npy")))
            )
        else:
            mel_full = compute_mel_spec(_load_audio(path, self.cfg), self.cfg)

        T_mel = mel_full.shape[1]

        # случайный отрезок
        start_mel = random.randint(0, max(0, T_mel - self.mel_segment))
        start_audio = start_mel * self.cfg.hop_length

        mel = mel_full[:, start_mel: start_mel + self.mel_segment]
        if mel.shape[1] < self.mel_segment:
            mel = torch.nn.functional.pad(mel, (0, self.mel_segment - mel.shape[1]))

        # читаем только нужный сегмент аудио (seek)
        data, sr = sf.read(
            str(path), dtype="float32", always_2d=False,
            start=start_audio, frames=self.segment_size,
        )
        audio = torch.from_numpy(data)
        if audio.dim() == 2:
            audio = audio.mean(dim=1)
        if sr != self.cfg.sample_rate:
            audio = torchaudio.functional.resample(audio, sr, self.cfg.sample_rate)
        if audio.shape[0] < self.segment_size:
            audio = torch.nn.functional.pad(audio, (0, self.segment_size - audio.shape[0]))
        else:
            audio = audio[: self.segment_size]

        return mel, audio


def collate_fn(batch):
    mels, audios = zip(*batch)
    return torch.stack(mels), torch.stack(audios)
