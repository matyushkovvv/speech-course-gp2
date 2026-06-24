from dataclasses import dataclass, field
from typing import List


@dataclass
class TrainConfig:
    # пути
    data_path: str = "./data/LJSpeech-1.1"
    mel_cache_dir: str = "./data/mel_cache"  # кеш мел-спектрограмм; "" — отключить
    segment_size: int = 8192       # сэмплов на один обучающий отрезок
    batch_size: int = 16
    num_workers: int = 4

    # аудио — должно совпадать с конфигом FastPitch
    sample_rate: int = 22050
    n_fft: int = 1024
    win_length: int = 1024
    hop_length: int = 256
    num_mels: int = 80
    mel_fmin: float = 0.0
    mel_fmax: float = 8000.0
    log_func: str = "np.log"
    mel_power: float = 1.5
    spec_gain: float = 1.0
    signal_norm: bool = False

    # генератор (HiFiGAN)
    upsample_rates: List[int] = field(default_factory=lambda: [8, 8, 2, 2])
    upsample_kernel_sizes: List[int] = field(default_factory=lambda: [16, 16, 4, 4])
    upsample_initial_channel: int = 128
    resblock_kernel_sizes: List[int] = field(default_factory=lambda: [3, 7, 11])
    resblock_dilation_sizes: List[List[int]] = field(
        default_factory=lambda: [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
    )

    # обучение
    learning_rate: float = 2e-4
    adam_b1: float = 0.8
    adam_b2: float = 0.99
    lr_decay: float = 0.999
    num_epochs: int = 300
    checkpoint_interval: int = 10
    checkpoint_dir: str = "./checkpoints"
    log_interval: int = 100

    # веса потерь
    lambda_mel: float = 45.0
    lambda_fm: float = 2.0
