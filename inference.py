"""
Запуск:
    python inference.py --checkpoint checkpoints/vocoder_final.pt \
                        --sentences test_sentences.txt \
                        --output_dir samples/
"""

import argparse
import os
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from config import TrainConfig
from model import Generator
from t2spec_converter import TextToSpecConverter


def load_generator(checkpoint_path: str, cfg: TrainConfig, device: str) -> Generator:
    generator = Generator(cfg).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    # поддерживаем оба формата чекпоинта: ключ "generator" и ключ "G"
    state = ckpt.get("generator", ckpt.get("G", ckpt))
    generator.load_state_dict(state)
    generator.eval()
    generator.remove_weight_norm()
    return generator


def mel_to_tensor(mel_np: np.ndarray, device: str) -> torch.Tensor:
    # мел [80, T] -> тензор [1, 80, T]
    return torch.tensor(mel_np, dtype=torch.float32).unsqueeze(0).to(device)


@torch.inference_mode()
def synthesize(text: str, t2s: TextToSpecConverter, generator: Generator, device: str) -> np.ndarray:
    mel_np = t2s.text2spec(text)
    if mel_np.shape[0] != 80:      # text2spec может вернуть [T, 80] вместо [80, T]
        mel_np = mel_np.T
    mel_tensor = mel_to_tensor(mel_np, device)

    waveform = generator(mel_tensor)   # [1, 1, T_audio]
    return waveform.squeeze().cpu().numpy()


def run_inference(checkpoint_path: str, sentences_path: str, output_dir: str):
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Устройство: {device}")

    cfg = TrainConfig()
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("Загружаем FastPitch...")
    t2s = TextToSpecConverter()

    print(f"Загружаем вокодер: {checkpoint_path}")
    generator = load_generator(checkpoint_path, cfg, device)

    with open(sentences_path, "r") as f:
        sentences = [line.strip() for line in f if line.strip()]

    print(f"Генерируем {len(sentences)} сэмплов...")
    for i, text in enumerate(sentences):
        print(f"  [{i+1}/{len(sentences)}] {text!r}")
        waveform = synthesize(text, t2s, generator, device)

        out_path = os.path.join(output_dir, f"sample_{i+1:02d}.wav")
        sf.write(out_path, waveform, cfg.sample_rate)
        print(f"    Сохранено: {out_path}")

    print("Готово.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sentences",  default="test_sentences.txt")
    parser.add_argument("--output_dir", default="samples")
    args = parser.parse_args()

    run_inference(args.checkpoint, args.sentences, args.output_dir)
