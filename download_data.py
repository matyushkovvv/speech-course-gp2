"""
Запуск:
    python download_data.py [--data_dir ./data]
"""

import argparse
import os
import tarfile
import urllib.request
from pathlib import Path

LJSPEECH_URL = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"
ARCHIVE_NAME  = "LJSpeech-1.1.tar.bz2"


def download_with_progress(url: str, dest: str):
    def progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        pct = downloaded / total_size * 100 if total_size > 0 else 0
        mb = downloaded / 1024 / 1024
        print(f"\r  {mb:.1f} МБ  ({pct:.1f}%)", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=progress)
    print()


def main(data_dir: str):
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    archive_path = os.path.join(data_dir, ARCHIVE_NAME)
    dataset_path = os.path.join(data_dir, "LJSpeech-1.1")

    if os.path.isdir(dataset_path):
        print(f"Датасет уже скачан: {dataset_path}")
        return

    if not os.path.isfile(archive_path):
        print(f"Скачиваем LJSpeech (~2.6 ГБ)...")
        download_with_progress(LJSPEECH_URL, archive_path)
    else:
        print(f"Архив уже есть: {archive_path}")

    print("Распаковываем...")
    with tarfile.open(archive_path, "r:bz2") as tar:
        tar.extractall(data_dir)

    print(f"Датасет готов: {dataset_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./data")
    args = parser.parse_args()
    main(args.data_dir)
