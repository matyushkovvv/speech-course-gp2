# Запуск вокодера HiFiGAN

```bash
python -m venv venv
source venv/bin/activate       
pip install -r requirements.txt
```

---

## 1. Скачать данные

```bash
python download_data.py
```

Чтобы указать другую папку:

```bash
python download_data.py --data_dir /path/to/data
```

---

## 2. Обучение

### Локально

```bash
python train.py
```

Основные аргументы:

| Аргумент | По умолчанию | Описание |
|---|---|---|
| `--data_path` | `./data/LJSpeech-1.1` | Путь к датасету |
| `--checkpoint_dir` | `./checkpoints` | Куда сохранять веса |
| `--num_epochs` | `10` | Количество эпох (рекомендуется 300) |
| `--batch_size` | `16` | Размер батча |
| `--resume` | — | Путь к чекпоинту для продолжения |

Пример на 300 эпох:

```bash
python train.py --num_epochs 300
```

Продолжить с чекпоинта:

```bash
python train.py --num_epochs 300 --resume checkpoints/vocoder_epoch0150.pt
```

Чекпоинты сохраняются каждые 10 эпох в `checkpoints/vocoder_epochXXXX.pt`.
Финальная модель — `checkpoints/vocoder_final.pt`.

---

## 3. Генерация сэмплов

### Через inference.py

```bash
python inference.py --checkpoint checkpoints/vocoder_final.pt
```

Аргументы:

| Аргумент | По умолчанию | Описание |
|---|---|---|
| `--checkpoint` | обязательный | Путь к файлу `.pt` |
| `--sentences` | `test_sentences.txt` | Файл с фразами (по одной на строку) |
| `--output_dir` | `samples` | Папка для сохранения `.wav` |

Пример с кастомными путями:

```bash
python inference.py \
    --checkpoint checkpoints/vocoder_final.pt \
    --sentences my_sentences.txt \
    --output_dir output/
```

Результат — файлы `sample_01.wav`, `sample_02.wav`, ... в папке `--output_dir`.
