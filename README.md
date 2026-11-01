# PingPongVision 🏓

**Отслеживание мяча и аналитика отскоков от стола** — полный пайплайн компьютерного зрения (Computer Vision) для настольного тенниса.

## Архитектура проекта

```
[Видеопоток / Датасет]
   │
   ├──> 1. Пайплайн данных и EDA        ← Шаг 1 (Выполнен)
   │       (Загрузчики датасета TTNet, EDA траектории и видимости)
   │
   ├──> 2. Детекция стола и гомография  ← Шаг 2 (Выполнен)
   │       (Стандарты ITTF, авто-поиск углов, вид сверху / Bird's Eye View)
   │
   ├──> 3. Детекция и трекинг мяча      ← Шаг 3 (Выполнен)
   │       (Фильтр Калмана, сглаживание размытия, скорость в км/ч, YOLO/CV)
   │
   └──> 4. Аналитика и веб-дашборд      ← Шаг 4 (Выполнен)
           (Интерактивный UI на Streamlit, 2D тепловые карты, отчеты о матче)
```

## Датасет

**Источник:** [lab.osai.ai](https://lab.osai.ai/) — академический датасет TTNet.

Каждый игровой клип содержит:
| Файл | Описание |
|------|-------------|
| `ball_markup.json` | `{frame_id: {x, y}}` — центр мяча в пикселях; `(-1,-1)` = мяч не виден |
| `events_markup.json` | `{frame_id: event}` — `"bounce"` (отскок), `"net"` (сетка), `"empty_event"` (пустое событие) |
| `segmentation_masks/{frame_id}.png` | RGB 320×128 семантическая сегментация (красный цвет = поверхность стола) |

Координаты мяча приведены относительно Full-HD кадров **1920×1080**.  
Маски сегментации масштабированы до **320×128**.

## Установка и настройка

```bash
# 1. Клонирование и создание виртуального окружения
python -m venv .venv
.venv\Scripts\activate       # Windows
# source .venv/bin/activate  # Linux/macOS

# 2. Установка проекта с зависимостями для разработки
pip install -e ".[dev]"

# Опционально: установка Ultralytics для обучения собственных моделей YOLO
pip install -e ".[models]"
```

## Структура проекта

```
table-tennis-cv/
├── app.py                    # Точка входа для запуска дашборда Streamlit
├── data/
│   ├── raw/                  # Исходные файлы датасета (из lab.osai.ai)
│   │   ├── ball_markup.json
│   │   ├── events_markup.json
│   │   └── segmentation_masks/
│   └── processed/            # Промежуточные артефакты / датасет YOLO
├── src/
│   └── ttcv/
│       ├── __init__.py
│       ├── app.py            # Основное веб-приложение на Streamlit
│       ├── analytics/        # Шаг 4: Игровые метрики и отчеты о матче
│       │   ├── __init__.py
│       │   ├── metrics.py    # RallyAnalyzer, RallyMetrics, ShotInfo
│       │   └── report.py     # Генераторы отчетов о матче в Markdown и JSON
│       ├── data/             # Шаг 1: Загрузчики данных и EDA
│       │   ├── __init__.py
│       │   ├── dataset.py    # TTNetDataset — базовый загрузчик данных
│       │   └── eda.py        # Вспомогательные функции для анализа данных
│       ├── geometry/         # Шаг 2: Стандарты ITTF и гомография
│       │   ├── __init__.py
│       │   ├── table.py      # Размеры ITTF, зоны стола и координаты
│       │   ├── homography.py # Детекция углов стола и модуль TableHomography
│       │   └── visualizer.py # Отрисовка 2D стола (BEV) и тепловых карт отскоков
│       └── tracking/         # Шаг 3: Отслеживание мяча и детекторы
│           ├── __init__.py
│           ├── kalman.py     # 6-параметрический фильтр Калмана постоянного ускорения
│           ├── tracker.py    # BallTracker (интерполяция, скорость, шлейф мяча)
│           ├── detector.py   # Детекторы: YOLOv8, ColorMotion и Mock
│           └── dataset_generator.py # Конвертер в YOLO и генератор синтетических розыгрышей
├── scripts/
│   ├── run_eda.py            # Запуск EDA по датасету и сохранение графиков
│   ├── run_homography.py     # Детекция стола, проекция BEV и тепловая карта
│   ├── infer.py              # Запуск инференса трекинга и 2D мини-карты на видео
│   ├── prepare_yolo_dataset.py # Конвертация ball_markup.json в формат YOLO
│   └── train_yolo.py         # Запуск обучения YOLOv8 через Ultralytics
├── tests/
│   ├── test_dataset.py       # 25 модульных тестов для структур данных
│   ├── test_eda.py           # 6 смоук-тестов для визуализаций EDA
│   ├── test_geometry.py      # 29 тестов геометрии стола и гомографии
│   ├── test_tracking.py      # 16 тестов для фильтра Калмана, трекера и детекторов
│   └── test_analytics.py     # 6 тестов для аналитики розыгрышей и отчетов
├── outputs/                  # Сгенерированные графики, видео и отчеты (.gitignore)
├── pyproject.toml
└── README.md
```

## Быстрый старт и примеры использования

### 1. Интерактивный веб-дашборд Streamlit (Шаг 4)
Запустите интерактивное веб-приложение в браузере:
```bash
streamlit run app.py
```
**Возможности:**
- Интерактивная аналитика стола с **тепловыми картами плотности отскоков (вид сверху / Bird's Eye View)**.
- Быстрое переключение между **реальными данными матча TTNet**, **физической симуляцией розыгрыша** и **загрузкой собственного видео**.
- Профиль скорости мяча (км/ч) и графики вертикальной траектории.
- Видеотрекинг с **эффектом кометного шлейфа** и живой **2D мини-картой** в углу экрана.
- Автоматический экспорт итогового отчета о матче (в форматах Markdown и JSON).

---

### 2. Инференс и трекинг на видео (Шаг 3)
```bash
# A. Физическая симуляция розыгрыша (генерирует видео с живой BEV мини-картой и шлейфом мяча):
python scripts/infer.py --synthetic --max-frames 120 --output outputs/tracking/synthetic_tracked.mp4

# B. Воспроизведение разметки датасета с фильтрацией Калмана и расчетом скорости:
python scripts/infer.py --replay-markup data/raw/ball_markup.json --start-frame 1298 --max-frames 120 --output outputs/tracking/real_markup_tracked.mp4

# C. Пользовательское видео с весами обученной модели YOLOv8:
python scripts/infer.py --video input.mp4 --weights best.pt --mask data/raw/segmentation_masks/1278.png --output outputs/tracking/match_tracked.mp4
```

---

### 3. Детекция стола и гомография вида сверху (Шаг 2)
```bash
python scripts/run_homography.py --mask data/raw/segmentation_masks/1278.png --output-dir outputs/homography
```
Извлекает углы из маски сегментации, вычисляет матрицу гомографии и проецирует все события отскоков в 2D-координаты стола по стандарту ITTF.

---

### 4. Разведочный анализ данных (EDA) (Шаг 1)
```bash
python scripts/run_eda.py data/raw --output-dir outputs/eda
```

---

### 5. Обучение детектора мяча YOLOv8
```bash
# 1. Конвертация датасета в формат YOLO:
python scripts/prepare_yolo_dataset.py data/raw/ball_markup.json --output-dir data/processed/yolo_dataset

# 2. Обучение YOLOv8-nano на датасете мячей:
python scripts/train_yolo.py --data data/processed/yolo_dataset/dataset.yaml --epochs 50 --imgsz 1280
```

---

### 6. Запуск тестов
```bash
python -m pytest tests/ -v
# 82 passed in ~5s
```

## План разработки (Roadmap)

- [x] **Шаг 1** — Пайплайн данных и EDA
- [x] **Шаг 2** — Детекция стола и гомография (BEV)
- [x] **Шаг 3** — Детекция и трекинг мяча (фильтр Калмана + YOLO / классические детекторы)
- [x] **Шаг 4** — Аналитический дашборд (интерактивное демо на Streamlit)
