# S04 – eda_cli: мини-EDA для CSV

Небольшое CLI-приложение для базового анализа CSV-файлов.
Используется в рамках Семинара 03 курса «Инженерия ИИ».

## Требования

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) установлен в систему

## Инициализация проекта

В корне проекта (S03):

```bash
uv sync
```

Эта команда:

- создаст виртуальное окружение `.venv`;
- установит зависимости из `pyproject.toml`;
- установит сам проект `eda-cli` в окружение.


## Запуск HTTP-сервиса

Для запуска FastAPI сервиса:

```bash
uvicorn eda_cli.api:app --reload --host 0.0.0.0 --port 8000
```


Системные
GET /health - проверка работоспособности сервиса

Оценка качества данных
POST /quality - оценка качества по агрегированным признакам

POST /quality-from-csv - оценка качества по CSV-файлу

POST /detailed-quality - расширенный анализ качества с новыми эвристиками из HW03

Новые возможности (HW03)
Эндпоинт POST /detailed-quality предоставляет:

Полный набор флагов качества, включая новые эвристики:

has_constant_columns - проверка на константные колонки

has_high_cardinality_categoricals - проверка на высокую кардинальность категориальных признаков

Детальную информацию о проблемных колонках

Статистику по типам данных

Интегральную оценку качества



## Запуск CLI

### Краткий обзор

```bash
uv run eda-cli overview data/example.csv
```

Параметры:

- `--sep` – разделитель (по умолчанию `,`);
- `--encoding` – кодировка (по умолчанию `utf-8`).

### Полный EDA-отчёт

```bash
uv run eda-cli report data/example.csv --out-dir reports
```

В результате в каталоге `reports/` появятся:

- `report.md` – основной отчёт в Markdown;
- `summary.csv` – таблица по колонкам;
- `missing.csv` – пропуски по колонкам;
- `correlation.csv` – корреляционная матрица (если есть числовые признаки);
- `top_categories/*.csv` – top-k категорий по строковым признакам;
- `hist_*.png` – гистограммы числовых колонок;
- `missing_matrix.png` – визуализация пропусков;
- `correlation_heatmap.png` – тепловая карта корреляций.


### Дополнительные параметры отчёта

Команда `report` поддерживает дополнительные параметры для настройки отчёта:

```bash
uv run eda-cli report data/example.csv \
  --out-dir reports \
  --title "Первичный анализ" \
  --top-k-categories 10 \
  --min-missing-share 0.2 \
  --max-hist-columns 8


## Тесты

```bash
uv run pytest -q
```
