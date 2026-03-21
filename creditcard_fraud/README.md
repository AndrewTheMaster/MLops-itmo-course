# Credit Card Fraud — MLOps

**Данные:** [OpenML 43627](https://www.openml.org/d/43627), файл в `data/raw/dataset` (под DVC).  
**EDA:** `notebooks/EDA.ipynb`  
**Цель / метрики:** не пропускать fraud → главный ориентир **recall по классу 1**, также **F1** по fraud (см. MLflow).

## Установка

Рекомендуется **Python 3.12 или 3.13**. Если pip всё равно собирает пакеты из исходников и пишет про `Python.h`, на Fedora поставьте заголовки:

```bash
sudo dnf install python3-devel
```

Дальше:

```bash
cd creditcard_fraud
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
pip install -e .
pip install "dvc>=3.0"
```

Данные уже в репозитории / подтянуты с remote:

```bash
dvc pull          # при настроенном remote
```

## Пайплайн (DVC)

Полное воспроизведение: загрузка ARFF → interim → preprocess (**проверки качества** в `ge_validation.py`, эквивалент базовому GE-suite; пакет `great-expectations` не ставим — он требует `numpy<2` и ломает стек под Python 3.13) → train (MLflow + `models/model.pkl`):

```bash
cd creditcard_fraud
dvc repro
```

Отдельные шаги:

```bash
python -m creditcard_fraud.load_data
python -m creditcard_fraud.preprocess
python -m creditcard_fraud.modeling.train
```

Вручную «всё подряд» без DVC (как раньше `dataset`):

```bash
python -m creditcard_fraud.dataset
```

## MLflow UI

```bash
docker compose -f docker-compose.mlflow.yml up -d
```

Открыть: http://localhost:5000  
Эксперимент по умолчанию: `creditcard_fraud_baseline`. Локальный `train` пишет в `./.mlruns` (или `$MLFLOW_TRACKING_URI`). Compose монтирует `./mlruns` в контейнер UI.

## REST API (FastAPI)

Локально (нужны `models/model.pkl` и `models/model_meta.json` после `dvc repro` / `train`):

```bash
make serve
# или: python -m uvicorn creditcard_fraud.api.main:app --host 0.0.0.0 --port 8000 --reload
```

- Документация: http://127.0.0.1:8000/docs  
- `GET /health`  
- `POST /predict` — тело `{"features": {"Time": 0.0, "V1": 0.1, ...}}` (все имена из `model_meta.json`).

## Docker (сервинг API)

```bash
docker build -t fraud-api .
docker run --rm -p 8000:8000 -v "$(pwd)/models:/app/models:ro" fraud-api
```

Jupyter в контейнере (если нужен отдельно):

```bash
docker run --rm -p 8888:8888 -v "$(pwd):/app" -w /app fraud-api \
  jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token=''
```

## Тесты

```bash
make test
# или: pip install pytest httpx && python -m pytest tests/pytest -q
```
