PYTHON ?= python3

.PHONY: requirements serve repro test

requirements:
	$(PYTHON) -m pip install -U pip
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e .

serve:
	$(PYTHON) -m uvicorn creditcard_fraud.api.main:app --host 0.0.0.0 --port 8000 --reload

repro:
	dvc repro

test:
	$(PYTHON) -m pytest tests/pytest -q
