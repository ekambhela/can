# Karkive — one-command workflows. See AUDIT.md (reproducibility).
.PHONY: help setup model test benchmark lint run clean

help:
	@echo "make setup      install runtime + dev dependencies"
	@echo "make model      build artifacts/model.joblib from committed GDSC data (seeded)"
	@echo "make test       run the pytest suite"
	@echo "make benchmark  reproduce the baseline -> ensemble accuracy study"
	@echo "make lint       ruff check"
	@echo "make run        start the web app (trains the model on first request if absent)"
	@echo "make clean      remove the built model artifact"

setup:
	python -m pip install -r requirements-dev.txt

# raw (committed) data -> trained model -> metrics.json, in one seeded command.
model:
	python -m model.train

test:
	pytest -q

# leakage-free study behind the benchmark table (decisions on VAL, confirmed on TEST).
benchmark:
	python -m experiments.run_baseline
	python -m experiments.run_ensemble

lint:
	ruff check app.py model tests conftest.py scripts

run:
	uvicorn app:app --reload

clean:
	rm -f artifacts/model.joblib
