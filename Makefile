PYTHON ?= .venv/bin/python
PIP ?= .venv/bin/pip
export PYTHONPATH := .

.PHONY: setup samples dataset prepare train eval baselines infer bench export test lint typecheck docker docker-jetson demo-game

setup:
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

samples:
	$(PYTHON) scripts/fetch_samples.py

dataset:
	$(PYTHON) scripts/download_dataset.py

prepare:
	$(PYTHON) scripts/prepare_dataset.py

train:
	$(PYTHON) scripts/train.py --config configs/train.yaml

eval:
	$(PYTHON) scripts/evaluate.py --weights runs/detect/traffic_lights_yolov8/weights/best.pt

baselines:
	$(PYTHON) scripts/compare_baselines.py

infer:
	$(PYTHON) scripts/infer.py --weights weights/best.pt --source samples --no-show --save outputs/infer

demo-game:
	$(PYTHON) scripts/demo_game.py --weights weights/best.pt --no-show --auto green:4,yellow:2,red:4,green:4 --duration 14 --record docs/demo_game.mp4

bench:
	$(PYTHON) scripts/benchmark.py --weights weights/best.pt --backend pytorch --iters 50

export:
	$(PYTHON) scripts/export_model.py --weights weights/best.pt --format onnx

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check src scripts tests

typecheck:
	$(PYTHON) -m mypy src --ignore-missing-imports

docker:
	docker build -f docker/Dockerfile --target infer -t traffic-light-detect:infer .

docker-jetson:
	docker build -f docker/Dockerfile.jetson -t traffic-light-detect:jetson .
