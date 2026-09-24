# TrafficTrak - common commands.  Python 3.10+.
PY ?= python3
VIDEOS ?= samples
PRED ?= predictions_samples.json

.PHONY: help install install-cpu install-demo weights test lint check predict validate visualize eda prior candidates dev-eval calibrate demo docker

help:
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | sed 's/:.*## /\t/'

install: ## runtime deps (GPU onnxruntime) + dev tools
	$(PY) -m pip install -r requirements.txt -r requirements-dev.txt
install-cpu: ## runtime deps for CPU-only machines
	$(PY) -m pip install -r requirements-cpu.txt -r requirements-dev.txt
install-demo: ## analysis / visualisation / demo deps
	$(PY) -m pip install -r requirements-demo.txt
weights: ## download + verify detector weights (run before offline evaluation)
	bash weights/download.sh

test: ## unit tests
	$(PY) -m pytest
lint: ## code-quality checks
	ruff check .
check: lint test ## lint + tests

predict: ## run solution.detect_events on every sample video -> $(PRED)
	$(PY) scripts/run_local.py --videos $(VIDEOS) --out $(PRED)
validate: ## schema check (+ official `python evaluate.py --pred $(PRED) --validate-only` when evaluate.py exists)
	$(PY) scripts/validate_predictions.py --pred $(PRED) --videos $(VIDEOS)
visualize: ## predictions + risk curves + annotated videos + timelines in outputs/
	$(PY) scripts/run_local.py --videos $(VIDEOS) --out $(PRED) --risk --visualize
eda: ## sample-video EDA in outputs/eda
	$(PY) scripts/analyze_samples.py --videos $(VIDEOS)
prior: ## EDA + learned scene prior (config/scene_prior.npz, config/background_reference.png)
	$(PY) scripts/analyze_samples.py --videos $(VIDEOS) --write-prior
calibrate: ## browser-based geometry calibration tool (outputs/calibration/calibrate.html)
	$(PY) scripts/calibrate_geometry.py html --image outputs/eda/reference_median.jpg
	$(PY) scripts/calibrate_geometry.py render --image outputs/eda/reference_median.jpg
candidates: ## candidate clips + review.csv for manual annotation
	$(PY) scripts/annotate_candidates.py mine --videos $(VIDEOS)
dev-eval: ## score against human-reviewed labels/dev_labels.json
	$(PY) scripts/eval_dev.py --pred $(PRED) --labels labels/dev_labels.json
demo: ## Streamlit upload demo
	streamlit run demo/app.py
docker: ## build the offline evaluation image
	docker build -t traffictrak .
