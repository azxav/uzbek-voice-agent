PYTHON ?= python
LIMIT ?= 8

.PHONY: test lint ci api eval-asr

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check src tests
	$(PYTHON) -m ruff format --check src tests

ci: lint test

api:
	$(PYTHON) -m uvicorn uzbek_voice_agent.api:app --host 0.0.0.0 --port 8000

eval-asr:
	$(PYTHON) -m uzbek_voice_agent.eval_asr --limit $(LIMIT) --output eval/fleurs_uz_prefix.json
