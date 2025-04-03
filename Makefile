.PHONY: install train serve test clean

install:
	pip install -r requirements.txt

train:
	python scripts/train_model.py

serve:
	uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

test:
	pytest tests/

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache