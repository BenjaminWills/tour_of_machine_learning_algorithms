lint:
	black .

test:
	python -m tests.initialise
	python -m unittest discover -s ./tests
	python -m tests.finalise
