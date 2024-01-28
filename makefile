lint:
	black .

test:
	docker build -t unit-tests .
	docker run unit-tests
