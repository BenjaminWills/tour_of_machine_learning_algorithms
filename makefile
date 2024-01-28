lint:
	black .

test:
	docker build -t my-python-app .
	docker run my-python-app
