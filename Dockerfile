# Use an official Python runtime as a parent image
FROM python:3.7-slim

# Set the working directory in the container to /app
WORKDIR /app

# Add the current directory contents into the container at /app
ADD machine_learning_algorithms /app/machine_learning_algorithms
ADD tests /app/tests
ADD requirements.txt /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Run unit tests when the container launches
RUN python -m tests.initialise
RUN python -m unittest discover -s ./tests
RUN python -m tests.finalise