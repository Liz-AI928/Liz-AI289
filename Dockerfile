# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /code

# Install system dependencies required by Whisper (ffmpeg)
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg

# Copy the requirements file into the container
COPY ./requirements.txt /code/requirements.txt

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy the rest of the application code into the container
COPY ./api /code/api
COPY ./public /code/public

# Command to run the application
# Hugging Face Spaces default port is 7860
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:7860", "api.app:app"]