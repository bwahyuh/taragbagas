# Use the official lightweight Python image.
# https://hub.docker.com/_/python
FROM python:3.11-slim

# Allow statements and log messages to be sent straight to the logs
ENV PYTHONUNBUFFERED True

# Install system dependencies required for Pillow, Torch, and psycopg2
# ffmpeg has been removed as it is not needed for video processing.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Copy over the requirements.txt file to leverage the Docker cache
COPY ./requirements.txt /requirements.txt

# Install the dependencies
RUN pip install --no-cache-dir -r /requirements.txt

# Set the working directory in the container
WORKDIR /app
# Copy the rest of the application code
COPY . .

# Expose the port that the application will run on
EXPOSE 8080
# Run the application
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.enableCORS=false"]