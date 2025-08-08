# Use an official Python runtime as a parent image.
FROM python:3.10-slim

# Set the working directory inside the container to /code
WORKDIR /code

# Set environment variables for Python and the cache location
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV SENTENCE_TRANSFORMERS_HOME=/tmp/sentence_transformers_cache

# Install system dependencies
RUN apt-get update && apt-get install -y build-essential

# Copy the requirements file into the container first for caching.
COPY requirements.txt .

# Install the Python dependencies.
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# --- FIX ---
# Pre-download the model during the Docker build phase.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"

# Copy the rest of your application's source code into the container
COPY . .

# Expose the port Google Cloud Run provides
EXPOSE $PORT

# Define the command to run your application.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "$PORT"]