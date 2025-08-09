# Use NVIDIA's official CUDA base image which includes the necessary GPU drivers.
# This is the key change to enable GPU functionality.
FROM nvidia/cuda:12.1.1-base-ubuntu22.04

# Install Python, pip, venv, and other essential build tools into the base image.
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-venv \
    build-essential \
    git \
 && rm -rf /var/lib/apt/lists/*

# Set python3.10 as the default python3 interpreter.
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Set the working directory inside the container.
WORKDIR /code

# Set environment variables for Python and the model cache location.
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV SENTENCE_TRANSFORMERS_HOME=/tmp/sentence_transformers_cache
ENV NLTK_DATA=/tmp/nltk_data

# Copy the requirements file into the container first to leverage Docker's caching.
COPY requirements.txt .

# Install the Python dependencies from your requirements.txt file.
# Because the container has CUDA, pip will automatically fetch the GPU-enabled
# versions of libraries like PyTorch and faiss-gpu.
RUN pip3 install --no-cache-dir -r requirements.txt

# Pre-download the sentence-transformer model during the build phase.
# This prevents timeouts when the application starts.
RUN python3 -c "from sentence_transformers import SentenceTransformer, CrossEncoder; import nltk; \
    SentenceTransformer('BAAI/bge-small-en-v1.5'); \
    CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2'); \
    nltk.download('punkt_tab', download_dir='/tmp/nltk_data')"
# Copy the rest of your application's source code into the container.
COPY . .

# Expose the port provided by Google Cloud Run.
EXPOSE $PORT

# Define the command to run your application using the installed uvicorn.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
