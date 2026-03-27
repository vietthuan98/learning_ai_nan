# Use a standard Python slim image for a smaller footprint
FROM python:3.12-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

# Install system dependencies required for building llama-cpp-python
# build-essential provides gcc/g++, needed for the C++ parts of llama.cpp
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy only the requirements file first to leverage Docker layer caching
COPY requirements.txt .

# Install Python dependencies
# This includes FastAPI, Llama-cpp-python, and LangChain components
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire backend source code into the container
COPY . .

# Ensure the models directory exists
RUN mkdir -p models

# Pre-download the TinyLlama model during the image build process
# This ensures that the model is already included in the image,
# allowing the server to start instantly without needing an internet connection.
RUN python3 -c 'from model_loader import download_model; download_model()'

# Expose the port that the FastAPI server will run on
EXPOSE 8000

# Command to run the FastAPI application using uvicorn
# We use the JSON array form to ensure signals (like SIGTERM) are handled correctly
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

