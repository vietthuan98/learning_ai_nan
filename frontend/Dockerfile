# Use Python 3.12 slim
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV BACKEND_URL=http://localhost:8000

# Install system dependencies for locales
RUN apt-get update && \
    apt-get install -y \
    locales \
    locales-all && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set locales
ENV LC_ALL=en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US.UTF-8

# Create non-root user and home directory
RUN groupadd -g 900 mesop && useradd -u 900 -m -s /bin/bash -g mesop mesop

# Set working directory to recommended path
WORKDIR /srv/mesop-app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and set ownership
COPY . .
RUN chown -R mesop:mesop /srv/mesop-app

# Switch to non-root user
USER mesop

# Mesop UI listens on port 32123
EXPOSE 32123

# Run mesop through gunicorn with threads and extended timeout
CMD ["gunicorn", "--bind", "0.0.0.0:32123", "--timeout", "120", "--threads", "8", "main:me"]
