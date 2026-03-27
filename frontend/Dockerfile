# Use Python 3.12 slim
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV BACKEND_URL=http://localhost:8000

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Mesop UI listens on port 32123
EXPOSE 32123

# Run mesop
# It defaults to listening on all interfaces (0.0.0.0) in most setups
CMD ["mesop", "main.py"]
