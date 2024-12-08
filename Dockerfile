# Utiliser une image Python officielle version 3.13
FROM python:3.13-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /app

# Copy all files into the container
COPY . /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && apt-get clean

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the necessary ports
EXPOSE 8080 8000

# Command to run both FastAPI and Flask
CMD ["sh", "-c", "gunicorn -w 2 -b 0.0.0.0:8080 app:fastapi & uvicorn server:app --host 0.0.0.0 --port 8000"]
