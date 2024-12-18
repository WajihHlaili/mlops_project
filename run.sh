#!/bin/bash

# Wait for the FastAPI service to be ready (optional, but useful if Flask depends on FastAPI)
# You can use tools like wait-for-it or just sleep for a few seconds
# Example: wait-for-it is a script that waits for a service to be available.
# For simplicity, we're just adding a short sleep time.
echo "Waiting for FastAPI to be ready..."
sleep 5

# Start FastAPI in the background
uvicorn server:app --host 0.0.0.0 --port 8000 &

# Start Flask app (make sure it's ready after FastAPI is started)
python3 app.py --host 0.0.0.0 --port 8080
