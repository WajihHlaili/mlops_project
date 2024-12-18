# Use an official Python runtime as a parent image
FROM python:3.11

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required for both Flask and FastAPI
RUN apt-get update && apt-get install -y python3-distutils python3-setuptools ca-certificates python3-dev

# Copy requirements.txt and install dependencies
COPY requirements.txt . 
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files into the container
COPY . /app

# Expose the ports your applications will run on
EXPOSE 8080 8000

# Add a script to run both Flask and FastAPI
COPY run.sh /app/run.sh
RUN chmod +x /app/run.sh

# Set the command to run the script
CMD ["/app/run.sh"]
