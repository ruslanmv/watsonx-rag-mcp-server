# Use an official Python runtime as a parent image
# Using python 3.10, adjust if your code needs a different version
FROM python:3.10-slim

# Set environment variables to prevent Python from buffering stdout/stderr
ENV PYTHONUNBUFFERED=1

# Optional: Set default Flask environment variables
# These can also be set via `docker run -e` or within the .env file if preferred
# ENV FLASK_APP=frontend.py
# ENV FLASK_RUN_HOST=0.0.0.0
# ENV FLASK_RUN_PORT=5001 # Match the port exposed later

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file first to leverage Docker cache
COPY requirements.txt ./

# Install system dependencies that might be needed by Python packages (example: build tools)
# RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*
# Uncomment the above RUN command if pip install fails due to missing build tools for some packages

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code and necessary files into the container
COPY frontend.py .
COPY server.py .
COPY templates ./templates/
COPY static ./static/
# COPY .env .              # <-- REMOVED this line: .env file is NOT copied into the image
COPY documents ./documents/  

# Create the directory for ChromaDB persistence within the image (optional, volume mount is better)
# RUN mkdir ./chroma_db_data

# Make port 5001 available to the world outside this container (the port Flask runs on)
EXPOSE 5001

# Define the command to run the application when the container starts
# Runs the Flask development server (suitable for testing, use gunicorn/waitress for production)
CMD ["python", "frontend.py"]

# --- Production Alternative CMD using Gunicorn ---
# Ensure gunicorn is in requirements.txt if using this
# CMD ["gunicorn", "--bind", "0.0.0.0:5001", "frontend:app"]
# --- Alternative CMD using Waitress for production ---
# Ensure waitress is in requirements.txt if using this