# FROM python:3.10.3-slim-buster

# WORKDIR /app

# COPY requirements.txt requirements.txt

# RUN pip install -r requirements.txt

# COPY . .

# ENV PYTHONUNBUFFERED=1

# ENV HOST 0.0.0.0

# EXPOSE 8080

# CMD ["python", "main.py"]

# Base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

ADD https://storage.googleapis.com/captone-bucket-model123/saved_model/1/saved_model.pb /app

RUN pip install --upgrade pip

# Copy requirements.txt
COPY requirements.txt .

# Install dependencies
RUN pip install -r requirements.txt

# Copy application files
COPY . .

# Expose the application port
EXPOSE 8080

# Run the application
# CMD ["uvicorn", "--host", "0.0.0.0", "--port", "8080", "main:app"]
CMD ["gunicorn", "main:app", "--bind", "0.0.0.0:8080"]