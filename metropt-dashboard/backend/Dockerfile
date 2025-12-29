# Use official lightweight Python image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .

# FIX: Install directly from the file so 'evidently' is included
RUN pip install --no-cache-dir -r requirements.txt

# Copy the backend code
COPY . .

# Expose the API port
EXPOSE 8000

# Run the API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]