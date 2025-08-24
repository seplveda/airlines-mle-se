# syntax=docker/dockerfile:1.2
FROM python:3.13

# Set working directory
WORKDIR /app

# Copy production requirements file
COPY production-requirements.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r production-requirements.txt

# Copy source code
COPY challenge/ ./challenge/
COPY data/ ./data/

# Expose port
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "challenge:app", "--host", "0.0.0.0", "--port", "8000"]