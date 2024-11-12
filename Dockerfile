# Dockerfile
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy necessary files
COPY src/ /app/src
COPY model/brain_tumor_model.keras /app/model/

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r /app/requirements.txt

# Expose port for Flask
EXPOSE 5000

# Run Flask app
CMD ["python", "/app/src/app.py"]